import os
import json
import math

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_json_data, construct_sft_prompt, construct_sft_prompt_1

# -------------------- USER CONFIG --------------------
MODEL_ID = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B" 
SPIDER_DATA_PATH = "/data0/xjh/text2sql_project/data/train_spider_simple.json"
SCHEMA_PATH = "/data0/xjh/text2sql_project/data/db_schemas_with_fk.json"
OUTPUT_DIR = "./spider-full-sft-qwen1.5B"
TRAIN_BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
EVAL_BATCH_SIZE = 4
EPOCHS = 5
LR = 2e-5
MAX_LENGTH = 512
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATE_EXAMPLES = 20  # 每个 epoch 导出多少个 validation 上的生成用于后续第三方评测
LOG_INTERVAL = 50
# -----------------------------------------------------

torch.manual_seed(SEED)

# ------- Dataset wrapper -------
class SpiderDataset(Dataset):
    def __init__(self, custom_data, tokenizer, max_length=512, split="train"):
        self.ds = custom_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        prompt = item[0]
        target = item[1]
        target += self.tokenizer.eos_token
        full = prompt + target

        enc_prompt = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, padding=False, return_tensors=None
        )
        enc_full = self.tokenizer(
            full, truncation=True, max_length=self.max_length, padding=False, return_tensors=None
        )

        input_ids = torch.tensor(enc_full["input_ids"], dtype=torch.long)
        # We'll use labels where prompt tokens are masked (set to -100) so loss is computed only on the response
        labels = input_ids.clone()
        prompt_len = len(enc_prompt["input_ids"])
        labels[:prompt_len] = -100  # mask prompt for causal LM loss

        attention_mask = torch.tensor(enc_full["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ------- collate fn with padding -------
def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        if pad_len > 0:
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        else:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            labels.append(item["labels"])
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels)}

# ------- load dataset & split -------
schemas = load_json_data(SCHEMA_PATH)
spider_data = load_json_data(SPIDER_DATA_PATH)
train_data = []
for spider_item in spider_data:
    sql_query = spider_item["query"]
    db_id = spider_item["db_id"]
    question = spider_item["question"]
    db_schema = schemas[db_id]
    prompt = construct_sft_prompt_1(db_schema, question)
    train_data.append((prompt, sql_query))
train_len = int(len(train_data) * 0.95)
val_data = train_data[train_len:]
train_data = train_data[:train_len]
print(f"train size: {len(train_data)}, val size: {len(val_data)}")

# ------- tokenizer & model -------
print("加载 tokenizer 和模型：", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# 确保 pad_token 存在
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print("tokenizer vocab size:", len(tokenizer))

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

# 单 GPU/CPU 环境下把 model 丢到 device（注意：若 model 很大需用 device_map 或 8bit 加载）
model.to(DEVICE)


# ------- Dataset -> DataLoader -------
train_dataset = SpiderDataset(train_data, tokenizer, max_length=MAX_LENGTH)
val_dataset = SpiderDataset(val_data, tokenizer, max_length=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=(TRAIN_BATCH_SIZE//ACCUMULATION_STEPS), shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

# ------- optimizer & scheduler -------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# 可选：加 LR Scheduler（这里不强制）

# ------- training loop with validation -------
os.makedirs(OUTPUT_DIR, exist_ok=True)
global_step = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    step_in_epoch = 0
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / ACCUMULATION_STEPS
        loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        running_loss += loss.item() * ACCUMULATION_STEPS
        step_in_epoch += 1

        if step_in_epoch % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        if step_in_epoch % 50 == 0:
            avg = running_loss / step_in_epoch
            print(f"Epoch {epoch} step {step_in_epoch} avg_train_loss={avg:.4f}")

    avg_epoch_loss = running_loss / max(1, step_in_epoch)
    print(f"Epoch {epoch} finished. avg_train_loss={avg_epoch_loss:.4f}")

    # ----- validation: compute avg loss & perplexity -----
    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for vb in val_loader:
            vb = {k: v.to(DEVICE) for k, v in vb.items()}
            out = model(**vb)
            vl = out.loss.item()
            val_loss += vl
            val_steps += 1
    avg_val_loss = val_loss / max(1, val_steps)
    try:
        ppl = math.exp(avg_val_loss)
    except OverflowError:
        ppl = float("inf")
    print(f"Epoch {epoch} validation: avg_loss={avg_val_loss:.4f} ppl={ppl:.4f}")

    # ----- 导出部分 validation 的生成结果，供第三方评测（或人工评审）使用 -----
    gen_examples = min(GENERATE_EXAMPLES, len(val_dataset))
    gen_outs = []
    with torch.no_grad():
        for i in range(gen_examples):
            prompt = val_data[i][0]
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            # 生成配置（可按需调整）
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            out_text = tokenizer.decode(gen_tokens[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            gen_outs.append({
                "id": i,
                "prompt": prompt,
                "prediction": out_text,
            })

    # 保存本 epoch 的生成结果（jsonl），可作为第三方评测输入
    gen_path = os.path.join(OUTPUT_DIR, f"val_gen_epoch{epoch}.jsonl")
    with open(gen_path, "w", encoding="utf-8") as fw:
        for item in gen_outs:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(gen_outs)} generated examples to {gen_path}")

    # ----- 保存 model checkpoint（每个 epoch 覆盖一次） -----
    save_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved checkpoint to {save_dir}")

