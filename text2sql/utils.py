import json
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, adapter_path: str = None):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"加载 peft 模型 {adapter_path}")
    model.eval()
    print("模型加载完成!")
    return tokenizer, model


def load_json_data(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def construct_full_prompt(question: str, schema: dict):
    full_prompt = "Database Schema:\n"
    for table_name in schema:
        columns = schema[table_name]
        full_prompt += f"Table: {table_name}\n"
        full_prompt += f"Columns: {columns}\n"
    
    full_prompt += f"Question: {question}\n"
    full_prompt += "Write a valid SQL query to answer the question. Only output the SQL query."

    return full_prompt


def model_generate(model, tokenizer, prompt: str, is_instruct = True, max_new_tokens=1024):
    if is_instruct:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if prompt in generated_text:
        generated_text = generated_text[len(prompt):].strip()
    else:
        generated_text = generated_text.strip()
    
    return generated_text


def extract_sql(generated_text: str):
    """从生成的文本中提取 sql 语句"""
    if "```sql" in generated_text:
        code = generated_text.split("```sql")[1].split("```")[0].strip()
    elif "```" in generated_text:
        code = generated_text.split("```")[1].split("```")[0].strip()
    else:
        code = generated_text.strip()

    return code