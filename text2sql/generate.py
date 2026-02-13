import json
import os
from tqdm import tqdm
from utils import load_json_data, load_model, construct_full_prompt, model_generate, extract_sql

model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct"
test_dataset_path = "/data0/xjh/spider_data/test.json"
db_schemas_path = "/data0/xjh/text2sql_project/data/test_db_schemas.json"
output_file_path = "/data0/xjh/text2sql_project/outputs/qwen-coder-7B.json"
output_dir = "/data0/xjh/text2sql_project/outputs"

def generate_all():
    tokenizer, model = load_model(model_path)
    test_data = load_json_data(test_dataset_path)
    print(f"load {len(test_data)} test data")
    db_schemas = load_json_data(db_schemas_path)
    output_content = []

    for i, item in tqdm(enumerate(test_data)):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        schema = db_schemas[db_id]
        full_prompt = construct_full_prompt(question, schema)
        generated_text = model_generate(model, tokenizer, full_prompt)
        extracted_sql = extract_sql(generated_text)
        output_item = {
            "db_id": db_id,
            "question": question,
            "full_prompt:": full_prompt,
            "gold_sql": gold_sql,
            "generated_sql": extracted_sql
        }
        output_content.append(output_item)
        with open(os.path.join(output_dir, f"task_{i}.json"), 'w', encoding='utf-8') as f1:
            json.dump(output_item, f1, indent=2)
        
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_content, f, indent=2)

def main():
    generate_all()

if __name__ == "__main__":
    main()