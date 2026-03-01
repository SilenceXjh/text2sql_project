import json
import os
from openai import OpenAI
from tqdm import tqdm
from utils import load_json_data, ds_api_generate, extract_sql, construct_normal_prompt, construct_prompt_with_fk

test_dataset_path = "/data0/xjh/spider_data/test.json"
has_fk = False
db_schemas_path = "/data0/xjh/text2sql_project/data/test_db_schemas.json"
# has_fk = True
# db_schemas_path = "/data0/xjh/text2sql_project/data/test_db_schemas_with_fk.json"
output_file_path = "/data0/xjh/text2sql_project/experiment_outputs/ds.json"
output_dir = "/data0/xjh/text2sql_project/experiment_outputs"

def generate_all():
    client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    test_data = load_json_data(test_dataset_path)
    print(f"load {len(test_data)} test data")
    db_schemas = load_json_data(db_schemas_path)
    output_content = []

    for i, item in tqdm(enumerate(test_data)):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        schema = db_schemas[db_id]
        # full_prompt = construct_full_prompt(question, schema)
        if has_fk:
            full_prompt = construct_prompt_with_fk(schema, question)
        else:
            full_prompt = construct_normal_prompt(schema, question)
        generated_text = ds_api_generate(full_prompt, client)
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