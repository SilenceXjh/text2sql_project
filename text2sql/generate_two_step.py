import json
import os
from openai import OpenAI
from tqdm import tqdm
from utils import load_json_data, load_model, model_generate, ds_api_generate, extract_sql, construct_schema_text_with_fk, construct_prompt_with_select_tables

model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct"
test_dataset_path = "/data0/xjh/spider_data/test.json"
use_ds_api = True
db_schemas_path = "/data0/xjh/text2sql_project/data/test_db_schemas_with_fk.json"
output_file_path = "/data0/xjh/text2sql_project/experiment_outputs/ds-2step.json"
output_dir = "/data0/xjh/text2sql_project/experiment_outputs"


def construct_select_table_prompt(question: str, schema: dict):
    schema_text = construct_schema_text_with_fk(schema)

    prompt = f"""You are an expert SQL generator.
Given the database schema and a natural language question, choose the tables that are related to the question.

### Database Schema:
{schema_text}
### Question:
{question}

Only output the table names that are related to the question in the following form:
[table_name1, table_name2, ...]
Table names should be exactly the same as the table names in the schema. Don't output any other content.
"""
    return prompt




def generate_all():
    if use_ds_api:
        client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    else:
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
        
        # 选表
        select_table_prompt = construct_select_table_prompt(question, schema)
        if use_ds_api:
            generated_text = ds_api_generate(select_table_prompt, client)
        else:
            generated_text = model_generate(model, tokenizer, select_table_prompt)

        start_index = generated_text.find("[")
        end_index = generated_text.find("]")
        if start_index == -1 or end_index == -1:
            print(f"{i}: select table error: {generated_text}")
            continue
        valid_content = generated_text[start_index+1:end_index]
        table_name_set = set()
        for table_name in valid_content.split(','):
            table_name_set.add(table_name.strip())
        table_names = list(table_name_set)
        # print("table names:", table_names)
        
        # 生成sql
        prompt = construct_prompt_with_select_tables(schema, question, table_names)
        # print("prompt:")
        # print(prompt)
        if use_ds_api:
            generated_text = ds_api_generate(prompt, client)
        else:
            generated_text = model_generate(model, tokenizer, prompt)
        extracted_sql = extract_sql(generated_text)
        output_item = {
            "db_id": db_id,
            "question": question,
            "full_prompt:": prompt,
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