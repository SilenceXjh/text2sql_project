import json


original_train_dataset_path = "/data0/xjh/spider_data/train_spider.json"

with open(original_train_dataset_path, 'r', encoding='utf-8') as f:
    original_train_data = json.load(f)

simple_train_data = []
for item in original_train_data:
    simple_item = {
        "db_id": item["db_id"],
        "query": item["query"],
        "question": item["question"]
    }
    simple_train_data.append(simple_item)

with open("train_spider_simple.json", "w") as out:
    json.dump(simple_train_data, out, indent=2)