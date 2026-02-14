import sqlite3
import os
from utils import load_json_data

def normalize(res):
    return sorted(
        [tuple("" if v is None else str(v) for v in row)
         for row in res]
    )

def eval_exec_match(db: str, p_str: str, g_str: str) -> bool:
    """
    return True if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
        except:
            return False

        cursor.execute(g_str)
        q_res = cursor.fetchall()

        if "order by" in g_str.lower():
            return p_res == q_res
        else:
            return normalize(p_res) == normalize(q_res)
    

db_dir = "/data0/xjh/spider_data/test_database"
result_file = "/data0/xjh/text2sql_project/outputs/qwen-coder-1.5B-sft3.json"
result_data = load_json_data(result_file)

total = 0
right = 0
for item in result_data:
    total += 1
    gold_sql = item["gold_sql"]
    gen_sql = item["generated_sql"]
    db_id = item["db_id"]
    db_path = os.path.join(db_dir, db_id, db_id+".sqlite")
    res = eval_exec_match(db_path, gen_sql, gold_sql)
    if res:
        right += 1

print(f"right rate: {right} / {total}")