import sqlite3
import os
import time
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
    # print("p_str:", p_str)
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        start_time = time.time()
        # 设置 progress handler
        def progress_handler():
            if time.time() - start_time > 3:
                print(f"[TIMEOUT >3s] SQL: {p_str}")
                raise Exception("Query timeout")
            return 0

        conn.set_progress_handler(progress_handler, 10000)

        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
        except:
            return False
        finally:
            conn.set_progress_handler(None, 0)

        cursor.execute(g_str)
        q_res = cursor.fetchall()

        if "order by" in g_str.lower():
            return p_res == q_res
        else:
            return normalize(p_res) == normalize(q_res)
    

db_dir = "/data0/xjh/spider_data/test_database"
result_file = "/data0/xjh/text2sql_project/experiment_outputs/qwen-coder-1.5B-full-sft2.json"
test_data_path = "/data0/xjh/text2sql_project/data/test_spider_simple.json"

result_data = load_json_data(result_file)
test_data = load_json_data(test_data_path)

total = 0
right = 0
right_array = [0,0,0]

for i, item in enumerate(result_data):
    total += 1
    test_problem = test_data[i]
    gold_sql = item["gold_sql"]
    gen_sql = item["generated_sql"]
    db_id = item["db_id"]

    assert gold_sql == test_problem["query"]
    table_count = test_problem["table_count"]

    db_path = os.path.join(db_dir, db_id, db_id+".sqlite")
    res = eval_exec_match(db_path, gen_sql, gold_sql)
    if res:
        right += 1
        if table_count == 1:
            right_array[0] += 1
        elif table_count == 2:
            right_array[1] += 1
        else:
            right_array[2] += 1

print(f"right rate: {right} / {total}")
print(right_array)