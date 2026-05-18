import sqlite3
import os
import time
from collections import Counter
from typing import Optional, Tuple
from utils import load_json_data


def normalize(res):
    return sorted(
        [tuple("" if v is None else str(v) for v in row)
         for row in res]
    )

def _sql_error_category(err: BaseException) -> str:
    msg = str(err).lower()
    exc_name = type(err).__name__
    if "query timeout" in msg:
        return "timeout"
    if not isinstance(err, sqlite3.Error):
        return f"non_sqlite_error:{exc_name}"
    if "you can only execute one statement at a time" in msg:
        return "multi_statement"
    if "selects to the left and right of" in msg and "do not have the same number of result columns" in msg:
        return "setop_column_mismatch"
    if "order by clause should come after" in msg and ("union" in msg or "intersect" in msg or "except" in msg):
        return "setop_order_by_position"
    if "syntax error" in msg:
        return "syntax_error"
    if "incomplete input" in msg:
        return "incomplete_input"
    if "unrecognized token" in msg:
        return "unrecognized_token"
    if "no such table" in msg:
        return "no_such_table"
    if "no such column" in msg:
        return "no_such_column"
    if "ambiguous column name" in msg:
        return "ambiguous_column"
    if "no such function" in msg:
        return "no_such_function"
    if "wrong number of arguments" in msg:
        return "wrong_args"
    if "distinct aggregates must have exactly one argument" in msg:
        return "distinct_agg_args"
    if "misuse of aggregate" in msg:
        return "misuse_aggregate"
    if "misuse of window function" in msg:
        return "misuse_window"
    if "datatype mismatch" in msg:
        return "datatype_mismatch"
    if "not authorized" in msg:
        return "not_authorized"
    if "too many terms in compound select" in msg:
        return "too_many_terms"
    if "parser stack overflow" in msg:
        return "parser_overflow"
    if "divide by zero" in msg:
        return "divide_by_zero"
    if "constraint failed" in msg:
        return "constraint_failed"
    return f"sqlite_other:{exc_name}"


def eval_exec_match(db: str, p_str: str, g_str: str) -> Tuple[str, Optional[BaseException]]:
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        start_time = time.time()

        def progress_handler():
            if time.time() - start_time > 3:
                raise Exception("Query timeout")
            return 0

        conn.set_progress_handler(progress_handler, 10000)
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
        except Exception as e:
            return "pred_error", e
        finally:
            conn.set_progress_handler(None, 0)

        try:
            cursor.execute(g_str)
            q_res = cursor.fetchall()
        except Exception as e:
            return "gold_error", e

        if "order by" in g_str.lower():
            return ("match" if p_res == q_res else "result_mismatch"), None
        p_norm = normalize(p_res)
        q_norm = normalize(q_res)
        return ("match" if p_norm == q_norm else "result_mismatch"), None
    

db_dir = "/data0/xjh/spider_data/test_database"
result_file = "/data0/xjh/text2sql_project/experiment_outputs/qwen-coder-7B-fk-exec-refine.json"
test_data_path = "/data0/xjh/text2sql_project/data/test_spider_simple.json"

result_data = load_json_data(result_file)
test_data = load_json_data(test_data_path)

total = 0
right = 0
right_array = [0, 0, 0]
outcome_counter = Counter()
pred_error_category = Counter()
pred_error_exception = Counter()
pred_error_other_message = Counter()
gold_error_category = Counter()

for i, item in enumerate(result_data):
    total += 1
    test_problem = test_data[i]
    gold_sql = item["gold_sql"]
    gen_sql = item["generated_sql"]
    db_id = item["db_id"]

    assert gold_sql == test_problem["query"]
    table_count = test_problem["table_count"]

    db_path = os.path.join(db_dir, db_id, db_id+".sqlite")
    outcome, err = eval_exec_match(db_path, gen_sql, gold_sql)
    outcome_counter[outcome] += 1
    if outcome == "match":
        right += 1
        if table_count == 1:
            right_array[0] += 1
        elif table_count == 2:
            right_array[1] += 1
        else:
            right_array[2] += 1
    elif outcome == "pred_error" and err is not None:
        cat = _sql_error_category(err)
        pred_error_category[cat] += 1
        pred_error_exception[type(err).__name__] += 1
        if cat.startswith("sqlite_other:"):
            pred_error_other_message[str(err)] += 1
    elif outcome == "gold_error" and err is not None:
        gold_error_category[_sql_error_category(err)] += 1

print(f"right rate: {right} / {total}")
print(right_array)
print("outcome breakdown:")
for k, v in outcome_counter.most_common():
    print(f"  {k}: {v}")
if pred_error_category:
    print("pred_error categories:")
    for k, v in pred_error_category.most_common(30):
        print(f"  {k}: {v}")
if pred_error_exception:
    print("pred_error exception types:")
    for k, v in pred_error_exception.most_common():
        print(f"  {k}: {v}")
if pred_error_other_message:
    print("pred_error other sqlite messages (top 20):")
    for k, v in pred_error_other_message.most_common(20):
        print(f"  {k}: {v}")
if gold_error_category:
    print("gold_error categories:")
    for k, v in gold_error_category.most_common(30):
        print(f"  {k}: {v}")
