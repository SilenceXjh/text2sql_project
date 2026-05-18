import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm

from utils import (
    load_json_data,
    load_model,
    model_generate,
    extract_sql,
    ds_api_generate,
    construct_schema_text,
    construct_schema_text_with_fk,
)


model_path = "/data0/xjh/Qwen2.5-Coder-0.5B-Instruct/"
test_dataset_path = "/data0/xjh/spider_data/test.json"
db_dir = "/data0/xjh/spider_data/test_database"
use_ds_api = False
has_fk = True
db_schemas_path = "/data0/xjh/text2sql_project/data/test_db_schemas_with_fk.json"
output_dir = "/data0/xjh/text2sql_project/experiment_outputs"
output_file_path = "/data0/xjh/text2sql_project/experiment_outputs/qwen-coder-0.5B-fk-exec-refine.json"

max_refine_rounds = 3
exec_timeout_s = 3


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


def _strip_sql_leading_comments(sql: str) -> str:
    s = sql.lstrip()
    while True:
        if s.startswith("--"):
            nl = s.find("\n")
            if nl == -1:
                return ""
            s = s[nl + 1 :].lstrip()
            continue
        if s.startswith("/*"):
            end = s.find("*/")
            if end == -1:
                return ""
            s = s[end + 2 :].lstrip()
            continue
        return s


def _validate_sql_is_single_select(sql: str) -> Optional[Tuple[str, str]]:
    s = sql.strip()
    if not s:
        return "empty_sql", "SQL is empty."
    if s.count(";") >= 2:
        return "multi_statement", "SQL contains multiple semicolons; only one statement is allowed."
    if s.count(";") == 1 and not s.rstrip().endswith(";"):
        return "multi_statement", "SQL contains multiple statements; only one statement is allowed."
    s0 = _strip_sql_leading_comments(s)
    s0 = s0.lstrip()
    lowered = s0.lower()
    if lowered.startswith("("):
        while lowered.startswith("("):
            s0 = s0[1:].lstrip()
            lowered = s0.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return "non_select_statement", "Only SELECT/CTE queries are allowed."
    return None


def try_execute_sqlite(db_path: str, sql: str, timeout_s: int) -> Tuple[bool, Optional[BaseException]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        start_time = time.time()

        def progress_handler():
            if time.time() - start_time > timeout_s:
                raise Exception("Query timeout")
            return 0

        conn.set_progress_handler(progress_handler, 10000)
        try:
            cursor.execute(sql)
            cursor.fetchall()
            return True, None
        except Exception as e:
            return False, e
        finally:
            conn.set_progress_handler(None, 0)


def schema_to_text(schema: Dict[str, Any], has_fk_flag: bool) -> str:
    if has_fk_flag:
        return construct_schema_text_with_fk(schema)
    return construct_schema_text(schema)


def build_refine_prompt(schema_text: str, question: str, prev_sql: str, error_message: str, error_category: str) -> str:
    return f"""You are an expert SQL generator.
Given the database schema and a natural language question, fix the SQL query so it executes successfully on SQLite.

### Database Schema:
{schema_text}
### Question:
{question}
### Previous SQL:
{prev_sql}
### SQLite Error:
[{error_category}] {error_message}

Please output a single corrected SQL query that matches the question and schema.
Only output the SQL query.
"""


def generate_one_with_exec_refine(
    model,
    tokenizer,
    client,
    schema_text: str,
    question: str,
    db_path: str,
    max_rounds: int,
    timeout_s: int,
) -> Dict[str, Any]:
    init_prompt = f"""You are an expert SQL generator.
Given the database schema and a natural language question, generate a correct SQL query.

### Database Schema:
{schema_text}
### Question:
{question}

Please only output the SQL query.
"""
    if client is not None:
        generated_text = ds_api_generate(init_prompt, client)
    else:
        generated_text = model_generate(model, tokenizer, init_prompt)
    sql = extract_sql(generated_text)

    rounds = []
    for round_idx in range(max_rounds + 1):
        validation = _validate_sql_is_single_select(sql)
        if validation is not None:
            err_cat, err_msg = validation
            rounds.append(
                {
                    "round": round_idx,
                    "sql": sql,
                    "exec_ok": False,
                    "error_category": err_cat,
                    "error_message": err_msg,
                }
            )
            if round_idx == max_rounds:
                break
            refine_prompt = build_refine_prompt(schema_text, question, sql, err_msg, err_cat)
            if client is not None:
                generated_text = ds_api_generate(refine_prompt, client)
            else:
                generated_text = model_generate(model, tokenizer, refine_prompt)
            sql = extract_sql(generated_text)
            continue

        ok, err = try_execute_sqlite(db_path, sql, timeout_s=timeout_s)
        if ok:
            rounds.append(
                {
                    "round": round_idx,
                    "sql": sql,
                    "exec_ok": True,
                    "error_category": None,
                    "error_message": None,
                }
            )
            break

        err_cat = _sql_error_category(err) if err is not None else "unknown_error"
        err_msg = str(err) if err is not None else "Unknown error."
        rounds.append(
            {
                "round": round_idx,
                "sql": sql,
                "exec_ok": False,
                "error_category": err_cat,
                "error_message": err_msg,
            }
        )
        if round_idx == max_rounds:
            break
        refine_prompt = build_refine_prompt(schema_text, question, sql, err_msg, err_cat)
        if client is not None:
            generated_text = ds_api_generate(refine_prompt, client)
        else:
            generated_text = model_generate(model, tokenizer, refine_prompt)
        sql = extract_sql(generated_text)

    final_sql = rounds[-1]["sql"] if rounds else sql
    exec_ok = bool(rounds and rounds[-1]["exec_ok"])
    return {
        "generated_sql": final_sql,
        "exec_ok": exec_ok,
        "rounds": rounds,
    }


def generate_all():
    if use_ds_api:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        tokenizer, model = None, None
    else:
        client = None
        tokenizer, model = load_model(model_path)

    test_data = load_json_data(test_dataset_path)
    print(f"load {len(test_data)} test data")
    db_schemas = load_json_data(db_schemas_path)
    os.makedirs(output_dir, exist_ok=True)
    output_content = []
    total = 0
    first_try_error = 0
    first_try_ok = 0
    fixed = 0
    still_error = 0

    for i, item in tqdm(enumerate(test_data)):
        total += 1
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        schema = db_schemas[db_id]
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        schema_text = schema_to_text(schema, has_fk)
        gen_res = generate_one_with_exec_refine(
            model=model,
            tokenizer=tokenizer,
            client=client,
            schema_text=schema_text,
            question=question,
            db_path=db_path,
            max_rounds=max_refine_rounds,
            timeout_s=exec_timeout_s,
        )

        rounds = gen_res.get("rounds") or []
        if rounds and rounds[0].get("exec_ok"):
            first_try_ok += 1
        else:
            first_try_error += 1
            if gen_res.get("exec_ok"):
                fixed += 1
            else:
                still_error += 1

        output_item = {
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": gen_res["generated_sql"],
            "exec_ok": gen_res["exec_ok"],
            "rounds": rounds,
        }
        output_content.append(output_item)
        with open(os.path.join(output_dir, f"task_{i}.json"), "w", encoding="utf-8") as f1:
            json.dump(output_item, f1, indent=2, ensure_ascii=False)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_content, f, indent=2, ensure_ascii=False)

    print("exec-refine summary:")
    print(f"  total: {total}")
    print(f"  first_try_ok: {first_try_ok}")
    print(f"  first_try_error: {first_try_error}")
    if first_try_error:
        print(f"  fixed: {fixed} (from first_try_error)")
        print(f"  still_error: {still_error} (from first_try_error)")


def main():
    generate_all()


if __name__ == "__main__":
    main()
