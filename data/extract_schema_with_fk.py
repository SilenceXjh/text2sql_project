import json


output_file = 'db_schemas_with_fk.json'
table_file = '/data0/xjh/spider_data/tables.json'
  

def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {}  #{'table': [col.lower, ..., ]} * -> __all__
        schema["tables"] = {}

        # get columns
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        # column_types = db['column_types']
        for i, table_name in enumerate(table_names_original):
            # table_name = str(table_name.lower())
            columns = []
            for col in column_names_original:
                if col[0] == i:
                    columns.append(col[1])
            schema["tables"][table_name] = columns
        
        # get foreign keys
        foreign_key_pairs = []
        for fk in db["foreign_keys"]:
            idx0 = fk[0]
            idx1 = fk[1]
            t0 = table_names_original[column_names_original[idx0][0]]
            c0 = column_names_original[idx0][1]
            t1 = table_names_original[column_names_original[idx1][0]]
            c1 = column_names_original[idx1][1]
            foreign_key_pairs.append((f"{t0}.{c0}", f"{t1}.{c1}"))
        schema["foreign_keys"] = foreign_key_pairs

        schemas[db_id] = schema

    return schemas


schemas = get_schemas_from_json(table_file)
     
with open(output_file, 'w') as out:
    json.dump(schemas, out, indent=4)