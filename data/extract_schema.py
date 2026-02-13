import json


output_file = 'test_db_schemas.json'
table_file = '/data0/xjh/spider_data/test_tables.json'
  

def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {}  #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        # column_types = db['column_types']
        for i, table_name in enumerate(table_names_original):
            # table_name = str(table_name.lower())
            columns = []
            for col in column_names_original:
                if col[0] == i:
                    columns.append(col[1])
            schema[table_name] = columns
        schemas[db_id] = schema

    return schemas


schemas = get_schemas_from_json(table_file)
     
with open(output_file, 'w') as out:
    json.dump(schemas, out, indent=4)