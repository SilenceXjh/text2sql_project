import sys
from pathlib import Path
# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text2sql.utils import construct_prompt_with_fk, construct_sft_prompt_1

schema = {
        "tables": {
            "perpetrator": [
                "Perpetrator_ID",
                "People_ID",
                "Date",
                "Year",
                "Location",
                "Country",
                "Killed",
                "Injured"
            ],
            "people": [
                "People_ID",
                "Name",
                "Height",
                "Weight",
                "Home Town"
            ]
        },
        "foreign_keys": [
            [
                "perpetrator.People_ID",
                "people.People_ID"
            ]
        ]
    }

question = "How many people"
prompt = construct_prompt_with_fk(schema, question)

print(prompt)

sft_prompt = construct_sft_prompt_1(schema, question)
print(sft_prompt)