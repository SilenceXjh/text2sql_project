import sys
from pathlib import Path
# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text2sql.utils import construct_sft_prompt

schema = {
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
}

question = "How many people"

sft_prompt = construct_sft_prompt(schema, question)
print(sft_prompt)