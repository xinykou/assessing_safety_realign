from datasets import load_dataset
import os
import json
# 设置HTTP代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:27999"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27999"


data_size = 2000

if data_size == 2000:
    output_json = f'BeaverTails_preference-2k.json'
else:
    raise ValueError("data_size must be 2000")

dataset =load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", cache_dir="../../../../data/cache")

index = 0
list_data_dict = []
for example in dataset["train"]:  # 26.9k examples
    if (example["is_response_0_safe"] and not example["is_response_1_safe"]) or \
            (not example["is_response_0_safe"] and example["is_response_1_safe"]):
        if index < 2000:
            instance = {}
            instance["instruction"] = example["prompt"]
            instance["input"] = ""
            if example["is_response_0_safe"]:
                instance["chosen"] = example["response_0"]
                instance["rejected"] = example["response_1"]
            else:
                instance["chosen"] = example["response_1"]
                instance["rejected"] = example["response_0"]

            list_data_dict += [instance]
            index += 1
        else:
            break

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(list_data_dict, f, indent=4)

