from datasets import load_dataset
import os
import json

# 设置HTTP代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:27999"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27999"

data_size = 4000
data_safe_size = 2000
data_unsafe_size = 2000

if data_size == 4000:
    output_json = f'BeaverTails_kto-2k.json'
else:
    raise ValueError("data_size must be 4000")

dataset = load_dataset("PKU-Alignment/BeaverTails", cache_dir="../../../../data/cache")

index = 0
list_data_dict = []

current_safe_size = 0
current_unsafe_size = 0


def build_dict(prompt, response, is_safe):
    instance = {"instruction": prompt, "input": "", "output": response, "label": is_safe}
    return instance


for example in dataset["30k_train"]:
    if example["is_safe"] and current_safe_size < data_safe_size:
        instance = build_dict(example["prompt"], example["response"], True)
        list_data_dict.append(instance)
        current_safe_size += 1
    elif not example["is_safe"] and current_unsafe_size < data_unsafe_size:
        instance = build_dict(example["prompt"], example["response"], False)
        list_data_dict.append(instance)
        current_unsafe_size += 1
    else:
        continue

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(list_data_dict, f, indent=4)
