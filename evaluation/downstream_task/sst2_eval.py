import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
import random
import numpy as np

# 设置随机种子函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--output_path", default='')
parser.add_argument("--cache_dir", default="")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)  # 添加随机种子参数
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=500)

args = parser.parse_args()
print(args)

set_seed(args.seed)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset
dataset = load_dataset("sst2", cache_dir='./data/cache')

index = 0
end_loc = 500 if args.end == 500 else args.end
start_loc = 0 if args.start == 0 else args.start
input_data_lst = []
for example in dataset["validation"]:
    if end_loc > index >= start_loc:
        if index < 3:
            print(f"the {index}th example: {example}")
        instance = {}
        instance["instruction"] = "Analyze the sentiment of the input, and respond only positive or negative"
        instance["input"] = example["sentence"]
        instance["label"] = example["label"]
        input_data_lst += [instance]
        index += 1
    else:
        index += 1

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True)
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(args.model_folder,
                                             cache_dir=args.cache_dir,
                                             load_in_8bit=False,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto"
                                             )

if args.lora_folder != "" or args.lora_folder2 != "":
    if args.lora_folder != "":
        print("Recover the first LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder,
            torch_dtype=torch.bfloat16,
        )
        model = model.merge_and_unload()

    if args.lora_folder2 != "":
        print("Recover the second LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder2,
            torch_dtype=torch.bfloat16,
        )
        model = model.merge_and_unload()
        print(model)
else:
    print("No LoRA weights recovered !!!")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def query(data_batch):
    prompts = [
        f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['instruction']}\n\n### Input:\n{data['input']}\n\n### Response:"
        for data in data_batch
    ]
    input_dict = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = input_dict['input_ids'].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in generation_output]
    results = [output.split("### Response:")[1].strip() if "### Response:" in output else output for output in outputs]
    res = [r.split("### Instruction:")[0].strip() if "### Instruction:" in r else r for r in results]
    return res


batch_size = args.batch_size
pred_lst = []
for i in tqdm(range(0, len(input_data_lst), batch_size), desc="Generating responses for sst2 evaluation"):
    data_batch = input_data_lst[i:i + batch_size]
    preds = query(data_batch)
    pred_lst.extend(preds)

output_lst = []
correct = 0
total = 0
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    if input_data["label"]:
        label1 = "positive"
        label2 = "Positive"
    else:
        label1 = "negative"
        label2 = "Negative"
        
    if label1 in pred or label2 in pred:
        correct += 1
        input_data["correct"] = "true"
    else:
        input_data["correct"] = "false"
    total += 1
    output_lst.append(input_data)
print("sst2 score: {:.2f}".format(correct/total*100))
output_lst.append("score={:.2f}".format(correct/total*100))
with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)
