import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default="../cache")
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("Output file exists. But no worry, we will overwrite it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

dataset = load_dataset("ag_news", cache_dir='./data/cache')
index = 0
input_data_lst = []
for example in dataset["test"]:
    if index < 500:
        instance = {}
        instance[
            "instruction"] = "Categorize the news article given in the input into one of the 4 categories:\n\nWorld\nSports\nBusiness\nSci/Tech\n"
        instance["input"] = example["text"]
        instance["label"] = example["label"]
        input_data_lst.append(instance)
        index += 1

tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True)
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(args.model_folder,
                                             cache_dir=args.cache_dir,
                                             load_in_8bit=False,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")

if args.lora_folder or args.lora_folder2:
    if args.lora_folder:
        print("Recovering the first LoRA weights..")
        model = PeftModel.from_pretrained(
            model,
            args.lora_folder,
            torch_dtype=torch.bfloat16,
        )
        model = model.merge_and_unload()

    if args.lora_folder2:
        print("Recovering the second LoRA weights..")
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


def query_batch(data_batch):
    prompts = [
        f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['instruction']}\n\n### Input:\n{data['input']}\n\n### Response:"
        for data in data_batch
    ]
    input_dict = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = input_dict['input_ids'].cuda()
    attention_mask = input_dict['attention_mask'].cuda()

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
for i in tqdm(range(0, len(input_data_lst), batch_size), desc="Generating responses for agnews evaluation"):
    data_batch = input_data_lst[i:i + batch_size]
    preds = query_batch(data_batch)
    pred_lst.extend(preds)

output_lst = []
correct = 0
total = 0
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    if input_data["label"] == 0:
        label1 = "World"
        label2 = "world"
    elif input_data["label"] == 1: 
        label1 = "Sports"
        label2 = "sports"
    elif input_data["label"] == 2: 
        label1 = "Business"
        label2 = "business"
    else:
        label1 = "Sci/Tech"
        label2 = "sci"
        
    if label1 in pred or label2 in pred:
        correct += 1
        input_data["correct"] = "true"
    else:
        # print(label + "  " + pred)
        input_data["correct"] = "false"
    total += 1
    output_lst.append(input_data)
print("agnews score: {:.2f}".format(correct/total*100))
output_lst.append("score={}".format(correct/total)*100)
with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)
