#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts
sub_dir=$(dirname "$parent_dir") # ./

cd $sub_dir


dataset_name=sst2
alignment_method=dpo
data_selected=total1000_n0_p0.05
model_path=./saves/lora/dpo/merged
lora_path=./saves/lora/finetune/${dataset_name}-${alignment_method}/${data_selected}

echo "model_path: ${model_path}"
echo "lora_path: ${lora_path}"
echo "dataset_name: ${dataset_name}"
echo "data_selected: ${data_selected}"

CUDA_VISIBLE_DEVICES=0 python ./evaluation/poison/pred.py \
  --model_folder ${model_path} \
  --lora_folder ${lora_path} \
	--instruction_path BeaverTails \
	--start 0 \
	--end 1000 \
	--output_path ./results/lora/finetune/${dataset_name}-${alignment_method}/${data_selected}-safety.json \

CUDA_VISIBLE_DEVICES=0 python ./evaluation/poison/eval_safety.py \
  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
  --input_path ./results/lora/finetune/${dataset_name}-${alignment_method}/${data_selected}-safety.json \

