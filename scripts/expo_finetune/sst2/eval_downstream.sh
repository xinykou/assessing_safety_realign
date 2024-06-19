#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/finetune
sub_dir=$(dirname "$parent_dir") # scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir

dataset_name=sst2
alignment_method=expo_dpo
data_selected=total1000_n0_p0.05
model_path=./saves/lora/expo-dpo/sft_to_dpo-alpha_0.9
lora_path=./saves/lora/finetune/${dataset_name}-${alignment_method}/${data_selected}
#lora_path=/media/data/3/yx/model_merging_v2/saves/lora/finetune/SST2/DPO/n1000_p0
echo "model_path: ${model_path}"
echo "lora_path: ${lora_path}"
echo "dataset_name: ${dataset_name}"
echo "data_selected: ${data_selected}"
#
CUDA_VISIBLE_DEVICES=1 python ./evaluation/downstream_task/sst2_eval.py \
  --model_folder ${model_path} \
  --lora_folder ${lora_path} \
  --data_path ./LLaMA_Factory/data/safety/finetune/sst2/val.json \
	--start 0 \
	--end 1000 \
	--output_path ./results/lora/finetune/${dataset_name}-${alignment_method}/${data_selected}-downstream.json \

