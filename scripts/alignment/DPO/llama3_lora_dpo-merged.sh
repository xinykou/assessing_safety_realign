#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/aligment
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # .

cd $main_dir

model_path=./saves/lora/sft/checkpoint-125-merged
save_path=./saves/lora/dpo

export CUDA_VISIBLE_DEVICES=0

#python main.py export config/alignment/llama3_lora_sft-merged.yaml

python export_merged.py \
    --org_model_path $model_path \
    --lora_path $save_path \
    --save_path $save_path/merged