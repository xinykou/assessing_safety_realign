#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts
sub_dir=$(dirname "$parent_dir") # ./

cd $sub_dir

model_path=./pretrained_model/Meta-Llama-3-8B

export CUDA_VISIBLE_DEVICES=0

python main.py export config/alignment/llama3_lora_sft-merged.yaml