#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/alignment
sub_dir=$(dirname "$parent_dir") # scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir

export WANDB_PROJECT="assessing_safety"
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=$main_dir
#CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config/alignment/SFT.yaml  # using the default config
python main.py train config/baselines/model_size/vaccine-qwen2_7b.yaml

python main.py train config/baselines/model_size/vaccine-mistral_7b.yaml