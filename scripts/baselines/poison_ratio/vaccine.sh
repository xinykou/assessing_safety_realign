#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts /baselines
sub_dir=$(dirname "$parent_dir") # ./scripts
subsub_dir=$(dirname "$sub_dir") # .


cd $subsub_dir
echo "Current working directory: $subsub_dir"

# vaccine build for defending against attack

export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=$subsub_dir
export WANDB_PROJECT="assessing_safety"

python main.py train config/baselines/poison_ratio/vaccine.yaml

