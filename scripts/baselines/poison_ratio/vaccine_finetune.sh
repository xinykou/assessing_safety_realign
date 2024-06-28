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

export WANDB_PROJECT="assessing_safety"

export PYTHONPATH=$subsub_dir
export CUDA_VISIBLE_DEVICES=0,1
poison_ratios=(0.01 0.05 0.1 0.15 0.2 0.3) # 0.01 0.05 0.1 0.15 0.2 0.3
## finetune for downstream task
# shellcheck disable=SC2034
for poison_ratio in "${poison_ratios[@]}"
do
python main.py train config/baselines/poison_ratio/vaccine_finetune-n1000_p${poison_ratio}.yaml
done