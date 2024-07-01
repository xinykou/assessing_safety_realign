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


main_name=aligned_finetune
alignment_methods=("sft" "simpo" "orpo" "kto")  # 0.01 0.05 0.1 0.2 0.3

# shellcheck disable=SC2068
for align_m in ${alignment_methods[@]}; do
    echo "The alignment method is: ${align_m}"

## finetune for downstream task
python main.py train config/baselines/alignment_techniques/${main_name}-"${align_m}"-n1000_p0.05.yaml
done