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
export CUDA_VISIBLE_DEVICES=4,5


model_names=("mistral_7b" "qwen2_7b")

# shellcheck disable=SC2068
for model_name in ${model_names[@]}; do
    echo "The model_name is: ${model_name}"

    ## finetune for downstream task
    python main.py train config/baselines/model_size/unaligned_finetune-${model_name}.yaml
done