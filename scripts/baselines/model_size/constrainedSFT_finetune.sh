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

poison_ratios=0.05  #  # 0.[01 0.1 0.2 0.3
model_names=("mistral_7b" "qwen2_7b")
# GPU IDs to use
gpu_ids=(0 1)
# shellcheck disable=SC2068
for i in ${!model_names[@]}; do
    echo "----->Running with model_name=${model_names[$i]}"
    # 启动进程时指定 GPU
    gpu=${gpu_ids[$i]}
    model_name=${model_names[$i]}
    CUDA_VISIBLE_DEVICES=$gpu python main.py train config/baselines/model_size/constrainedSFT_finetune-${model_name}.yaml &
done


