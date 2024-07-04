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

# GPU IDs to use
gpus=(4 5)  # 你可以根据实际情况调整此数组

main_name=vlguard_finetune
dataset_n=("ag_news" "gsm8k")  # sst2 agnews gsm8k

# shellcheck disable=SC2068
for i in ${!dataset_n[@]}; do
    d_name=${dataset_n[$i]}
    gpu_id=${gpus[$i % ${#gpus[@]}]}  # 循环使用 GPU IDs

    echo "The dataset is: ${d_name} on GPU: ${gpu_id}"

    # 启动进程时指定 GPU，并放到后台运行
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py train config/baselines/dataset_name/"${d_name}"-${main_name}-n1000_p0.05.yaml &
done


echo "All tasks completed."
