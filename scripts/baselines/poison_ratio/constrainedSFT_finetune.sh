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

main_name=constrainedSFT_finetune
poison_ratios=(0.01 0.1 0.2 0.3)  #  # 0.01 0.1 0.2 0.3

# GPU IDs to use
gpus=(0 1 2 3)  # 你可以根据实际情况调整此数组

# shellcheck disable=SC2068
for i in ${!poison_ratios[@]}; do
    p_ratio=${poison_ratios[$i]}
    gpu_id=${gpus[$i % ${#gpus[@]}]}  # 循环使用 GPU IDs

    echo "The alignment method is: ${p_ratio} on GPU: ${gpu_id}"

    # 启动进程时指定 GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py train config/baselines/poison_ratio/${main_name}-dpo-n1000_p"${p_ratio}".yaml &
done

# 等待所有后台任务完成
wait

echo "All tasks completed."
