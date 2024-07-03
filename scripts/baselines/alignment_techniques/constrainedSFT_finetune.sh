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
alignment_methods=("sft" "orpo" "kto" "simpo" "dpo")  #

# GPU IDs to use
gpus=(5 1 2)  # 你可以根据实际情况调整此数组

# shellcheck disable=SC2068
for i in ${!alignment_methods[@]}; do
    align_m=${alignment_methods[$i]}
    gpu_id=${gpus[$i % ${#gpus[@]}]}  # 循环使用 GPU IDs

    echo "The alignment method is: ${align_m} on GPU: ${gpu_id}"

    # 启动进程时指定 GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py train config/baselines/alignment_techniques/${main_name}-${align_m}-n1000_p0.05.yaml &
done

# 等待所有后台任务完成
wait

echo "All tasks completed."
