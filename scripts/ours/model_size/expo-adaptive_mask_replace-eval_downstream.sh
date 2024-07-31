#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ours
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd "$main_dir"

region_method=low_rank

export CUDA_VISIBLE_DEVICES=1

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.8
prune_rates=(0.5)
epsilon=0.2
model_names=("mistral_7b" "qwen2_7b")
# 生成预测文件
for model_name in "${model_names[@]}"; do
    echo "-----> Running with model_names=$model_names"

    for prune_rate in "${prune_rates[@]}"; do
        echo "------> Running with prune_rate=$prune_rate"
        # 将文件列表传递给 Python 脚本
        python ./evaluation/downstream_task/sst2_eval.py \
              --model_folder ./saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged \
              --lora_folder ./saves/lora/realign/expo-adaptive_mask_replace-${model_name}/${region_method}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"  \
              --start 0 \
              --end 1000 \
              --output_path ./results/lora/realign/expo-adaptive_mask_replace-${model_name}/${region_method}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-downstream.json \
              --add

    done
done



