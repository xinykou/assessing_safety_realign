#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ours
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir
echo "Current working directory: $sub_dir"

export CUDA_VISIBLE_DEVICES=1

prune_type=low_rank
sparsity_ratios=(0.8)
model_names=("mistral_7b" "qwen2_7b")

# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}
do
    for model_name in ${model_names[@]}
    do
      echo "----->sparsity_ratio: ${sparsity_ratio}..."
      echo "--->model_name: ${model_name}..."

      python ./prune_regions/sparsity_ratio_low.py \
            --rank_path ./saves/lora/prune_regions/${model_name}-${prune_type}-${sparsity_ratio}/rank_bottom_${sparsity_ratio} \
            --output_dir ./saves/lora/prune_regions/${model_name}-${prune_type}-${sparsity_ratio} \
            --sparsity_ratio ${sparsity_ratio}
    done
done

