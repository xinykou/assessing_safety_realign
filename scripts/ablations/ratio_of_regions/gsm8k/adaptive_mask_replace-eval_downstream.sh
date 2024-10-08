#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ablations/probability_vs_value_layer_pruning
sub_dir=$(dirname "$parent_dir") # ./scripts/ablations
main_dir=$(dirname "$sub_dir") # ./scripts
main_dir=$(dirname "$main_dir") # ./

cd $main_dir


dataset_name=gsm8k
alignment_method=dpo
region_method=wanda
data_selected=n1000_p0.05
model_path=./saves/lora/ablations/ratio_of_regions/adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}

export CUDA_VISIBLE_DEVICES=0


# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)
prune_rates=(0.5)
epsilon=0.2
# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}; do
    echo "-----> Running with sparsity_ratio=$sparsity_ratio"

    for prune_rate in "${prune_rates[@]}"; do
        echo "------> Running with prune_rate=$prune_rate"
        python ./evaluation/downstream_task/gsm8k_eval.py \
              --model_folder ./saves/lora/sft/checkpoint-125-merged \
              --lora_folder ${model_path}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}" \
              --output_path ./results/lora/ablations/ratio_of_regions/adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}/sparsity_ratio_${sparsity_ratio}-prune_rate_${prune_rate}_epsilon_${epsilon}-downstream.json \
              --add
    done

done