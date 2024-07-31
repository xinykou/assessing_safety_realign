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

export CUDA_VISIBLE_DEVICES=0,1
prune_type=low_rank
sparsity_ratios=(0.8) #  0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
model_names=("mistral_7b" "qwen2_7b") # ("dpo" "kto" "simpo" "orpo" "expo_dpo_lora" "expo_kto_lora" "expo_simpo_lora" "expo_orpo_lora")
alpha=0.9
## 1. Prune regions
# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}
do
    for model_name in ${model_names[@]}
    do
        echo "----->sparsity_ratio: ${sparsity_ratio}..."
        echo "----->model_name: ${model_name}..."

        CUDA_VISIBLE_DEVICES=0,1 python ./prune_regions/identify_neurons_or_ranks.py \
             --model_path ./saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged \
             --lora_path ./saves/lora/baselines/model_size/align/expo-alpha_${alpha}-"${model_name}" \
             --sparsity_ratio ${sparsity_ratio} \
             --prune_method ${prune_type} \
             --data_path ./LLaMA_Factory/data/safety/prune_regions/"${model_name}"-safety_regions-filtered.json \
             --output_dir ./saves/lora/prune_regions/"${model_name}"-${prune_type}-${sparsity_ratio} \
             --save_mask \
             --nsamples 2000 \

    done

done
