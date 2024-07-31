#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/safe_lora
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir
# build alignment matrix
export CUDA_VISIBLE_DEVICES=1

model_names=("mistral_7b" "qwen2_7b")
region_method=low_rank  # wanda, wandg, or low_rank

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.8
prune_rates=(0.5)
epsilon=0.2
alpha=0.9

# shellcheck disable=SC2068
for model_name in ${model_names[@]}; do
    echo "----->Running with model_name=$model_name"

# realign lora
for prune_rate in ${prune_rates[@]}; do
    echo "----->Running with prune_rate=$prune_rate"
    python ./safe_lora/identify_realign.py \
         --model_path ./saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged \
         --lora_path ./saves/lora/baselines/model_size/finetune/aligned-finetune-"${model_name}" \
         --aligned_path ./saves/lora/baselines/model_size/align/expo-alpha_${alpha}-"${model_name}" \
         --mask_path ./saves/lora/prune_regions/${model_name}-${region_method}-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt \
         --sparsity_ratio "${sparsity_ratio}" \
         --prune_rate "${prune_rate}" \
         --epsilon ${epsilon} \
         --realign_type adaptive_mask_replace \
         --output_path ./saves/lora/realign/expo-adaptive_mask_replace-${model_name}/${region_method}


done

done