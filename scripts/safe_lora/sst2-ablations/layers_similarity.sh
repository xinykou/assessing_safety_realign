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


dataset_name="sst2"
dataset_selected="n1000_p0.05"
region_method=low_rank  # wanda, wandg, or low_rank

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratios=(0.5)
first_method_name="wandg"
second_method_name="low_rank"

# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}; do
    echo "----->Running with sparsity_ratio=$sparsity_ratio"

taus=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55)
# realign lora
for tau in ${taus[@]}; do
    echo "----->Running with tau=$tau"
    python ./safe_lora/layer_similarity.py \
         --first_lora_layer_distributions ./saves/lora/realign/mask_replace-safe_lora/${first_method_name}-layers_ratio/modified_layers_sparsity_ratio_${sparsity_ratio}-tau_${tau}.txt \
         --second_lora_layer_distributions ./saves/lora/realign/mask_replace-safe_lora/${second_method_name}-layers_ratio/modified_layers_sparsity_ratio_${sparsity_ratio}-tau_${tau}.txt \
         --tau ${tau} \
         --sparsity_ratio ${sparsity_ratio} \
         --output_path ./saves/lora/realign/mask_replace-safe_lora/${first_method_name}_vs_${second_method_name}-similarity

done

done