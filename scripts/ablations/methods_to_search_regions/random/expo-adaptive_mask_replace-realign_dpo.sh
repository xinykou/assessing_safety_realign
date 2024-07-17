#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ablations/methods_to_search_regions
sub_dir=$(dirname "$parent_dir") # ./scripts/ablations
main_dir=$(dirname "$sub_dir") # ./scripts
main_dir=$(dirname "$main_dir") # ./

cd $main_dir
# build alignment matrix
export CUDA_VISIBLE_DEVICES=0


dataset_name="sst2"
dataset_selected="n1000_p0.05"
alignment_types=("expo_dpo_lora") # ("dpo" "kto" "simpo" "orpo" "expo-dpo-lora" "expo-kto-lora" "expo-simpo-lora" "expo-orpo-lora")
fusion_effect=sft_to_dpo-alpha_0.9
region_method=random  # wanda, wandg, or low_rank

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratios=(0.8)
prune_rates=(0.5)
epsilon=0.2
# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}; do
    echo "----->Running with sparsity_ratio=$sparsity_ratio"


    # realign lora
    for prune_rate in ${prune_rates[@]}; do
        echo "----->Running with prune_rate=$prune_rate"
        # shellcheck disable=SC2154
        for alignment_name in ${alignment_types[@]}
        do
            python ./safe_lora/identify_realign.py \
                 --model_path ./saves/lora/sft/checkpoint-125-merged \
                 --lora_path ./saves/lora/baselines/poison_ratio/aligned-finetune-${dataset_selected} \
                 --aligned_path ./saves/lora/"${alignment_name}"/${fusion_effect} \
                 --mask_path ./saves/lora/prune_regions/"${alignment_name}"-${region_method}-"${sparsity_ratio}"/mask_bottom_${sparsity_ratio}.pt \
                 --sparsity_ratio "${sparsity_ratio}" \
                 --prune_rate "${prune_rate}" \
                 --epsilon ${epsilon} \
                 --realign_type adaptive_mask_replace \
                 --output_path ./saves/lora/ablations/methods_to_search_regions/expo-adaptive_mask_replace-safe_lora/${dataset_name}-"${alignment_name}"-${dataset_selected}-${region_method}
        done

    done

done