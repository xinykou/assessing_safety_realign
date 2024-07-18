#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ablations/expo_sparsity
sub_dir=$(dirname "$parent_dir") # ./scripts/ablations
main_dir=$(dirname "$sub_dir") # ./scripts
main_dir=$(dirname "$main_dir") # ./

cd $main_dir
# build alignment matrix
export CUDA_VISIBLE_DEVICES=1


dataset_name="ag_news"
dataset_selected="n1000_p0.05"
fusion_effect=sft_to_dpo-alpha_0.9
alignment_type="expo_dpo_lora"  # dpo
region_method=wanda  # wanda, wandg, or low_rank

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratios=(0.1 0.3 0.5 0.7 0.9)
prune_rates=(0.5)
epsilon=0.2
# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}; do
    echo "----->Running with sparsity_ratio=$sparsity_ratio"

    if [[ "$alignment_type" = *"expo"* ]]; then
        modified_alignment_name="${alignment_type}/${fusion_effect}"
    else
        modified_alignment_name="${alignment_type}"
    fi

    # realign lora
    for prune_rate in ${prune_rates[@]}; do
        echo "----->Running with prune_rate=$prune_rate"
        python ./safe_lora/identify_realign.py \
             --model_path ./saves/lora/sft/checkpoint-125-merged \
             --lora_path ./saves/lora/baselines/dataset_name/${dataset_name}-aligned-finetune-${dataset_selected} \
             --aligned_path ./saves/lora/${modified_alignment_name} \
             --mask_path ./saves/lora/prune_regions/${alignment_type}-${region_method}-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt \
             --sparsity_ratio "${sparsity_ratio}" \
             --prune_rate "${prune_rate}" \
             --epsilon ${epsilon} \
             --realign_type adaptive_mask_replace \
             --output_path ./saves/lora/ablations/expo_sparsity/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_type}-${dataset_selected}-${region_method}


    done
done