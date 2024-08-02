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
export CUDA_VISIBLE_DEVICES=0

model_names=("llama3_8b" "mistral_7b" "qwen2_7b")  # "mistral_7b" "qwen2_7b"
region_method=low_rank  # wanda, wandg, or low_rank

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.8
prune_rate=0.5
poison_ratios=(0.01 0.05 0.1 0.2 0.3)
epsilon=0.2
alpha=0.9

# shellcheck disable=SC2068
for model_name in ${model_names[@]}; do

# realign lora
for poison_ratio in ${poison_ratios[@]}; do
    echo "----->Running with poison_ratio=$poison_ratio"
    data_selected="n1000_p${poison_ratio}"
    if [[ "${model_name}" == *qwen2_7b* ]] || [[ "${model_name}" == *mistral_7b* ]]; then
        model_path=./saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged
        data_selected="aligned-finetune-"${model_name}"-n1000_p${poison_ratio}"
        aligned_path=./saves/lora/baselines/model_size/align/expo-alpha_${alpha}-"${model_name}"
        mask_path=${model_name}-${region_method}-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt
        echo "----->Running with model_name=$model_name"
    else
        model_path=./saves/lora/sft/checkpoint-125-merged
        data_selected="aligned-finetune-n1000_p${poison_ratio}"
        aligned_path=./saves/lora/expo_dpo_lora/sft_to_dpo-alpha_0.9
        mask_path=expo_dpo_lora-${region_method}-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt
        echo "----->Running with model_name=$model_name"
    fi

    python ./safe_lora/identify_realign.py \
         --model_path ${model_path} \
         --lora_path ./saves/lora/baselines/clean_data/${data_selected} \
         --aligned_path ${aligned_path} \
         --mask_path ./saves/lora/prune_regions/${mask_path} \
         --sparsity_ratio "${sparsity_ratio}" \
         --prune_rate "${prune_rate}" \
         --epsilon ${epsilon} \
         --realign_type adaptive_mask_replace \
         --output_path ./saves/lora/baselines/clean_data/expo-adaptive_mask_replace-"${data_selected}"


done

done