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
echo "Current working directory: $main_dir"

export CUDA_VISIBLE_DEVICES=1

prune_type=low_rank
sparsity_ratios=(0.8)
alignment_types=("expo_sft_lora" "expo_kto_lora" "expo_simpo_lora" "expo_orpo_lora")
fusion_effects=("sft_to_sft-alpha_0.9" "sft_to_kto-alpha_0.9" "sft_to_simpo-alpha_0.9" "sft_to_orpo-alpha_0.9")



# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}
echo "----->sparsity_ratio: ${sparsity_ratio}..."
do
    for i in ${!alignment_types[@]}
    do
      alignment_name="${alignment_types[$i]}"
      echo "--->Alignment type: ${alignment_name}..."

      if [[ "$alignment_name" == *"expo"* ]]; then  # if alignment_name contains "expo"
          modified_alignment_name="${alignment_name}/${fusion_effects[$i]}"
      else
          modified_alignment_name="${alignment_name}"
      fi
      python ./prune_regions/sparsity_ratio_low.py \
            --rank_path ./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio}/rank_bottom_${sparsity_ratio} \
            --output_dir ./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio} \
            --sparsity_ratio ${sparsity_ratio}
    done
done

