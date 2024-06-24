#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts
sub_dir=$(dirname "$parent_dir") # ./

cd $sub_dir
echo "Current working directory: $sub_dir"

prune_type=wandg
sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
alignment_type=("dpo") # ("dpo" "kto" "simpo" "orpo" "expo-dpo-lora" "expo-kto-lora" "expo-simpo-lora" "expo-orpo-lora")

## 1. Prune regions
# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}
do
for alignment_name in ${alignment_type[@]}
do
  echo "----->sparsity_ratio: ${sparsity_ratio}..."
  echo "--->Alignment type: ${alignment_name}..."

  modified_alignment_name="$alignment_name"
  if [[ "$alignment_name" = *"expo"* ]]; then  # if alignment_name contains "expo"
      modified_alignment_name="${alignment_name}/${fusion_effect}"
  fi

  CUDA_VISIBLE_DEVICES=0,1 python ./prune_regions/identify_neurons_or_ranks.py \
       --model_path ./saves/lora/sft/checkpoint-125-merged \
       --lora_path ./saves/lora/"${modified_alignment_name}" \
       --sparsity_ratio ${sparsity_ratio} \
       --prune_method ${prune_type} \
       --data_path ./LLaMA_Factory/data/safety/prune_regions/${alignment_name}-safety_regions-filtered.json \
       --output_dir ./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio} \
       --save_mask \
       --nsamples 2000 \

done
done
