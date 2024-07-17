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

prune_type=wanda
sparsity_ratios=(0.8) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
alignment_types=("expo_dpo_lora") # ("dpo" "kto" "simpo" "orpo" "expo_dpo_lora" "expo_kto_lora" "expo_simpo_lora" "expo_orpo_lora")
fusion_effects=("sft_to_dpo-alpha_0.9")  # "sft_to_dpo-alpha_0.9" "sft_to_kto-alpha_0.9" "sft_to_simpo-alpha_0.9" "sft_to_orpo-alpha_0.9"


## 1. Prune regions
# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]}
do
for i in "${!alignment_types[@]}"
do
  echo "----->sparsity_ratio: ${sparsity_ratio}..."
  echo "--->Alignment type: ${alignment_types[$i]}..."

  alignment_name="${alignment_types[$i]}"
  if [[ "$alignment_name" = *"expo"* ]]; then  # if alignment_name contains "expo"
      modified_alignment_name="${alignment_name}/${fusion_effects[$i]}"
  else
      modified_alignment_name="${alignment_name}"
  fi

  CUDA_VISIBLE_DEVICES=0,1 python ./prune_regions/identify_neurons_or_ranks.py \
       --model_path ./saves/lora/sft/checkpoint-125-merged \
       --lora_path ./saves/lora/"${modified_alignment_name}" \
       --sparsity_ratio ${sparsity_ratio} \
       --prune_method ${prune_type} \
       --data_path ./LLaMA_Factory/data/safety/prune_regions/"${alignment_name}"-safety_regions-filtered.json \
       --output_dir ./saves/lora/prune_regions/"${alignment_name}"-${prune_type}-${sparsity_ratio} \
       --save_mask \
       --nsamples 2000 \




## 1. Prune regions score distribution analysis
#  CUDA_VISIBLE_DEVICES=1 python ./prune_regions/identify_neurons_or_ranks.py \
#       --model_path ./saves/lora/sft/checkpoint-125-merged \
#       --lora_path ./saves/lora/"${alignment}" \
#       --sparsity_ratio ${sparsity_ratio} \
#       --prune_method ${prune_type} \
#       --data_path ./LLaMA_Factory/data/safety/prune_regions/${alignment}-safety_regions-filtered.json \
#       --output_dir ./saves/lora/prune_regions/${alignment}-${prune_type}-${sparsity_ratio} \
#       --dump_wanda_score \
#       --save ./saves/lora/prune_regions/${alignment}-${prune_type}-${sparsity_ratio} \
#       --nsamples 2000
done
done
