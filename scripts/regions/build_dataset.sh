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

# 同时导入两个路径
export PYTHONPATH="${sub_dir}:${sub_dir}/evaluation/poison"
export CUDA_VISIBLE_DEVICES=0,1

# "sft_to_sft-alpha_0.9"  "sft_to_dpo-alpha_0.9" "sft_to_kto-alpha_0.9" "sft_to_orpo-alpha_0.9" "sft_to_simpo-alpha_0.9"
fusion_effects=("sft_to_kto-alpha_0.9" "sft_to_orpo-alpha_0.9" "sft_to_simpo-alpha_0.9")
# "dpo" "kto" "orpo" "simpo" "expo_sft_lora" "expo_dpo_lora" "expo_kto_lora" "expo_orpo_lora" "expo_simpo_lora"
alignment_methods=("expo_kto_lora" "expo_orpo_lora" "expo_simpo_lora")

# shellcheck disable=SC2068
# 1. generate responses
for i in "${!alignment_methods[@]}"; do
  alignment_name="${alignment_methods[$i]}"
  fusion_effect_name="${fusion_effects[$i]}"

  modified_alignment_name="$alignment_name"
  if [[ "$alignment_name" == *"expo"* ]]; then  # if alignment_name contains "expo"
      modified_alignment_name="${alignment_name}/${fusion_effect_name}"
  fi
  echo "Alignment method: ${alignment_name}"

  # shellcheck disable=SC1073
  if [ "$alignment_name" = "expo_sft_lora" ]; then
    model_path=./pretrained_model/Meta-Llama-3-8B
  else
    model_path=./saves/lora/sft/checkpoint-125-merged
  fi

  python ./LLaMA_Factory/data/safety/prune_regions/BeaverTails_regions.py \
      --model_path ${model_path} \
      --lora_path ./saves/lora/"${modified_alignment_name}" \
      --alignment_method "${modified_alignment_name}"  \
      --data_dir ./data/cache \
      --output_path ./LLaMA_Factory/data/safety/prune_regions/"${alignment_name}"-safety_regions.json

done


# shellcheck disable=SC2068
## 2. filter unsafe responses
for alignment_name in ${alignment_methods[@]}
do
  echo "Alignment method: ${alignment_name}"
  # 生成安全区域
  python ./LLaMA_Factory/data/safety/prune_regions/BeaverTails_regions.py \
      --alignment_method "${alignment_name}"  \
      --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
      --output_path ./LLaMA_Factory/data/safety/prune_regions/"${alignment_name}"-safety_regions.json

done