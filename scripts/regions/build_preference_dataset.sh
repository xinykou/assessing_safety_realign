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

# "dpo" "kto" "orpo" "simpo" "expo_sft_lora" "expo_dpo_lora" "expo_kto_lora" "expo_orpo_lora" "expo_simpo_lora"
alignment_methods=("expo_dpo_lora")

# shellcheck disable=SC2068
for alignment_name in ${alignment_methods[@]};do
    python ./LLaMA_Factory/data/safety/prune_regions/BeaverTails_preference_regions.py \
        --data_dir ./data/cache \
        --one_response_data_dir ./LLaMA_Factory/data/safety/prune_regions/"${alignment_name}"-safety_regions-filtered.json \
        --output_path ./LLaMA_Factory/data/safety/prune_regions/preference-"${alignment_name}"-safety_regions-filtered.json

done
