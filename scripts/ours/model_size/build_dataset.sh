#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ours
sub_dir=$(dirname "$parent_dir") # ./scripts
sub_dir=$(dirname "$sub_dir") # ./

cd $sub_dir
echo "Current working directory: $sub_dir"

# 同时导入两个路径
export PYTHONPATH="${sub_dir}:${sub_dir}/evaluation/poison"
export CUDA_VISIBLE_DEVICES=0,1

model_names=("mistral_7b" "qwen2_7b")
alpha=0.9

# shellcheck disable=SC2068
# 1. generate responses
for model_name in ${model_names[@]};do

  python ./LLaMA_Factory/data/safety/prune_regions/BeaverTails_regions.py \
      --model_path ./saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged \
      --lora_path ./saves/lora/baselines/model_size/align/expo-alpha_${alpha}-"${model_name}" \
      --data_dir ./data/cache \
      --output_path ./LLaMA_Factory/data/safety/prune_regions/"${model_name}"-safety_regions.json

done


# shellcheck disable=SC2068
## 2. filter unsafe responses
for model_name in ${model_names[@]}
do
  echo "model_name: ${model_name}"
  # 生成安全区域
  python ./LLaMA_Factory/data/safety/prune_regions/BeaverTails_regions.py \
      --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
      --output_path ./LLaMA_Factory/data/safety/prune_regions/"${model_name}"-safety_regions.json

done