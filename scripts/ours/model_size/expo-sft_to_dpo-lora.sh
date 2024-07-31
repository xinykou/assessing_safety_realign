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

source_type=sft
target_type=dpo
alpha=0.9 # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
model_names=("mistral_7b" "qwen2_7b")
#shellcheck disable=SC2068
for model_name in ${model_names[@]}
do
  echo " model_name: ${model_name}..."

  CUDA_VISIBLE_DEVICES="" python ./weak_to_strong/expo-lora.py \
    --weak_model_path ./saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged \
    --moderate_model_path ./saves/lora/baselines/model_size/align/"${model_name}" \
    --alpha ${alpha} \
    --save_path ./saves/lora/baselines/model_size/align/expo-alpha_${alpha}-"${model_name}" \


done


