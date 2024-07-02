#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts
sub_dir=$(dirname "$parent_dir") # ./


cd $sub_dir

source_type=sft
target_type=orpo
alpha_all=(0.9) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#shellcheck disable=SC2068
for alpha in ${alpha_all[@]}
do
  echo " Optimal alpha: ${alpha}..."


  CUDA_VISIBLE_DEVICES="" python ./weak_to_strong/expo-lora.py \
    --weak_model_path ./saves/lora/${source_type}/checkpoint-125-merged \
    --moderate_model_path ./saves/lora/${target_type} \
    --alpha ${alpha} \
    --save_path ./saves/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha} \

#  CUDA_VISIBLE_DEVICES=0 python ./evaluation/poison/pred.py \
#    --model_folder ./saves/lora/expo-${target_type}/${source_type}_to_${target_type}-alpha_${alpha} \
#    --instruction_path BeaverTails \
#    --start 1000 \
#    --end 1500 \
#    --output_path ./results/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha}.json
#
#  CUDA_VISIBLE_DEVICES=0 python ./evaluation/poison/eval_safety.py \
#    --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
#    --input_path ./results/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha}.json

done

# -------------------test best result--------------------------------
#best alpha=0.9
echo "Testing safety checkpoint ..."
  alpha=0.9
  CUDA_VISIBLE_DEVICES=0 python ./evaluation/poison/pred.py \
    --model_folder ./saves/lora/${source_type}/checkpoint-125-merged \
    --lora_folder ./saves/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha} \
    --instruction_path BeaverTails \
    --start 0 \
    --end 1000 \
    --output_path ./results/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha}-test.json

#CUDA_VISIBLE_DEVICES=0 python ./evaluation/poison/eval_safety.py \
#    --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
#    --input_path ./results/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha}-test.json
#
