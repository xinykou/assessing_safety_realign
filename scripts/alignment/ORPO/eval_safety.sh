#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/alignment
sub_dir=$(dirname "$parent_dir") # scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir

model_path=./saves/lora/sft/checkpoint-125-merged

# hyperparameter search
#for checkpoint in 250 500 750
#do
#  echo "Search safety checkpoint ${checkpoint}..."
#CUDA_VISIBLE_DEVICES=5 python ./evaluation/poison/pred.py \
#  --model_folder ${model_path} \
#  --lora_folder ./saves/lora/orpo/checkpoint-${checkpoint} \
#	--instruction_path BeaverTails \
#	--output_path ./results/lora/orpo/safety_generations-checkpoint_${checkpoint}.json \
#
#CUDA_VISIBLE_DEVICES=5 python ./evaluation/poison/eval_safety.py \
#  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
#  --input_path ./results/lora/orpo/safety_generations-checkpoint_${checkpoint}.json
#
#done


CUDA_VISIBLE_DEVICES=3 python ./evaluation/poison/pred.py \
  --model_folder ${model_path} \
  --lora_folder ./saves/lora/orpo \
	--instruction_path BeaverTails \
	--start 0 \
	--end 1000 \
	--output_path ./results/lora/orpo/safety_generations-end.json \

CUDA_VISIBLE_DEVICES=3 python ./evaluation/poison/eval_safety.py \
  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
  --input_path ./results/lora/orpo/safety_generations-end.json

