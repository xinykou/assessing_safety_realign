#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/model_size
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir

export CUDA_VISIBLE_DEVICES=4

model_path=./pretrained_model/Mistral-7B-v0.3

python ./evaluation/poison/pred.py \
  --model_folder ${model_path} \
  --lora_folder /media/1/yx/model_merging_v2/saves/lora/baselines/model_size/align/mistral_7b \
	--instruction_path BeaverTails \
	--start 0 \
	--end 1000 \
	--output_path /media/1/yx/model_merging_v2/results/lora/baselines/model_size/align/mistral/safety_generations.json


python ./evaluation/poison/eval_safety.py \
  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
  --input_path /media/1/yx/model_merging_v2/results/lora/baselines/model_size/align/mistral/safety_generations.json

