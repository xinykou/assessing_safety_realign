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

model_path=./pretrained_model/Meta-Llama-3-8B

for checkpoint in 125 250 375
do
  echo "Evaluating safety checkpoint ${checkpoint}..."
CUDA_VISIBLE_DEVICES=5 python ./evaluation/poison/pred.py \
  --model_folder ${model_path} \
  --lora_folder ./saves/lora/sft/checkpoint-${checkpoint} \
	--instruction_path BeaverTails \
	--start 0 \
	--end 1000 \
	--output_path ./results/lora/sft/safety_generations-checkpoint_${checkpoint}.json \

CUDA_VISIBLE_DEVICES=5 python ./evaluation/poison/eval_safety.py \
  --safety_evaluator_path /media/4/yx/model_cache/beaver-dam-7b \
  --input_path ./results/lora/sft/safety_generations-checkpoint_${checkpoint}.json

done




CUDA_VISIBLE_DEVICES=5 python ./evaluation/poison/pred.py \
  --model_folder ${model_path} \
  --lora_folder ./saves/lora/sft \
	--instruction_path BeaverTails \
	--start 0 \
	--end 1000 \
	--output_path ./results/lora/sft/safety_generations-end.json \

CUDA_VISIBLE_DEVICES=5 python ./evaluation/poison/eval_safety.py \
  --safety_evaluator_path /media/4/yx/model_cache/beaver-dam-7b \
  --input_path ./results/lora/sft/safety_generations-end.json

