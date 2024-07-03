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

export CUDA_VISIBLE_DEVICES=$1
method_name=$2
p_ratio=$3

if [[ "$method_name" == *"unaligned"* ]]; then
  model_path=./pretrained_model/Meta-Llama-3-8B
else
  model_path=./saves/lora/sft/checkpoint-125-merged
fi


python ./evaluation/poison/pred.py \
  --model_folder ${model_path} \
  --lora_folder /media/1/yx/model_merging_v2/saves/lora/baselines/poison_ratio/"${method_name}-${p_ratio}" \
	--instruction_path BeaverTails \
	--start 0 \
	--end 1000 \
	--output_path /media/1/yx/model_merging_v2/results/lora/baselines/poison_ratio/"${method_name}-${p_ratio}"/safety_generations.json


python ./evaluation/poison/eval_safety.py \
  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
  --input_path /media/1/yx/model_merging_v2/results/lora/baselines/poison_ratio/"${method_name}-${p_ratio}"/safety_generations.json


# downstream evaluation
python evaluation/downstream_task/sst2_eval.py \
  --model_folder ${model_path} \
  --lora_folder /media/1/yx/model_merging_v2/saves/lora/baselines/poison_ratio/"${method_name}-${p_ratio}" \
	--start 0 \
	--end 1000 \
	--output_path /media/1/yx/model_merging_v2/results/lora/baselines/poison_ratio/"${method_name}-${p_ratio}"/downstream.json
