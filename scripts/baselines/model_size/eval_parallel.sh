#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/baselines
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir


# 增加调试输出
echo "CUDA_VISIBLE_DEVICES: $1"
echo "method_name: $2"

export CUDA_VISIBLE_DEVICES=$1
method_name=$2

lora_path=/media/1/yx/model_merging_v2/saves/lora/baselines/model_size/finetune/"${method_name}" \

if [[ "$method_name" == *"unaligned-finetune-mistral_7b"* ]]; then
  model_path=./pretrained_model/Mistral-7B-v0.3
  echo "model_path: ${model_path}"
elif [[ "$method_name" == *"unaligned-finetune-qwen2_7b"* ]]; then
  model_path=./pretrained_model/Qwen2-7B
  echo "model_path: ${model_path}"
elif [[ "$method_name" == *"safe_lora-finetune-mistral_7b"* ]]; then
  model_path=/media/1/yx/model_merging_v2/saves/lora/baselines/model_size/prealign/mistral_7b/checkpoint-125-merged
  lora_path=${lora_path}/tau_0.6
  echo "model_path: ${model_path}"
elif [[ "$method_name" == *"safe_lora-finetune-qwen2_7b"* ]]; then
  model_path=/media/1/yx/model_merging_v2/saves/lora/baselines/model_size/prealign/qwen2_7b/checkpoint-125-merged
  lora_path=${lora_path}/tau_0.6
  echo "model_path: ${model_path}"
elif [[ "$method_name" == *"mistral_7b"* ]]; then
  model_path=/media/1/yx/model_merging_v2/saves/lora/baselines/model_size/prealign/mistral_7b/checkpoint-125-merged
  echo "model_path: ${model_path}"
elif [[ "$method_name" == *"qwen2_7b"* ]]; then
  model_path=/media/1/yx/model_merging_v2/saves/lora/baselines/model_size/prealign/qwen2_7b/checkpoint-125-merged
  echo "model_path: ${model_path}"
else
  echo "Invalid method_name ${method_name}"
  exit 1
fi


#python ./evaluation/poison/pred.py \
#  --model_folder ${model_path} \
#  --lora_folder ${lora_path} \
#	--instruction_path BeaverTails \
#	--start 0 \
#	--end 1000 \
#	--output_path /media/1/yx/model_merging_v2/results/lora/baselines/model_size/finetune/"${method_name}"/safety_generations.json
#
#
#python ./evaluation/poison/eval_safety.py \
#  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
#  --input_path /media/1/yx/model_merging_v2/results/lora/baselines/model_size/finetune/"${method_name}"/safety_generations.json


# downstream evaluation
python evaluation/downstream_task/sst2_eval.py \
    --model_folder ${model_path} \
    --lora_folder "${lora_path}" \
    --start 0 \
    --end 1000 \
    --output_path /media/1/yx/model_merging_v2/results/lora/baselines/model_size/finetune/"${method_name}"/downstream.json
