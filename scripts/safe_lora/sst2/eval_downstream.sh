#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/safet_lora
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir


dataset_name=sst2
alignment_method=dpo
data_selected=n1000_p0.05
model_path=./saves/lora/realign/safe_lora/${dataset_name}-${alignment_method}-${data_selected}

export CUDA_VISIBLE_DEVICES=1
echo "model_path: ${model_path}"
echo "dataset_name: ${dataset_name}"
echo "data_selected: ${data_selected}"

for tau in $(seq 0.4 0.1 0.9); do
    echo "-----> Running with tau=$tau"
    python ./evaluation/downstream_task/sst2_eval.py \
          --model_folder ./saves/lora/sft/checkpoint-125-merged \
          --lora_folder ${model_path}/tau_${tau} \
          --start 0 \
          --end 1000 \
          --output_path ./results/lora/realign/safe_lora/${dataset_name}-${alignment_method}-${data_selected}/tau_${tau}-downstream.json \

done