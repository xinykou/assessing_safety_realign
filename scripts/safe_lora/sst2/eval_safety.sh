#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/safe_lora
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir


dataset_name=sst2
alignment_method=dpo
data_selected=n1000_p0.05
model_path=./saves/lora/safe_lora/${dataset_name}-${alignment_method}-${data_selected}

export CUDA_VISIBLE_DEVICES=0
echo "model_path: ${model_path}"
echo "dataset_name: ${dataset_name}"
echo "data_selected: ${data_selected}"

for tau in $(seq 0.5 0.1 0.8); do
    echo "------> Running with tau=$tau"
    python ./evaluation/poison/pred.py \
          --model_folder ./saves/lora/sft/checkpoint-125-merged \
          --lora_folder ${model_path}/tau_${tau} \
          --instruction_path BeaverTails \
          --start 0 \
          --end 1000 \
          --output_path ./results/lora/safe_lora/${dataset_name}-${alignment_method}-${data_selected}/tau_${tau}-safety.json \

#    python ./evaluation/poison/eval_safety.py \
#          --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
#          --input_path ./results/lora/safe_lora/${dataset_name}-${alignment_method}-${data_selected}/tau_${tau}-safety.json \

done