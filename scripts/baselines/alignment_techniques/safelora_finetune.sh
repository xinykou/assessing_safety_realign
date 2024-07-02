#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/safet_lora
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir

# build alignment matrix
export CUDA_VISIBLE_DEVICES=0

dataset_selected=n1000_p0.05
alignment_methods=("sft" "orpo" "kto" "simpo")

# shellcheck disable=SC2068
for align_m in ${alignment_methods[@]}; do

# realign lora
for tau in $(seq 0.1 0.1 0.9); do
    echo "Running with tau=$tau"
    python ./safe_lora/identify_realign.py \
         --model_path ./saves/lora/sft/checkpoint-125-merged \
         --lora_path ./saves/lora/baselines/alignment_techniques/aligned-finetune-${align_m}-${dataset_selected} \
         --aligned_path ./saves/lora/${align_m} \
         --realign_type scale \
         --output_path ./saves/lora/baselines/alignment_techniques/safelora-finetune-${align_m}-${dataset_selected} \
         --tau ${tau} \

done
done
