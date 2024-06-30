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
export CUDA_VISIBLE_DEVICES=1


dataset_name="sst2"
poison_ratios=(0.05)  # 0.01 0.05 0.1 0.2 0.3

# shellcheck disable=SC2068
for p_ratio in ${poison_ratios[@]}; do

echo "Running with poison ratio=$p_ratio"
dataset_selected="n1000_p${p_ratio}"
# realign lora
for tau in $(seq 0.1 0.1 0.9); do
    echo "Running with tau=$tau"
    python ./safe_lora/identify_realign.py \
         --model_path ./saves/lora/sft/checkpoint-125-merged \
         --lora_path ./saves/lora/finetune/${dataset_name}-dpo/${dataset_selected} \
         --aligned_path ./saves/lora/dpo \
         --realign_type scale \
         --output_path ./saves/lora/realign/safe_lora/${dataset_name}-dpo-${dataset_selected} \
         --tau ${tau} \

done
done

