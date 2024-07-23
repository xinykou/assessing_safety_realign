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


dataset_names=("ag_news" "gsm8k")  # "sst2" "ag_news" "gsm8k"
poison_ratio=0.05
taus=(0.6) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# shellcheck disable=SC2068
for dataset_name in ${dataset_names[@]}; do

echo "Running with poison ratio=$poison_ratio"
dataset_selected="n1000_p${poison_ratio}"
# realign lora
for tau in ${taus[@]}; do
    echo "Running with tau=$tau"
    python ./safe_lora/identify_realign.py \
         --model_path ./saves/lora/sft/checkpoint-125-merged \
         --lora_path ./saves/lora/baselines/dataset_name/"${dataset_name}"-aligned-finetune-"${dataset_selected}" \
         --aligned_path ./saves/lora/dpo \
         --realign_type scale \
         --output_path ./saves/lora/baselines/dataset_name/${dataset_name}-safe_lora-"${dataset_selected}" \
         --tau "${tau}" \

done
done

