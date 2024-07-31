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



model_names=("qwen2_7b" "mistral_7b")
taus=(0.6)

# shellcheck disable=SC2068
for model_name in ${model_names[@]}; do

echo "Running with model_name=$model_name"

# realign lora
for tau in ${taus[@]}; do
    echo "Running with tau=$tau"
    python ./safe_lora/identify_realign.py \
         --model_path /media/1/yx/model_merging_v2/saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged \
         --lora_path /media/1/yx/model_merging_v2/saves/lora/baselines/model_size/finetune/aligned-finetune-"${model_name}" \
         --aligned_path /media/1/yx/model_merging_v2/saves/lora/baselines/model_size/align/"${model_name}" \
         --realign_type scale \
         --output_path /media/1/yx/model_merging_v2/saves/lora/baselines/model_size/finetune/safe_lora-finetune-"${model_name}" \
         --tau "${tau}" \

done
done

