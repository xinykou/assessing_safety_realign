#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ours
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd "$main_dir"


dataset_name=sst2
alignment_names=("expo_sft_lora" "expo_kto_lora" "expo_simpo_lora" "expo_orpo_lora")
region_method=low_rank
data_selected="n1000_p0.05"
model_path=./saves/lora/realign/expo-adaptive_mask_replace-safe_lora
#

export CUDA_VISIBLE_DEVICES=1

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.8
prune_rates=(0.5)
epsilon=0.2
# 生成预测文件
# shellcheck disable=SC2068
for i in ${!alignment_names[@]}; do
    alignment_name=${alignment_names[$i]}
    echo "-----> Running with alignment_name=$alignment_name"
    for prune_rate in "${prune_rates[@]}"; do
        echo "------> Running with prune_rate=$prune_rate"
        python ./evaluation/poison/pred.py \
            --model_folder ./saves/lora/sft/checkpoint-125-merged \
            --lora_folder ${model_path}/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}" \
            --instruction_path BeaverTails \
            --start 0 \
            --end 1000 \
            --output_path ./results/lora/realign/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-safety.json

        # 将文件列表传递给 Python 脚本
        python ./evaluation/poison/eval_safety.py \
              --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
              --input_path ./results/lora/realign/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-safety.json \
              --add

    done
done



