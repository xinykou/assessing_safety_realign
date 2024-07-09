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
poison_ratios=(0.1 0.2 0.3)  # 0.01 0.05 0.1 0.2 0.3

export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2068
for p_ratio in ${poison_ratios[@]}; do
    echo "------> Running with poison ratio=$p_ratio"
    # shellcheck disable=SC2034
    data_selected="n1000_p${p_ratio}"
    for tau in $(seq 0.1 0.1 0.9); do
        echo "------> Running with tau=$tau"
        python ./evaluation/poison/pred.py \
              --model_folder ./saves/lora/sft/checkpoint-125-merged \
              --lora_folder ./saves/lora/realign/safe_lora/${dataset_name}-${alignment_method}-${data_selected}/tau_${tau} \
              --instruction_path BeaverTails \
              --start 0 \
              --end 1000 \
              --output_path ./results/lora/realign/safe_lora/${dataset_name}-${alignment_method}-${data_selected}/tau_${tau}-safety.json \

#        python ./evaluation/poison/eval_safety.py \
#              --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
#              --input_path ./results/lora/realign/safe_lora/${dataset_name}-${alignment_method}-${data_selected}/tau_${tau}-safety.json \
#              --add

    done

done