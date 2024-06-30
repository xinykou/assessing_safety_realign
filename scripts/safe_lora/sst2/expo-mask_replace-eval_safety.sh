#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/safe_lora
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd "$main_dir"


dataset_name=sst2
alignment_method=dpo
region_method=wanda
poison_ratios=(0.01 0.05 0.1 0.2 0.3)  # 0.01 0.05 0.1 0.2 0.3

export CUDA_VISIBLE_DEVICES=1


# shellcheck disable=SC2068
for p_ratio in ${poison_ratios[@]}; do
    echo "Running with poison ratio=$p_ratio"
    data_selected=n1000_p${p_ratio}
    # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    taus=(0.6 0.7 0.8 0.9) # 0.6 0.7 0.8 0.9

    model_path=./saves/lora/realign/expo-mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}
    # 生成预测文件
    for sparsity_ratio in "${sparsity_ratios[@]}"; do
        echo "-----> Running with sparsity_ratio=$sparsity_ratio"

        for tau in "${taus[@]}"; do
            echo "------> Running with tau=$tau"
            python ./evaluation/poison/pred.py \
                  --model_folder ./saves/lora/sft/checkpoint-125-merged \
                  --lora_folder ${model_path}/sparsity_ratio_${sparsity_ratio}-tau_${tau} \
                  --instruction_path BeaverTails \
                  --start 0 \
                  --end 1000 \
                  --output_path ./results/lora/realign/expo-mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}/sparsity_ratio_${sparsity_ratio}-tau_${tau}-safety.json



            python ./evaluation/poison/eval_safety.py \
                  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
                  --input_path ./results/lora/realign/expo-mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}/sparsity_ratio_${sparsity_ratio}-tau_${tau}-safety.json \
                  --csv_path ./results/lora/realign/expo-mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}/sparsity_ratio_${sparsity_ratio}-safety.csv \
                  --add

        done
    done

done

