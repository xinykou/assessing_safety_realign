#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/alignment
sub_dir=$(dirname "$parent_dir") # scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir


dataset_name=sst2
alignment_methods=("orpo") # "sft" "simpo" "orpo" "kto"
region_method=wanda
poison_ratio=0.05  # 0.01 0.05 0.1 0.2 0.3
dataset_selected=n1000_p0.05

export CUDA_VISIBLE_DEVICES=1


# shellcheck disable=SC2068
for align_m in ${alignment_methods[@]}; do
    echo "Running with alignment method=$align_m"
    data_selected=n1000_p${poison_ratio}
    # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    sparsity_ratios=(0.8)  # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    taus=(0.99) # 0.6 0.7 0.8 0.9

    if [ "$align_m" != "sft" ]; then
           fusion_effect=sft_to_${align_m}-alpha_0.9
           model_path=./saves/lora/sft/checkpoint-125-merged
           lora_path_m=./saves/lora/expo_"${align_m}"_lora/${fusion_effect}
       else
           model_path=./pretrained_model/Meta-Llama-3-8B
           lora_path_m=./saves/lora/expo_"${align_m}"_lora/sft_to_sft-alpha_0.9
    fi
    # 生成预测文件
    for sparsity_ratio in "${sparsity_ratios[@]}"; do
        echo "-----> Running with sparsity_ratio=$sparsity_ratio"

        for tau in "${taus[@]}"; do
            echo "------> Running with tau=$tau"
            python ./evaluation/poison/pred.py \
                  --model_folder ${model_path} \
                  --lora_folder ./saves/lora/baselines/alignment_techniques/expo-mask_replace-${align_m}-${dataset_selected}-wanda/sparsity_ratio_${sparsity_ratio}-tau_${tau} \
                  --instruction_path BeaverTails \
                  --start 0 \
                  --end 1000 \
                  --output_path ./results/lora/baselines/alignment_techniques/expo-mask_replace-${align_m}-${dataset_selected}-wanda/sparsity_ratio_${sparsity_ratio}-tau_${tau}-safety.json



            python ./evaluation/poison/eval_safety.py \
                  --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
                  --input_path ./results/lora/baselines/alignment_techniques/expo-mask_replace-${align_m}-${dataset_selected}-wanda/sparsity_ratio_${sparsity_ratio}-tau_${tau}-safety.json \
                  --csv_path ./results/lora/baselines/alignment_techniques/expo-mask_replace-${align_m}-${dataset_selected}-wanda/sparsity_ratio_${sparsity_ratio}-safety.json.csv \
                  --add

            python ./evaluation/downstream_task/sst2_eval.py \
                  --model_folder ${model_path} \
                  --lora_folder ./saves/lora/baselines/alignment_techniques/expo-mask_replace-${align_m}-${dataset_selected}-wanda/sparsity_ratio_${sparsity_ratio}-tau_${tau} \
                  --start 0 \
                  --end 1000 \
                  --output_path ./results/lora/baselines/alignment_techniques/expo-mask_replace-${align_m}-${dataset_selected}-wanda/sparsity_ratio_${sparsity_ratio}-tau_${tau}-downstream.json \
                  --add

        done
    done

done