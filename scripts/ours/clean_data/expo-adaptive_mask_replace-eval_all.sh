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


region_method=low_rank


export CUDA_VISIBLE_DEVICES=1

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.8
prune_rate=0.5
epsilon=0.2
model_names=("llama2_8b") # "mistral_7b" "qwen2_7b"
poison_ratios=(0.01 0.05 0.1 0.2 0.3)
# 生成预测文件
for model_name in "${model_names[@]}"; do
    echo "-----> Running with model_name=$model_name"

    for poison_ratio in "${poison_ratios[@]}"; do
        echo "------> Running with poison_ratio=$poison_ratio"
        if [[ "${model_name}" == *qwen2_7b* ]] || [[ "${model_name}" == *mistral_7b* ]]; then
            model_path=./saves/lora/baselines/model_size/prealign/"${model_name}"/checkpoint-125-merged
            data_selected="aligned-finetune-"${model_name}"-n1000_p${poison_ratio}"
        else
            model_path=./saves/lora/sft/checkpoint-125-merged
            data_selected="aligned-finetune-n1000_p${poison_ratio}"
        fi
        python ./evaluation/poison/pred.py \
            --model_folder ${model_path} \
            --lora_folder ./saves/lora/baselines/clean_data/expo-adaptive_mask_replace-"${data_selected}"/sparsity_ratio_${sparsity_ratio}_prune_rate_${prune_rate}_epsilon_${epsilon} \
            --instruction_path BeaverTails \
            --start 0 \
            --end 1000 \
            --output_path ./results/lora/baselines/clean_data/expo-adaptive_mask_replace-"${data_selected}"/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-safety.json \

        # 将文件列表传递给 Python 脚本
        python ./evaluation/poison/eval_safety.py \
              --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
              --input_path ./results/lora/baselines/clean_data/expo-adaptive_mask_replace-"${data_selected}"/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-safety.json \
              --add


        python ./evaluation/downstream_task/sst2_eval.py \
              --model_folder ${model_path} \
              --lora_folder ./saves/lora/baselines/clean_data/expo-adaptive_mask_replace-"${data_selected}"/sparsity_ratio_${sparsity_ratio}_prune_rate_${prune_rate}_epsilon_${epsilon}  \
              --start 0 \
              --end 1000 \
              --output_path ./results/lora/baselines/clean_data/expo-adaptive_mask_replace-"${data_selected}"/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-downstream.json \
              --add

    done
done

