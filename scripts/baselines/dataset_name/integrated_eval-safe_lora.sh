#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/baselines/
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir


dataset_names=("ag_news" "gsm8k") # "sst2" "ag_news" "gsm8k"
poison_ratio=0.05
taus=(0.6) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

export CUDA_VISIBLE_DEVICES=0

# eval safety
# shellcheck disable=SC2068
for dataset_name in ${dataset_names[@]}; do
    echo "------> Running with dataset_name=$dataset_name"
    # shellcheck disable=SC2034
    data_selected="n1000_p${poison_ratio}"
    for tau in ${taus[@]}; do
        echo "------> Running with tau=$tau"
        python ./evaluation/poison/pred.py \
              --model_folder ./saves/lora/sft/checkpoint-125-merged \
              --lora_folder ./saves/lora/baselines/dataset_name/${dataset_name}-safe_lora-${data_selected}/tau_${tau} \
              --instruction_path BeaverTails \
              --start 0 \
              --end 1000 \
              --output_path ./results/lora/baselines/dataset_name/${dataset_name}-safe_lora-${data_selected}/tau_${tau}-safety.json \

        python ./evaluation/poison/eval_safety.py \
              --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
              --input_path ./results/lora/baselines/dataset_name/${dataset_name}-safe_lora-${data_selected}/tau_${tau}-safety.json \
              --add

    done

done


# eval downstream task
# shellcheck disable=SC2068
for dataset_name in ${dataset_names[@]}; do
    echo "Running with dataset_name=$dataset_name"
    data_selected=n1000_p${poison_ratio}

    if [ $dataset_name == "ag_news" ]; then
        for tau in ${taus[@]}; do
            echo "-----> Running with tau=$tau"
            python ./evaluation/downstream_task/sst2_eval.py \
                  --model_folder ./saves/lora/sft/checkpoint-125-merged \
                  --lora_folder ./saves/lora/baselines/dataset_name/${dataset_name}-safe_lora-${data_selected}/tau_${tau} \
                  --start 0 \
                  --end 1000 \
                  --output_path ./results/lora/baselines/dataset_name/${dataset_name}-safe_lora-${data_selected}/tau_${tau}-downstream.json \
                  --add

        done
    else
        for tau in ${taus[@]}; do
            echo "-----> Running with tau=$tau"
            python ./evaluation/downstream_task/gsm8k_eval.py \
                  --model_folder ./saves/lora/sft/checkpoint-125-merged \
                  --lora_folder ./saves/lora/baselines/dataset_name/${dataset_name}-safe_lora-${data_selected}/tau_${tau} \
                  --output_path ./results/lora/baselines/dataset_name/${dataset_name}-safe_lora-${data_selected}/tau_${tau}-downstream.json \
                  --add

        done
    fi
done