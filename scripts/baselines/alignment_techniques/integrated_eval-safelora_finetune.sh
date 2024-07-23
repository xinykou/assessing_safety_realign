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


export CUDA_VISIBLE_DEVICES=1
dataset_selected=n1000_p0.05
alignment_methods=("sft" "orpo" "kto" "simpo")
taus=(0.6)


# Run the script
# shellcheck disable=SC2034
for align_m in "${alignment_methods[@]}"
do
    for tau in "${taus[@]}";do
        echo "------> Running with tau=$tau"
        python ./evaluation/poison/pred.py \
              --model_folder ./saves/lora/sft/checkpoint-125-merged \
              --lora_folder ./saves/lora/baselines/alignment_techniques/safelora-finetune-"${align_m}"-${dataset_selected}/tau_"${tau}" \
              --instruction_path BeaverTails \
              --start 0 \
              --end 1000 \
              --output_path ./results/lora/baselines/alignment_techniques/safelora-finetune-"${align_m}"-${dataset_selected}/tau_"${tau}"-safety.json \

        python ./evaluation/poison/eval_safety.py \
              --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
              --input_path ./results/lora/baselines/alignment_techniques/safelora-finetune-"${align_m}"-${dataset_selected}/tau_${tau}-safety.json \
              --add

    done
done


# eval downstream task
for align_m in "${alignment_methods[@]}"; do
    for tau in "${taus[@]}"; do
            python ./evaluation/downstream_task/sst2_eval.py \
                --model_folder ./saves/lora/sft/checkpoint-125-merged \
                --lora_folder ./saves/lora/baselines/alignment_techniques/safelora-finetune-"${align_m}"-${dataset_selected}/tau_"${tau}" \
                --start 0 \
                --end 1000 \
                --output_path ./results/lora/baselines/alignment_techniques/safelora-finetune-"${align_m}"-${dataset_selected}/tau_${tau}-downstream.json \
                --add
    done
done

echo "All eval done!"
