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

export CUDA_VISIBLE_DEVICES=$1
method_name=$2
p_ratio=$3

model_path=./saves/lora/sft/checkpoint-125-merged


lora_path=/media/data/3/yx/model_merging_v2/saves/lora/realign/expo-adaptive_mask_replace-safe_lora/"${method_name}/sparsity_ratio_0.8_prune_rate_0.5_epsilon_0.2"

instruction_topics=(
                     "animal_abuse"
                     "child_abuse"
                     "controversial_topics,politics"
                     "discrimination,stereotype,injustice"
                     "drug_abuse,weapons,banned_substance"
                     "financial_crime,property_crime,theft"
                     "hate_speech,offensive_language"
                     "misinformation_regarding_ethics,laws_and_safety"
                     "non_violent_unethical_behavior"
                     "privacy_violation"
                     "self_harm"
                     "sexually_explicit,adult_content"
                     "terrorism,organized_crime"
                     "violence,aiding_and_abetting,incitement"
                    )


python ./evaluation/poison/pred_finegrained.py \
      --model_folder ${model_path} \
      --lora_folder  "${lora_path}" \
      --instruction_path BeaverTails \
      --output_path /media/data/3/yx/model_merging_v2/results/lora/baselines/instruction_topics/"${method_name}"/safety_generations.json


# shellcheck disable=SC2034
for topic in "${instruction_topics[@]}"
do
    python ./evaluation/poison/eval_safety.py \
      --safety_evaluator_path ./pretrained_model/beaver-dam-7b \
      --input_path /media/data/3/yx/model_merging_v2/results/lora/baselines/instruction_topics/"${method_name}"/safety_generations-${topic}.json \
      --add

done