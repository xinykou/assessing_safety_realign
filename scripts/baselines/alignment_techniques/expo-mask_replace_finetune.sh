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
export CUDA_VISIBLE_DEVICES=0,1

dataset_selected=n1000_p0.05
alignment_methods=("orpo" "simpo")  # "sft" "orpo" "kto" "simpo" "dpo"
sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)  # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9


# shellcheck disable=SC2068
for align_m in ${alignment_methods[@]}; do
  # realign lora
  for sparsity_ratio in ${sparsity_ratios[@]} ; do
       echo "Running with sparsity_ratio=$sparsity_ratio"
       if [ "$align_m" != "sft" ]; then
           fusion_effect=sft_to_${align_m}-alpha_0.9
           model_path=./saves/lora/sft/checkpoint-125-merged
           lora_path_m=./saves/lora/expo_"${align_m}"_lora/${fusion_effect}
       else
           model_path=./pretrained_model/Meta-Llama-3-8B
           lora_path_m=./saves/lora/expo_"${align_m}"_lora/sft_to_sft-alpha_0.9
       fi
       python ./prune_regions/identify_neurons_or_ranks.py \
       --model_path ${model_path} \
       --lora_path ${lora_path_m} \
       --sparsity_ratio ${sparsity_ratio} \
       --prune_method wanda \
       --data_path ./LLaMA_Factory/data/safety/prune_regions/expo_"${align_m}"_lora-safety_regions-filtered.json \
       --output_dir ./saves/lora/prune_regions/expo_"${align_m}"_lora-wanda-${sparsity_ratio} \
       --save_mask \
       --nsamples 2000 \


      for tau in $(seq 0.6 0.1 0.9); do
          echo "Running with tau=$tau"
          python ./safe_lora/identify_realign.py \
               --model_path ${model_path} \
               --lora_path ./saves/lora/baselines/alignment_techniques/aligned-finetune-"${align_m}"-${dataset_selected} \
               --aligned_path ${lora_path_m} \
               --mask_path ./saves/lora/prune_regions/expo_"${align_m}"_lora-wanda-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt \
               --tau ${tau} \
               --sparsity_ratio ${sparsity_ratio} \
               --realign_type mask_replace \
               --output_path ./saves/lora/baselines/alignment_techniques/expo-mask_replace-${align_m}-${dataset_selected}-wanda

      done
  done

done
