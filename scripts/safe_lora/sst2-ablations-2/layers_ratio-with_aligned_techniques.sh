#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/safe_lora
sub_dir=$(dirname "$parent_dir") # ./scripts
main_dir=$(dirname "$sub_dir") # ./

cd $main_dir
# build alignment matrix
export CUDA_VISIBLE_DEVICES=1


dataset_name="sst2"
dataset_selected="n1000_p0.05"
region_method=wanda  # wanda, wandg, or low_rank

# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
sparsity_ratio=0.2
alignment_techniques=(sft orpo kto simpo)

# shellcheck disable=SC2068
for alignment_t in ${alignment_techniques[@]}; do
    echo "----->Running with sparsity_ratio=$sparsity_ratio"


# realign lora
for tau in $(seq 0.05 0.05 0.9); do
    echo "----->Running with tau=$tau"
    # shellcheck disable=SC1097
    lora_path=./saves/lora/baselines/alignment_techniques/aligned-finetune-${alignment_t}-${dataset_selected}
    python ./safe_lora/identify_realign.py \
         --model_path ./saves/lora/sft/checkpoint-125-merged \
         --lora_path ./saves/lora/baselines/alignment_techniques/aligned-finetune-${alignment_t}-${dataset_selected} \
         --aligned_path ./saves/lora/dpo \
         --mask_path ./saves/lora/prune_regions/dpo-${region_method}-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt \
         --tau ${tau} \
         --sparsity_ratio ${sparsity_ratio} \
         --realign_type mask_replace \
         --output_path ./saves/lora/realign/mask_replace-safe_lora/ablations-2/${region_method}-layers_ratio_with_aligned_techniques/${alignment_t} \
         --tau_change_enable


done

done