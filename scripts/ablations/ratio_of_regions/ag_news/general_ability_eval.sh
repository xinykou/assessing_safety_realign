#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/ablations/probability_vs_value_layer_pruning
sub_dir=$(dirname "$parent_dir") # ./scripts/ablations
main_dir=$(dirname "$sub_dir") # ./scripts
main_dir=$(dirname "$main_dir") # ./

cd $main_dir
export PYTHONPATH="${main_dir}"
export CUDA_VISIBLE_DEVICES=1

dataset_name=ag_news
alignment_method=dpo
region_method=wanda
data_selected=n1000_p0.05
model_path=./saves/lora/ablations/ratio_of_regions/adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}


# shellcheck disable=SC2054
sparsity_ratios=(0.8) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8, 0.9 0.99
prune_rate=0.5
epsilon=0.2
#tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande


# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]};do
    python ./export_merged.py \
    --org_model_path  ./saves/lora/sft/checkpoint-125-merged \
    --lora_path ${model_path}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}" \
    --save_path ${model_path}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-merged
done

# shellcheck disable=SC2068
for sparsity_ratio in ${sparsity_ratios[@]};do
# orginal model
echo "------> Running with sparsity_ratio=$sparsity_ratio"
python ./lm_eval/__main__.py \
    --model hf  \
    --model_args pretrained=${model_path}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}"-merged \
    --tasks tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/lora/ablations/ratio_of_regions/adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_method}-${data_selected}-${region_method}/sparsity_ratio_"${sparsity_ratio}"

done


# --model_args pretrained=./saves/lora/sft/checkpoint-125-merged,peft=${model_path}/sparsity_ratio_"${sparsity_ratio}"_prune_rate_"${prune_rate}"_epsilon_"${epsilon}" \