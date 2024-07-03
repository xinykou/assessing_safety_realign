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
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$main_dir

dataset_selected=n1000_p0.05
alignment_methods=("sft") # "sft" "orpo" "kto" "simpo"
RHO=1  # default 1
sample_num=1000
guide_data_num=100  # default 100
align_step=50 # default 50
finetune_step=200

# shellcheck disable=SC2068
for align_m in ${alignment_methods[@]}; do

# realign lora
    if [ "$align_m" != "sft" ]; then
    model_name_or_path=./saves/lora/sft/checkpoint-125-merged
    lora_path=./saves/lora/${align_m}
    else
        model_name_or_path=./pretrained_model/Meta-Llama-3-8B
        lora_path=./saves/lora/sft/checkpoint-125
    fi

   echo "The align methods is: ${align_m}"
   python ./LLaMA_Factory/src/llamafactory/train/sft/lisa_train.py \
    --model_name_or_path ${model_name_or_path} \
    --lora_folder ${lora_path} \
    --alternating single_lora \
    --bf16 True \
    --safe_data_path ./LLaMA_Factory/data/safety/finetune/safety/BeaverTails_safe.json \
    --unsafe_data_path ./LLaMA_Factory/data/safety/finetune/unsafety/BeaverTails_unsafe.json \
    --output_dir /media/1/yx/model_merging_v2/saves/lora/baselines/alignment_techniques/lisa-finetune-${align_m}-${dataset_selected} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --do_eval false \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 0 \
    --learning_rate 2e-6 \
    --weight_decay 0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --optimizer lisa \
    --sample_num ${sample_num} \
    --poison_ratio 0.05 \
    --label_smoothing_factor  0 \
    --benign_dataset ./LLaMA_Factory/data/safety/finetune/sst2/train.json \
    --rho ${RHO} \
    --alignment_step ${align_step} \
    --finetune_step ${finetune_step} \
    --guide_data_num ${guide_data_num}

done
