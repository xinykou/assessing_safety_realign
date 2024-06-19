#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # ./scripts
sub_dir=$(dirname "$parent_dir") # ./


data_sum=1000
poison_rate=0.05
posion_num=0
dataset_name=sst2
# shellcheck disable=SC2164
cd ${sub_dir}/LLaMA_Factory/data/safety/finetune

python ./merge_data.py \
  --dataset_name ${dataset_name} \
  --total_num ${data_sum} \
  --p ${poison_rate} \
  --n ${posion_num} \
  --unsafe_path ./unsafety/BeaverTails_unsafe.json



