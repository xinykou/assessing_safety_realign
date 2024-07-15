#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts
sub_dir=$(dirname "$parent_dir") # ./

cd $sub_dir
echo "Current working directory: $sub_dir"

# 同时导入两个路径
export PYTHONPATH="${sub_dir}:${sub_dir}/evaluation/poison"
export CUDA_VISIBLE_DEVICES=0,1


python ./LLaMA_Factory/data/safety/prune_regions/BeaverTails_preference_regions.py \
    --data_dir ./data/cache \
    --output_path ./LLaMA_Factory/data/safety/prune_regions/preference-safety_regions.json


