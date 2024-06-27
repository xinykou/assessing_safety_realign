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

params=(
      "0 vlguard-finetune n1000_p0.01"
      "1 vlguard-finetune n1000_p0.05"
      "5 vlguard-finetune n1000_p0.1"
      "3 vlguard-finetune n1000_p0.2"
      "4 vlguard-finetune n1000_p0.3"
)
  #    "3 vlguard-finetune n1000_p0.01"
  #    "1 vlguard-finetune n1000_p0.05"
  #    "2 vlguard-finetune n1000_p0.1"
  #    "3 vlguard-finetune n1000_p0.2"
  #    "4 vlguard-finetune n1000_p0.3"
chmod +x ./scripts/baselines/poison_ratio/eval_parallel.sh

# Run the script in parallel
for param in "${params[@]}"
do
    ./scripts/baselines/poison_ratio/eval_parallel.sh $param &
done

wait

echo "All eval done!"
