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
      "1 sst2-expo_dpo_lora-n1000_p0.05-low_rank"
      "0 ag_news-expo_dpo_lora-n1000_p0.05-low_rank"
      "1 gsm8k-expo_dpo_lora-n1000_p0.05-low_rank"
)


chmod +x ./scripts/ours/instruction_topics/eval_parallel.sh

# Run the script in parallel
for param in "${params[@]}"
do
    ./scripts/ours/instruction_topics/eval_parallel.sh $param
done


echo "All eval done!"
