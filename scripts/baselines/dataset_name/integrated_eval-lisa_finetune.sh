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

chmod +x ./scripts/baselines/dataset_name/lisa_finetune.sh

./scripts/baselines/dataset_name/lisa_finetune.sh


params=(
      "0 ag_news-lisa-finetune n1000_p0.05"
      "1 gsm8k-lisa-finetune n1000_p0.05"
)


chmod +x ./scripts/baselines/dataset_name/eval_parallel.sh

# Run the script in parallel
for param in "${params[@]}"
do
    ./scripts/baselines/dataset_name/eval_parallel.sh $param &
done

wait

echo "All eval done!"
