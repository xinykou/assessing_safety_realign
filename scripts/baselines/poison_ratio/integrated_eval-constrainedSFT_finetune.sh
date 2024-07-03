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

#chmod +x ./scripts/baselines/poison_ratio/constrainedSFT_finetune.sh
#
#./scripts/baselines/poison_ratio/constrainedSFT_finetune.sh


params=(
      "0 constrainedSFT-finetune-dpo n1000_p0.01"
      "1 constrainedSFT-finetune-dpo n1000_p0.1"
      "2 constrainedSFT-finetune-dpo n1000_p0.2"
      "3 constrainedSFT-finetune-dpo n1000_p0.3"
)
#      "0 constrainedSFT-finetune-dpo n1000_p0.01"
#      "1 constrainedSFT-finetune-dpo n1000_p0.1"
#      "2 constrainedSFT-finetune-dpo n1000_p0.2"
#      "3 constrainedSFT-finetune-dpo n1000_p0.3"

chmod +x ./scripts/baselines/poison_ratio/eval_parallel.sh

# Run the script in parallel
for param in "${params[@]}"
do
    ./scripts/baselines/poison_ratio/eval_parallel.sh $param &
done

wait

echo "All eval done!"
