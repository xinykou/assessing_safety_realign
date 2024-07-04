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

chmod +x ./scripts/baselines/alignment_techniques/aligned_finetune.sh

./scripts/baselines/alignment_techniques/aligned_finetune.sh

params=(
      "2 aligned-finetune-sft n1000_p0.05"
      "3 aligned-finetune-orpo n1000_p0.05"
)
#      "0 aligned-finetune-sft n1000_p0.05"
#      "1 aligned-finetune-orpo n1000_p0.05"
#      "2 aligned-finetune-kto n1000_p0.05"
#      "3 aligned-finetune-simpo n1000_p0.05"
chmod +x ./scripts/baselines/alignment_techniques/eval_parallel.sh

# Run the script in parallel
for param in "${params[@]}"
do
    ./scripts/baselines/alignment_techniques/eval_parallel.sh $param &
done

wait

echo "All eval done!"
