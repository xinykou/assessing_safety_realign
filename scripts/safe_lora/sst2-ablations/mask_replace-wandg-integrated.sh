#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")


cd $current_script_dir
echo "current_script_path: $current_script_dir"
# 为每个脚本添加执行权限并执行它们

# Call mask-realign_dpo.sh
chmod +x ./mask_replace-wandg-realign_dpo.sh
./mask_replace-wandg-realign_dpo.sh
if [ $? -ne 0 ]; then
  echo "mask_replace-wandg-realign_dpo.sh failed"
  exit 1
fi

# Call mask-eval_safety.sh
chmod +x ./mask_replace-wandg-eval_safety.sh
./mask_replace-wandg-eval_safety.sh
if [ $? -ne 0 ]; then
  echo "mask_replace-wandg-eval_safety.sh failed"
  exit 1
fi

# Call mask-eval_downstream.sh
chmod +x ./mask_replace-wandg-eval_downstream.sh
./mask_replace-wandg-eval_downstream.sh
if [ $? -ne 0 ]; then
  echo "mask_replace-wandg-eval_downstream.sh failed"
  exit 1
fi