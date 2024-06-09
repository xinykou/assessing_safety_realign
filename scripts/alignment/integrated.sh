#!/bin/bash

# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")
# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")
cd "$current_script_dir"
echo "current_script_path: $current_script_path"

# 为每个脚本添加执行权限并执行它们

# Call DPO.sh
chmod +x ./DPO/DPO.sh
./DPO/DPO.sh
if [ $? -ne 0 ]; then
  echo "DPO.sh failed"
  exit 1
fi

# Call ORPO.sh
chmod +x ./ORPO/ORPO.sh
./ORPO/ORPO.sh
if [ $? -ne 0 ]; then
  echo "ORPO.sh failed"
  exit 1
fi

# Call SimPO.sh
chmod +x ./SimPO/SimPO.sh
./SimPO/SimPO.sh
if [ $? -ne 0 ]; then
  echo "SimPO.sh failed"
  exit 1
fi

# Call KTO.sh
chmod +x ./KTO/KTO.sh
./KTO/KTO.sh
if [ $? -ne 0 ]; then
  echo "KTO.sh failed"
  exit 1
fi

# 添加执行权限并并行执行 eval_safety.sh 脚本
chmod +x ./DPO/eval_safety.sh
chmod +x ./ORPO/eval_safety.sh
chmod +x ./SimPO/eval_safety.sh
chmod +x ./KTO/eval_safety.sh

# 并行执行
./DPO/eval_safety.sh &
pid1=$!
./ORPO/eval_safety.sh &
pid2=$!
./SimPO/eval_safety.sh &
pid3=$!
./KTO/eval_safety.sh &
pid4=$!

# 等待所有并行作业完成
wait $pid1
if [ $? -ne 0 ]; then
  echo "DPO/eval_safety.sh failed"
  exit 1
fi

wait $pid2
if [ $? -ne 0 ]; then
  echo "ORPO/eval_safety.sh failed"
  exit 1
fi

wait $pid3
if [ $? -ne 0 ]; then
  echo "SimPO/eval_safety.sh failed"
  exit 1
fi

wait $pid4
if [ $? -ne 0 ]; then
  echo "KTO/eval_safety.sh failed"
  exit 1
fi

echo "All scripts executed successfully"
