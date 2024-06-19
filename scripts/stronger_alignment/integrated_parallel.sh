# 获取当前脚本的绝对路径
current_script_path=$(realpath "$0")
# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")
cd "$current_script_dir"
echo "current_script_path: $current_script_path"

# 添加执行权限并并行执行 eval_safety.sh 脚本
chmod +x ./expo-sft_to_dpo-lora.sh
chmod +x ./expo-sft_to_orpo-lora.sh
chmod +x ./expo-sft_to_simpo-lora.sh
chmod +x ./expo-sft_to_kto-lora.sh

# 并行执行
./expo-sft_to_dpo-lora.sh &
pid1=$!
./expo-sft_to_orpo-lora.sh &
pid2=$!
./expo-sft_to_simpo-lora.sh &
pid3=$!
./expo-sft_to_kto-lora.sh &
pid4=$!

# 等待所有并行作业完成
wait $pid1
if [ $? -ne 0 ]; then
  echo "stronger_alignment/expo-sft_to_dpo-lora.sh failed"
  exit 1
fi

wait $pid2
if [ $? -ne 0 ]; then
  echo "stronger_alignment/expo-sft_to_orpo-lora.sh failed"
  exit 1
fi

wait $pid3
if [ $? -ne 0 ]; then
  echo "stronger_alignment/expo-sft_to_simpo-lora.sh failed"
  exit 1
fi

wait $pid4
if [ $? -ne 0 ]; then
  echo "stronger_alignment/expo-sft_to_kto-lora.sh failed"
  exit 1
fi

echo "All scripts executed successfully"