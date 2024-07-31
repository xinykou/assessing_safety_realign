#!/bin/bash

# Get the current script's absolute path
current_script_path=$(realpath "$0")
# Get the directory containing the current script
current_script_dir=$(dirname "$current_script_path")
# Get the parent directory
parent_dir=$(dirname "$current_script_dir")  # scripts/alignment
# Get the subdirectory
sub_dir=$(dirname "$parent_dir")          # scripts
# Get the main directory
main_dir=$(dirname "$sub_dir")           # .

cd "$main_dir"

params=("2  safe_lora-finetune-qwen2_7b"
        "3  safe_lora-finetune-mistral_7b"
        )

# "0  unaligned-finetune-qwen2_7b"
# "1  unaligned-finetune-mistral_7b"
# "0  aligned-finetune-qwen2_7b"
# "1  aligned-finetune-mistral_7b"
# "0  vlguard-finetune-qwen2_7b"
# "1  vlguard-finetune-mistral_7b"
# "2  vaccine-finetune-qwen2_7b"
# "3  vaccine-finetune-mistral_7b"
# "2  lisa-finetune-qwen2_7b"
# "3  lisa-finetune-mistral_7b"
# "2  safe_lora-finetune-qwen2_7b"
# "3  safe_lora-finetune-mistral_7b"
# "2  constrainedSFT-finetune-qwen2_7b"
# "3  constrainedSFT-finetune-mistral_7b"

# Make the script executable if needed
chmod +x "./scripts/baselines/model_size/eval_parallel.sh"

# Run the script in parallel
for param in "${params[@]}"; do
    set -- $param
    ./scripts/baselines/model_size/eval_parallel.sh $1 $2 &
done

# Wait for all background processes to finish
wait

echo "All eval done!"