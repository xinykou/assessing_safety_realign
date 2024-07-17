current_script_path=$(realpath "$0")

# 获取当前脚本所在的目录
current_script_dir=$(dirname "$current_script_path")

# 获取上一级目录
parent_dir=$(dirname "$current_script_dir")  # scripts/stronger_alignment
sub_dir=$(dirname "$parent_dir") # ./scripts
sub_sub_dir=$(dirname "$sub_dir") # ./


cd $sub_sub_dir
echo "current working directory: ${sub_sub_dir}"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${sub_sub_dir}"
source_type=sft
target_type=dpo

# shellcheck disable=SC2054
alpha_all=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8, 0.9
#tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande
# shellcheck disable=SC2068

# shellcheck disable=SC1061
for alpha in ${alpha_all[@]};do
    python ./export_merged.py \
    --org_model_path  ./saves/lora/${source_type}/checkpoint-125-merged \
    --lora_path /media/data/3/yx/model_merging_v2/saves/lora/expo_dpo_lora/${source_type}_to_${target_type}-alpha_"${alpha}" \
    --save_path /media/data/3/yx/model_merging_v2/saves/lora/expo_dpo_lora/${source_type}_to_${target_type}-alpha_"${alpha}"-merged
done

# shellcheck disable=SC2068
for alpha in ${alpha_all[@]};do
# orginal model
echo "------> Running with alpha=$alpha"
python ./lm_eval/__main__.py \
    --model hf  \
    --model_args pretrained=/media/data/3/yx/model_merging_v2/saves/lora/expo_dpo_lora/${source_type}_to_${target_type}-alpha_"${alpha}"-merged \
    --tasks tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande \
    --batch_size 1 \
    --log_samples \
    --output_path  ./results/lora/ablations/expo_hyparameter/alpha_"${alpha}"
done
