main_dir="/home/zsf/project/assessing_safety_realign"

cd $main_dir

model_path=./saves/lora/sft/checkpoint-125-merged
lora_path=SST2/DPO/n1000_p0 #
output_path=./results/lora/finetune/${lora_path}_downstream.json

export CUDA_VISIBLE_DEVICES=4

python evaluation/downstream_task/sst2_eval.py \
  --model_folder ${model_path} \
  --lora_folder ./saves/lora/finetune/${lora_path} \
	--start 0 \
	--end 1000 \
	--output_path ${output_path} \
