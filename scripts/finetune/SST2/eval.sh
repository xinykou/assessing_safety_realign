main_dir="/home/zsf/project/assessing_safety_realign"

cd $main_dir

model_path=./saves/lora/sft/checkpoint-125-merged
lora_path=dpo #
output_path=./results/lora/finetune/${lora_path}_safety.json

export CUDA_VISIBLE_DEVICES=2

# safety evaluation
python ./evaluation/poison/pred.py \
  --model_folder ${model_path} \
  --lora_folder ./saves/lora/finetune/${lora_path} \
	--instruction_path BeaverTails \
	--start 0 \
	--end 1000 \
	--output_path ${output_path} \

python ./evaluation/poison/eval_safety.py \
  --safety_evaluator_path /media/4/yx/model_cache/beaver-dam-7b \
  --input_path ${output_path}

output_path=./results/lora/finetune/${lora_path}_downstream.json

# downstream evaluation
python evaluation/downstream_task/sst2_eval.py \
  --model_folder ${model_path} \
  --lora_folder ./saves/lora/finetune/${lora_path} \
	--start 0 \
	--end 1000 \
	--output_path ${output_path} \
