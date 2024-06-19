main_dir="/home/zsf/project/assessing_safety_realign"

cd $main_dir

export WANDB_PROJECT="assessing_safety"
export PYTHONPATH=$main_dir
export CUDA_VISIBLE_DEVICES=0,1
python main.py train config/finetune/SST2.yaml