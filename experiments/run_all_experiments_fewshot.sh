#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi


#python run_opensource_model.py  --model_id mixtral --prompt_type full
#python run_opensource_model.py  --model_id mixtral --prompt_type separate

#python run_opensource_model.py  --model_id command-r --prompt_type full
#python run_opensource_model.py  --model_id command-r --prompt_type separate

#python run_opensource_model.py  --model_id llama-7b --prompt_type full
python run_opensource_model.py  --model_id llama-7b --prompt_type separate

#python run_opensource_model.py  --model_id llama-70b --prompt_type full
python run_opensource_model.py  --model_id llama-70b --prompt_type separate