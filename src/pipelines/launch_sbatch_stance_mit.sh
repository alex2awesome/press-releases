#!/bin/bash
#SBATCH --job-name=coref-resolution
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --cpus-per-task 32
#SBATCH --mem-per-cpu 10GB
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

#module load Conda/3
# Activate conda environment and run job commands
cd $HOME
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alex
cd /pool001/spangher/pycharm_project_624/src/pipelines

python nli_pipeline_hf_datasets.py  \
    --input-file ../../data/s_p_500_backlinks/all-coref-resolved  \
    --read-type one-dataset \
    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv \
    --cache-dir ../../data/s_p_500_backlinks/stance-cache \
    --output-file coref-resolved-articles.jsonl \
    --start-pct "${1:-.55}" \
    --end-pct "${2:-.60}" \
    --batch-size 356 \
    --reload-data \
    --model_name_or_path 'alex2awesome/stance-detection-classification-model'

