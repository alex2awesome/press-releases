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
python coref_pipeline_hf_datasets.py  \
    --input-file ../../data/s_p_500_backlinks/parsed-articles-no-pdfs-for-download.jsonl.gz  \
    --output-file ../../data/s_p_500_backlinks/coref-resolved-articles.jsonl \
    --outer-error-filter-gpu-batch-size 5000   \
    --inner-error-filter-gpu-batch-size 600 \
    --outer-coref-resolution-gpu-batch-size 5000  \
    --inner-coref-resolution-gpu-batch-size 360
#    --recalculate


