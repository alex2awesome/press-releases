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

python nli_pipeline_hf_datasets.py  \
    --input-file ../../data/s_p_500_backlinks/all-coref-resolved  \
    --read-type one-dataset \
    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv \
    --cache-dir ../../data/s_p_500_backlinks/nli-cache \
    --output-file coref-resolved-articles.jsonl \
    --start-pct "${1:-.55}" \
    --end-pct "${2:-.60}" \
    --batch-size 356 \
    --reload-data