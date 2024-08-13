#!/bin/bash
#SBATCH --job-name=nli
#SBATCH --nodes 1 # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem-per-gpu=50GB
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

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
    --model_name_or_path alex2awesome/stance-detection-classification-model


#python nli_pipeline_hf_datasets.py  \
#    --input-file ../../data/s_p_500_backlinks/all-coref-resolved  \
#    --read-type one-dataset \
#    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv \
#    --cache-dir ../../data/s_p_500_backlinks/stance-cache \
#    --output-file coref-resolved-articles.jsonl \
#    --start-pct 0 \
#    --end-pct .05 \
#    --batch-size 256 \
#    --reload-data \
#    --model_name_or_path alex2awesome/stance-detection-classification-model


# --gres=gpu:a40:4 # number of GPUs