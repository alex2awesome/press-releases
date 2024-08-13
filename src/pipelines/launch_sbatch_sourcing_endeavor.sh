#!/bin/bash
#SBATCH --job-name=sourcing
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

python sourcing_pipeline_hf_datasets.py \
    --input-file ../../data/s_p_500_backlinks/all-coref-resolved \
    --read-type one-dataset \
    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv.gz \
    --output-file sourcing-scores.jsonl \
    --cache-dir ../../data/s_p_500_backlinks/sourcing-scores-cache \
    --do-quote-type \
    --do-attribution \
    --start-pct "${1:-.55}" \
    --end-pct "${2:-.56}" \
    --quote-type-batch-size 16 \
    --attribution-batch-size 2 \
    --do-quote-type \
    --do-attribution




