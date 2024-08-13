#!/bin/bash
#SBATCH --job-name=coref-resolution
#SBATCH --nodes 1 # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem-per-gpu=60GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

python coref_pipeline_hf_datasets.py  \
    --input-file ../../data/s_p_500_backlinks/parsed-articles-no-pdfs-for-download.jsonl.gz \
    --output-file coref-resolved-articles.jsonl \
    --cache-dir ../../data/s_p_500_backlinks/coref-resolved-articles \
    --read-type parquet \
    --outer-error-filter-gpu-batch-size 5000 \
    --inner-error-filter-gpu-batch-size 240 \
    --outer-coref-resolution-gpu-batch-size 500 \
    --inner-coref-resolution-gpu-batch-size 1 \
    --read-type parquet \
    --start-pct "${1:-55}" \
    --end-pct "${2:-60}" \
    --recalculate

#python coref_pipeline_hf_datasets.py  \
#    --input-file ../../data/s_p_500_backlinks/parsed-articles-no-pdfs-for-download.jsonl.gz \
#    --output-file coref-resolved-articles.jsonl \
#    --cache-dir ../../data/s_p_500_backlinks/coref-resolved-articles \
#    --read-type parquet \
#    --outer-error-filter-gpu-batch-size 5000 \
#    --inner-error-filter-gpu-batch-size 240 \
#    --outer-coref-resolution-gpu-batch-size 60 \
#    --inner-coref-resolution-gpu-batch-size 10 \
#    --url-list-file ../../notebooks/cache/gpt-checked-articles.csv
#
#
