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
    --cache-dir ../../data/s_p_500_backlinks/nli-cache \
    --output-file coref-resolved-articles.jsonl \
    --start-pct "${1:-.55}" \
    --end-pct "${2:-.60}" \
    --batch-size 3560 \
    --reload-data

#
#python nli_pipeline_hf_datasets.py \
#    --input-file ../../data/s_p_500_backlinks/coref-resolved-articles__split_gpt-checked-articles/coref-resolved-articles.jsonl \
#    --read-type one-dataset \
#    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv.gz \
#    --output-file nli-scores-with-coref-gpt-scored-sample.jsonl \
#    --cache-dir ../../data/nli-scores-cache-with-coref-gpt-scored-sample \
#    --batch-size 8

# --gres=gpu:a40:4 # number of GPUs
#anti-join newly processed:
# -----------------------------------------------------------------------
#import glob
#from datasets import Dataset, concatenate_datasets
#files = glob.glob('nli-cache__*/coref-resolved-articles*')
#from tqdm.auto import tqdm
#d = Dataset.load_from_disk('partially-full-nli-dataset')
#already_processed_a_urls = set(d['article_url'])
#already_processed_pr_urls = set(d['press_release_url'])
#
#new_dis = []
#for f_i in tqdm(files):
#    d_i = Dataset.load_from_disk(f_i)
#    print(f_i)
#    t = (d_i[0]['article_url'] not in already_processed_a_urls) or (d_i[0]['press_release_url'] not in already_processed_pr_urls)
#    if t:
#        new_dis.append(d_i)
#
#new_di = concatenate_datasets(new_dis)
#new_di.save_to_disk('newly-processed-nli-dataset')