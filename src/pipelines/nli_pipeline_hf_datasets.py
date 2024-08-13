import sys, os
here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, '..'))

from datasets import Dataset, concatenate_datasets
import glob
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BartForSequenceClassification
)
from torch.nn.functional import softmax
from utils_basic import get_device_memory, compile_model, batchifier, chunk_into_sublists, transpose_dict, get_rank
from more_itertools import flatten
import pandas as pd
import datasets
import multiprocess
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


TESTED_GPU_SIZE = 11011.5
def score_nli(
        example, rank, model=None, tokenizer=None, config=None,
        **kwargs
):
    """
    Function to be parallelized across a `dataset.map` operation. Takes in a pair of documents and scores them.

    Input:
    * example: a dict containing the following keys:
    [
         'article_url', ...
         'word_lists', 'sent_lists', 'best_class'
    ]
    each key points to either a list (if `batch=True`) or a single element

    * rank: of process, assigned by call.
    * model, tokenizer, config: pretrained model needed to score.
    """
    def _score_sent_list(press_release_sents, article_sents):
        """
        Scores a list of sentences and returns (1) error probability (2) a best-class label.
        """
        return_token_type_ids = not isinstance(model, BartForSequenceClassification)
        inputs = tokenizer(
            list(press_release_sents),
            list(article_sents),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=40,
            return_token_type_ids=return_token_type_ids,
        ).to(f"cuda:{rank}")
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = softmax(outputs.logits, dim=1).detach().cpu().numpy().T
        prob_dict = {
            config.id2label[int(x)]: p.tolist() for x, p in enumerate(probabilities)
        }
        return prob_dict

    rank = get_rank(rank)
    model.to(f"cuda:{rank}")

    compile_model(model)
    # handle a non-batched scenario
    if isinstance(example['article_url'], str):
        p, a = [example['press_release_sents']], [example['article_sents']]
    else:
        p, a = example['press_release_sents'], example['article_sents']
    return _score_sent_list(p, a)


def load_data(args, mapping_file):
    """
    Determines the split of files to use, based on slices of the original mapping file

    """
    mapping_file = mapping_file.iloc[args.start_idx: args.end_idx]
    if args.read_type == 'one-dataset':
        dataset = Dataset.load_from_disk(args.input_file)
    else:
        input_files = glob.glob(args.input_file)  # 'cache__split*/cache_files_with_coref.parquet'
        datasets = list(map(Dataset.from_file, input_files))
        dataset = concatenate_datasets(datasets)

    target_urls = set(mapping_file[['URL', 'Target URL']].stack().pipe(set))
    dataset = dataset.filter(
        lambda x: x['article_url'] in target_urls,
        desc='filtering dataset...',
        num_proc=10,
    )
    df = dataset.select_columns(['article_url', 'coref_resolved_sents']).to_pandas()

    # match
    print('matching...')
    mapped_articles = (
        mapping_file
            .merge(df, left_on='URL', right_on='article_url')
            .rename(columns={args.sent_col_to_compare: 'article_sents'})
            .drop(columns=['article_url'])
            .merge(df, left_on='Target URL', right_on='article_url')
            .rename(columns={args.sent_col_to_compare: 'press_release_sents'})
            .drop(columns=['article_url'])
    )

    # cross join on `URL` and `Target URL`
    mapped_articles.set_index(['URL', 'Target URL'], inplace=True)

    # assign sentence idx
    print('exploring...')
    article_sents = (
        mapped_articles
            .assign(article_idx=lambda df: df['article_sents'].apply(lambda x: list(range(len(x)))))
            [['article_idx', 'article_sents']]
    ).explode(['article_idx', 'article_sents'])
    press_release_sents = (
        mapped_articles
        .assign(press_release_idx=lambda df: df['press_release_sents'].apply(lambda x: list(range(len(x)))))
        [['press_release_idx', 'press_release_sents']]
    ).explode(['press_release_idx', 'press_release_sents'])

    print('cross-joining...')
    # cross join
    joined_df = (
        pd.DataFrame(article_sents)
            .join(
            pd.DataFrame(press_release_sents), how='outer', lsuffix='_article', rsuffix='_press_release')
            .reset_index()
            .rename(columns={'URL': 'article_url', 'Target URL': 'press_release_url'})
    ).drop_duplicates(['article_url', 'press_release_url', 'article_sents', 'press_release_sents'])
    ds_to_process = Dataset.from_pandas(joined_df)
    print('done!!!')
    return ds_to_process


# for stance detection, run: 'alex2awesome/stance-detection-classification-model'

if __name__ == '__main__':
    multiprocess.set_start_method("spawn")
    datasets.enable_caching()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='path to input file, or file_pattern')
    parser.add_argument('--read-type', type=str, help='how to ingest the files (one-dataset, dataset-cache-pattern)')
    parser.add_argument('--cache-dir', type=str, default=None, help='path to cache dir')
    parser.add_argument('--output-file', type=str, help='path to output file')
    parser.add_argument('--mapping-file', type=str, help='path to mapping file')
    parser.add_argument('--reload-data', action='store_true', help='whether to reload data')
    parser.add_argument(
        '--sent-col-to-compare', type=str,
        default='coref_resolved_sents',
        help='column name of sentences to compare. Options: ["coref_resolved_sents", "sent_lists"]'
    )

    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--start-pct', type=float, default=None)
    parser.add_argument('--end-pct', type=float, default=None)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default="ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    )

    args = parser.parse_args()

    mapping_file = pd.read_csv(args.mapping_file)
    if args.start_pct is not None:
        args.start_idx = int(args.start_pct * len(mapping_file))
    if args.end_pct is not None:
        args.end_idx = int(args.end_pct * len(mapping_file))
    if (args.start_idx is not None) and (args.end_idx is not None):
        args.output_file = args.output_file.replace('.jsonl', f'__{args.start_idx}-{args.end_idx}.jsonl')
        args.cache_dir = args.cache_dir + f'__{args.start_idx}-{args.end_idx}'

    args.cache_dir = os.path.join(here, args.cache_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    print('loading data...')
    os.environ['HF_DATASETS_CACHE'] = args.cache_dir
    os.environ['HF_HOME'] = args.cache_dir
    # lame caching...
    dataset_path = os.path.join(args.cache_dir, 'filtered-joined-matched.parquet')
    if os.path.exists(dataset_path) and (not args.reload_data):
        ds_to_process = Dataset.load_from_disk(dataset_path)
    else:
        ds_to_process = load_data(args, mapping_file)
        ds_to_process.save_to_disk(dataset_path)

    # load model
    print('loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    # score
    print('scoring...')
    gpu_mem = get_device_memory()
    GPU_SIZE_RATIO = TESTED_GPU_SIZE / gpu_mem.get(0)
    GPU_BATCH_SIZE = int(args.batch_size / GPU_SIZE_RATIO)
    scored_dataset = (
        ds_to_process
            .map(
                score_nli,
                batched=True,
                with_rank=True,
                batch_size=GPU_BATCH_SIZE,
                desc='scoring dataset...',
                num_proc=torch.cuda.device_count(),
                fn_kwargs={'model':model, 'tokenizer': tokenizer, 'config': config},
                cache_file_name=f'{args.cache_dir}/cache_files_nli.parquet',
            )
        )

    # save
    scored_dataset.save_to_disk(os.path.join(args.cache_dir, args.output_file))

"""
python nli_pipeline_hf_datasets.py \
    --input-file 'cache__split*/cache_files_with_coref.parquet' \
    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv.gz \
    --output-file nli-scores.jsonl \
    --cache-dir ../../data/nli-scores-cache \
    --start-pct 0.0 \
    --end-pct 0.1 \
    --batch-size 8 \
    
python nli_pipeline_hf_datasets.py \
    --input-file ../../data/s_p_500_backlinks/coref-resolved-articles__split_gpt-checked-articles/<filename> \
    --read-type one-dataset \
    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv.gz \
    --output-file nli-scores-with-coref-gpt-scored-sample.jsonl \
    --cache-dir ../../data/nli-scores-cache-with-coref-gpt-scored-sample \
    --batch-size 8
"""