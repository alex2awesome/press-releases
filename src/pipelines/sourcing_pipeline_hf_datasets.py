import sys, os
here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, '..'))

from datasets import Dataset, concatenate_datasets
import glob
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoConfig,
    BartForSequenceClassification
)
from utils_model import QAModel
from torch.nn.functional import softmax
from utils_basic import get_device_memory, compile_model, batchifier, chunk_into_sublists, transpose_dict, get_rank
from more_itertools import flatten
import pandas as pd
import datasets
import multiprocess
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import re
from unidecode import unidecode

CLEANR = re.compile('<.*?>')


def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

def normalize(text):
    text = '' if pd.isnull(text) else text
    text = re.sub('\s+', ' ', text)
    return cleanhtml(unidecode(text).strip())


TESTED_GPU_SIZE = 11011.5
def score_sent_list(
    example, rank, batch_size, model=None, tokenizer=None, config=None,
    **kwargs
):
    """
    Function to be parallelized across a `dataset.map` operation. Takes in a set of sentences and scores them.

    Input:
    * example: a dict containing the following keys:
    [
         'article_url', 'word_lists', 'sent_lists'
    ]
    each key points to either a list (if `batch=True`) or a single element

    * rank: of process, assigned by call.
    * model, tokenizer, config: pretrained model needed to score.
    """
    def _score_sent_list(sent_lists):
        """
        Scores a list of sentences and returns (1) error probability (2) a best-class label.
        """
        return_token_type_ids = not isinstance(model, BartForSequenceClassification)
        inputs = tokenizer(
            sent_lists,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=200,
            return_token_type_ids=return_token_type_ids,
        ).to(f"cuda:{rank}")
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = softmax(outputs.logits, dim=1).detach().cpu().numpy()
        prob_dict = {
            config.id2label[int(x)]: p.tolist() for x, p in enumerate(probabilities.T)
        }
        best_label = list(map(lambda x: config.id2label[x], probabilities.argmax(axis=1)))
        return transpose_dict(prob_dict), best_label

    rank = get_rank(rank)
    model.to(f"cuda:{rank}")

    compile_model(model)
    doc_lens = list(map(len, example['sent_lists']))
    all_sent_list = list(flatten(example['sent_lists']))
    all_p, all_b = [], []
    for doc_chunk in batchifier(all_sent_list, batch_size):
        p, b = _score_sent_list(doc_chunk)
        all_p += p
        all_b += b
    chunked_p = chunk_into_sublists(all_p, doc_lens)
    chunked_b = chunk_into_sublists(all_b, doc_lens)

    docs = transpose_dict(example)
    output_examples = []
    for p, b, doc in zip(chunked_p, chunked_b, docs):
        doc['quote_type_probs'] = p
        doc['quote_type'] = b
        output_examples.append(doc)
    return transpose_dict(output_examples)


def run_attribution(
    example, rank, batch_size, model=None, tokenizer=None, config=None, **kwargs
):
    def _score_sent_lists_for_attribution(sent_lists, article_text):
        return_token_type_ids = not isinstance(model, BartForSequenceClassification)
        inputs = tokenizer(
            article_text,
            sent_lists,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=int(.8 * config.max_position_embeddings),
            return_token_type_ids=return_token_type_ids,
        ).to(f"cuda:{rank}")
        with torch.no_grad():
            outputs = model(**inputs)
        start_tokens, end_tokens = outputs.start_logits.argmax(dim=1), outputs.end_logits.argmax(dim=1)
        start_tokens, end_tokens = torch.minimum(start_tokens, end_tokens), torch.maximum(start_tokens, end_tokens)

        spans = []
        for ids, start_token, end_token in zip(inputs['input_ids'], start_tokens, end_tokens):
            span = ids[start_token: end_token + 1].detach().cpu().numpy()
            span_text = tokenizer.decode(span)
            spans.append(span_text)
        return spans

    # attribution_dataset.process_output
    rank = get_rank(rank)
    model.to(f"cuda:{rank}")
    compile_model(model)
    doc_lens = list(map(len, example['sent_lists']))
    all_sent_list = list(flatten(example['sent_lists']))
    all_text = list(map(lambda x: [x[1]] * doc_lens[x[0]], enumerate(example['article_text'])))
    all_text = list(flatten(all_text))
    input_data = zip(all_sent_list, all_text)
    all_attributions = []
    for doc_chunk in batchifier(input_data, batch_size):
        batch_sents, batch_text = zip(*doc_chunk)
        all_attributions += _score_sent_lists_for_attribution(batch_sents, batch_text)
    chunked_attributions = chunk_into_sublists(all_attributions, doc_lens)

    docs = transpose_dict(example)
    output_examples = []
    for doc_attributions, doc in zip(chunked_attributions, docs):
        doc['attributions'] = doc_attributions
        output_examples.append(doc)
    return transpose_dict(output_examples)


def load_data(args, url_list):
    """
    Determines the split of files to use, based on slices of the original mapping file

    """
    if args.read_type == 'one-dataset':
        dataset = Dataset.load_from_disk(args.input_file)
    else:
        input_files = glob.glob(args.input_file)  # 'cache__split*/cache_files_with_coref.parquet'
        datasets = list(map(Dataset.from_file, input_files))
        dataset = concatenate_datasets(datasets)

    url_list = set(url_list)
    dataset = dataset.filter(lambda x: x['article_url'] in url_list, num_proc=10)
    return dataset


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
    parser.add_argument('--articles-to-process', type=str, default='articles', help='path to articles to process')

    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--start-pct', type=float, default=None)
    parser.add_argument('--end-pct', type=float, default=None)

    parser.add_argument('--quote-type-batch-size', type=int, default=32)
    parser.add_argument('--attribution-batch-size', type=int, default=2)
    parser.add_argument('--do-quote-type', action='store_true', default=False)
    parser.add_argument('--quote_type_model_name_or_path', type=str,
                        default="alex2awesome/quote-type-prediction__basic")
    parser.add_argument('--do-attribution', action='store_true', default=False)
    parser.add_argument('--attribution_model_name_or_path', type=str,
                        default="alex2awesome/quote-attribution__qa-model-v3")
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.mapping_file)
    if args.articles_to_process == 'articles':
        url_list = mapping_file['URL'].unique()
    else:
        url_list = mapping_file['Target URL'].unique()

    # start/stop idx
    if args.start_pct is not None:
        args.start_idx = int(args.start_pct * len(url_list))
    if args.end_pct is not None:
        args.end_idx = int(args.end_pct * len(url_list))
    if (args.start_idx is not None) and (args.end_idx is not None):
        args.output_file = args.output_file.replace('.jsonl', f'__{args.start_idx}-{args.end_idx}.jsonl')
        args.cache_dir = args.cache_dir + f'__{args.start_idx}-{args.end_idx}'
        url_list = url_list.tolist()[args.start_idx: args.end_idx]

    args.cache_dir = os.path.join(here, args.cache_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    print('loading data...')
    os.environ['HF_DATASETS_CACHE'] = args.cache_dir
    # os.environ['HF_HOME'] = args.cache_dir
    # lame caching...
    dataset_path = os.path.join(args.cache_dir, 'filtered-joined-matched.parquet')
    if os.path.exists(dataset_path) and (not args.reload_data):
        ds_to_process = Dataset.load_from_disk(dataset_path)
    else:
        ds_to_process = load_data(args, url_list)
        ds_to_process.save_to_disk(dataset_path)

    gpu_mem = get_device_memory()
    GPU_SIZE_RATIO = TESTED_GPU_SIZE / gpu_mem.get(0)

    if args.do_quote_type:
        print('loading model...')
        model = AutoModelForSequenceClassification.from_pretrained(args.quote_type_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.quote_type_model_name_or_path)
        config = AutoConfig.from_pretrained(args.quote_type_model_name_or_path)
        GPU_BATCH_SIZE = int(args.quote_type_batch_size / GPU_SIZE_RATIO)
        scored_dataset = (
            ds_to_process
                .map(
                    score_sent_list,
                    batched=True,
                    with_rank=True,
                    batch_size=100,
                    desc='quote-type scoring dataset...',
                    num_proc=torch.cuda.device_count(),
                    fn_kwargs={
                        'model': model, 'tokenizer': tokenizer, 'config': config, 'batch_size': GPU_BATCH_SIZE
                    },
                    cache_file_name=f'{args.cache_dir}/cache_files_quote_type.parquet',
                )
        )
        del model, tokenizer, config

    # load model
    if args.do_attribution:
        # concatenate articles to get full-text, then join full-text with sentences.
        scored_dataset = scored_dataset.map(
            lambda x: {
                'article_text': normalize('journalist passive-voice ' + ' '.join(x['sent_lists'])),
                'sent_lists': list(map(normalize, x['sent_lists'])),
            },
            desc='adding full-text to dataset...',
            num_proc=10,
        )

        model = QAModel.from_pretrained(args.attribution_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.attribution_model_name_or_path)
        config = AutoConfig.from_pretrained(args.attribution_model_name_or_path)
        GPU_BATCH_SIZE = int(args.attribution_batch_size / GPU_SIZE_RATIO)
        scored_dataset = (
            scored_dataset
                .map(
                    run_attribution,
                    batched=True,
                    with_rank=True,
                    batch_size=10,
                    desc='running attribution on dataset...',
                    num_proc=torch.cuda.device_count(),
                    fn_kwargs={
                        'model': model, 'tokenizer': tokenizer, 'config': config, 'batch_size': GPU_BATCH_SIZE
                    },
                    cache_file_name=f'{args.cache_dir}/cache_files_attribution.parquet',
                )
        )

    if args.do_attribution or args.do_quote_type:
        # save
        scored_dataset.save_to_disk(os.path.join(args.cache_dir, args.output_file))


"""
python sourcing_pipeline_hf_datasets.py \
    --input-file cache__split*/cache_files_with_coref.parquet \
    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv.gz \
    --output-file sourcing-scores.jsonl \
    --cache-dir ../../data/sourcing-scores-cache \
    --start-pct 0.0 \
    --end-pct 0.01 \
    --quote-type-batch-size 8 \
    --attribution-batch-size 1
 
    
# endeavor    
python sourcing_pipeline_hf_datasets.py \
    --input-file ../../data/s_p_500_backlinks/all-coref-resolved \
    --read-type one-dataset \
    --mapping-file ../../data/s_p_500_backlinks/article-to-pr-mapper.csv.gz \
    --output-file ../../data/sourcing-scores.jsonl \
    --cache-dir ../../data/sourcing-scores-cache \
    --do-quote-type \
    --do-attribution \
    --start-pct 0.0 \
    --end-pct 0.01 \
    --quote-type-batch-size 32 \
    --attribution-batch-size 2 

"""

"""
# group datasets from server
# assume that attribution-cache and quote-type-cache are in separate files

import glob
import pandas as pd
from datasets import Dataset, concatenate_datasets
from tqdm.auto import tqdm

def first_third_nums(x):
    found_ints = re.findall('\d+', x)
    return (found_ints[0], found_ints[2])

def get_fileparts(df):
    parts_df = (
        df['fname']
            .str.findall('\d+')
            .pipe(lambda s: pd.DataFrame(s.tolist(), columns=['start-idx', 'end-idx', 'part', 'all-parts']))
    )
    return pd.concat([df, parts_df], axis=1)
    
cols_to_use = ['fname', 'start-idx', 'part']
attribution_f = glob.glob('sourcing-scores-cache__*/cache_files_attribution_*_of_00002.parquet')
att_f_df = pd.DataFrame(attribution_f, columns=['fname']).pipe(get_fileparts)
quote_type_f = glob.glob('sourcing-scores-cache__*/cache_files_quote_type_*_of_00002.parquet')
quote_type_df = pd.DataFrame(quote_type_f, columns=['fname']).pipe(get_fileparts)
file_df = att_f_df[cols_to_use].merge(
    quote_type_df[cols_to_use], on=['start-idx', 'part',], suffixes=('_attribution', '_quote_type')
)

file_df = file_df[['fname_attribution', 'fname_quote_type']]
all_dfs = []
for _, (a_f, q_f) in tqdm(file_df.iterrows(), total=len(file_df)):
    a_df = (
        Dataset
            .from_file(a_f)
            .to_pandas()
            [['article_url', 'links', 'target_timestamp', 'coref_resolved_sents', 'attributions']]
            .drop_duplicates(['article_url'])
    )
    q_df = (
        Dataset
            .from_file(q_f)
            .to_pandas()
            [['article_url', 'best_class', 'quote_type_probs', 'quote_type', ]]
            .drop_duplicates(['article_url'])
    )
    full_df = a_df.merge(q_df, on=['article_url']).drop_duplicates(['article_url'])
    (full_df
        .drop(columns=['coref_resolved_sents', 'quote_type_probs'])
        .to_json('partial-source-scored-data.jsonl',  orient='records', lines=True, mode='a')
    )

    
=======================================================================================================
attribution_f = sorted(attribution_f, key=first_third_nums)
quote_type_f = sorted(quote_type_f, key=first_third_nums)
"""