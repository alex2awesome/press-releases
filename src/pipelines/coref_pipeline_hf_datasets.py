import sys, os

import datasets

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))

# import sys
# sys.path.insert(0, '..')
import pandas as pd
from datasets import load_dataset, Value
from datasets import Features, Dataset, DatasetDict
import multiprocess
import torch
import os
import re
from pyarrow import json
from pyarrow.json import ReadOptions
import pyarrow as pa
import time
import logging
import warnings
import spacy
from fastcoref import spacy_component
from utils_basic import (
    sentencize_docs, transpose_dict, batchifier, get_rank, get_device_memory, compile_model, chunk_into_sublists
)
from utils_filtering import get_discourse_model_torch
from tqdm.auto import tqdm
from utils_coref import get_coref_model, resolve_coref_with_fastcoref, get_coref_model_fastcoref
from torch.nn.functional import softmax

from fastcoref.utilities.util import encode
from fastcoref.utilities.collate import DynamicBatchSampler
from fastcoref.utilities.collate import LeftOversCollator, DynamicBatchSampler, PadCollator
from fastcoref.utilities.util import create_mention_to_antecedent, create_clusters, align_to_char_level
from fastcoref.modeling import CorefResult

from utils_basic import get_spacy_sentencizer_model, to_disable
import numpy as np
from more_itertools import flatten
from resolve_coref import resolve_corefs_sentence_level, convert_word_clusters_to_char_clusters

warnings.simplefilter(action='ignore', category=FutureWarning)

TESTED_GPU_SIZE = 11011.5

# fastcoref_logger = logging.getLogger('fastcoref')
# fastcoref_logger.setLevel(logging.ERROR)

datasets_logger = logging.getLogger('datasets')
datasets_logger.setLevel(logging.ERROR)

USE_SPACY_COREF = True

def load_data(
    input_file, cache_dir, n_rows=None,
    start_idx=None, end_idx=None, start_pct=None, end_pct=None, url_list_file=None,
    read_type='datasets'
):
    dataset_cache_path = os.path.join(cache_dir, 'orig-dataset')
    if os.path.exists(dataset_cache_path):
        return Dataset.load_from_disk(dataset_cache_path)

    print('loading data...')
    cols_df = pd.read_json(input_file, lines=True, nrows=n_rows or 100)
    block_size_10MB = 100 << 20
    split = 'train'
    if (start_pct is not None) or (end_pct is not None):
        if start_pct is None:
            start_pct = 0
        if end_pct is None:
            end_pct = 100
        split = f'train[{start_pct}%:{end_pct}%]'
    if (start_idx is not None) or (end_idx is not None):
        if start_idx is None:
            start_idx = ''
        if end_idx is None:
            end_idx = ''
        split = f'train[{start_idx}:{end_idx}]'

    if args.n_rows is None:
        if read_type == 'datasets':
            cols_table = pa.Table.from_pandas(cols_df)
            features = Features.from_arrow_schema(cols_table.schema)
            dataset = load_dataset(
                'json',
                data_files=args.input_file,
                chunksize=block_size_10MB,
                features=features,
                # num_proc=10,
                cache_dir=cache_dir,
                split=split,
                # download_mode=DownloadMode.FORCE_REDOWNLOAD
            )
        elif read_type == 'parquet':
            now = time.time()
            table = json.read_json(args.input_file, read_options=ReadOptions(block_size=block_size_10MB))
            print(f'reading pyarrow {time.time() - now}')
            df = pa.Table.to_pandas(table)
            print(f'done transforming to pandas: {time.time() - now}')
            if start_pct is not None and end_pct is not None:
                start_idx = int(start_pct / 100 * len(df))
                end_idx = int(end_pct / 100 * len(df))
            if start_idx is not None and end_idx is not None:
                df = df.iloc[start_idx:end_idx]
            dataset = Dataset.from_pandas(df)
    else:
        dataset = Dataset.from_pandas(cols_df)
    if url_list_file is not None:
        url_list = set(pd.read_csv(url_list_file, index_col=0)['0'].tolist())
        dataset = dataset.filter(lambda x: x['article_url'] in url_list, num_proc=10)

    print('done loading data...')
    dataset.save_to_disk(dataset_cache_path)
    return dataset


def filter_errors(
        example, rank, default_gpu_batch_size,
        model=None, tokenizer=None, config=None, verbose=True,
        **kwargs
):
    """
    Function to be parallelized across a `dataset.map` operation. Does several things:
    1. Takes in a document or batch of documents
    2. Scores each with a discourse classifier
    3. Identifies and filters out error sentences
    4. Assigns a best-class score

    Input:
    * example: a dict containing the following keys:
    [
         'article_url', 'wayback_url',

        # key/timestamp values (pretty useless here)
         'target_timestamp_key', 'target_timestamp', 'wayback_timestamp',

        # values set by scraping process or other. 100% useless.
         'sort_criteria', 'method', 'links', 'article_text',

        # added attributes
         'word_lists', 'sent_lists', 'best_class'
    ]

    each key points to either a list or a single element (if `batch=True`)

    * rank: of process, assigned by call.
    * model, tokenizer, config: pretrained model needed to score.
    """
    def _score_sent_list(sent_list):
        """
        Scores a list of sentences and returns (1) error probability (2) a best-class label.
        """
        inputs = tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt", max_length=40)
        inputs = {k: inputs[k].to(f"cuda:{rank}") for k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=1).cpu()
        #
        target_class = config.label2id['NA']
        error_probs = probabilities[:, target_class].tolist()
        best_class = list(map(lambda x: config.id2label[int(x)], probabilities.argmax(dim=1)))
        return error_probs, best_class

    def _filter_one_doc(doc, error_probs, best_class):
        """
        Filters one document if p(error) > .1

        Returns:
            * dict or None
                dict: {key: <filtered list> for key in ['sent_lists', 'word_lists', ...]}
                None: if \nexists rows where p(error) < .1
        """
        packet = zip(doc['sent_lists'], doc['word_lists'], best_class, error_probs)
        packet = list(filter(lambda x: x[3] < .1, packet))
        if len(packet) > 0:
            doc['sent_lists'], doc['word_lists'], best_class, _ = zip(*packet)
            doc['best_class'] = best_class
            return doc

    rank = get_rank(rank)
    model.to(f"cuda:{rank}")
    gpu_mem = get_device_memory()
    GPU_SIZE_RATIO = TESTED_GPU_SIZE / gpu_mem[rank]
    GPU_BATCH_SIZE = int(default_gpu_batch_size / GPU_SIZE_RATIO)

    compile_model(model)
    # handle a non-batched scenario
    if isinstance(example['article_url'], str):
        error_scores, best_class = _score_sent_list(example['sent_lists'])
        return _filter_one_doc(example, error_scores, best_class)
    # handle a batched scenario
    else:
        output_examples = []

        # if verbose:
        #     docs = tqdm(docs, desc=f'inner-{rank}, error filter:')
        # heuristics to filter garbage sentences
        # 1. filter out sentences that are too long
        example['sent_lists'] = list(map(lambda x: list(filter(lambda y: len(y) < 1000, x)), example['sent_lists']))
        # 2. filter out sentences with too many \n
        example['sent_lists'] = list(map(lambda x: list(filter(lambda y: y.count('\n') < 10, x)), example['sent_lists']))
        # 3. filter out sentences too few words in betweeh each \n, based on mean counts
        def _count_words_between_newlines(sent):
            counts = list(map(lambda x: len(x.split()), sent.split('\n')))
            return np.mean(counts)
        example['sent_lists'] = list(map(lambda x: list(filter(lambda y: _count_words_between_newlines(y) > 4, x)), example['sent_lists']))

        doc_lens = list(map(len, example['sent_lists']))
        all_sent_list = list(flatten(example['sent_lists']))
        all_e, all_b = [], []
        for doc_chunk in batchifier(all_sent_list, GPU_BATCH_SIZE):
            e, b = _score_sent_list(doc_chunk)
            all_e += e
            all_b += b
        chunked_e = chunk_into_sublists(all_e, doc_lens)
        chunked_b = chunk_into_sublists(all_b, doc_lens)
        assert list(map(len, chunked_e)) == list(map(len, chunked_b)) == doc_lens

        docs = transpose_dict(example)
        for e, b, doc in zip(chunked_e, chunked_b, docs):
            doc = _filter_one_doc(doc, e, b)
            if doc is not None:
                output_examples.append(doc)
        return transpose_dict(output_examples)


def resolve_coref_with_spacy_model(
        data, rank, default_gpu_batch_size,
        tokenizer=None, model=None, config=None, verbose=True,
        **kwargs
):
    rank = get_rank(rank)
    device = f"cuda:{rank}"
    spacy.require_gpu(rank)
    sentencizer_model = get_spacy_sentencizer_model()
    coref_model = spacy.load('en_core_web_lg')
    coref_model.add_pipe(
        "fastcoref",
        config={
            'model_architecture': 'LingMessCoref',
            'model_path': 'biu-nlp/lingmess-coref',
            'device': device,
            # 'max_tokens_in_batch': 100_000,
        }
    )
    # needs to run the sentences through spacy again to sentencize, which is a little inefficient, but ...
    list_of_full_text = list(map(lambda x: ' '.join(x), data['sent_lists']))
    list_of_full_text = list(map(lambda x: x.replace(' .', '.').replace('\n', ' '), list_of_full_text))
    list_of_full_text = list(map(lambda x: re.sub('\s+', ' ', x), list_of_full_text))
    # filter out long docs
    max_doc_len = 3_000
    list_of_tokenized_text = tokenizer(list_of_full_text)['input_ids']
    list_of_tokenized_text = list(map(lambda x: x[:max_doc_len], list_of_tokenized_text))
    list_of_full_text = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True).strip(), list_of_tokenized_text))

    res = list(coref_model.pipe(
        list_of_full_text,
        # batch_size=5,
        component_cfg={"fastcoref": {
            'resolve_text': True,
            'attempt_recover': True,
            'batch_size': default_gpu_batch_size,
            'inference_progress': default_gpu_batch_size > 100,
        }}
    ))
    res = list(map(lambda doc: doc._.resolved_text, res))
    res = list(sentencizer_model.pipe(res, disable=to_disable, ))
    data['coref_resolved_sents'] = list(map(lambda x: list(map(str, x.sents)), res))
    return data


def resolve_coref_with_implemented_model(
        data, rank, default_gpu_batch_size,
        tokenizer=None, model=None, config=None, verbose=True,
        **kwargs
):
    """
    Caution: The character idxes used in the end of this function don't work.
    However, the  clusters generated in `_post_process_lingmess` seem to be
    pretty good, even if they don't have all the clusters that get automatically resolved by
    the spacy function above.
    """
    def _post_process_lingmess(outputs_np, batch):
        # post-process
        span_starts, span_ends, mention_logits, coref_logits = outputs_np
        doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)
        results = []
        for i in range(len(batch['tokens'])):
            doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
            predicted_clusters = create_clusters(doc_mention_to_antecedent)
            char_map, reverse_char_map = align_to_char_level(
                span_starts[i], span_ends[i], batch['offset_mapping'][i], batch['subtoken_map'][i]
            )
            result = CorefResult(
                text=batch['tokens'][i], clusters=predicted_clusters,
                char_map=char_map, reverse_char_map=reverse_char_map,
                coref_logit=coref_logits[i], text_idx=batch['idx'][i]
            )
            results.append(result)
        return results

    def _run_lingmess(_word_lists, _tokenizer, _model, device):
        # encode
        _word_lists = list(map(lambda x: list(flatten(x)), _word_lists))
        offset_mapping = [(list(zip(range(len(w)), range(1, 1 + len(w))))) for w in _word_lists]
        encoded_batch = _tokenizer(
            _word_lists, add_special_tokens=True, is_split_into_words=True, return_length=True,
            padding=True, truncation=True, return_tensors="pt", max_length=4096
        )
        batch = {
            'tokens': _word_lists,
            'input_ids': encoded_batch['input_ids'].to(device),
            'attention_mask': encoded_batch['attention_mask'].to(device),
            'length': encoded_batch['length'],
            'subtoken_map': [enc.word_ids for enc in encoded_batch.encodings],
            'new_token_map': [list(range(len(w))) for w in _word_lists],
            'offset_mapping': offset_mapping,
            'idx': range(len(_word_lists))
        }
        # run through model
        with torch.no_grad():
            outputs = model(batch, return_all_outputs=True)
        outputs_np = []
        for tensor in outputs:
            outputs_np.append(tensor.cpu().numpy())
        results = _post_process_lingmess(outputs_np, batch)
        return results

    rank = get_rank(rank)
    device = f"cuda:{rank}"

    # post-process
    list_of_full_text = list(map(lambda x: ' '.join(x), data['sent_lists']))
    list_of_full_text = list(map(lambda x: x.replace(' .', '.').replace('\n', ' '), list_of_full_text))
    list_of_full_text = list(map(lambda x: re.sub('\s+', ' ', x), list_of_full_text))
    model.to(device)
    model = compile_model(model)

    # run model
    list_of_clusters = []
    batches = batchifier(data['word_lists'], default_gpu_batch_size)
    if verbose:
        batches = tqdm(batches, total=int(len(data['word_lists']) / default_gpu_batch_size), desc='inner-loop...')
    for word_list_batch in batches:
        list_of_clusters += _run_lingmess(word_list_batch, tokenizer, model, device)
    #
    spacy_model = get_spacy_sentencizer_model()
    list_of_docs = list(spacy_model.pipe(list_of_full_text, disable=to_disable,))
    list_of_resolved_sents = []
    for clusters, doc in zip(list_of_clusters, list_of_docs):
        cluster_word_idxs = clusters.get_clusters(as_strings=False)
        cluster_char_idxs = convert_word_clusters_to_char_clusters(cluster_word_idxs, doc)
        sents = list(map(lambda x: x.text, doc.sents))
        resolved_sents = resolve_corefs_sentence_level(
            cluster_char_idxs, doc, sents=sents, is_char_clusters=True, use_non_nouns_as_head=True
        )
        list_of_resolved_sents.append(resolved_sents)
    data['coref_resolved_sents'] = list_of_resolved_sents
    return data

# 1. load and tokenize data
# 2. spark score with discourse to remove errors
# 3. resolve coreferences
# 4. spark score NLI
if __name__ == '__main__':
    multiprocess.set_start_method("spawn")
    datasets.enable_caching()
    NUM_CPU_PROCS = 10
    NUM_GPU_PROCS = torch.cuda.device_count()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--n-rows', type=int, default=None)
    parser.add_argument('--num-processes', type=int, default=None)
    parser.add_argument('--recalculate', action='store_true',)
    #
    parser.add_argument('--outer-error-filter-gpu-batch-size', type=int, default=1000)
    parser.add_argument('--inner-error-filter-gpu-batch-size', type=int, default=8)
    parser.add_argument('--outer-coref-resolution-gpu-batch-size', type=int, default=1000)
    parser.add_argument('--inner-coref-resolution-gpu-batch-size', type=int, default=2)
    #
    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--start-pct', type=int, default=None)
    parser.add_argument('--end-pct', type=int, default=None)
    parser.add_argument('--url-list-file', type=str, default=None)
    parser.add_argument('--read-type', type=str, default='datasets')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    if (args.start_idx is not None) or (args.end_idx is not None):
        if args.start_idx is None:
            args.start_idx = 'start'
        if args.end_idx is None:
            args.end_idx = 'end'
        args.cache_dir = f'{args.cache_dir}__split_{args.start_idx}-{args.end_idx}'
    if (args.start_pct is not None) or (args.end_pct is not None):
        if args.start_pct is None:
            args.start_pct = 0
        if args.end_pct is None:
            args.end_pct = 100
        args.cache_dir = f'{args.cache_dir}__split_{args.start_pct}-{args.end_pct}'
    if args.url_list_file is not None:
        args.cache_dir = f'{args.cache_dir}__split_{os.path.basename(args.url_list_file).split(".")[0]}'

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # 1. load and tokenize data
    os.environ['HF_DATASETS_CACHE'] = os.path.join(here, args.cache_dir)
    os.environ['HF_HOME'] = os.path.join(here, args.cache_dir)
    dataset = load_data(
        args.input_file, args.cache_dir, args.n_rows, args.start_idx, args.end_idx, args.start_pct, args.end_pct,
        args.url_list_file,
        args.read_type
    )
    dataset = dataset.filter(
        lambda x: (len(x['article_text']) > 0) and (len(x['article_text']) < 1_000_000),
        batch_size=10_000,
        num_proc=10  # int(os.cpu_count() / 2)
    )
    dataset = dataset.map(
        sentencize_docs,
        fn_kwargs={'text_col': 'article_text'},
        batched=True,
        num_proc=10,  # args.num_processes or os.cpu_count(),
        cache_file_name=f'{args.cache_dir}/cache_sentencized_files.parquet',
        load_from_cache_file=not args.recalculate,
        batch_size=100,
        desc='Sentencizing...'
    )

    dataset = dataset.filter(
        lambda x: len(x['sent_lists']) > 0,
        batch_size=10_000,
        num_proc=10 # os.cpu_count()
    )

    # 2. score with discourse to remove errors
    kwargs = get_discourse_model_torch()
    kwargs['verbose'] = args.verbose
    kwargs['default_gpu_batch_size'] = args.inner_error_filter_gpu_batch_size
    GPU_SIZE_RATIO = TESTED_GPU_SIZE / get_device_memory().get(0)
    GPU_BATCH_SIZE = int(kwargs['default_gpu_batch_size'] / GPU_SIZE_RATIO)
    print(f'Running error filtering with batch size: {GPU_BATCH_SIZE}')
    dataset_with_discourse = dataset.map(
        filter_errors,
        with_rank=True,
        batched=True,
        num_proc=args.num_processes or torch.cuda.device_count(),
        fn_kwargs=kwargs,
        batch_size=args.outer_error_filter_gpu_batch_size,
        cache_file_name=f'{args.cache_dir}/cache_files_with_discourse.parquet',
        load_from_cache_file=not args.recalculate,
        desc='Filtering errors and discourse...'
    )

    del kwargs['model'], kwargs['tokenizer'], kwargs['config'], kwargs
    # 3. resolve coreferences
    USE_MAP = True
    if USE_MAP:
        kwargs = get_coref_model()
        kwargs['verbose'] = args.verbose
        kwargs['default_gpu_batch_size'] = args.inner_coref_resolution_gpu_batch_size
        resolve_coref_with_model = resolve_coref_with_spacy_model if USE_SPACY_COREF else resolve_coref_with_implemented_model
        dataset_coref_resolved = dataset_with_discourse.map(
            resolve_coref_with_model,
            with_rank=True,
            batched=True,
            num_proc=torch.cuda.device_count(),  # Lingmess can't handle parallel processing... torch.cuda.device_count(),
            fn_kwargs=kwargs,
            batch_size=args.outer_coref_resolution_gpu_batch_size,
            load_from_cache_file=False,
            cache_file_name=f'{args.cache_dir}/cache_files_with_coref.parquet',
            desc='Resolving Coref...'
        )

    else:
        dataset_coref_resolved = resolve_coref_with_fastcoref(
            dataset_with_discourse,
            coref_model=get_coref_model_fastcoref()
        )

    dataset_coref_resolved.save_to_disk(os.path.join(args.cache_dir, args.output_file))



    """
    python coref_pipeline_hf_datasets.py \
        --input-file ../../data/s_p_500_backlinks/parsed-articles-no-pdfs-for-download.jsonl.gz \
        --output-file coref-resolved-articles.jsonl \
        --cache-dir ../../data/s_p_500_backlinks/coref-resolved-articles \
        --read-type parquet \
        --outer-error-filter-gpu-batch-size 5000 \
        --inner-error-filter-gpu-batch-size 240 \
        --outer-coref-resolution-gpu-batch-size 5000 \
        --inner-coref-resolution-gpu-batch-size 180 \
        --url-list-file ../../notebooks/cache/gpt-checked-articles.csv 
         
    --input-file ../../data/s_p_500_backlinks/parsed-articles-no-pdfs.jsonl \
    --input-file ../../data/s_p_500_backlinks/bigger-parsed-articles-for-tests.jsonl.gz \
    """

    # from pyarrow import json
    # full_fn_no_pdfs = '../../data/s_p_500_backlinks/parsed-articles-no-pdfs.jsonl'
    # block_size_10MB = 100 << 20
    # from pyarrow.json import ReadOptions
    # import pyarrow as pa
    # import time
    # now = time.time()
    # table = json.read_json(full_fn_no_pdfs, read_options=ReadOptions(block_size=block_size_10MB))
    # print(f'reading pyarrow {time.time() - now}')
    # df = pa.Table.to_pandas(table)
    # print(f'done transforming to pandas: {time.time() - now}')
    #


"""
0-10: 7
10-20: 6
20-30: 10
30-40: 11
40-50: 12
50-60: 13
60-70: 14
70-80: 5
80-90: 2
90-100: 1
"""

