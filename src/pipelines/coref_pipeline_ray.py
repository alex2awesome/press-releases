import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datasets
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))

from pyarrow import json
from pyarrow.json import ReadOptions
import ray
from typing import Dict, Any, List
import numpy as np
import torch
from utils_basic import get_spacy_sentencizer_model, to_disable, is_acceptable_sentence
from utils_filtering import get_discourse_model_torch
from utils_basic import batchifier, get_rank, get_device_memory, compile_model, transpose_dict
from utils_coref import get_coref_model
from torch.nn.functional import softmax
from tqdm.auto import tqdm
from more_itertools import flatten
from resolve_coref import resolve_corefs_sentence_level, convert_word_clusters_to_char_clusters

from fastcoref.utilities.util import create_mention_to_antecedent, create_clusters, align_to_char_level
from fastcoref.modeling import CorefResult

TESTED_GPU_SIZE = 11011.5

# @ray.remote
class Sentencizer:
    '''Take a document and split it into sentences and words.'''
    def __init__(self):
        self.spacy_model = get_spacy_sentencizer_model()

    def __call__(self, batch: Dict[str, List[Any]], text_col: str) -> Dict[str, List[Any]]:
        '''
        Parameters:
        -----------
            * `batch`: Dict[List] with column `text_col`, which contains a list of full-text.
            * `text_col`: the column name of the full-text in `data`.
        '''
        list_of_full_text = batch[text_col]
        list_of_docs = list(self.spacy_model.pipe(list_of_full_text, disable=to_disable, ))
        list_of_list_of_sents = list(map(lambda doc: list(filter(is_acceptable_sentence, doc.sents)), list_of_docs))
        batch['word_lists'] = '' # list(
        #     map(lambda sents: list(map(lambda x: list(map(str, x)), sents)), list_of_list_of_sents))
        batch['sent_lists'] = list(map(lambda x: list(map(str, x)), list_of_list_of_sents))
        return batch

# @ray.remote
class ErrorFilter:
    '''Take a document and filter out the sentences that are likely to have errors.'''
    def __init__(self, verbose=False):
        discourse_package = get_discourse_model_torch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = discourse_package['tokenizer']
        self.config = discourse_package['config']
        self.model = discourse_package['model']
        self.model = compile_model(self.model).to(self.device)
        self.GPU_SIZE_RATIO = TESTED_GPU_SIZE / get_device_memory()[0]
        self.GPU_BATCH_SIZE = int(10 / self.GPU_SIZE_RATIO)
        self.verbose = verbose


    def _score_one_doc(self, doc: List[str]):
        """
            Scores a single document and (1) filters out errors (2) assigns a best-class label.
        """
        error_probs, best_class = [], []
        for doc_chunk in batchifier(doc, self.GPU_BATCH_SIZE):
            inputs = self.tokenizer(doc_chunk, padding=True, truncation=True, return_tensors="pt", max_length=40)
            inputs = {
                k: inputs[k].to(self.device) for k in ['input_ids', 'attention_mask']
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
            probabilities = softmax(outputs.logits, dim=1).cpu()
            target_class = self.config.label2id['NA']
            error_probs += probabilities[:, target_class].tolist()
            best_class += list(map(lambda x: self.config.id2label[int(x)], probabilities.argmax(dim=1)))
        return error_probs, best_class

    def _filter_one_doc(self, doc, error_probs, best_class):
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

    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        if isinstance(batch['article_url'], str):
            error_scores, best_class = self._score_one_doc(batch['sent_lists'])
            return self._filter_one_doc(batch, error_scores, best_class)
        else:
            output_examples = []
            docs = transpose_dict(batch)
            if self.verbose:
                docs = tqdm(docs, desc=f'inner error filter:')
            for idx, doc in enumerate(docs):
                e, b = self._score_one_doc(doc['sent_lists'])
                doc = self._filter_one_doc(doc, e, b)
                if doc is not None:
                    output_examples.append(doc)
            return transpose_dict(output_examples)

# @ray.remote
class CorefResolution:
    '''Take a document and resolve the coreferences.'''

    def __init__(self, verbose=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coref_model = get_coref_model()
        self.model = compile_model(coref_model['model'].to(self.device))
        self.tokenizer = coref_model['tokenizer']
        self.verbose = verbose
        self.spacy_model = get_spacy_sentencizer_model()

    def _tokenize_data(self, word_lists):
        # encode
        word_lists = list(map(lambda x: list(flatten(x)), word_lists))
        encoded_batch = self.tokenizer(
            word_lists, add_special_tokens=True, is_split_into_words=True, return_length=True,
            padding=True, truncation=True, return_tensors="pt", max_length=4096
        )
        batch = {
            'tokens': word_lists,
            'input_ids': encoded_batch['input_ids'].to(self.device),
            'attention_mask': encoded_batch['attention_mask'].to(self.device),
            'length': encoded_batch['length'],
            'subtoken_map': [enc.word_ids for enc in encoded_batch.encodings],
            'new_token_map': [list(range(len(w))) for w in word_lists],
            'offset_mapping': [(list(zip(range(len(w)), range(1, 1 + len(w))))) for w in word_lists],
        }
        return batch

    def _run_lingmess(self, _word_lists):
        # encode
        batch  = self._tokenize_data(_word_lists)
        # run through model
        with torch.no_grad():
            outputs = self.model(batch, return_all_outputs=True)
        outputs_np = []
        for tensor in outputs:
            outputs_np.append(tensor.cpu().numpy())
        results = self._post_process_lingmess(outputs_np, batch)
        return results

    def _post_process_lingmess(self, outputs_np, batch):
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

    def __call__(self, data):
        # run model
        list_of_clusters = []
        batch_size = 8
        batches = batchifier(data['word_lists'], batch_size)
        if self.verbose:
            batches = tqdm(batches, total=int(len(data['word_lists']) / batch_size), desc='inner-loop...')
        for word_list_batch in batches:
            list_of_clusters += self._run_lingmess(word_list_batch)

        # post-process
        list_of_full_text = list(map(lambda x: ' '.join(x), data['sent_lists']))
        list_of_docs = list(self.spacy_model.pipe(list_of_full_text, disable=to_disable, ))
        list_of_resolved_sents = []
        for clusters, doc, sents in zip(list_of_clusters, list_of_docs, data['sent_lists']):
            cluster_word_idxs = clusters.get_clusters(as_strings=False)
            cluster_char_idxs = convert_word_clusters_to_char_clusters(cluster_word_idxs, doc)
            resolved_sents = resolve_corefs_sentence_level(cluster_char_idxs, doc, sents=sents, is_char_clusters=True)
            list_of_resolved_sents.append(resolved_sents)
        data['coref_resolved_sents'] = list_of_resolved_sents
        return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)

    args = parser.parse_args()

    block_size_10MB = 100 << 20
    table = json.read_json(args.input_file, read_options=ReadOptions(block_size=block_size_10MB))
    ds = ray.data.from_arrow(table)
    ds = ds.select_columns(['article_url', 'article_text'])

    ds = ( ds
            .repartition(1000)
            .filter(
                lambda x: (len(x['article_text']) > 0) and (len(x['article_text']) < 1_000_000),
            )
            .map_batches(
                Sentencizer,
                concurrency=10,
                batch_size=1000,
                fn_kwargs={'text_col': 'article_text'},
                batch_format="pandas"

            )
            .filter(lambda x: len(x['sent_lists']) > 0)
            .map_batches(
                ErrorFilter,
                concurrency=4,
                batch_size=1000,
                num_gpus=1, # num GPUs per
                batch_format="numpy"
            )
        )
    print(ds.take(1))


"""
python coref_pipeline_ray.py \
   --input-file ../../data/s_p_500_backlinks/parsed-articles-no-pdfs.jsonl \
   --output-file ../../data/s_p_500_backlinks/parsed-articles-no-pdfs-coref.jsonl
"""


'''

import pyarrow.json as pajson
block_size = 10 << 20 # Set block size to 10MB
ray.data.read_json(  
    "s3://anonymous@ray-example-data/log.json",
    read_options=pajson.ReadOptions(block_size=block_size)
)

'''