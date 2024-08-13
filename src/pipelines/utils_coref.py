import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
import torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from fastcoref.coref_models.modeling_lingmess import LingMessModel
from utils_basic import batchifier, get_rank, compile_model, get_spacy_sentencizer_model, to_disable
from more_itertools import flatten
from resolve_coref import resolve_corefs_sentence_level, convert_word_clusters_to_char_clusters
from fastcoref import LingMessCoref
from tqdm.auto import tqdm
import sys, os


def get_coref_model_fastcoref():
    coref_model = LingMessCoref(enable_progress_bar=True, nlp=None)
    # for i in range(torch.cuda.device_count()):  # send model to every GPU
    #     coref_model.model.to(f'cuda:{i}')
    # return {'coref_model': coref_model}
    coref_model.model = compile_model(coref_model.model, 'LingMessCoref')
    return coref_model


def resolve_coref_with_fastcoref(data: Dict[List, str], rank=None, coref_model=None):
    """
    Take a tokenized document and resolve it's coreferences.

    * `data`: Dict[List] with columns `word_lists`, `docs`, and `sent_lists`.
        * `word_lists`: a list of lists of words.
        * `docs`: a list of spacy documents.
        * `sent_lists`: a list of lists of sentences.
    """
    assert isinstance(data['article_url'], list), "Must be batched..."
    if rank is not None:
        coref_model.model.to(f'cuda:{rank}')
        coref_model.collator.device = torch.device(rank)

    # cluster corefs
    flattened_list_of_words = list(map(lambda x: list(flatten(x)), data['word_lists']))
    list_of_clusters = coref_model.predict(texts=flattened_list_of_words, is_split_into_words=True)
    list_of_full_text = list(map(lambda x: ' '.join(x), data['sent_lists']))
    spacy_model = get_spacy_sentencizer_model()
    list_of_docs = list(spacy_model.pipe(list_of_full_text, disable=to_disable,))
    list_of_resolved_sents = []
    for clusters, doc, sents in zip(list_of_clusters, list_of_docs, data['sent_lists']):
        cluster_word_idxs = clusters.get_clusters(as_strings=False)
        cluster_char_idxs = convert_word_clusters_to_char_clusters(cluster_word_idxs, doc)
        resolved_sents = resolve_corefs_sentence_level(cluster_char_idxs, doc, sents=sents, is_char_clusters=True)
        list_of_resolved_sents.append(resolved_sents)
    if len(list_of_resolved_sents) < len(data):
        list_of_resolved_sents += [None] * (len(data) - len(list_of_resolved_sents))
    data = data.add_column('coref_resolved_sents', list_of_resolved_sents)
    return data


def get_coref_model():
    print('loading coref model...')
    model_name_or_path = 'biu-nlp/lingmess-coref'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = LingMessModel.from_pretrained(model_name_or_path, config=config).eval()
    for i in range(torch.cuda.device_count()):  # send model to every GPU
        model.to(f'cuda:{i}')
    return {'model': model, 'tokenizer': tokenizer, 'config': config}


