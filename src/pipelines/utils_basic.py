import torch
import sys, os
import itertools
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from filter_short_sentences import filter_stopwords, SENT_LEN_TO_EXCLUDE
import spacy
from typing import Dict, List



def get_device_memory():
    n_gpus = torch.cuda.device_count()
    device_memory = {}
    for i in range(n_gpus):
        device_memory[i] = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2
    return device_memory


def compile_model(model, model_name=None):
    if '3.9' in sys.version:
        if model_name is not None:
            print(f'compiling model {model_name}...')
        model = torch.compile(model)
    return model


def batchifier(iterable, n):
    iterable = iter(iterable)
    return iter(lambda: list(itertools.islice(iterable, n)), [])



def chunk_into_sublists(l, n):
    """
    Chunk a flat list into uneven sublists, each of length n[i].

    Parameters:
    * l: list
    * n: list of int, each with the size of the sublist
    """
    total = 0
    list_chunks = []
    for j in range(len(n)):
        chunk = l[total: total + n[j]]
        list_chunks.append(chunk)
        total += n[j]
    return list_chunks



def transpose_dict(d):
    if isinstance(d, list):
        output = {}
        for key in d[0].keys():
            output[key] = list(map(lambda d_i: d_i[key], d))
    else:
        keys = dict(d).keys()
        lists = list(map(lambda k: d[k], keys))
        lists = list(zip(*lists))
        output = list(map(lambda x: dict(zip(keys, x)), lists))

    return output


def get_rank(rank):
    if rank is None:
        # mimic multiprocessing environment
        # import random
        # rank = random.choice([0, 1, 2, 3])
        rank = 0
    return rank


def is_acceptable_sentence(x):
    if not isinstance(x, str):
        x = str(x)
    x = filter_stopwords(x)
    n_words = len(x.split())
    return (n_words > SENT_LEN_TO_EXCLUDE) and (n_words < 50)


to_disable = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "textcat"]
_nlp_sentencizer = None
def get_spacy_sentencizer_model():
    global _nlp_sentencizer
    if _nlp_sentencizer is None:
        _nlp_sentencizer = spacy.load('en_core_web_lg')
        _nlp_sentencizer.add_pipe('sentencizer')
    return _nlp_sentencizer


def sentencize_docs(data: Dict[List, str], text_col: str):
    """
    Take a document and split it into sentences and words.

    * `data`: Dict[List] with column `text_col`, which contains a list of full-text.
    * `text_col`: the column name of the full-text in `data`.
    """

    spacy_model = get_spacy_sentencizer_model()
    list_of_full_text = data[text_col]
    list_of_docs = list(spacy_model.pipe(list_of_full_text, disable=to_disable,))
    list_of_list_of_sents = list(map(lambda doc: list(filter(is_acceptable_sentence, doc.sents)), list_of_docs))
    output = {}
    output['word_lists'] = list(map(lambda sents: list(map(lambda x: list(map(str, x)), sents)), list_of_list_of_sents))
    output['sent_lists'] = list(map(lambda x: list(map(str, x)), list_of_list_of_sents))
    # output['doc_lists'] = list_of_docs
    # output['joined_docs'] = list(map(lambda x: '<NEW_SENT>'.join(x), output['sent_lists']))
    return output

