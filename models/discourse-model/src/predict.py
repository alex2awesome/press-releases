from transformers import AutoConfig, AutoTokenizer
import torch
import sys
import jsonlines
from tqdm.auto import tqdm
sys.path.insert(0, '.')
from copy import copy
from more_itertools import flatten
from unidecode import unidecode
import re
from util import label_mapper
import numpy as np
import pandas as pd
import sqlite3

MAX_DOC_LENGTH = 8192
MAX_SENT_LENGTH = 450
MAX_NUM_SENTS = 110

def get_tokenizer_name(model_name, tokenizer_name):
    if tokenizer_name is not None:
        return tokenizer_name
    if 'roberta-base' in model_name:
        return 'roberta-base'
    elif 'roberta-large' in model_name:
        return 'roberta-large'


def get_model_and_dataset_class(model_type):
    if model_type == 'sentence' :
        from sentence_model import SentenceClassificationModel as model_class
        from sentence_model import TokenizedDataset as dataset_class
    else:
        from full_sequence_model import LongRangeClassificationModel as model_class
        from full_sequence_model import TokenizedDataset as dataset_class
    return model_class, dataset_class


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def load_data(args, tokenizer):
    if '.csv' in args.dataset_name:
        print('reading csv...')
        df = pd.read_csv(args.dataset_name)
    elif '.json' in args.dataset_name:
        print('reading json...')
        df = pd.read_json(args.dataset_name, lines=True)
    elif '.db' in args.dataset_name:
        print('reading from db...')
        conn = sqlite3.connect(args.dataset_name)
        df = pd.read_sql(args.sql_command, conn)

    df = (
        df.loc[lambda df: df[args.text_col].notnull()]
          .drop_duplicates(args.text_col)
          .loc[lambda df: df[args.text_col].str.len() < MAX_DOC_LENGTH * 2]
    )
    if 'sentences' not in df.columns:
        df['sentences'] = sentencize_col(df[args.text_col], tokenizer)
    return list(df.to_dict(orient='records'))


spacy_model = None
def get_spacy_model():
    global spacy_model
    if spacy_model is None:
        import spacy
        print('loading spacy model...')
        spacy_model = spacy.load('en_core_web_lg')
        spacy_model.add_pipe('sentencizer')
    return spacy_model

def sentencize_col(col, tokenizer):
    spacy_model = get_spacy_model()
    doc_sentences = []
    col = (
        col
        .apply(lambda x: tokenizer.encode(x)[:MAX_DOC_LENGTH])
        .apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
    )
    print('sentencizing...')
    for doc in tqdm(spacy_model.pipe(col, disable=[
        "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "textcat"
    ]), total=len(col)):
        sents = list(map(str, doc.sents))
        sents = list(flatten(map(lambda x: x.split('\n'), sents)))
        sents = list(map(lambda x: unidecode(x).strip(), sents))
        sents = list(filter(lambda x: x != '', sents))
        sents = list(map(lambda x: re.sub(r' +', ' ', x), sents))
        sent_lens = list(map(lambda x: len(tokenizer.encode(x)), sents))
        cum_sent_lens = np.cumsum(sent_lens)
        start = np.where(cum_sent_lens > MAX_DOC_LENGTH)[0]
        if len(start) > 0:
            sents = sents[:start[0]]
        if len(sents) > MAX_NUM_SENTS:
            sents = sents[:MAX_NUM_SENTS]
        doc_sentences.append(sents)
    return doc_sentences


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--model_type', default='sentence', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument(
        '--sql-command',
        default='''
            SELECT common_crawl_url as article_url, article_text body 
            FROM article_data 
            WHERE is_press_release_article + is_archival_article = 1
            LIMIT 5
            ''',
        type=str
    )
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--config_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default='roberta-base', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--text-col', default='body', type=str)
    parser.add_argument('--id-col', default='suid', type=str)
    args = parser.parse_args()

    # set naming
    tokenizer_name = get_tokenizer_name(args.model_name_or_path, args.tokenizer_name)
    model_class, dataset_class = get_model_and_dataset_class(args.model_type)

    # load model
    config = AutoConfig.from_pretrained(args.config_name or args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # load in dataset
    data = load_data(args, tokenizer)
    dataset = dataset_class(
        tokenizer=tokenizer, do_score=True, label_mapper=label_mapper, max_length=MAX_SENT_LENGTH
    )
    device = get_device()
    model.eval()
    model = model.to(device)
    output_data = []
    for doc in tqdm(data, total=len(data)):
        if len(doc[args.text_col]) < 5:
            continue

        input_ids, attention_mask, _ = dataset.process_one_doc(doc)
        datum = {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device)
        }

        # score
        scores = model.get_proba(**datum)
        preds = dataset.transform_logits_to_labels(scores, num_docs=len(input_ids))

        # process data
        output_datum = []
        for sent_idx, sent in enumerate(doc['sentences']):
            output_packet = {
                'discourse_preds': preds[sent_idx],
                'sentences': sent,
                'doc_id': doc[args.id_col],
                'sent_idx': sent_idx,
            }
            output_datum.append(output_packet)
        output_data.append(output_datum)

    #
    with open(args.outfile, 'w') as f:
        jsonlines.Writer(f).write_all(output_data)
