import itertools
import os

from transformers import (AutoModel, AutoTokenizer, AutoModelForSequenceClassification)
import xopen
import orjson
from itertools import product
import pandas as pd
import torch
from tqdm.auto import tqdm
from more_itertools import flatten
from datetime import datetime, timedelta
from dateutil.parser import parse as date_parse
import numpy as np

qa_fact_eval_kwargs = {
    "cuda_device": os.environ.get('CUDA_VISIBLE_DEVICES', 0),
    "use_lerc_quip": True,
    "verbose": True,
    "generation_batch_size": 32,
    "answering_batch_size": 32,
    "lerc_batch_size": 8
}

candidate_models =[
    # 'google/t5_xxl_true_nli_mixture',
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"
]

all_label_mapping = {
    'ynie': {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction',
    },
    'google': {
        0: 'entailment',
        1: 'no entailment',
    }
}


def batchifier(iterable, n):
    iterable = iter(iterable)
    return iter(lambda: list(itertools.islice(iterable, n)), [])


def get_prediction(model_name, model, premises, hypotheses, args, device='cpu'):
    if model_name == 'bartscore':
        return get_prediction_bart_score(model, premises, hypotheses, args)
    else:
        return get_prediction_nli(
            tokenizer=model['tokenizer'],
            model=model['model'],
            premise=premises,
            hypothesis=hypotheses,
            device=device
        )

def get_prediction_bart_score(model, premises, hypotheses, args, device='cpu'):
    logits = model['model'].score(premises, hypotheses, batch_size=args.batch_size)
    probs = np.exp(logits)
    return list(map(float, probs))


def get_prediction_nli(tokenizer, model, premise, hypothesis, max_length=512, device='cpu'):
    tokenized_input_seq_pair = tokenizer(
        list(premise),
        list(hypothesis),
        max_length=max_length,
        return_token_type_ids=True,
        truncation=True,
        return_tensors='pt',
        padding='longest'
    )
    input_ids = tokenized_input_seq_pair['input_ids'].to(device)
    token_type_ids = tokenized_input_seq_pair['token_type_ids'].to(device)
    attention_mask = tokenized_input_seq_pair['attention_mask'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
    logits = outputs.logits.detach().cpu().numpy()
    predicted_index = logits.argmax(axis=1).tolist()
    return logits[:, 0], predicted_index


def list_of_discourse_sents_to_df(row):
    return pd.DataFrame(row[['sentences','discourse_preds']].to_dict())


def match_archive_articles(doc_id, articles_discourse_df, archive_df, args):
    article = (
        articles_discourse_df
            .loc[lambda df: df[args.doc_id_col] == doc_id]
            .iloc[0]
    )
    article_matching_keys = set(article[args.key_match_col])
    article_date = article[args.date_col]

    if archive_df is None:
        archive_df = articles_discourse_df

    start_dt = article_date - timedelta(days=1 * 30)
    archive = (
        archive_df
            .loc[lambda df: df[args.date_col] < article_date]
            .loc[lambda df: df[args.date_col] > start_dt]
            .loc[lambda df: df[args.doc_id_col] != doc_id]
            .loc[lambda df: df[args.key_match_col].apply(lambda x: len(x & article_matching_keys) > 0)]
            .sort_values(by=args.date_col, ascending=False)
            .iloc[:20]
    )

    return article, archive


def read_news_archive_df(path, args):
    if path is None:
        return

    print('reading news archive...')
    news_archive_df = []
    for chunk in tqdm(pd.read_json(path, lines=True, orient='records', chunksize=1000)):
        news_archive_df.append(chunk)
    news_archive_df = pd.concat(news_archive_df)
    news_archive_df['security_figi'] = (
        news_archive_df['derived_tickers']
        .apply(lambda x: [] if isinstance(x, float) else x)
        .apply(lambda x: list(map(lambda y: y.get('security_figi'), x)))
    )
    news_archive_df = news_archive_df.drop(columns=['nicodes', 'tickers', 'derived_tickers'])
    news_archive_df['security_figi'] = news_archive_df['security_figi'].apply(set)
    news_archive_df['timeofarrival'] = news_archive_df['timeofarrival'].pipe(pd.to_datetime)
    return news_archive_df


def summarize_archive_columns(news_archive_df, summ_model, summ_tokenizer):
    def get_summary(src_text):
        batch = summ_tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = summ_model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    output_archive = []
    for _, row in tqdm(news_archive_df.head(100).iterrows(), total=len(news_archive_df)): # each row is a separate article
        expanded_archive_article = list_of_discourse_sents_to_df(row)
        for discourse_group, sentences in (
                expanded_archive_article
                        .groupby('discourse_preds')['sentences']
                        .aggregate(list).items()
        ):
            output_archive.append({
                'doc_id': row['doc_id'],
                'discourse_pred': discourse_group,
                'sentences': ' '.join(sentences),
            })

    archive_df = pd.DataFrame(output_archive)
    archive_df['summary'] = archive_df['sentences'].apply(get_summary)



def get_archive_comparison_pairs(sample_archive):
    if 'headline' in sample_archive.columns:
        return sample_archive.set_index('suid')['headline']
    elif 'sentences' in sample_archive.columns:
        output_archive = []
        for _, row in sample_archive.iterrows():
            for sent_idx, sent in enumerate(row['sentences']):
                output_archive.append({
                    'doc_id': row['doc_id'] + ':::' + str(sent_idx),
                    'sent': sent,
                })
        if len(output_archive) > 0:
            return pd.DataFrame(output_archive).set_index('doc_id')['sent']
        else:
            return pd.Series()
    else:
        output_archive = []
        for _, row in sample_archive.iterrows():
            expanded_archive_article = list_of_discourse_sents_to_df(row)
            for discourse_group, sentences in (
                    expanded_archive_article
                                .groupby('discourse_preds')['sentences']
                                .aggregate(list).items()
            ):
                output_archive.append({
                    'doc_id': row['doc_id'] + ':::' + discourse_group,
                    'premise': ' '.join(sentences),
                })
        return pd.DataFrame(output_archive).set_index('doc_id')['premise']


def get_excluded_entites(df, col):
    return set(
        df[col]
            .apply(list)
            .pipe(lambda s: pd.Series(list(flatten(s.tolist()))))
            .value_counts()
            .pipe(lambda s: s/s.sum())
            .loc[lambda s: s > .01]
            .index.tolist()
    )


def fix_matching_col(news_discourse_df, news_archive_df, matching_col):
    # excluded entities
    excluded_entity_list = set([
        'Dow Jones', 'Reuters', 'Thomson Reuters', 'Thomson Reuters Corp', 'Thomson Reuters Corp.',
        'The New York Times', 'New York Times', 'The New York Times Company', 'New York Times Company', 'The New York Times\'s ',
    ])
    news_discourse_df[matching_col] = (
        news_discourse_df[matching_col].apply(lambda x: set(list(filter(lambda y: y is not None, x))))
    )
    excluded_entity_list |= get_excluded_entites(news_discourse_df, matching_col)
    news_discourse_df[matching_col] = news_discourse_df[matching_col].apply(lambda x: x - excluded_entity_list)

    if news_archive_df is not None:
        news_archive_df[matching_col] = (
            news_archive_df[matching_col].apply(lambda x: set(list(filter(lambda y: y is not None, x))))
        )
        excluded_entity_list |= get_excluded_entites(news_archive_df, matching_col)
        news_archive_df[matching_col] = news_archive_df[matching_col].apply(lambda x: x - excluded_entity_list)
    return news_discourse_df, news_archive_df


def fix_date_col(news_discourse_df, date_col):
    if date_col == 'date':
        news_discourse_df = news_discourse_df.loc[lambda df: df['date'].notnull()]
        news_discourse_df['date'] = news_discourse_df['date'].astype(int).astype(str).apply(date_parse)
    else:
        news_discourse_df['timeofarrival'] = news_discourse_df['timeofarrival'].pipe(pd.to_datetime)
    return news_discourse_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--factual_consistency_model_name_or_path', type=str, default=candidate_models[0])
    parser.add_argument('--summary_model_name_or_path', type=str, default=None)
    parser.add_argument('--article-dataset', type=str)
    parser.add_argument('--archive-dataset', type=str, default=None) # None if exists a column `is_target_article` in `article-dataset`
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--doc-id-col', type=str, default='suid')
    parser.add_argument('--date-col', type=str, default='timeofarrival')
    parser.add_argument('--key-match-col', type=str, default='security_figi')
    parser.add_argument('--batch-size', type=int, default=3)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.factual_consistency_model_name_or_path == 'bartscore':
        import sys
        sys.path.insert(0, 'BARTScore')
        from bart_score import BARTScorer
        model = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        model.load(path='BARTScore/bart_score.pth')
        model = {
            'model': model,
            'label_mapping': {}
        }


    elif args.factual_consistency_model_name_or_path == 'qafacteval':
        from qafacteval import QAFactEval
        model_folder = "models/"
        model = QAFactEval(
            lerc_quip_path=f"{model_folder}/quip-512-mocha",
            generation_model_path=f"{model_folder}/generation/model.tar.gz",
            answering_model_dir=f"{model_folder}/answering",
            lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
            lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
            **qa_fact_eval_kwargs
        )
        model = {'model': model, 'label_mapping': {}}
    else:
        model = {
            'tokenizer': AutoTokenizer.from_pretrained(args.factual_consistency_model_name_or_path),
            'model': AutoModelForSequenceClassification
                        .from_pretrained(args.factual_consistency_model_name_or_path)
                        .to(device),
            'label_mapping': all_label_mapping[args.factual_consistency_model_name_or_path.split('/')[0]]
        }

    news_archive_df = read_news_archive_df(args.archive_dataset, args)
    news_discourse_df = pd.read_json(args.article_dataset, lines=True, orient='records', convert_dates=False)
    news_discourse_df = fix_date_col(news_discourse_df, args.date_col)
    if args.key_match_col == 'ents':
        news_discourse_df, news_archive_df = fix_matching_col(news_discourse_df, news_archive_df, args.key_match_col)

    # summary model
    if args.summary_model_name_or_path is not None:
        from transformers import PegasusForConditionalGeneration
        summ_tokenizer = AutoTokenizer.from_pretrained(args.summary_model_name_or_path)
        summ_model = PegasusForConditionalGeneration.from_pretrained(args.summary_model_name_or_path).to(device)
    else:
        summ_tokenizer = summ_model = None

    # if summ_model is not None:
    #     if news_archive_df is None:
    #         news_archive_df = news_discourse_df.copy()
    #     news_archive_df = summarize_archive_columns(news_archive_df, summ_model, summ_tokenizer)

    doc_ids = news_discourse_df[args.doc_id_col]
    if 'is_target_article' in news_discourse_df.columns:
        doc_ids = news_discourse_df.loc[lambda df: df['is_target_article'] == True][args.doc_id_col]
    doc_ids = doc_ids.drop_duplicates().sample(frac=1).tolist()
    with xopen.xopen(args.outfile, 'wb') as f:
        for doc_id in tqdm(doc_ids):
            sample_article, sample_archive = match_archive_articles(
                doc_id, news_discourse_df, news_archive_df, args
            )
            a = sample_article.pipe(list_of_discourse_sents_to_df)
            h = get_archive_comparison_pairs(sample_archive)

            all_items = list(product(enumerate(a['sentences']), h.items()))
            all_items = list(map(lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]), all_items))
            batches = list(batchifier(all_items, args.batch_size))
            for b in batches:
                article_sent_idxes, article_sents, archive_sent_idxs, archive_sents = list(zip(*b))
                pred_labels = get_prediction(
                    model_name=args.factual_consistency_model_name_or_path,
                    model=model,
                    premises=archive_sents,
                    hypotheses=article_sents,
                    args=args,
                    device=device
                )
                for a_idx, a_i, h_idx, h_i, pred_label in zip(
                        article_sent_idxes,
                        article_sents,
                        archive_sent_idxs,
                        archive_sents,
                        pred_labels
                ):
                    output = {
                        'doc_idx': doc_id,
                        'article_sentence': a_i,
                        'article_sent_idx': a_idx,
                        'archive_headline': h_i,
                        'archive_headline_idx': h_idx,
                        'model_prediction': model['label_mapping'].get(pred_label, pred_label)
                    }
                    f.write(orjson.dumps(output))
                    f.write(b'\n')



