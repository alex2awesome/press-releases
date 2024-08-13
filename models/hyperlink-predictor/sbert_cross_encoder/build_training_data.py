from retriv import DenseRetriever
import pandas as pd
from tqdm.auto import tqdm
import json
from datetime import timedelta


# python build_training_data.py --index-name test-index --positive-label-dataset-name ../data/positive_label_dataset.csv --timestep-data-file data/canonical_timestamps.csv --output-file data/negative_label_dataset.csv
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--index-name', type=str, default="test-index")
    parser.add_argument('--positive-label-dataset-name', type=str)
    parser.add_argument('--timestep-data-file', type=str, default="doc_id")
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    dr = DenseRetriever.load(args.index_name)

    positive_training_examples = pd.read_csv(args.positive_label_dataset_name)
    canonical_timestamps = pd.read_csv(args.timestep_data_file)
    canonical_timestamps['canonical_timestamp'] = pd.to_datetime(canonical_timestamps['canonical_timestamp'])
    positive_training_examples['canonical_timestamp'] = pd.to_datetime(positive_training_examples['canonical_timestamp'])

    negative_examples = []

    with open(args.output_file, 'w') as f:
        for _, row in tqdm(positive_training_examples.iterrows(), total=len(positive_training_examples)):
            to_search_text = row.pipe(lambda s: s['sentences'] + ' ' + s['article_head'])
            to_search_url, to_search_date = row[['article_url', 'canonical_timestamp']]

            documents = dr.search(to_search_text, cutoff=1_000)

            q_results = (
                pd.DataFrame(documents)
                     .merge(canonical_timestamps, right_on='canonical_domain', left_on='id')
                     .drop(columns='canonical_domain')
            )

            neg_sample = (
                q_results
                     .loc[lambda df: df['id'] != to_search_url]
                     .loc[lambda df: df['canonical_timestamp'] <= (to_search_date)]
                     .loc[lambda df: df['canonical_timestamp'] > (to_search_date - timedelta(days=60))]
                     .iloc[:10]
                     .pipe(lambda df: df.sample().iloc[0] if len(df) > 0 else None)
            )

            if neg_sample is not None:
                f.write(json.dumps({
                    'source_url': to_search_url,
                    'source_text': to_search_text,
                    'target_url': neg_sample['id'],
                    'target_text': neg_sample['text'],
                }) + '\n')
