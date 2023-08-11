import requests
import json
import time
from tqdm.auto import tqdm
import jsonlines
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from random import choice
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs, urljoin
import itertools 
import datetime
import os


ENDPOINT_URLS = [
    'https://wayback-scrape-v2-1-ukvxfz3sya-wl.a.run.app',
    'https://wayback-scrape-v2-2-ukvxfz3sya-ue.a.run.app',
    'https://wayback-scrape-v2-3-ukvxfz3sya-uc.a.run.app'
]


def clean_url(to_get_url):
    return urljoin(to_get_url, urlparse(to_get_url).path)


def simple_gcf_wrapper(data, url):
    output = requests.post(
            url,
            headers={
                'Content-Type': 'application/json'
            },
            data=json.dumps(data)
        )
    if output.status_code == 200:
        if output.text == 'No items in Wayback Machine':
            print(output.text)
            print(str(data))
            return

        return output.json()
    else:
        print(output.status_code)
        if output.status_code == 500:
            print(str(data))
        return


def get_article_for_df(data, article_timestamp_col='article_timestamp', parallelize=True, article_timestamp_default=None, max_workers=None):
        """
        For a dataframe of article URLs, hit the GCF endpoint and return the article text.

        Parameters:
        -----------
        data: pd.DataFrame
            with columns 'article_url' and 'article_timestamp'
                (if available. else `article_timestamp` will be set to the current date.)

        """
        # create cycle through URLs 
        c = itertools.cycle(ENDPOINT_URLS)
        api_urls_to_hit = [next(c) for _ in range(len(data))]
        
        # infill `homepage_key`
        if article_timestamp_col not in data.columns:
            if article_timestamp_default is None:
                article_timestamp_default = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            data[article_timestamp_col] = article_timestamp_default

        data = data.to_dict(orient='records')
        if parallelize:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for output in tqdm(executor.map(simple_gcf_wrapper, data, api_urls_to_hit), total=len(data)):
                    yield output
        else:
            for output in tqdm(map(simple_gcf_wrapper, data, api_urls_to_hit), total=len(data)):
                yield output

# run: python wayback_gce_runner.py --input-file <input_file> --output-file <output_file> --num-concurrent-workers 3 --already-fetched-file <already_fetched_file>
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', dest='input_file', type=str)
    parser.add_argument('--output-file', dest='output_file', type=str)
    parser.add_argument('--article-id-key', default='href', type=str)
    parser.add_argument('--article-timestamp-key', default='article_timestamp', type=str)
    parser.add_argument('--num-concurrent-workers', dest='workers', type=int, default=3)
    parser.add_argument('--already-fetched-file', dest='already_fetched', type=str,
                        help='List of URLs that we already have from prior runs.')

    args = parser.parse_args()
    article_df = pd.read_csv(args.input_file)
    if 'Unnamed: 0' in article_df.columns:
        article_df = article_df.drop(columns='Unnamed: 0')

    if args.article_id_key != 'article_url':
        if 'article_url' in article_df.columns:
            article_df = article_df.drop(columns='article_url')
            article_df = article_df.rename(columns={args.article_id_key: 'article_url'})

    data = (
        article_df
            .assign(article_url=lambda df: df['article_url'].apply(clean_url))
            .drop_duplicates(['article_url'])
    )

    if (args.already_fetched is not None) or os.path.exists(args.output_file):
        print('filtering articles...')
        fn = args.already_fetched or args.output_file
        if fn.endswith('jsonl'):
            already_fetched_urls = list(jsonlines.open(
                fn, loads=lambda x: json.loads(x)[args.article_id_key]
            ))
        else:
            with open(fn) as f:
                already_fetched_urls = f.read()
        already_fetched_urls = list(map(clean_url, already_fetched_urls))
        data = data.loc[lambda df: ~df['article_url'].isin(already_fetched_urls)]

    print('fetching articles...')
    with open(args.output_file, 'w') as f:
        w = jsonlines.Writer(f)
        for output in get_article_for_df(
                data,
                article_timestamp_col=args.article_timestamp_key,
                parallelize=True,
                max_workers=args.workers
        ):
            if output is not None:
                output.pop('article_html', None)
                w.write(output)




python predict.py --model_name_or_path alex2awesome/newsdiscourse-model --model_type sentence --dataset_name ../../../data/open-sourced-articles/all-articlesin-db.csv.gz --outfile ../data/all-articles-in-db-news-discourse.jsonl --tokenizer_name roberta-base --text-col body --id-col article_url