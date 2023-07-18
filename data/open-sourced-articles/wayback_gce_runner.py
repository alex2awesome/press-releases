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


def get_article_for_df(data, parallelize=True, homepage_key_default=None, max_workers=None):
        """
        For a dataframe of article URLs, hit the GCF endpoint and return the article text.

        Parameters:
        -----------
        data: pd.DataFrame
            with columns 'article_url' and 'homepage_key' (if available. else `homepage_key` will be set to the current date.)

        """
        # create cycle through URLs 
        c = itertools.cycle(ENDPOINT_URLS)
        api_urls_to_hit = [next(c) for _ in range(len(data))]
        
        # infill `homepage_key`
        if 'homepage_key' not in data.columns:
            if homepage_key_default is None:
                homepage_key_default = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            data['homepage_key'] = homepage_key_default

        data = data.to_dict(orient='records')
        if parallelize:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for output in tqdm(executor.map(simple_gcf_wrapper, data, api_urls_to_hit), total=len(data)):
                    yield output
        else:
            for output in tqdm(map(simple_gcf_wrapper, data, api_urls_to_hit), total=len(data)):
                yield output

#  --input-file ../data/latimes-article-urls-to-fetch.csv --output-file ../data/latimes-articles-8-years.jsonl --num-concurrent-workers 10

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', dest='input_file', type=str)
    parser.add_argument('--output-file', dest='output_file', type=str)
    parser.add_argument('--num-concurrent-workers', dest='workers', type=int)
    parser.add_argument('--already-fetched-file', dest='already_fetched', type=str,
                        help='List of URLs that we already have from prior runs.')

    args = parser.parse_args()

    compression = None if not args.input_file.endswith('gzip') else 'gzip'
    article_df = pd.read_csv(args.input_file, compression=compression)

    data = (
        article_df
            .rename(columns={'href': 'article_url', 'key': 'homepage_key'})
            .assign(article_url=lambda df: df['article_url'].apply(clean_url))
            .assign(homepage_key=lambda df: df['homepage_key'].astype(str))
            .drop_duplicates(['article_url', 'homepage_key'])
    )

    if args.already_fetched is not None:
        print('filtering articles...')
        fn = args.already_fetched
        if fn.endswith('jsonl'):
            already_fetched_urls = list(jsonlines.open(fn, loads=lambda x: json.loads(x)['article_url']))
        else:
            with open(fn) as f:
                already_fetched_urls = f.read()
        already_fetched_urls = list(map(clean_url, already_fetched_urls))
        data = data.loc[lambda df: ~df['article_url'].isin(already_fetched_urls)]

    print('fetching articles...')
    with open(args.output_file, 'w') as f:
        w = jsonlines.Writer(f)
        for output in get_article_for_df(data, parallelize=True, max_workers=args.workers):
            if output is not None:
                output.pop('article_html', None)
                w.write(output)