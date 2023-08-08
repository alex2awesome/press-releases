import os
from subprocess import Popen, PIPE
import glob
from tqdm.auto import tqdm
import pandas as pd
import time
import requests
import json
import random
import glob
from simplejson import JSONDecodeError
from requests_futures.sessions import FuturesSession
import itertools
from concurrent.futures import as_completed
import jsonlines
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs, urljoin
import orjson
import datetime
from concurrent.futures import ThreadPoolExecutor
import time
from requests import ConnectionError
from bs4 import BeautifulSoup
import re

def clean_url(to_get_url):
    return urljoin(to_get_url, urlparse(to_get_url).path)


WAYBACK_GCR_URLS = [
    # west: west1, west2, west3
   'https://wayback-html-and-pdf-1-ukvxfz3sya-uw.a.run.app',
   'https://wayback-html-and-pdf-1-ukvxfz3sya-wl.a.run.app',
   'https://wayback-html-and-pdf-1-ukvxfz3sya-wm.a.run.app',
    # east: east1, east4, east5
   'https://wayback-html-and-pdf-1-ukvxfz3sya-ue.a.run.app',
   'https://wayback-html-and-pdf-1-ukvxfz3sya-ul.a.run.app',
   'https://wayback-html-and-pdf-1-ukvxfz3sya-uk.a.run.app',
    # northeast: northamerica-northeast1, northamerica-northeast2
   'https://wayback-html-and-pdf-1-ukvxfz3sya-nn.a.run.app',
   'https://wayback-html-and-pdf-1-ukvxfz3sya-pd.a.run.app',
    # europewest: europe-west1, europe-west2
   'https://wayback-html-and-pdf-1-ukvxfz3sya-ew.a.run.app',
   'https://wayback-html-and-pdf-1-ukvxfz3sya-nw.a.run.app'
]


def clean_html(html_str):
    soup = BeautifulSoup(html_str, 'lxml')
    for s in soup(["script", "style", "meta", "noscript", "svg", "button", "figcaption", "head", "title", "nav"]):
        _ = s.extract()
    text = re.sub('\n+\s*', '\n', soup.get_text(' '))
    return text


def simple_gcf_wrapper(data, url):
    def hit_wayback(api_url, data):
        ts = data['canonical_timestamp']
        if pd.isnull(ts):
            ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return requests.post(
            api_url,
            headers={
                'Content-Type': 'application/json'
            },
            data=json.dumps({
                'article_url': data['href'],
                'target_timestamp_key': 'target_timestamp',
                'target_timestamp': ts,
                'sort_criteria': 'past'
            })
        )

    def hit_playwright(api_url, data):
        return requests.post(
            api_url,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data={
                'url': data['href'],
            }
        )

    for _ in range(3):
        try:
            if 'wayback' in url:
                output = hit_wayback(url, data)
            else:
                output = hit_playwright(url, data)
            if output.status_code == 200:
                if output.text == '':
                    print(output.text)
                    print(str(data))
                    return
                try:
                    output_json = output.json()
                    output_json['method'] = 'wayback'
                except JSONDecodeError:
                    return {
                        'article_url': data['href'],
                        'data': clean_html(output.text),
                        'method': 'playwright'
                    }

            else:
                print(output.status_code)
                time.sleep(3)
                print(str(data))
                print(output.text)
                print('trying again...')

        except ConnectionError:
            pass


def get_article_for_df(
        data,
        parallelize=True,
        max_workers=None
):
    """
    For a dataframe of article URLs, hit the GCF endpoint and return the article text.

    Parameters:
    -----------
    data: pd.DataFrame
        with columns 'article_url' and 'article_timestamp'
            (if available. else `article_timestamp` will be set to the current date.)

    """
    # create cycle through URLs
    c = itertools.cycle(WAYBACK_GCR_URLS)
    api_urls_to_hit = [next(c) for _ in range(len(data))]

    data = data.to_dict(orient='records')
    if parallelize:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for output in tqdm(executor.map(simple_gcf_wrapper, data, api_urls_to_hit), total=len(data)):
                yield output
    else:
        for output in tqdm(map(simple_gcf_wrapper, data, api_urls_to_hit), total=len(data)):
            yield output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default=None)
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--already-fetched-file', dest='already_fetched', type=str, default=None)
    parser.add_argument('--article-id-key', default='href', type=str)
    parser.add_argument('--article-timestamp-key', default='article_timestamp', type=str)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--start-idx', type=float, default=None)
    parser.add_argument('--end-idx', type=float, default=None)
    parser.add_argument('--clean-url', action='store_true', )
    args = parser.parse_args()

    args = parser.parse_args()
    article_df = pd.read_csv(args.input_file)
    article_df['article_url'] = article_df[args.article_id_key]
    if args.clean_url:
        article_df['article_url'] = article_df['article_url'].apply(clean_url)
    article_df = article_df.drop_duplicates(['article_url'])

    if (args.already_fetched is not None) or os.path.exists(args.output_file):
        print('filtering articles...')
        fn = args.already_fetched or args.output_file
        with open(fn) as f:
            if fn.endswith('jsonl'):
                already_fetched_urls = []
                for line in f:
                    try:
                        url = json.loads(line).get('article_url', '')
                        already_fetched_urls.append(url)
                    except Exception as e:
                        print(f'error: {str(e)}')
                        continue
            else:
                already_fetched_urls = f.read().split('\n')
        already_fetched_urls = list(filter(lambda x: x != '', already_fetched_urls))
        if args.clean_url:
            already_fetched_urls = list(map(clean_url, already_fetched_urls))
        article_df = article_df.loc[lambda df: ~df['article_url'].isin(already_fetched_urls)]

    with open(args.output_file, 'a') as f_output:
        for output in get_article_for_df(article_df, parallelize=args.workers > 1, max_workers=args.workers):
            if (output is not None) and (output != {}):
                f_output.write(json.dumps(output) + '\n')














# c = itertools.cycle(WAYBACK_GCR_URLS)
#     api_urls = list(map(lambda x: next(c), range(len(article_df))))
#
#     USE_FUTURE_SESSION = False
#     if USE_FUTURE_SESSION:
#         with (
#             open(args.output_file, 'a') as f_output,
#             FuturesSession(max_workers=args.workers) as session
#         ):
#             futures = []
#             for site, timestamp, api_url in zip(
#                     article_df['article_url'],
#                     article_df[args.article_timestamp_key],
#                     api_urls
#             ):
#                 f = session.post(
#                     api_url,
#                     headers={'Content-Type': 'application/json'},
#                     data=json.dumps({
#                         'article_url': site,
#                         'target_timestamp_key': 'target_timestamp',
#                         'target_timestamp': timestamp,
#                         'sort_criteria': 'past'
#                     })
#                 )
#                 futures.append(f)
#
#             for output in tqdm(as_completed(futures), total=len(article_df)):
#                 if output is not None:
#                     resp = output.result()
#                     f_output.write(orjson.dumps(resp.json()) + '\n')
#
#     else: