##
# This is the initial step:
#    Takes in a list of candidate common crawl URLs
#    Gets the HTML from the common crawl URL.
#    Does some rudimentary cleaning of the HTML using the newspaper4k library.
#

import jsonlines
import requests
import datetime
import orjson as json
from tqdm.auto import tqdm
import pandas as pd
from random import choice
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs, urljoin
import gzip
import xopen
import re
import os
from bs4 import BeautifulSoup
from concurrent.futures import as_completed

import itertools
from requests_futures.sessions import FuturesSession


CONTAINER_URLS = [
    "https://common-crawl-scrape-v1-ukvxfz3sya-uw.a.run.app",
    'https://common-crawl-scrape-v1-2-ukvxfz3sya-ue.a.run.app',
    'https://common-crawl-scrape-v1-3-ukvxfz3sya-uc.a.run.app',
    'https://common-crawl-scrape-v1-4-ukvxfz3sya-wl.a.run.app',
    'https://common-crawl-scrape-v1-5-ukvxfz3sya-wm.a.run.app',
    'https://common-crawl-scrape-v1-6-ukvxfz3sya-wn.a.run.app',
    'https://common-crawl-scrape-v1-7-ukvxfz3sya-uk.a.run.app',
    'https://common-crawl-scrape-v1-8-ukvxfz3sya-ew.a.run.app',
    'https://common-crawl-scrape-v1-9-ukvxfz3sya-nw.a.run.app',
    'https://common-crawl-scrape-v1-10-ukvxfz3sya-ey.a.run.app',
    'https://common-crawl-scrape-v1-11-ukvxfz3sya-ez.a.run.app',
    'https://common-crawl-scrape-v1-12-ukvxfz3sya-rj.a.run.app',
]


get_key = lambda x: re.sub('\{.*?\}', '', x).strip()
def get_json_body(x):
    """
    Takes in a common-crawl index and returns the part of it that contains the JSON body of data to hit the GCF endpoint with.

    Parameters
    ----------
    * x : (str) common-crawl index line.
    """
    try:
        to_return = re.findall('\{.*?\}', x)[-1]
        to_return = '{' + to_return.split('{')[-1]
        to_return = json.loads(to_return)
        parts = x.split()
        date_part = list(filter(lambda x: x.isdigit(), parts))
        if len(date_part) > 0:
            to_return['date'] = date_part[0]
    except:
        print(f'error: {x}')
        to_return = {}
    return to_return


def make_output_data(row):
    """ 
    Make the output data to send to the GCF endpoint. Minimal processing on some fields that are already part of the data.

    Parameters
    ----------
    * row : (dict) of data to send to the GCF endpoint, crucially, contains `json_packet` key.
    """
    output_packet = row['json_packet']
    output_packet['article_url'] = row['url'].split('?')[0]
    output_packet['scrape_timestamp'] = row.get('timestamp', datetime.datetime.now().isoformat())
    return output_packet


def filter_article_df(article_df, fn):
    """
    Filter `article_df` for articles that have already been fetched.

    Parameters
    ----------
    * article_df : (pd.DataFrame) of articles to fetch.
    * fn : (str) path to file containing already fetched articles. 
        Must be in a `.jsonl` format, although can be compressed.
    """
    already_fetched_urls = []
    try:
        with xopen.xopen(filename=fn) as f_handle:
            for line in f_handle:
                try:
                    json_obj = json.loads(line)
                    already_fetched_urls.append(json_obj['article_url'])
                except:
                    continue
    except:
        pass

    already_fetched_urls = list(map(clean_url, already_fetched_urls))
    already_fetched_urls = list(map(lambda x: x.split(')')[-1].strip(), already_fetched_urls))
    article_df = article_df.loc[lambda df: ~df['url'].str.split(')').str.get(-1).str.strip().isin(already_fetched_urls)]
    print(f'filtering articles to those already fetched... {len(article_df)} left.')
    return article_df


def clean_url(to_get_url):
    """Remove query params from URL. """
    return urljoin(to_get_url, urlparse(to_get_url).path)


def simple_gcf_wrapper(data, url):
    """
    Function helper to hit a GCF endpoint with the POST request.

    Parameters
    ----------
    * data : (dict) of data to POST to the GCF endpoint.
    * url : (str) URL of the GCF endpoint.
    """
    output = requests.post(
            url,
            headers={
                'Content-Type': 'application/json'
            },
            data=json.dumps(data)
        )
    if output.status_code == 200:
        data = output.json()
        if data.get('status_code') is not None:
            print(str(data))
            return

        return output.json()
    else:
        print(output.status_code)
        if output.status_code == 500:
            print(str(data))
        return


def handle_response(resp):
    """ 
    Handle the response from the GCF endpoint.

    Parameters
    ----------
    * resp : (requests.Response) object that the GCF endpoint returns.
    """
    if resp.status_code == 200:
        data = resp.json()
        if data.get('status_code') is not None:
            print(str(data))
            return
        return resp.json()
    else:
        print(resp.status_code)
        if resp.status_code == 500:
            print(str(resp.text))
            print(resp.request.url)
        return

def post_process(datum, args):
    """Post-process the response from the GCF endpoint.
    
    Parameters
    ----------
    * datum : (dict) of data returned from the GCF endpoint.
    * args : (argparse.Namespace) of arguments passed to the script. Can include:
        * drop_html : (bool) whether to drop the HTML from the response.
        * extract_links : (bool) whether to extract the links from the HTML.
    """
    if args.extract_links:
        html = datum['html']
        soup = BeautifulSoup( html)

        # find all links with non-null href
        links = soup.find_all('a', href=True)
        links = list(filter(lambda x: x.get_text() != '', links))
        links_obj = list(map(lambda x: {'text': x.get_text(), 'href': x['href']}, links))
        datum['links'] = links_obj

    if args.drop_html:
        datum.pop('html', None)
    
    return datum


#  python common_crawl_gce_runner.py --input-file <i> --output-file <o> --num-concurrent-workers 3 --date-filter 20210101
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', dest='input_file', type=str)
    parser.add_argument('--output-file', dest='output_file', help="Should be a `.jsonl` file.", type=str)
    parser.add_argument('--num-concurrent-workers', dest='workers', type=int)
    parser.add_argument('--url-filter', default=None, type=str, help='A string to check for in the URL.')
    parser.add_argument('--date-filter', default=None, type=str, help='Only get articles not in this common crawl date-range.')
    parser.add_argument('--status-filter', nargs='+', default=None, type=int, help='A list of status codes to filter out.')
    parser.add_argument(
        '--already-fetched-file',
        dest='already_fetched', 
        type=str,
        help='List of URLs that we already have from prior runs.'
    )
    parser.add_argument(
        '--url-selection-style',
        dest='url_cycle',
        default='round-robin',
        help='values= ["random", "round-robin"].'
    )
    parser.add_argument('--drop-html', dest='drop_html', action='store_true', help='Drop the HTML from the response.')
    parser.add_argument('--extract-links', dest='extract_links', action='store_true', help='Extract the links from the HTML.')

    args = parser.parse_args()

    if args.input_file.endswith('gzip') or args.input_file.endswith('gz'):
        f_handle = gzip.open(args.input_file, 'rb')
    else:
        f_handle = open(args.input_file)

    seen = set([])
    unique_article_lines = []
    for line in f_handle:
        if isinstance(line, bytes):
            line = line.decode()
        if len(line) > 2:
            url = line.split()[0].split('?')[0]
            if (url not in seen):
                if args.url_filter is not None:
                    if args.url_filter not in url:
                        continue
                if args.status_filter is not None:
                    json_body = get_json_body(line)
                    if int(json_body.get('status', 0)) not in args.status_filter:
                        continue
                if args.date_filter is not None:
                    json_body = get_json_body(line)
                    if json_body.get('date', '1') < args.date_filter:
                        continue
                unique_article_lines.append(line)
                seen.add(url)

    article_df = (
        pd.Series(unique_article_lines)
            .to_frame('data')
            .assign(json_packet=lambda df: df['data'].apply(get_json_body))
            .assign(key=lambda df: df['data'].apply(get_key).str.split())
            .loc[lambda df: df['key'].str.len() == 2]
            .assign(url=lambda df: df['key'].str.get(0).apply(clean_url))
            .assign(timestamp=lambda df: df['key'].str.get(1))
            .sample(frac=1)
    )
    print(f'Total articles: {len(article_df)}')

    if args.already_fetched is not None:
        article_df = filter_article_df(article_df, args.already_fetched)
    
    if os.path.exists(args.output_file):
        article_df = filter_article_df(article_df, args.output_file)

    if args.url_cycle == 'random':
        api_urls = list(map(lambda x: choice(CONTAINER_URLS), range(len(article_df))))
    else:
        c = itertools.cycle(CONTAINER_URLS)
        api_urls = list(map(lambda x: next(c), range(len(article_df))))

    data = article_df.apply(make_output_data, axis=1)
    data = data.tolist()
    print('fetching articles...')
    
    # stream to file    
    if not args.output_file.endswith('.gz'):
        args.output_file = args.output_file + '.gz'

    with xopen.xopen(filename=args.output_file, mode='ab') as outfile_handle:
        w = jsonlines.Writer(outfile_handle)
        with FuturesSession(max_workers=args.workers) as session:
            futures = []
            for row_to_get, api_url in zip(data, api_urls):
                f = session.post(
                    api_url,
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps(row_to_get)
                )
                futures.append(f)

            for output in tqdm(as_completed(futures), total=len(data)):
                if output is not None:
                    resp = output.result()
                    resp_json = handle_response(resp)
                    if resp_json is not None:
                        resp_json = post_process(resp_json, args)
                        w.write(resp_json)








# filter out articles with a bad status (before uploading the file)
#   seen = set([])
#   unique_article_lines = []
#   for line in f_handle:
#       if isinstance(line, bytes):
#           line = line.decode()
#       if len(line) > 2:
#           url = line.split()[0].split('?')[0]
#           if (url not in seen) and ('/business/' in url):
#               json_body = json.loads(get_json_body(line))
#               if json_body.get('status') == '200':
#                   unique_article_lines.append(line)
#                   seen.add(url)

# seen = set([])
# unique_article_lines = []
# for line in f_handle:
#     if isinstance(line, bytes):
#         line = line.decode()
#     if len(line) > 2:
#         url = line.split()[0].split('?')[0]
#         if (url not in seen):
#             unique_article_lines.append(line)
#             seen.add(url)
