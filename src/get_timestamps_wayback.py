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
from requests_futures.sessions import FuturesSession
import itertools
from concurrent.futures import as_completed


def get_timestamp_from_wayback(site, user_agent):
    SEARCH_URL = "https://web.archive.org/cdx/search/cdx"
    num_tries = 3
    for i in range(num_tries):
        res = requests.get(
            SEARCH_URL,
            params={
                "url": site,
                "fl": "timestamp",
                "limit": "5",
            },
            headers={
                "User-Agent": user_agent,
            }
        )
        if res.status_code == 200:
            r = res.text.split()
            if len(r) > 0:
                return sorted(r)[0]
            else:
                return 'Not Found'

        else:
            print(f'error: {res.status_code} {res.text}')
            time.sleep(1)


GCF_URLS = [
    'https://us-west1-usc-research.cloudfunctions.net/wayback-timestamp',
    'https://us-west2-usc-research.cloudfunctions.net/wayback-timestamp',
    'https://us-west3-usc-research.cloudfunctions.net/wayback-timestamp',
    #
    'https://us-east1-usc-research.cloudfunctions.net/wayback-timestamp',
    'https://us-east4-usc-research.cloudfunctions.net/wayback-timestamp',
    #
    'https://us-central1-usc-research.cloudfunctions.net/wayback-timestamp',
    'https://southamerica-east1-usc-research.cloudfunctions.net/wayback-timestamp',
    'https://northamerica-northeast1-usc-research.cloudfunctions.net/wayback-timestamp',
    #
    'https://europe-west6-usc-research.cloudfunctions.net/wayback-timestamp',
    'https://europe-west2-usc-research.cloudfunctions.net/wayback-timestamp',
    'https://europe-west1-usc-research.cloudfunctions.net/wayback-timestamp'
]

def get_timestamp_from_wayback_gcf(site, api_url):
    for i in range(3):
        try:
            t = requests.post(
                api_url,
                headers={ 'Content-Type': 'application/json'},
                data=json.dumps({'article_url': site})
            )
            return t.text
        except:
            time.sleep(1)
            pass


cc_to_web_map = {
    'com,forbes': 'https://www.forbes.com',
    'com,barrons': 'https://www.barrons.com',
    'com,foxbusiness': 'https://www.foxbusiness.com',
    'com,businessinsider': 'https://www.businessinsider.com',
    'com,cnbc': 'https://www.cnbc.com',
    'com,marketwatch': 'https://www.marketwatch.com',
    'com,nytimes': 'https://www.nytimes.com',
    'com,reuters': 'https://www.reuters.com',
    'com,techcrunch': 'https://techcrunch.com',
    'com,washingtonpost': 'https://www.washingtonpost.com',
    'com,wsj': 'https://www.wsj.com'
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default=None)
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--start-idx', type=float, default=None)
    parser.add_argument('--end-idx', type=float, default=None)
    args = parser.parse_args()

    # if args.output_file is None:
    #     args.output_file = args.input_file.replace('.csv', '_wayback_timestamps.csv')
    # df = pd.read_csv(args.input_file)
    # urls = pd.concat([
    #     df['target_article_url'].drop_duplicates(),
    #     df['archival_url'].drop_duplicates()
    # ])


    a_maps = glob.glob('article-map*')
    a_maps = list(filter(lambda x: 'wayback' not in x, a_maps))
    all_urls = []
    print('reading input files...')
    for f in tqdm(a_maps):
        df = pd.read_csv(f)
        urls = pd.concat([
            df['target_article_url'].drop_duplicates(),
            df['archival_url'].drop_duplicates()
        ])
        all_urls.append(urls)

    urls = pd.concat(all_urls).drop_duplicates()
    urls = (
        urls
            .apply(lambda x: x.split(')'))
            .apply(lambda x: cc_to_web_map[x[0]] + x[-1])
            .drop_duplicates()
    )

    if os.path.exists(args.output_file):
        fetched_df = pd.read_csv(args.output_file, index_col=None, header=None, on_bad_lines='warn')
        fetched_df.columns = ['url', 'timestamp']
        urls = urls.loc[lambda df: ~df.isin(fetched_df['url'])]

    if args.start_idx is not None and args.end_idx is not None:
        if args.start_idx < 1:
            args.start_idx = int(args.start_idx * len(urls))
        if args.end_idx <= 1:
            args.end_idx = int(args.end_idx * len(urls))

        urls = urls.iloc[args.start_idx:args.end_idx]
        args.output_file = args.output_file.replace('.csv', f'_{args.start_idx}_{args.end_idx}.csv')

    # print('getting timestamps...')
    # with open(args.output_file, 'w') as f:
    #     for idx, url in tqdm(enumerate(urls), total=len(urls)):
    #         t = get_timestamp_from_wayback_gcf(url)# user_agent=f'spangher@usc.edu {args.input_file}')
    #         f.write(f'{url},{t}\n')

    c = itertools.cycle(GCF_URLS)
    api_urls = list(map(lambda x: next(c), range(len(urls))))

    with (
        open(args.output_file, 'a') as f_output,
        FuturesSession(max_workers=args.workers) as session
    ):
        futures = []
        for site, api_url in zip(urls, api_urls):
            f = session.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps({'article_url': site})
            )
            f.site = site
            futures.append(f)

        for output in tqdm(as_completed(futures), total=len(urls)):
            if output is not None:
                resp = output.result()
                f_output.write(f'{output.site},{resp.text}\n')

