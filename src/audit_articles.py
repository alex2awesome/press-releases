import os
import itertools
import re
import orjson, xopen
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs, urljoin
from tqdm.auto import tqdm


def clean_url(to_get_url):
    """Remove query params from URL. """
    url = urljoin(to_get_url, urlparse(to_get_url).path)
    return url.split(')')[-1].strip()

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
        to_return = orjson.loads(to_return)
        parts = x.split()
        date_part = list(filter(lambda x: x.isdigit(), parts))
        if len(date_part) > 0:
            to_return['date'] = date_part[0]
    except:
        print(f'error: {x}')
        to_return = {}
    return to_return


def parse_urls_in_to_fetch_file(to_fetch_file, args):
    seen = set([])
    with xopen.xopen(filename=to_fetch_file) as f_handle:
        for line in tqdm(f_handle):
            if isinstance(line, bytes):
                line = line.decode()
            if len(line) > 2:
                url = clean_url(line.split()[0])
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
                        if json_body.get('date') < args.date_filter:
                            continue
                    seen.add(url)
    return list(seen)

def parse_urls_in_article_file(article_file, n_files=None, args=None):
    already_fetched_urls = []
    try:
        with xopen.xopen(filename=article_file) as f_handle:
            for line in tqdm(f_handle):
                if n_files is not None:
                    if len(already_fetched_urls) > n_files:
                        break
                try:
                    json_obj = orjson.loads(line)
                    fetched_url = clean_url(json_obj['article_url'])
                    already_fetched_urls.append(fetched_url)
                except:
                    continue
    except:
        print(f'error: {article_file}')
    return already_fetched_urls


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=str, default='article_fetch')
    parser.add_argument('--input-dir', type=str, default='.')
    parser.add_argument('--date-filter', type=str, default=None)
    parser.add_argument('--url-filter', type=str, default=None)
    parser.add_argument('--status-filter', type=str, default=None)

    args = parser.parse_args()
    files_on_disk = os.listdir(args.input_dir)
    files_on_disk = list(filter(lambda x: '-' in x, files_on_disk))
    files_on_disk = sorted(files_on_disk, key=lambda x: x.split('-')[0])
    file_groups = itertools.groupby(files_on_disk, key=lambda x: x.split('-')[0])
    file_groups = list(map(lambda x: (x[0], list(x[1])), file_groups))

    if args.comparison == 'article_fetch':
        for g, article_list in file_groups:
            print(f'group: {g}')

            url_file = list(filter(lambda x: 'to-fetch' in x, article_list))[0]
            article_file = list(filter(lambda x: 'articles.json' in x, article_list))[0]

            to_fetch_urls = parse_urls_in_to_fetch_file(url_file, args)
            already_fetched_urls = parse_urls_in_article_file(article_file, n_files=None, args=args)

            print(f'urls to fetch: {len(to_fetch_urls)}')
            print(f'urls already fetched: {len(already_fetched_urls)}')
            print(f'num urls to fetch: {len(set(to_fetch_urls) - set(already_fetched_urls))}')
            print('')



