from bs4 import BeautifulSoup
import jsonlines 
import xopen
import orjson
import orjsonl
from tqdm.auto import tqdm
import gzip

def post_process(datum, extract_links, drop_html):
    """Post-process the response from the GCF endpoint.
    
    Parameters
    ----------
    * datum : (dict) of data returned from the GCF endpoint.
    * drop_html : (bool) whether to drop the HTML from the response.
    * extract_links : (bool) whether to extract the links from the HTML.
    """
    key = 'html' if 'html' in datum else 'article_html'
    if extract_links:
        html = datum[key]
        soup = BeautifulSoup( html, 'lxml')

        # find all links with non-null href
        links = soup.find_all('a', href=True)
        links = list(filter(lambda x: x.get_text() != '', links))
        links_obj = list(map(lambda x: {'text': x.get_text(), 'href': x['href']}, links))
        datum['links'] = links_obj

    if drop_html:
        datum.pop(key, None)
    
    return datum


import mmap

def mapcount(filename):
    with open(filename, 'r+') as f:
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
        return lines

# to run as a script:
# python parse_links.py --input-file=<input> --output-file=<output> --extract-links --drop-html
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--extract-links', action='store_true')
    parser.add_argument('--drop-html', action='store_true')
    args = parser.parse_args()

    # read
    num_lines = mapcount(args.input_file)
    print(f'num lines: {num_lines}')
    with (
        xopen.xopen(args.input_file) as f_in,
        xopen.xopen(args.output_file, 'wb') as f_out
    ):
        for line in tqdm(f_in, total=num_lines):
            try: 
                datum = orjson.loads(line)
                datum = post_process(datum, args.extract_links, args.drop_html)
                datum_str = orjson.dumps(datum)
                f_out.write(datum_str + b'\n')
            except:
                continue

