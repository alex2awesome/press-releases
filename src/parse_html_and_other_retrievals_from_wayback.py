##
# Takes in .jsonl.gz files where each row is an article containing raw HTML.
# Outputs two files:
#   1. One file with all plain text articles, extracted entities
#   2. A second file with all the above, only for articles that contain press release links.
#
#

import os.path
import xopen
import orjson
from tqdm.auto import tqdm
import gzip
from itertools import islice
from utils import get_num_lines_robust, extract_links_from_html_with_char_num_and_post_process, retrieve_ents_for_col, find_press_release

num_lines_cache = {
    'wsj-articles.jsonl.gz': 217200,
}


# to run as a script:
# python parse_links.py --input-file=<i> --all-articles-output-file=<o1> --target-links-output-file=<o2>
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--all-articles-output-file', type=str, default=None)
    parser.add_argument('--target-links-output-file', type=str, default=None)
    parser.add_argument('--extract-links', action='store_true')
    parser.add_argument('--drop-html', action='store_true')
    parser.add_argument('--num-articles', type=int, default=None)
    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--no-calculate-lines', action='store_true')
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()

    if args.all_articles_output_file is None:
        args.all_articles_output_file = args.input_file.replace('.jsonl.gz','') + '-parsed.jsonl.gz'

    if args.target_links_output_file is None:
        args.target_links_output_file = args.input_file.replace('.jsonl.gz','') + '-links.jsonl.gz'

    # filter out urls that we have already fetched
    existing_urls = set()
    if os.path.exists(args.all_articles_output_file) and os.path.getsize(args.all_articles_output_file) > 100:
        try:
            with xopen.xopen(args.all_articles_output_file) as f:
                for line in f:
                    try:
                        datum = orjson.loads(line)
                    except:
                        continue
                    existing_urls.add(datum['article_url'])
        except:
            pass

    # read
    if args.no_calculate_lines:
        num_lines = num_lines_cache[args.input_file]
    else:
        try:
            with xopen.xopen(args.input_file) as f:
                num_lines = get_num_lines_robust(f)
        except:
            with gzip.open(args.input_file) as f:
                num_lines = get_num_lines_robust(f)

    print(f'num lines: {num_lines}')
    print(f'num existing lines: {len(existing_urls)}')
    with (
        xopen.xopen(args.input_file) as f_in,
        xopen.xopen(args.all_articles_output_file, 'a') as f_all_articles_out,
        xopen.xopen(args.target_links_output_file, 'a') as f_target_links_out
    ):
        curr_file_idx = 0
        total = int(num_lines / args.batch_size)
        if args.start_idx is not None:
            total = int(args.end_idx / args.batch_size)

        for batch_lines in tqdm(iter(lambda: tuple(islice(f_in, args.batch_size)), ()), total=total):
            if (args.start_idx is not None) and (curr_file_idx < args.start_idx):
                curr_file_idx += args.batch_size
                continue

            if (args.end_idx is not None) and (curr_file_idx >= args.end_idx):
                import sys
                sys.exit(0)

            batch = []
            for line in batch_lines:
                try:
                    datum = orjson.loads(line)
                except:
                    continue
                if datum['article_url'] in existing_urls:
                    continue

                datum = extract_links_from_html_with_char_num_and_post_process(datum)
                batch.append(datum)

            # read next line
            batch_ents = retrieve_ents_for_col(list(map(lambda x: x['article_text'], batch)))
            for datum, ents in zip(batch, batch_ents):
                datum['ents'] = ents

            # identify target links
            target_links = []
            for datum in batch:
                found = False
                if 'links' not in datum:
                    continue
                for link in datum['links']:
                    if find_press_release(link['href'], link['text']):
                        found = True

                if found:
                    target_links.append(datum)

            # write to disk
            datum_strs = list(map(lambda x: orjson.dumps(x).decode(), batch))
            f_all_articles_out.write('\n'.join(datum_strs) + '\n')

            link_strs = list(map(lambda x: orjson.dumps(x).decode(), target_links))
            f_target_links_out.write('\n'.join(link_strs) + '\n')

            curr_file_idx += args.batch_size
