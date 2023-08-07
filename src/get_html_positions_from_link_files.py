###
## This is implemented as a downstream filtering step for speed, but it is also implemented
# in the `parse_links.py` script.
#
# This method takes in a `links` file, matches it to the raw HTML and reprocesses it
import orjson
import pandas as pd
import glob
from utils import extract_links_with_char_num_and_post_process, find_press_release
import xopen
from tqdm.auto import tqdm

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default=None)
    parser.add_argument('--output-file', type=str, default=None)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace('.jsonl.gz', '-links-with-chars.jsonl.gz')

    outlet_name = args.input_file.split('-')[0]
    links_file = list(filter(lambda x: outlet_name in x, glob.glob('*link*')))
    links_file = list(filter(lambda x: 'chars' not in x, links_file))[0]
    links_df = pd.read_json(links_file, orient='records', lines=True)
    key = 'article_url' if 'article_url' in links_df.columns else 'url'
    links_df = (
        links_df
            .drop_duplicates('article_url')
            .set_index('article_url')
    )
    link_urls = set(links_df.index.tolist())

    with (
        xopen.xopen(args.input_file, 'rb') as f_in,
        xopen.xopen(args.output_file, 'wb') as f_out,
        tqdm(total=len(link_urls)) as pbar
    ):
        for line in f_in:
            try:
                line = orjson.loads(line)
            except:
                continue
            batch = []
            if line['article_url'] in link_urls:
                line_output = extract_links_with_char_num_and_post_process(line)
                link_row = links_df.loc[line['article_url']]
                line_output['ents'] = link_row['ents']
                links_with_ind = []
                for link in line_output['links']:
                    link['is_press_release'] = find_press_release(row=link)
                    links_with_ind.append(link)
                line_output['links'] = links_with_ind
                batch.append(line_output)
                pbar.update(1)

            if len(batch) > 0:
                lines_to_write = list(map(orjson.dumps, batch))
                f_out.write(b'\n'.join(lines_to_write) + b'\n')
                batch = []

        # write remaining batch
        if len(batch) > 0:
            lines_to_write = list(map(orjson.dumps, batch))
            f_out.write(b'\n'.join(lines_to_write) + b'\n')
            batch = []



