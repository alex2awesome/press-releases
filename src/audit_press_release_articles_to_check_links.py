##
# Checks the .jsonl.gz links files to filter out links that are not press releases.
# Outputs a new article file with articles that contain press release links.
#
import xopen
import gzip
from utils import audit_links
import orjson


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
    parser.add_argument('--num-articles', type=int, default=None)
    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--no-calculate-lines', action='store_true')
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()

    # hack to filter files
    if '-links.' in args.input_file:
        with (
            gzip.open(args.input_file) as f_in,
            xopen.xopen(args.target_links_output_file, 'ab') as f_target_links_out
        ):
            for batch_output in audit_links(f_in, args):
                dict_strs = list(map(orjson.dumps, batch_output))
                f_target_links_out.write(b'\n'.join(dict_strs) + b'\n')

        # break out of the program
        import sys
        sys.exit(0)