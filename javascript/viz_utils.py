# data_to_render = uda.match_sentences(one_article_matched_df, one_article_sentences_df)
# data_dict_for_rendering = uda.dump_output_to_app_readable(data_to_render)

import pandas as pd
import unidecode
import json

def match_sentences(matched_sentences, entailment_cutoff=.3, contradiction_cutoff=.3):
    """
    Takes as input a `matched_sentences` DF and a `split_sentences` DF and returns a merged DF that can be
    dumped as output for the app, endpoint `/view_task_match`.
    """
    article_sent_idx_col = 'article_sent_idx'
    if article_sent_idx_col not in matched_sentences.columns:
        article_sent_idx_col = 'article_idx'
    press_release_sent_idx_col = 'press_release_sent_idx'
    if press_release_sent_idx_col not in matched_sentences.columns:
        press_release_sent_idx_col = 'press_release_idx'

    # get HTML diffs
    grouped_arcs = (
        matched_sentences
         .groupby(['press_release_url', 'article_url'])
         .apply(lambda df:
                df[[press_release_sent_idx_col, article_sent_idx_col]]
                .loc[lambda df: df['entailment'] > entailment_cutoff]
                .loc[lambda df: df['contradiction'] > contradiction_cutoff]
                .sort_values()
                .to_dict(orient='records')
         )
         .to_frame('arcs')
    )
    url2url_dict = {k:v for k, v in (
        matched_sentences.index.tolist() + list(map(lambda x: (x[1], x[0]), matched_sentences.index.tolist()))
    )}

    split_sentences = pd.concat([
        matched_sentences[['article_url', 'article_sent_idx', 'article_sents']]
            .drop_duplicates().assign(version='article').rename(columns={
                    'article_sent_idx': 'sent_idx', 'article_sents': 'sentence', 'article_url': 'url'
            }),
        matched_sentences[['press_release_url', 'press_release_sent_idx', 'press_release_sents']]
            .drop_duplicates().assign(version='press_release').rename(columns={
                    'press_release_sent_idx': 'sent_idx', 'press_release_sents': 'sentence', 'press_release_url': 'url'
            }),
    ])
    split_sentences['sentence'] = split_sentences['sentence'].apply(unidecode.unidecode)
    split_sentences['sentence'] = split_sentences['sentence'].str.replace('"', '\'\'')
    split_sentences['sentence'] = split_sentences['sentence'].str.replace('<p>', '').str.replace('</p>', '').str.strip()
    grouped_nodes = (
        split_sentences
             .assign(other_url=lambda df: df['url'].map(url2url_dict))
             .groupby(['url', 'other_url'])
             .apply(
                  lambda df: df[['version', 'sent_idx', 'sentence']].to_dict(orient='records')
             )
             .to_frame('nodes').reset_index()
    )
    output = pd.concat([grouped_nodes, grouped_arcs], axis=1)
    return output


def dump_output_to_app_readable(output_df, outfile=None):
    output = output_df[['nodes', 'arcs']].to_dict(orient='index')
    output = {str(k): v for k, v in output.items()}
    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(output, f)
    else:
        return output
