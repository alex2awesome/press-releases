from more_itertools import flatten
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import xopen
import orjson
import pandas as pd
import glob
import os
import gzip
from sklearn.feature_extraction.text import CountVectorizer


to_exclude_list = set([
    'bloomberg  correction:  clarification', 'the federal reserve’s',
    'the federal communications commission', 'ftc', 'tech', 'post', 'refinitiv', 'who',
    'the new york times', 'congress', 'times', 'twitter',
    'fox', 'congress', 'senate', 'reuters breakingviews',
    'reuters.com register', 'register', 'reuters', 'reuters.com register',
    'reuters.com register  reporting', 'the thomson reuters trust principles',
    'thomson reuters', 'company', 'company information',
    'the new york stock exchange', 'nyse', 'the nasdaq stock market',
    'the european union', 'fed', 'eu', 'the federal reserve',
    'getty images', 'the etf channel flexible growth investment portfolio',
    'etf channel', 'etfchannel.com', 'ai', 'nfl', 'nasd', 'exchange', 'nasdaq',
    'bloomberg finance', 'afp', 'the associated press', 'ap', 'associated press',
    'getty images  key facts', 'the etf finder', 'house', 'sec', 'ipad', 'fda', 'ftx',
    'faa', 'the federal aviation administration', 'the federal aviation administration  faa',
    'visit business insider\'s', 'business insider', 'intelligence', 'the wall street journal',
    'fox news', 'fbi', 'eps', 'barrons.com', 'factset', 'the securities and exchange commission',
    'the labor department', 'federal reserve', 'morningstar', 'goldman sachs', 'the wall street journal',
    'bloomberg', 'barron\'s', 'citigroup', 'ipo', 'index', 'globe newswire', 'the globe and mail',
    'cnbc', 'cramer', 'board', 'board of directors', 'the board of directors', 'the board',
    'board', 'ecb', 's&p', 'the s&p', 'the s&p 500', 'the s&p 500 index', 'the s&p 500 index spx',
    'times square', 'flash', 'javascript', 'citigroup', 'bank of america', 'the european central bank',
    'jp morgan', 'jp morgan chase', 'dow jones', 'nyt', 'c.e.o.', 'ceo', 'cfo', 'c.f.o.', 'cfo',
    'facebook messenger', 'tc messenger', 'techcrunch', 'washington post', 'post writer badge',
    'culture connoisseur badge culture connoisseurs ', 'washingtologist badge',  'weather watcher badge weather watchers',
    'superfan', 'superfans', 'post contributor badge', 'post contributor', 'post contributors',
    'the washington post', 'the washington post opinions', 'the discussion', 'the discussion policy',
    'the discussion guidelines', 'the discussion guidelines and faqs', 'the discussion guidelines and faqs',
    'service', 'bloomberg television\'s', 'bloomberg television', 'bloomberg correction: clarification ',
    'tech friend', 'market data', 'world markets', 'wonkbook  personal finance',
    'wsj', 'the wall street journal', 'journal', 'potomac watch', 'propublica', 'pbs frontline',
    'free expression', 'the wall street journal’s', 'customer service', 'cbs news', 'cbs',
    'cbs sports', 'the customer center', 'the associated press and who wants',
    'wnyc', 'marketwatch front page', 'search marketwatch', 'search  search marketwatch',
    'comtex', 'comtex/', 'nls', 'marketwire', 'marketwatch community', 'the college-industrial',
    'thomson first call', 'dow', 'expresswire', 'the securities and exchange commission',
    'factset', 'irs', 'federal reserve', 'merrill lynch', 'the justice department', 'european union',
    'citigroup', 'the food and drug administration', 's&p global market intelligence', 'flickr',
    'motley fool', 'the motley fool', 'the motley fool\'s', 'the motley fool\'s new personal finance brand',
    'nasdaqoth', 'facebook share', 'twitter tweet', 'linkedin share', 'email this', 'print this',
    'facebook share article', 'twitter tweet article', 'linkedin share article', 'email this article',
    'twitter share article', 'linkedin share article', 'watchlist +',
    'treasury', 'ubs', 'gop', 'mad money', 'file/associated press', 'pentagon', 'arrowright', 'the justice department',
    'bloomberg news', 'the u.s. treasury', 'cop27',
    'wall street journal', 'npr', 'opb', 'mutual fund', 'factset digital solutions',
    'fox business', 'ticker security last change change', 'fox business’', 'fox business',
    'the fox business network’s',
    'the treasury department', 'the federal trade commission',
    'ticker security last change change', 'the treasury department', 'abc', 'standard & poor\'s',
    'globenewswire', 'social security', 'market.us', 'gigantic report online store',
    'the marketwatch news department', 'fitch ratings', 'cbs marketwatch', 'wiredrelease',
    'the new york mercantile exchange', 'first call',  'wall street journal',
    'culture connoisseur badge culture connoisseurs', 'company llc d', 'factset research',
    'bulletin', 'dow jones reprints', 'vc', 'quotes?email', 'un', 'supreme court ', 'state',
    'the supreme court', 'mutual fund', 'ticker security last change change', 'factset digital solutions',
])

def load_and_filter_jsonl_fault_tolerant(filename):
    lines = []
    try:
        with gzip.open(filename) as f:
            for line in f:
                try:
                    lines.append(orjson.loads(line))
                except:
                    continue
    except:
        pass

    lines_df = pd.DataFrame(lines)
    lines_df = (
        lines_df
            .loc[lambda df: df['ents'].str.len() > 0]
            .reset_index(drop=True)
    )
    lines_df['ents'] = lines_df['ents'].apply(lambda x: list(map(lambda y: y.lower().strip(), x)))
    return lines_df

def print_top_terms(df, cv):
    global to_exclude_list

    vec = cv.fit_transform(df)
    top_terms = (
        pd.DataFrame((vec > 0).todense(), columns=sorted(cv.vocabulary_))
        .sum()
        .sort_values(ascending=False)
        .pipe(lambda s: s / len(df))
        .reset_index()
        .loc[lambda df: ~df['index'].isin(to_exclude_list)]
        .set_index('index')[0]
        .head(50)
    )

    to_exclude_list = to_exclude_list.union(set(top_terms.loc[lambda s: s> .03].index))
    print(top_terms)

# to run:
# python link_archive_files.py --data-directory <d> --output-file <o> --archive-article-subset <s>
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-directory', type=str, default='.')
    parser.add_argument('--archive-article-subset', nargs='*', default=None)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    press_release_article_files = glob.glob(os.path.join(args.data_directory, '*-links*') )
    archive_article_files = glob.glob(os.path.join(args.data_directory, '*-parsed*'))
    if args.archive_article_subset is not None:
        check_if_in_subset = lambda c: any(map(lambda f: f in c, args.archive_article_subset))
        archive_article_files = list(filter(check_if_in_subset, archive_article_files))
        archive_file_names = list(map(lambda x: x.split('articles')[0], archive_article_files))
        archive_file_names = list(map(lambda x: x.replace('/', '').replace('.', ''), archive_file_names))
        p = args.output_file.split('.')
        args.output_file = p[0] + f'__{"-".join(archive_file_names)}.' + p[1]

    str_join = lambda x: '\n'.join(x)
    print(f'press release files: {str_join(press_release_article_files)}')
    print(f'archive files: {str_join(archive_article_files)}')

    print('reading files and calculating top entities...')
    cv = CountVectorizer(analyzer=lambda x: x)
    press_release_dfs = []
    archival_dfs = []
    for f in (press_release_article_files + archive_article_files):
        articles_df = load_and_filter_jsonl_fault_tolerant(f)
        print(f'top terms:\n{f}')
        print_top_terms(articles_df['ents'], cv)
        articles_df['ents'] = (
            articles_df['ents']
            .apply(lambda x: x if isinstance(x, (list, set)) else [])
            .apply(lambda x: list(set(x) - to_exclude_list))
        )
        if 'links' in f: # press release
            press_release_dfs.append(articles_df)
        else:
            archival_dfs.append(articles_df)

    #
    press_release_articles_df = pd.concat(press_release_dfs).reset_index(drop=True)
    archival_articles_df = pd.concat(archival_dfs).reset_index(drop=True)

    print('filtering entities down to one file...')
    # get only archival articles with at least 1 overlap
    target_ents = press_release_articles_df['ents'].pipe(lambda s: set(list(flatten(s.tolist()))))
    relevant_archival_articles = (
        archival_articles_df
            .loc[lambda df: df['ents'].apply(lambda x: len(set(x) - target_ents) > 0)]
            .reset_index(drop=True)
    )

    print('calculating article map...')
    ent_vecs = cv.fit_transform(pd.concat([
        press_release_articles_df['ents'],
        relevant_archival_articles['ents'],
    ]))
    pr_ent_vecs = ent_vecs[:len(press_release_articles_df)]
    archive_ent_vecs = ent_vecs[len(press_release_articles_df):]
    sims = cosine_similarity(pr_ent_vecs, archive_ent_vecs)
    x, y = np.where(sims > 0)
    archive_article_map = pd.concat([
        press_release_articles_df['article_url']
        .iloc[x]
        .reset_index(drop=True)
        .to_frame('target_article_url'),
        relevant_archival_articles['article_url']
        .iloc[y]
        .reset_index(drop=True)
        .to_frame('archival_url')
    ], axis=1)

    print('writing to disk...')
    archive_article_map.to_csv(args.output_file, index=False)




