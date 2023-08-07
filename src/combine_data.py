##
# Reads the following files:
# -----------------------------------------------------------
# 1. parsed article files
# 2. article map files
# 3. wayback timestamps files
#
# Combines them all together and writes them to a sqlite3 database.
#


import glob
import pandas as pd
import sqlite3

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

    print('reading article maps...')
    all_maps = glob.glob('article-map__*')
    all_maps = list(map(pd.read_csv, all_maps))
    all_maps_df = pd.concat(all_maps)
    target_article_urls = all_maps_df['target_article_url'].drop_duplicates()
    archival_article_urls = all_maps_df['archival_url'].drop_duplicates()

    print('reading timestamp files...')
    all_timestamps = glob.glob('all-wayback-timestamps*')
    all_timestamps = list(map(
        lambda f: pd.read_csv(f, index_col=None, header=None, on_bad_lines='warn'), all_timestamps
    ))
    all_timestamps_df = pd.concat(all_timestamps)
    all_timestamps_df.columns = ['url', 'timestamp']

    conn = sqlite3.connect('press_release_articles.db')
    all_maps_df.to_sql('article_map', conn, if_exists='replace', index=False)

    # merge article data
    for article_file in glob.glob('*parsed.jsonl.gz'):
    # for article_file in [
    #     'techcrunch-articles-parsed.jsonl.gz',
    #     'fox-business-articles-parsed.jsonl.gz',
    #     'wsj-articles-parsed.jsonl.gz',
    # ]:
        print(f'reading {article_file}...')
        source = article_file.split('-articles')[0]
        links_file = article_file.replace('parsed.jsonl.gz', 'links-with-chars.jsonl.gz')
        article_df = (
            pd.read_json(article_file, lines=True)
                .assign(source=source)
        )

        slim_article_df = (
            article_df
                # .loc[lambda df: df['article_url']
                # .isin(pd.concat([target_article_urls, archival_article_urls]))]
        )

        links_article_df = pd.read_json(links_file, lines=True)
        articles_to_href_df = (
            links_article_df
            .set_index('article_url')['links']
            .explode()
            .dropna()
            .pipe(lambda s: pd.DataFrame(s.tolist(), index=s.index))
            .assign(source=source)
        )

        slim_article_df = (
            slim_article_df.assign(
                timestamp_join_key=lambda df: df['article_url']
                                                  .apply(lambda x: x.split(')'))
                                                  .apply(lambda x: cc_to_web_map[x[0]] + x[-1])
            )
            .merge(all_timestamps_df, left_on='timestamp_join_key', right_on='url', how='left')
            .drop(columns=['url'])
            .rename(columns={'article_url': 'common_crawl_url'})
            .assign(is_press_release_article=lambda df: df['common_crawl_url'].isin(target_article_urls))
            .assign(is_archival_article=lambda df: df['common_crawl_url'].isin(archival_article_urls))
        )

        ents_df = (
            slim_article_df[['common_crawl_url', 'ents']]
                .explode('ents')
                .dropna()
                .assign(source=source)
        )

        (
            slim_article_df
                .drop(columns=['ents', 'links', 'article_video'])
                .assign(article_authors=lambda df: df['article_authors'].str.join('; '))
                .to_sql('article_data', conn, if_exists='append', index=False)
        )
        ents_df.to_sql('article_ents', conn, if_exists='append', index=False)
        articles_to_href_df.reset_index().to_sql('article_to_href', conn, if_exists='append', index=False)



