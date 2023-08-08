from bleach import clean
import copy
from unidecode import unidecode
from bs4 import BeautifulSoup
import re
import spacy
import orjson
from tqdm.auto import tqdm

def get_num_lines_robust(f):
    num_lines = 0
    try:
        for _ in f:
            num_lines += 1
    except:
        pass
    return num_lines


def strip_html_leave_links(src, allowed=['a']):
    return clean(src, tags=allowed, strip=True, strip_comments=True)


def extract_links_with_char_num_and_post_process(datum):
    """Extract text and links from HTML string, and get the position of the links in the text."""
    #
    datum = copy.deepcopy(datum)
    key = 'html' if 'html' in datum else 'article_html'
    html_str = datum[key]
    html_str = unidecode(html_str)
    soup = BeautifulSoup(html_str, 'lxml')
    for s in soup(["script", "style", "meta", "noscript", "svg", "button", "figcaption", "head", "title", "nav"]):
        _ = s.extract()
    #
    text_with_links = strip_html_leave_links(str(soup))
    text_with_links = re.sub(r'\s+', ' ', text_with_links).strip().replace('> <', '><')
    #
    # remove <a> tags that don't have an href element
    pattern = r'<a\b(?![^>]*\bhref\b)[^>]*>(.*?)</a>'
    text_with_links = re.sub(pattern, ' ', text_with_links, flags=re.IGNORECASE | re.DOTALL)
    #
    # split on the links
    text_sans_links_split = re.split(r'<a href="[^"]+".*?>[^<]*</a>', text_with_links)
    text_sans_links_split = list(map(lambda x: x.strip(), text_sans_links_split))
    links_and_link_text = re.findall(r'<a href="([^"]+)".*?>([^<]*)</a>', text_with_links)
    links_and_link_text = list(map(lambda x: (x[0].strip(), f' {x[1].strip()} '), links_and_link_text))
    #
    # count number of tokens before each link
    char_num = 0
    output = []
    output_text = []
    for text_chunk, link in zip(text_sans_links_split, links_and_link_text):
        char_num += len(text_chunk)
        output.append({
            'href': link[0],
            'text': link[1],
            'char_start_idx': char_num,
            'char_end_idx': char_num + len(link[1]),
        })
        char_num += len(link[1])
        output_text.append(text_chunk)
        output_text.append(link[1])
    #
    # add the last chunk of text
    output_text.append(text_sans_links_split[-1])
    datum.pop(key)
    datum['links'] = output
    datum['article_text'] = ''.join(output_text)
    return datum


def extract_links_and_process(datum):
    """Post-process the response from the GCF endpoint.

    Parameters
    ----------
    * datum : (dict) of data returned from the GCF endpoint.
    """
    key = 'html' if 'html' in datum else 'article_html'
    html = datum[key]
    soup = BeautifulSoup( html, 'lxml')
    # find all links with non-null href
    links = soup.find_all('a', href=True)
    links = list(filter(lambda x: x.get_text() != '', links))
    links_obj = list(map(lambda x: {'text': x.get_text(), 'href': x['href']}, links))
    datum['links'] = links_obj
    datum.pop(key)
    return datum


_spacy_model = None
def get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        _spacy_model = spacy.load('en_core_web_lg')
        _spacy_model.add_pipe('sentencizer')
    return _spacy_model

DESIRED_ENTS = set(['ORG', 'PRODUCT', 'FAC', 'LAW', 'EVENT'])
def retrieve_ents_for_col(text_col, num_sentences=5, desired_ents=DESIRED_ENTS, verbose=False):
    text_col = list(map(lambda x: x.replace('\n', ' '), text_col))
    # get first n sentences
    sent_pipe = get_spacy_model().pipe(
        text_col,
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "textcat", "ner"]
    )
    if verbose:
        sent_pipe = tqdm(sent_pipe, total=len(text_col))
    text_col = list(map(lambda x: list(x.sents)[:num_sentences], sent_pipe))
    text_col = list(map(lambda x: ' '.join(list(map(str, x))), text_col))

    # ner
    ner_pipe = get_spacy_model().pipe(
        text_col,
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "textcat", "sentencizer"]
    )
    if verbose:
        ner_pipe = tqdm(ner_pipe, total=len(text_col))
    entities = []
    for doc in ner_pipe:
        if desired_ents is not None:
            ents = list(filter(lambda x: x.label_ in desired_ents, doc.ents))
        ents = list(map(str, ents))
        entities.append(ents)

    return entities

from urllib.parse import urlparse
from tldextract import tldextract
domain_exclusions = open('../data/utility-files/domain-exclusions-master-list.txt').read().split('\n')
domain_exclusions = set(list(filter(lambda x: x != '', domain_exclusions)))
text_candidates = ['press release', 'news release', 'announce', 'earnings call']
href_whitelist_candidates = [
    'prnewswire',
    'businesswire',
    'press',
    'release',
    'globenewswire',
    'news',
    'earnings',
    'call-transcript'
]


def find_press_release(href=None, text=None, row=None):
    if href is None:
        href = row['href']
    if text is None:
        text = row['text']
    for s in ['/', '#', 'mailto:', '@']:
        if href.startswith(s):
            return False
    # parse domain
    try:
        domain = urlparse(href).netloc
        domain = tldextract.extract(domain).domain
    except ValueError as e:
        print(f'Error {str(e)} parsing {href}...')
        return False
    # blacklist
    if domain in domain_exclusions:
        return False
    # text
    for t in text_candidates:
        if t in text:
            return True
    # href
    for h in href_whitelist_candidates:
        if h in href:
            return True
    return False


def count_str_splits(links=None, text=None, row=None):
    if links is None:
        links = row['links']
    if text is None:
        text = row['article_text']
    t_link = list(filter(lambda x: find_press_release(x['href'], x['text']), links))
    # get sentences
    return list(map(lambda x: len(text.split(x['text'])), t_link))


def audit_links(line_list, args):
    """
    Audit the links file to see how many links are press releases.
    """
    file_batch = []
    for idx, line in tqdm(enumerate(line_list)):
        if args.start_idx is not None and idx < args.start_idx:
            continue
        if args.end_idx is not None and idx >= args.end_idx:
            break
        try:
            d = orjson.loads(line)
        except:
            continue
        found = False
        for link in d['links']:
            if find_press_release(link['href'], link['text']):
                found = True
                break
        if found:
            file_batch.append(d)

        if len(file_batch) >= args.batch_size:
            yield file_batch
            file_batch = []

