import requests
import json
import random
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import mimetypes
import numpy as np
from flask import Flask, request, Response
import os
import re
from bs4 import BeautifulSoup
import textract
from dateutil.parser import parse as date_parse
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs, urljoin
from bs4 import BeautifulSoup
import re


CDX_TEMPLATE = 'https://web.archive.org/cdx/search/cdx?url={url}&output=json&fl=timestamp,original,statuscode,digest'
ARCHIVE_TEMPLATE = "https://web.archive.org/web/{timestamp}{flag}/{url}"
server = Flask(__name__)

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2919.83 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2866.71 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux i686 on x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2820.59 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2762.73 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2656.18 Safari/537.36',
]

def clean_url(to_get_url):
    return urljoin(to_get_url, urlparse(to_get_url).path)


def get_wayback_url(article_url, target_timestamp, sort_by='past'):
    orig_article_url = clean_url(article_url)
    target_timestamp = date_parse(target_timestamp).strftime('%Y%m%d%H%M%S')
    target_timestamp = int(target_timestamp)

    cdx_url = CDX_TEMPLATE.format(url=orig_article_url)
    cdx_response = requests.get(cdx_url).json()

    if len(cdx_response) < 2:
        return

    cols = cdx_response[0]
    timestamp_col = cols.index('timestamp')
    status_col = cols.index('statuscode')

    rows = cdx_response[1:]
    rows = list(filter(lambda x: x[status_col] != '-', rows))

    # get the nearest timestamp to the target timestamp by a criteria (e.g. "past", "closest", or "future")
    timestamps = list(map(lambda r: r[timestamp_col], rows))
    if sort_by == 'past':
        timestamps = list(filter(lambda t: int(t) < target_timestamp, timestamps))
    if sort_by == 'future':
        timestamps = list(filter(lambda t: int(t) > target_timestamp, timestamps))
    timestamps = sorted(timestamps, key=lambda t: abs(int(t) - target_timestamp))
    if len(timestamps) == 0:
        return

    article_timestamp = timestamps[0]

    # fetch URL
    return ARCHIVE_TEMPLATE.format(timestamp=article_timestamp, flag='', url=orig_article_url)


@server.route('/', methods=['POST'])
def get_webpage_pdf_and_convert_to_text():
    file_data = request.json
    target_timestamp_key = file_data.get('target_timestamp_key') or 'homepage_key'
    article_url_key = file_data.get('article_url_key') or 'article_url'
    sort_critera = file_data.get('sort_criteria') or 'past'

    wayback_url = get_wayback_url(file_data[article_url_key], file_data[target_timestamp_key], sort_by=sort_critera)
    if wayback_url is None:
        return '{}'

    resp = requests.get(
        wayback_url,
        headers={'User-Agent': random.choice(USER_AGENTS)}
    )
    content_type_header = resp.headers['Content-Type'].split(';')[0]
    doc_extension = mimetypes.guess_extension(content_type_header)

    # file is HTML
    if doc_extension == '.html':
        soup = BeautifulSoup(resp.text, 'lxml')
        for s in soup(["script", "style", "meta", "noscript", "svg", "button", "figcaption", "head", "title", "nav"]):
            _ = s.extract()
        text = re.sub('\n+\s*', '\n', soup.get_text(' '))
        file_data['data'] = text

    # file is a PDF
    elif doc_extension == '.pdf':
        pdf_content = resp.content
        doc = convert_from_bytes(pdf_content)
        all_pages_text = []
        # loop through the pages of the pdf file and extract text
        for page_number, page_data in enumerate(doc):
            page_data_array = np.asarray(page_data)
            txt = pytesseract.image_to_string(Image.fromarray(page_data_array))
            all_pages_text.append(txt)
        file_data['data'] = '\n\n'.join(all_pages_text)

    elif doc_extension == '.doc':
        tmp = NamedTemporaryFile()
        with open(tmp.name, 'wb') as f:
            f.write(resp.content)
        text = textract.process(tmp.name, extension='.doc').decode()
        file_data['data'] = text

    else:
        return '{}'

    return json.dumps(file_data)


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)


