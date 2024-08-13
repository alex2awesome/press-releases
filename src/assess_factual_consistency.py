from bart_score import BARTScorer
import os
from utils import get_spacy_model
from tqdm.auto import tqdm


def get_bart_model():
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_scorer_fn = 'bart_score.pth'
    if not os.path.exists(bart_scorer_fn):
        import gdown
        remote_file = 'https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing'
        gdown.download(remote_file, bart_scorer_fn, fuzzy=True, quiet=False)
    bart_scorer.load(path=bart_scorer_fn)
    return bart_scorer


def get_sentences(text_col, verbose=False):
    text_col = list(map(lambda x: x.replace('\n', ' '), text_col))
    # get first n sentences
    sent_pipe = get_spacy_model().pipe(
        text_col,
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "textcat", "ner"]
    )
    if verbose:
        sent_pipe = tqdm(sent_pipe, total=len(text_col))
    text_col = list(map(lambda x: list(x.sents), sent_pipe))
    text_col = list(map(lambda x: ' '.join(list(map(str, x))), text_col))
    return text_col


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default=None)
    parser.add_argument('--text-col', type=str, default='body')
    parser.add_argument('--id-map-file', type=str, default=None)
    parser.add_argument('--output-file', type=str, default=None)

    bart_scorer = get_bart_model()
    bart_scorer.score(['This is interesting.'], ['This is fun.'], batch_size=4)