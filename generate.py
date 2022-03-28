"""

"""

import sys; sys.path.append('.')
import os.path as osp, os
import argparse
from tqdm import tqdm
import re
from unidecode import unidecode
import multiprocessing as mp
from collections import defaultdict
from icecream import ic
import string; alphabet = set(string.ascii_lowercase)

from core.utils import save_object, get_mem_usage
from core.ngram import build_ngram
from core.nlp import word_tokenize, sent_tokenize, lemmatize, pos_tag_sents, stopwords, embeddings


PATH = osp.dirname(osp.realpath(__file__))
CORPUS_PATH = osp.join(PATH, 'corpus')
MODEL_PATH = osp.join(PATH, 'model')


invalid_ptn = re.compile(r'([][/&^#@~=+\\|}{()]|https?|www[.]|[.]html|reddit|tl;?dr|^edit|chapter)')
valid_one_char = {'a', 'i', 'u'}
valid_one_char_comp = alphabet - valid_one_char

to_remove = {
    re.compile(r'[\n<>*`]'),
    re.compile(r'^[-* ]+'),
    re.compile(r'\d+[:]\d+'),
}

to_replace = {
    re.compile(r'_'): ' ',
    re.compile(r'[.](?=.)'): '. ',
    re.compile(r'u[.]s[.](a[.])?'): 'usa',
}


def process_doc(doc_path: str) -> tuple[str, list[list[str]], dict[str, int], list[list[tuple[str, str]]]]:
    with open(doc_path, 'r') as f: raw_sents = list(sent_tokenize(unidecode(f.read())))
    sent_tokens = []
    vocab = defaultdict(int)
    while raw_sents:
        raw_sent = raw_sents.pop(0)
        for ptn in to_remove: raw_sent = re.sub(ptn, '', raw_sent)
        for ptn, repl in to_replace.items(): raw_sent = re.sub(ptn, repl, raw_sent)
        words = word_tokenize(raw_sent)
        if not invalid_ptn.findall(' '.join(words).lower()):
            lowered = [lowered for word in words if len(lowered := word.lower()) > 1 or lowered not in valid_one_char_comp]
            for word in lowered: vocab[word] += 1
            sent_tokens.append(lowered)

    pos = pos_tag_sents(sent_tokens)
    return doc_path, sent_tokens, vocab, pos


def make_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    return argparser


def main() -> None:
    argparser = make_argparser()
    args = argparser.parse_args()
    doc_paths = [osp.join(CORPUS_PATH, doc) for doc in os.listdir(CORPUS_PATH) if doc.endswith('.txt')]

    all_vocab = defaultdict(int)
    all_tokens = []
    all_sent_tokens = []
    all_pos_tokens = []

    with mp.Pool() as p:
        for path, sent_tokens, vocab, pos in tqdm(
            p.imap_unordered(process_doc, doc_paths, chunksize=1),
            total = len(doc_paths),
            desc = 'processing docs'
        ):
            for sent_token in sent_tokens:
                for word in sent_token:
                    all_tokens.append(word)
                all_sent_tokens.append(sent_token)
            for term, count in vocab.items(): all_vocab[term] += count
            for sent_token in pos:
                for token_pair in sent_token:
                    all_pos_tokens.append(token_pair)


    sorted_vocab = sorted(list(all_vocab.items()), key = lambda pair: (lambda _, r: r)(*pair), reverse = True)

    with tqdm(total = 5, desc = 'saving objects') as pbar:
        save_object(sorted_vocab, osp.join(MODEL_PATH, 'sorted_vocab')); pbar.update()
        save_object(all_vocab, osp.join(MODEL_PATH, 'vocab_freq')); pbar.update()
        save_object(all_tokens, osp.join(MODEL_PATH, 'tokens')); pbar.update()
        save_object(all_sent_tokens, osp.join(MODEL_PATH, 'sent_tokens')); pbar.update()
        save_object(all_pos_tokens, osp.join(MODEL_PATH, 'pos_tokens')); pbar.update()

    # generate ngram, currently single process due to potential memory constraints
    pos_vals = [pair[1] for pair in all_pos_tokens]
    with tqdm(total = 4, desc = 'generating ngrams') as pbar:
        for i, prefix in {(3, 'three'), (4, 'four')}:
            save_object(build_ngram(all_tokens, i), osp.join(MODEL_PATH, f'{prefix}_gram')); pbar.update()
            save_object(build_ngram(pos_vals, i), osp.join(MODEL_PATH, f'{prefix}_gram_pos')); pbar.update()

    # train embeddings
    lemmatized = [[lemmatize(token) for token in sent_tokens if token not in stopwords] for sent_tokens in tqdm(all_sent_tokens, desc = 'lemmatizing')]
    embedding_model = embeddings(lemmatized)
    embedding_model.save(osp.join(MODEL_PATH, 'embeddings_l.model'))
    
    # sentiment analysis?


if __name__ == '__main__':
    main()