"""

"""

import os.path as osp, os
import argparse
from icecream import ic
from gensim.models import FastText
import compress_pickle
from tqdm import tqdm
import random
from wordsegment import segment as wordsegment, load as ws_load; ws_load()
from nltk.corpus import words; en_words = set(words.words())
import numpy as np
import re

from core.nlp import lemmatize, stopwords
from core.utils import get_mem_usage


PATH = osp.dirname(osp.realpath(__file__))
MODEL_PATH = osp.join(PATH, 'model')

term_chars = {'?', '!', '.', ';', ':'}
special_chars = {'.', ',', '!', '?', ':', ';', '%'}
extra_stops = stopwords | special_chars
word_match = re.compile(r'[\'a-zA-Z]+')

embedding_model = FastText.load(osp.join(MODEL_PATH, 'embeddings_l.model'))
def word_sim(w1: str, w2: str):
    return embedding_model.wv.similarity(w1, w2)


def score_fn(i: int, size_r: int, o: str, r: str):
    return np.sqrt(1 + size_r - i) * (word_sim(o, r) ** 2)

def synth_text(query: str, model: dict, n: int, target_length: int, n_samples: int = 15) -> str:
    result_tokens = query.split(' ')
    sent_len = 0
    while len(result_tokens) < target_length or result_tokens[-1] not in term_chars - {';', ':'}:
        key = ' '.join(result_tokens[-n:])
        try:
            options = list({option for _ in range(n_samples) if (option := model[key].sample()) != result_tokens[-1]})
            if result_tokens[-1] in term_chars:
                for char in term_chars:
                    if char in options:
                        options.remove(char)
        except Exception as e:
            print(query)
            raise e
        r_words = [lemmatize(r) for r in result_tokens if r not in extra_stops]  # have so far
        o_words = [lemmatize(o) for o in options if o not in extra_stops]  # could add next

        if r_words and len(o_words) > 1:  # we have some results and have multiple options
            scores = [(o, sum([score_fn(i, len(r_words), o, r) for i, r in enumerate(r_words)])) for o in o_words]
            new_word = options[scores.index(max(scores, key = lambda pair: (lambda o, score: score)(*pair)))]
        else:
            try:
                new_word = options.pop()
            except Exception as e:
                print(query)
                raise e
        if new_word in term_chars: sent_len = 0
        else: sent_len += 1
        result_tokens.append(new_word)
        
    tokens = []
    for token in result_tokens:
        segmented = wordsegment(token)
        if word_match.fullmatch(token) and all([segment in en_words and len(segment) > 2 for segment in segmented]):
            for segment in segmented:
                tokens.append(segment)
        else:
            tokens.append(token)

    return ' '.join(tokens)


def make_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    return argparser

def main():
    argparser = make_argparser()
    args = argparser.parse_args()

    ngram = compress_pickle.load(osp.join(MODEL_PATH, 'three_gram.gz'))

    starters = [
        'you have to try',
        'i believe that we',
        'i need to stop',
        'my throat is parched',
        'dishonour is worse than',
        'after the lapse of',
        'i know that you',
        'if it be love',
        'the majority of these',
        'i see now they',
        'i tried to find',
    ]

    simple_subs = [
        (' \' ', ' '),
        (' " ', ' '),
        (' \'.', '.'),
        (' ".', '.'),
        ('$ ', '$'),
        (' i ', ' I '),
        (' i\'', ' I\''),
    ] + [(' ' + char, char) for char in special_chars]
    
    re_subs = [
        (re.compile(r'([.!?]) (\w)'), lambda m: m.groups()[0] + ' ' + m.groups()[1].upper()),
        (re.compile(r'[.!?] \d+[.]'), '.'),
        (re.compile(r'^(.)'), lambda m: m.groups()[0].upper()),
    ]

    while True:
        resp = synth_text((random.choice(starters)).lower(), ngram, 3, 150)
        for text, sub in simple_subs: resp = resp.replace(text, sub)
        for ptn, sub in re_subs: resp = re.sub(ptn, sub, resp)
        print(resp)
        if input('\n') in {'exit', 'exit()', 'quit', 'quit()'}: break


if __name__ == '__main__':
    main()