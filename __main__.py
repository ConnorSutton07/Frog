"""

"""

import os.path as osp, os
import argparse
from icecream import ic
from gensim.models import FastText
import compress_pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
import random
import numpy as np
import re

from core.utils import get_mem_usage



PATH = osp.dirname(osp.realpath(__file__))
MODEL_PATH = osp.join(PATH, 'model')
specials = {'.', ',', '!', '?', ':', ';', '%'}

embedding_model = FastText.load(osp.join(MODEL_PATH, 'embeddings_l.model'))
# ic(embedding_model.wv.most_similar('love', topn = 25))


LEMMATIZER = WordNetLemmatizer()

def lemmatize(target: str) -> str:
    return LEMMATIZER.lemmatize(target)


stops = set(stopwords.words('english')) | specials


def synth_text(query: str, model: dict, n: int, length: int) -> str:
    og_query = query
    query: list = ((query.split(' '))[-n:])
    result = query[:]
    sentences = 0
    length_reached = False
    at_sentence = False
    i = 0
    while not length_reached or not at_sentence:
        at_sentence = False
        key = ' '.join(query)

        options = set()
        for _ in range(15):
            new_word = query[-1]
            cc = 0
            while new_word == query[-1] and cc < 5:
                try:  # TODO figure out why err here
                    new_word = model[key].sample()
                except Exception as e:
                    print(og_query)
                    raise e
                cc += 1
            options.add(new_word)
        options = list(options)
        q_w = [lemmatize(q) for q in result if q not in stops]
        o_w = [lemmatize(o) for o in options if o not in stops]

        if q_w and len(o_w) > 1:
            scores = []
            for o in o_w:
                score = 0
                for i, q in enumerate(q_w, start = 0):
                    score += np.sqrt(1 + len(q_w) - i) * (embedding_model.wv.similarity(o, q) ** 2)
                scores.append((o, score))
            winner = max(scores, key = lambda pair: pair[1])
            idx = scores.index(winner)
            new_word = options[idx]

        del query[0]
        query.append(new_word)
        result.append(new_word)

        if new_word in {'?', '!', '.'}:
            sentences += 1
            at_sentence = True
        i += 1
        if i == length:
            length_reached = True
    return ' '.join(result)


def cap_f(match: re.Match):
    return match.groups()[0] + ' ' + match.groups()[1].upper()


def make_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    return argparser


def main():
    argparser = make_argparser()
    args = argparser.parse_args()

    ngram = compress_pickle.load(osp.join(MODEL_PATH, 'three_gram.gz'))
    ic('loaded')

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
    
    cap_ptn = re.compile(r'([.!?]) (\w)')
    loose_num_ptn = re.compile(r'[.] \d+[.]')
    usr = ''
    while not usr:
        resp = synth_text((random.choice(starters)).lower(), ngram, 3, 50)
        for special in specials:
            resp = resp.replace(f' {special}', special)
        resp = resp.replace(f'$ ', '$')
        resp = resp.replace(' \' ', ' ').replace(' " ', ' ').replace(' \'.', '.').replace(' ".', '.')
        resp = re.sub(cap_ptn, cap_f, resp)
        resp = re.sub(cap_ptn, cap_f, resp)
        resp = re.sub(loose_num_ptn, '.', resp)
        resp = resp[0].upper() + resp[1:]
        print(resp)
        usr = input('\nContinue?')


if __name__ == '__main__':
    main()