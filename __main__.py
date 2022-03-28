"""
TODO: Use multiple vals of n to sample, then create a dist that prefers
      higher n but still allows less. This would make things more coherent
      but still allow flexibility.
"""

import os.path as osp, os
import argparse
from icecream import ic
from gensim.models import FastText
import random
from nltk.corpus import words; en_words = set(words.words())
import numpy as np
from num2words import num2words
import re

from core.nlp import lemmatize, stopwords, pos_tag
from core.utils import load_object


PATH = osp.dirname(osp.realpath(__file__))
MODEL_PATH = osp.join(PATH, 'model')

term_chars = {'?', '!', '.', ';', ':'}  # these terminate a psuedo sentence
special_chars = {'.', ',', '!', '?', ':', ';', '%', '$'}  # special chars
special_chars_rhs = {'.', ',', '!', '?', ':', ';', '%'}  # special chars
special_chars_lhs = {'$'}  # special chars
extra_stops = stopwords | special_chars | {'\'', '"'}
word_match = re.compile(r'[\'a-zA-Z]+')
trivial_pos = {'CC', 'CD', 'CD', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'WDT', 'WP', 'WRB'}

embedding_model = FastText.load(osp.join(MODEL_PATH, 'embeddings_l.model'))
def word_sim(w1: str, w2: str):
    return (embedding_model.wv.similarity(w1, w2) + 1) / 2  # now in [0, ]


def option_relevance(i: int, size_r: int, o: str, r: str):
    # Helps pick which option to pick in word synth
    # i: index of meaningful word already generated, max = size_r - 1
    # size_r: number of meaningful already generated words
    # o: current option
    # r: current meaningful word
    dist_from_curr = i / size_r  # in (0, 1]
    similarity = word_sim(o, r)  # in [0, 1]
    return np.log(dist_from_curr + 2) * similarity ** 2  # words generated first are more important, score in [0, 1]

def relevance_score(o, r_words):
    return np.sqrt(sum([option_relevance(i, len(r_words), o, r) for i, r in enumerate(r_words)]) / len(r_words) ** 2)

def synth_text(query: str, model: dict, n: int, target_length: int, n_samples: int = 10) -> str:
    if n < 2: raise ValueError('n must be at least 2')
    result_tokens = query.split(' ')
    #sent_len = 0
    while len(result_tokens) < target_length or (result_tokens[-1] not in term_chars - {';', ':'} and len(result_tokens) < 1.5 * target_length):
        # while (we haven't generated enough words) OR (we have generated enough words but we have not completed current sentence yet AND that sentence isn't too long)
        dist = model[' '.join(result_tokens[-n:])]
        options = list({dist.sample() for _ in range(n_samples)})
        r_words = [lemmatize(r) for r, pos in pos_tag(result_tokens) if r not in extra_stops and pos not in trivial_pos]  # meaningful words generated so far
        o_words = [lemmatize(o) for o in options]  # could add these words next

        if r_words and len(o_words) > 1:  # we have some results and have multiple options
            # for each option, lets compute its average relevance score across all the meaningful words generated so far
            relevance_scores = [relevance_score(o, r_words) for o in o_words]
            # randomly pick option, based on 
            weights = [score * dist.get_weight(option) for option, score in zip(options, relevance_scores)]
            new_word = random.choices(options, weights, k = 1).pop()
        else: new_word = options.pop()
        #if new_word in term_chars: sent_len = 0  # reset current sentence length
        #else: sent_len += 1
        result_tokens.append(new_word)
        
    return result_tokens


def make_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    return argparser

def main():
    argparser = make_argparser()
    args = argparser.parse_args()

    n = 3

    ngram: dict[str, 'core.ngram.FreqDistribution'] = load_object(osp.join(MODEL_PATH, f'{num2words(n)}_gram.gz'))

    # starters = [
    #     'you have to try',
    #     'i believe that we',
    #     'i need to stop',
    #     'my throat is parched',
    #     'dishonour is worse than',
    #     'after the lapse of',
    #     'i know that you',
    #     'if it be love',
    #     'the majority of these',
    #     'i tried to find',
    #     'now obey the instruction',
    #     'the displacement of a',
    #     'it is unnecessary to',
    #     'we might expect that',
    #     'there are a number',
    #     'now I speak of',
    #     'a society that has',
    # ]
    starters = [gram for gram in ngram.keys() if not any(gram.startswith(char) for char in special_chars)]

    simple_subs = [
        ('"', ''),
        (' \' ', ' '),
        ('\' ', ' '),
        (' \'', ' '),
        (' " ', ' '),
        (' \'.', '.'),
        (' ".', '.'),
        (' i ', ' I '),
        (' i\'', ' I\''),
    ] + [(' ' + char, char) for char in special_chars_rhs] #+ [(char + ' ', char) for char in special_chars_lhs]
    
    re_subs = [
        (re.compile(r' {2,}'), ' '),
        (re.compile(r'([.!?]) (\w)'), lambda m: m.groups()[0] + ' ' + m.groups()[1].upper()),
        (re.compile(r'[.!?] \d+[.]'), '.'),
        (re.compile(r'^(.)'), lambda m: m.groups()[0].upper()),
    ]

    while True:
        query = random.choice(starters)
        synthesized_tokens = synth_text((query).lower(), ngram, n, 150)
        synthesized = ' '.join(synthesized_tokens)
        for text, sub in simple_subs: synthesized = synthesized.replace(text, sub)
        for ptn, sub in re_subs: synthesized = re.sub(ptn, sub, synthesized)
        if synthesized[-1] not in term_chars: synthesized += '...'
        print(synthesized)

        if input() in {'exit', 'exit()', 'quit', 'quit()'}: break

if __name__ == '__main__':
    main()