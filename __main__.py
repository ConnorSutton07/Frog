"""

"""

print('Importing libraries...')
import os.path as osp, os
import argparse
from icecream import ic
from gensim.models import FastText
import random
from tqdm import tqdm
from nltk.corpus import words; en_words = set(words.words())
from nltk.tokenize import word_tokenize
import numpy as np
import pyttsx3
from num2words import num2words
import re

from core.nlp import lemmatize, stopwords, pos_tag
from core.utils import load_object, irange, Timer

print('Instantiating resources...')
PATH = osp.dirname(osp.realpath(__file__))
MODEL_PATH = osp.join(PATH, 'model')

speech_engine = pyttsx3.init()
speech_engine.setProperty('rate', 200)

term_chars = {'?', '!', '.', ';', ':'}  # these terminate a psuedo sentence
special_chars = {'.', ',', '!', '?', ':', ';', '%', '$'}
special_chars_rhs = {'.', ',', '!', '?', ':', ';', '%'}
special_chars_lhs = {'$'}
extra_stops = stopwords | special_chars | {'\'', '"'}
word_match = re.compile(r'[\'a-zA-Z]+')
trivial_pos = {'CC', 'CD', 'CD', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'WDT', 'WP', 'WRB'}

embedding_model = FastText.load(osp.join(MODEL_PATH, 'embeddings_l.model'))
def word_sim(w1: str, w2: str):
    return (embedding_model.wv.similarity(w1, w2) + 1) / 2  # now in [0, 1]

def get_similar(w: str, topn: int = 10) -> list[str]:
    return [term for term, pos in pos_tag([t for t, _ in embedding_model.wv.most_similar(w, topn=topn)]) if pos not in trivial_pos]

def extract_meaningful(tokens: list[str]):
    return [t for t, pos in pos_tag(tokens) if t not in extra_stops and pos not in trivial_pos]


def score_result(
        anchors: list[str],
        synth_anchors: list[str],
        relevance_weight: float = 100,
        presence_weight: float = 1,
    ) -> float:
    # gives score to a complete response
    targets = [s for a in anchors for s in get_similar(a, 50)] + anchors  # what user wants + similar to what user wants
    target_set = set(targets)
    target_count = {t: targets.count(t) for t in target_set}  # how important is each thing user might want
    synth_anchor_count = {t: synth_anchors.count(t) for t in target_set}  # how many times did we produce each thing user might want
    relevance = sum([tc * synth_anchor_count[t] for t, tc in target_count.items() if t in synth_anchor_count]) / (len(target_set) or 1)  # compares each thing user wants vs how often it occurred
    presence = sum([synth_anchors.count(a) for a in anchors])
    return relevance_weight * relevance + presence_weight * presence
    # return sum([sum([word_sim(a, s) ** 2 for a in anchors]) / len(anchors) for s in synth_anchors]) / len(synth_anchors)


def transition_score(i: int, r_words: list[str], option: str, result_word: str) -> float:
    # Helps pick which option to pick in word synth
    dist_from_curr = (len(r_words) - i) / len(r_words)  # in (0, 1]
    similarity = word_sim(option, result_word)  # in [0, 1]
    return np.log(dist_from_curr + 1) * similarity  # words generated first are more important, score in [0, 1]


def relevance_score(
        option: str,
        r_words: list[str],
        anchors: list[str],
        transition_weight: float = 1,
        anchor_weight: float = 2,
        max_window: int = 15
    ) -> float:
    target_r_words = r_words[-max_window:]
    transition_term = sum([transition_score(i, r_words, option, r) for i, r in enumerate(target_r_words)]) / len(target_r_words)
    anchor_term = sum([word_sim(option, anchor) for anchor in anchors]) / (len(anchors) or 1)
    return transition_weight * transition_term + anchor_weight * anchor_term

class GramNotFoundError(Exception): pass

def synth_text(
        query: str,
        model: dict,
        n: int,
        target_length: int,
        anchors: list[str],
        relevance_weight: float = 1,
        liklihood_weight: float = 1,
        n_samples: int = 20
    ) -> list[str]:
    # higher n_samples -> higher probability of semantic matching
    log = """"""
    if n < 2: raise ValueError('n must be at least 2')
    result_tokens = word_tokenize(query)

    log += f"Tokenized Query: {result_tokens} \n\n"
    #sent_len = 0  # could use this to prevent run ons
    while len(result_tokens) < target_length or (result_tokens[-1] not in term_chars - {';', ':'} and len(result_tokens) < 1.5 * target_length):
        log += "----------------------\n"
        log += f"Result: {' '.join(result_tokens)} \n\n" 
        # while (we haven't generated enough words) OR (we have generated enough words but we have not completed current sentence yet AND that sentence isn't too long)
        try: dist = model[' '.join(result_tokens[-n:])]  # can rarely sometimes raise error... for some reason? TODO
        except KeyError: raise GramNotFoundError()
        options = list({dist.sample() for _ in range(n_samples)})
        r_words = extract_meaningful(list(map(lemmatize, result_tokens)))  # meaningful words generated so far
        o_words = [lemmatize(o) for o in options]  # candidates for next word

        if r_words and len(o_words) > 1:  # we have some results and have multiple options
            # for each option, lets compute its relevance score across all the meaningful words generated so far
            relevance_scores = [relevance_score(o, r_words, anchors) for o in o_words]

            options_log = list(zip(options, relevance_scores))
            options_log.sort(key = lambda x: x[1], reverse = True)
            log += "Options: "
            for i, x in enumerate(options_log): 
                log += f"{x[0]}, {round(x[1], 3)}"
                if i != 0 and i % 5 == 0: log += "\n"
                elif i != len(options_log) - 1: log += " | "
            log += "\n\n"

            #log += "Options :" + " | ".join(options_log) + "\n"
            # randomly pick option, based on both original liklihood and on relevance score
            weights = [relevance_weight * score + liklihood_weight * dist.get_weight(option) for option, score in zip(options, relevance_scores)]

            options_log = list(zip(options, weights))
            options_log.sort(key = lambda x: x[1], reverse = True)
            log += "Weighted options: "
            for i, x in enumerate(options_log): 
                log += f"{x[0]}, {round(x[1], 3)}"
                if i != 0 and i % 5 == 0: log += ",\n"
                elif i != len(options_log) - 1: log += " | "
            log += "\n\n"


            new_word = random.choices(options, weights, k = 1).pop()
            log += f"Chosen word: {new_word}\n\n"
        else: 
            new_word = options.pop()
            log += f"Only one option: {new_word}\n\n"
        #if new_word in term_chars: sent_len = 0  # reset current sentence length
        #else: sent_len += 1
        result_tokens.append(new_word)
    
    with open('log.txt', 'w') as f: f.write(log)
        
    return result_tokens


def make_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(description = 'talk with The Frog')
    argparser.add_argument('--nval', '-n', type = int, default = 3, choices = {2, 3, 4, 5}, help = 'n value for ngram model')
    argparser.add_argument('--resp_size', '-r', default = 150, help = 'roughly number of terms per repsonse')
    argparser.add_argument('--samples', '-s', default = 1, help = 'number of responses to generate before ranking')
    argparser.add_argument('--talk', '-t', action = 'store_const', const=True, help = 'perform basic text to speech on result')
    return argparser

def main():
    argparser = make_argparser()
    args = argparser.parse_args()
    n: int = args.nval
    resp_size: int = args.resp_size
    n_samples: int = args.samples
    speak: bool = bool(args.talk)

    print('Loading objects...')
    ngram: dict[str, 'core.ngram.FreqDistribution'] = load_object(osp.join(MODEL_PATH, f'{num2words(n)}_gram.gz'))
    
    print('Collecting starters...')
    starters = [gram for gram in ngram.keys() if not any(gram.startswith(char) for char in special_chars)]
    
    print('Compiling filters...')
    simple_subs = [
        ('"', ''), (' \'', ' ')  # no double quotes, no floating single quotes
    # below fixes spacing for special chars (including punctuation)
    ] + [(' ' + char, char) for char in special_chars_rhs] + [(char + ' ', char) for char in special_chars_lhs]
    
    re_subs = [
        (re.compile(r'([.!?]) +(\w)'), lambda m: m.groups()[0] + ' ' + m.groups()[1].upper()),  # capitalize start of sentence
        (re.compile(r'([.!?])( +\d+)+[.]'), lambda m: m.groups()[0]),  # remove any sentences that only contain numbers.
        (re.compile(r'^ *(.)'), lambda m: m.groups()[0].upper()),  # capitalize very first sentence
        (re.compile(r' {2,}'), ' '),  # there should only be one consecutive space
        (re.compile(r' [.!?] '), ' '),  # floating punctuation should be removed
    ]

    print('Ready!\n')
    for i in irange(start = 1):
        if (query := input('>> ').lower()) in {'exit', 'exit()', 'quit', 'quit()'}: break
        with Timer() as t:
            anchors = extract_meaningful(list(map(lemmatize, word_tokenize(query))))
            ic(anchors)

            candidates = []
            for _ in tqdm(range(n_samples), desc='musing over ideas'):
                fails = 0
                while fails < 3:
                    try: synthesized_tokens = synth_text(random.choice(starters), ngram, n, resp_size, anchors)
                    except GramNotFoundError: fails += 1
                    else: break
                synth_anchors = extract_meaningful(synthesized_tokens)
                candidates.append((synthesized_tokens, score_result(anchors, synth_anchors)))

            # ranking
            print('Ranking candidates...')
            final_synth_tokens = max(candidates, key = lambda pair: (lambda _, score: score)(*pair))[0]
            final_synth = ' '.join(final_synth_tokens)

            # final filtering
            print('Filtering response...')
            for text, sub in simple_subs: final_synth = final_synth.replace(text, sub)
            for ptn, sub in re_subs: final_synth = re.sub(ptn, sub, final_synth)
            if final_synth[-1] not in term_chars: final_synth += '...'
        print('\nResponse:')
        print(final_synth)
        ic(i, float(t))
        print()
        if speak:
            speech_engine.say(final_synth)
            speech_engine.runAndWait()
    print('Deallocating...')

if __name__ == '__main__':
    print('Loaded definitions.')
    main()