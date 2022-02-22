import os 
import re 
from tqdm import tqdm 
from nltk import pos_tag
import random 
import pickle
from core.distribution import Distribution
import json 

def load_models(): #n_values: List[int]) -> dict:
    # models = {}
    # for n in n_values:
    #     with open(os.path.join("Models", f"{n}.pickle"), 'rb') as handle:
    #         models[n] = pickle.load(handle)
    # return models
    with open(os.path.join("Models", "trigram_model.pickle"), 'rb') as handle:
        language_model = pickle.load(handle)
    with open(os.path.join("Models", "pos_model.pickle"), 'rb') as handle:
        pos_model = pickle.load(handle)
    return language_model, pos_model

# def sample(query: str, models: List[dict], n: int, length: int):
#     og_query = query
#     query: list = ((query.split(' '))[-n:])
#     result = query[:]
#     sentences = 0
#     length_reached = False
#     at_sentence = False
#     i = 0
#     while not length_reached or not at_sentence:
#         model = models[n]
#         at_sentence = False
#         key = " ".join(query)
#         if key not in model and n - 1:
#             return sample(og_query, models, n - 1, length)
#         new_word = query[-1]
#         cc = 0
#         while new_word == query[-1] and cc < 5:
#             new_word = model[key].sample()
#             cc += 1
#         del query[0]
#         query.append(new_word)
#         result.append(new_word)

#         if new_word in {'?', '!', '.'}:
#             sentences += 1
#             at_sentence = True
#         i += 1
#         if i == length:
#             length_reached = True
#     return " ".join(result), n

def sample(query: str, language_model, pos_model, vocab, n, length):
    og_query = query 
    query: list = query.split(' ')[-n:]
    result = query[:]
    sentences = 0 
    length_reached = False 
    at_sentence = False 
    i = 0

    while not length_reached or not at_sentence:
        at_sentence = False 
        key = " ".join(query)
        if not key in language_model: 
            pos_query = " ".join(pos_tag(query))
            if not pos_query in pos_model: return "fail"
            new_word = random.choice(vocab[pos_model[" ".join(result[-4:])]])
        else:
            new_word = query[-1]
            cc = 0
            while new_word == query[-1] and cc < 5:
                new_word = language_model[key].sample()
                cc += 1
        del query[0]
        query.append(new_word)
        result.append(new_word)

        if new_word in {'?', '!', '.'}:
            sentences += 1
            at_sentence = True
        i += 1
        if i == length:
            length_reached = True
    return " ".join(result)
            

if __name__ == "__main__":
    language_model, pos_model = load_models()
    with open(os.path.join("Documents", "master_scroll_pos_vocab.json"), 'r') as handle:
        pos_vocab = json.load(handle)
    n = 3
    resp_len = 75
    # THESE WORK: note that upon failing a value of n, it will atempt to decrease n for remainder of generation.
    # i guess he could have just
    # jocks bring the nectar to
    # the fat of the peace
    # i came for questions not to
    # this book is largely concerned with hobbits
    query = 'this book is largely concerned with hobbits'.lower()
    stop_pattern = re.compile(r'[.?!][.?!]+')
    comma_pattern = re.compile(r'[,][,]+')
    space_pattern = re.compile(r'  +')

    bad_comma_pattern = re.compile(r'[,](?=<[!.?]+>)')
    for i in range(1, 11):
        #gen, actual_n = sample(query, models, n, length = resp_len)
        gen = sample(query, language_model, pos_model, pos_vocab, n, length = resp_len)
        gen = gen.replace(' .', '.')
        gen = gen.replace(' ;', ';')
        gen = gen.replace(' :', ':')
        raw_resp = ' '.join(query.split()[:len(query.split()) - n]) + ' ' + gen
        resp = '. '.join(sent.strip().capitalize() for sent in raw_resp.split('. '))
        resp = resp.replace(' i ', ' I ')
        resp = stop_pattern.sub(random.choice(('.', '?', '!')), resp)
        resp = space_pattern.sub(' ', resp)
        resp = comma_pattern.sub(',', resp)
        resp = bad_comma_pattern.sub('', resp)
        print(f'{i}) "{resp}"')
        print()