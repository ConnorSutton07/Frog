
import numpy as np
import random
from icecream import ic
from typing import List
import os
from tqdm import tqdm
import nltk
import pickle
from core.distribution import Distribution

# def save_models(models: dict):
#     for n in tqdm(list(models.keys())):
#         with open(os.path.join("Models", f"{n}.pickle"), 'wb') as handle:
#             pickle.dump(models[n], handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_model(model: dict, title: str):
    with open(os.path.join("Models", f"{title}.pickle"), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def ngram(n: int, text: List[str], message: str = ""):
    print(message)
    ngrams = {}
    for i in tqdm(range(len(text) - n)):
        seq = " ".join(text[i : i + n])
        if seq not in ngrams: ngrams[seq] = [text[i + n]]
        else: ngrams[seq].append( text[i + n] )
        
    for key, choices in ngrams.items():
        ngrams[key] = Distribution(choices)
        choices.clear()
    return ngrams

if __name__ == "__main__":
    with open(os.path.join("Documents", "master_scroll_tagged"), 'rb') as infile:
        data = pickle.load(infile)
    text = [x[0] for x in data]
    pos  = [x[1] for x in data]

    n = 3
    pos_model = ngram(n, pos, f"Creating {n}-gram POS model...")
    language_model = ngram(n, text, message = f"Creating {n}-gram language model...")
    print("Saving models...")
    save_model(pos_model, "pos_model")
    save_model(language_model, "trigram_model")