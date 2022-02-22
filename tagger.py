from typing import List, Tuple
import os.path as osp, os
import nltk
import pickle 
import json
import re
from tqdm import tqdm
import itertools

BASE_PATH = osp.dirname(osp.realpath(__file__))

def preprocess(text: str ):
    print("Preprocessing...")
    text = text.lower()
    text = re.sub(r'[\n ]+', ' ', text)
    return re.split(r"([.!?])", text)

def tag(sentences: List[str]):
    print("Tagging sentences...")
    text = []
    for sentence in tqdm(sentences):
        text.append(nltk.pos_tag(sentence.split(' ')))
    return list(itertools.chain(*text))

def build_pos_vocab(text: List[Tuple]):
    print ("Building POS vocab...")
    vocab = {}
    for word, pos in tqdm(text):
        if pos not in vocab: vocab[pos] = [word]
        elif word not in vocab[pos]: vocab[pos].append(word)
    return vocab

def main():
    doc_path = osp.join(BASE_PATH, 'Documents')
    target_path = osp.join(doc_path, 'master_scroll_tokenized.txt')
    tag_out_path = osp.join(doc_path, 'master_scroll_tagged')
    vocab_out_path = osp.join(doc_path, 'master_scroll_pos_vocab.json')
    with open(target_path) as infile:
        text = infile.read()

    sentences = preprocess(text)
    tagged_text = tag(sentences)
    #print(tagged_text[:100])
    vocab = build_pos_vocab(tagged_text)

    with open(tag_out_path, 'wb') as outfile:
        pickle.dump(tagged_text, outfile)

    with open(vocab_out_path, 'w') as outfile:
        json.dump(vocab, outfile)
    

if __name__ == "__main__":
    main()