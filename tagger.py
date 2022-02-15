import os.path as osp, os
import nltk
import pickle 
import re

BASE_PATH = osp.dirname(osp.realpath(__file__))

def preprocess(text: str ):
    text = text.lower()
    text = re.sub(r'[\n ]+', ' ', text)
    #text = re.sub(r'[^A-Za-z.;:!?,\"\']', '', text)
    text = text.split(' ')
    return text

def main():
    doc_path = osp.join(BASE_PATH, 'Documents')
    target_path = osp.join(doc_path, 'master_scroll_tokenized.txt')
    out_path = osp.join(doc_path, 'master_scroll_tagged')
    with open(target_path) as infile:
        text = infile.read()
    print("Preprocessing...")
    text = preprocess(text)
    print("Tagging...")
    text = nltk.pos_tag(text)
    ("Writing...")
    with open(out_path, 'wb') as outfile:
        pickle.dump(text, outfile)

if __name__ == "__main__":
    main()