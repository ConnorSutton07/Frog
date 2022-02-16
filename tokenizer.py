"""

"""

import os.path as osp, os
from tqdm import tqdm
import re
from icecream import ic


BASE_PATH = osp.dirname(osp.realpath(__file__))
banned_p = osp.join(BASE_PATH, 'extra_banned.txt')
with open(banned_p, 'r') as f:
    banned = set(ln.rstrip() for ln in f.readlines())

space_pattern = re.compile(r' +')
number_pattern = re.compile(r'[-+]?\d*[.]\d+|\d+')
# chapter_pattern = re.compile(r'chapter [-+]?\d*\.\d+|\d+')


def clean_line(line: str) -> str:
    result = line.replace('\'', '')
    result = result.replace('"', '')
    result = result.replace('„', '')
    result = result.replace('‘', '')
    result = result.replace('’', '')
    result = result.replace('“', '')
    result = result.replace('/', ' ')
    result = result.replace('(', '')
    result = result.replace('·', ' ')
    result = result.replace('^', '')
    result = result.replace('|', ' ')
    result = result.replace('+', ' ')
    result = result.replace('”', '')
    result = result.replace('{', '')
    result = result.replace('}', '')
    result = result.replace('#', '')
    result = result.replace('®', ' ')
    result = result.replace('$k', 'money')
    result = result.replace('$', '')
    result = result.replace('_', ' ')
    result = result.replace('‬', '')
    result = result.replace(')', '')
    result = result.replace('.', ' . ')
    result = result.replace('-', ' ')
    result = result.replace('–', ' ')
    result = result.replace('\t', ' ')
    result = result.replace('—', ' ')
    result = result.replace(';', ' ; ')
    result = result.replace('\'', '')
    result = result.replace(',', ' , ')
    result = result.replace('!', ' ! ')
    result = result.replace('?', ' ? ')
    result = result.replace(':', ' : ')
    result = result.replace('subreddit', 'fiefdom')
    result = result.replace('upvoted', 'funded')
    result = result.replace('reddit', 'bazaar')
    result = result.replace('comments', 'messages')
    result = result.replace('downvote', 'diminish')
    result = result.replace(' mods ', ' enforcers ')
    result = result.replace(' mod ', ' enforcer ')
    result = result.replace('chapter', 'void')
    result = result.replace('spelling edit', ';')
    result = result.replace(' thread ', ' great hall ')
    result = result.replace(' link ', ' trail ')
    result = result.replace(' shitposting ', ' bamboozlement ')
    for b in banned:
        result = result.replace(b, '')
    # result = result.replace('\n', ' ')
    # result = chapter_pattern.sub('', result)
    result = number_pattern.sub('', result)
    result = space_pattern.sub(' ', result)
    return result.strip() + '\n'


def main():
    doc_path = osp.join(BASE_PATH, 'Documents')
    target_path = osp.join(doc_path, 'master_scroll_clean.txt')
    out_path = osp.join(doc_path, 'master_scroll_tokenized.txt')
    vocab_out_path = osp.join(doc_path, 'vocab_out.txt')

    with open(target_path, 'r') as f:
        line_count = sum(1 for line in f)

    vocab = set()
    with open(target_path, 'r') as in_f, open(out_path, 'w') as out_f:
        for line in tqdm(in_f, total = line_count):
            clean = clean_line(line)
            for token in clean.split():
                vocab.add(token)
            out_f.write(clean)

    with open(vocab_out_path, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')

    print(len(vocab))







if __name__ == '__main__':
    main()