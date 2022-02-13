"""

"""

import os.path as osp, os
from tqdm import tqdm
from pprint import pprint
from icecream import ic


BASE_PATH = osp.dirname(osp.realpath(__file__))


blocked = {
    'master_scroll.txt',
    'master_scroll_clean.txt',
    'master_scroll_tokenized.txt',
    'vocab_out.txt',
    'nsfw_stories_gone_wild_reddit.txt',
    'copy_pasta_reddit.txt'
}

def main():
    doc_path = osp.join(BASE_PATH, 'Documents')
    ouput_path = osp.join(doc_path, 'master_scroll.txt')
    src_files = [osp.join(doc_path, src) for src in os.listdir(doc_path) if src.endswith('.txt') and osp.basename(src) not in blocked]
    with open(ouput_path, 'w') as out_f:
        for src in tqdm(src_files):
            with open(src, 'r') as in_f:
                out_f.write(in_f.read() + '\n')


if __name__ == '__main__':
    main()