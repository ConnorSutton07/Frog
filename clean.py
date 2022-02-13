"""

"""

import os.path as osp, os
from tqdm import tqdm
import re
from icecream import ic


BASE_PATH = osp.dirname(osp.realpath(__file__))

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)

url_pattern = re.compile(r'https?[:][/][/]\S+')
bad_char_pattern = re.compile(r'[]\\<>;%&@~`=*[]')

def de_emojify(text):
    return emoji_pattern.sub('',text)

def de_url(text):
    return url_pattern.sub('', text)

def de_bad_char(text):
    return bad_char_pattern.sub('', text)

def clean_line(line: str) -> str:
    result = line.strip().lower()
    result = result.replace('[removed]', '')
    result = result.replace('[deleted]', '')
    result = result.replace('&#x200b', '')
    result = result.replace('edit:', '')
    result = result.replace('tldr:', '')
    result = result.replace('…', '')
    result = result.replace('•', '')
    result = result.replace('()', '')
    result = result.replace('...', '.')
    result = de_emojify(result)
    result = de_url(result)
    result = de_bad_char(result)
    if set(result.strip()) < ({'-', '_', '=', ' ', '.'} | set('1234567890')):
        return ''
    if result: result += '\n'
    return result


def main():
    doc_path = osp.join(BASE_PATH, 'Documents')
    target_path = osp.join(doc_path, 'master_scroll.txt')
    out_path = osp.join(doc_path, 'master_scroll_clean.txt')

    with open(target_path, 'r') as f:
        line_count = sum(1 for line in f)

    with open(target_path, 'r') as in_f, open(out_path, 'w') as out_f:
        for line in tqdm(in_f, total = line_count):
            out_f.write(clean_line(line))







if __name__ == '__main__':
    main()