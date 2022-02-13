"""
Refactor this v messyyyyy
"""

import os.path as osp, os
import praw
from praw.models import MoreComments
import json
from time import sleep
from tqdm import tqdm
from icecream import ic


BASE_PATH = osp.dirname(osp.realpath(__file__))


def main():
    credentials_path = osp.join(BASE_PATH, 'credentials.json')
    output_path = osp.join(BASE_PATH, 'Documents', 'dump_reddit.txt')

    with open(credentials_path, 'r') as f: creds = json.load(f)

    reddit = praw.Reddit(client_id = creds['client'], client_secret = creds['secret'], user_agent = creds['user_agent'])

    random = False
    if random:
        res = ''
        while res.lower().strip() not in {'y', 'yes'}:
            sub = reddit.random_subreddit()
            print(sub)
            res = input('yes? ')
    else:
        sub = reddit.subreddit('FanTheories')
        
    limit = 1000
    comment_limit = 10
    skip = 2
    post = True
    comments = False
    posts = sub.top(limit=limit)

    c = 0
    for post in tqdm(posts, total = limit):
        c += 1
        if c < skip: continue
        with open(output_path, 'a') as f:
            if comments:
                buffer = ''
                cc = 0
                for top_level_comment in post.comments:
                    if cc > comment_limit:
                        break
                    if isinstance(top_level_comment, MoreComments):
                        continue
                    buffer += top_level_comment.body + '\n'
                    cc += 1
                f.write(buffer)
            if post:
                f.write(post.selftext)
        sleep(0.05)


if __name__ == '__main__':
    main()