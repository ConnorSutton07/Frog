"""

"""

from tqdm import tqdm
from collections import defaultdict
from math import log
import random


class Distribution:
    def __init__(self, choices: list, smoothing: float = 2.5) -> None:
        # the higher the base, the more the smoothing
        assert smoothing > 1
        self.counts = {choice: log(choices.count(choice) + 1, smoothing) for choice in set(choices)}
        self.size = len(self.counts)

    def sample(self):
        return random.choices(list(self.counts.keys()), weights = list(self.counts.values()), k = 1)[0]


def build_ngram(text: list[str], n: int) -> dict:
    ngram = defaultdict(list)
    for i in range(len(text) - n):
        ngram[' '.join(text[i : i + n])].append(text[i + n])
    ngram = {key: Distribution(choices) for key, choices in ngram.items()}
    return ngram