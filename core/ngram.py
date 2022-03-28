"""

"""

from collections import defaultdict
from typing import Any
from math import log
import random
from icecream import ic


class FreqDistribution:
    def __init__(self, choices: list[Any], smoothing: float = 2.5) -> None:
        assert smoothing > 1  # higher smoothing makes less common words more likely
        self.elements, raw_weights = list(zip(*[(choice, log(choices.count(choice) + 1, smoothing)) for choice in set(choices)]))
        self.weights = [weight / max(raw_weights) for weight in raw_weights]  # now in (0, 1]


    def sample(self) -> Any:
        # test w generator
        return random.choices(self.elements, self.weights, k = 1).pop()

    def get_weight(self, element: Any) -> float:
        # would be fast to create instance hash map from element to weight, but this way has lower memory footprint
        return self.weights[self.elements.index(element)]


def build_ngram(text: list[str], n: int) -> dict[str, FreqDistribution]:
    ngram = defaultdict(list)
    for i in range(len(text) - n): ngram[' '.join(text[i:i + n])].append(text[i + n]) 
    return {key: FreqDistribution(choices) for key, choices in ngram.items()}