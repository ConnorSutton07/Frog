import random 
from math import log

class Distribution:
    def __init__(self, choices: list, smoothing: float = 2) -> None:
        # the higher the base, the more the smoothing
        assert smoothing > 1
        self.counts = {choice: log(choices.count(choice) + 1, smoothing) for choice in set(choices)}
        self.size = len(self.counts)

    def sample(self):
        return random.choices(list(self.counts.keys()), weights = list(self.counts.values()), k = 1)[0]