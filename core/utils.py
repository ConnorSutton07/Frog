"""

"""

import os
from typing import Any
import psutil
import time
import compress_pickle
from icecream import ic


def get_mem_usage(precision: int = 2) -> float:
    """Returns current process' memory usage in MB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1000000, precision)


def irange(start = 0):
    i = start
    while True:
        yield i
        i += 1


class Timer:
    """This context manager allows timing blocks of code."""
    def __init__(self):
        self._timer = None
        self._elapsed = None
    
    def __enter__(self) -> None:
        self._timer = time.time()
        return self

    def __exit__(self, *_: list) -> None:
        self._elapsed = time.time() - self._timer

    def __float__(self):
        return self._elapsed


def save_object(obj: Any, file_path: str):
    compress_pickle.dump(obj, file_path, 'gzip')

def load_object(file_path: str) -> Any:
    return compress_pickle.load(file_path)