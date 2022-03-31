"""

"""
from curses import delay_output
from nltk.tokenize import sent_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stops
from functools import lru_cache
from nltk import pos_tag_sents, pos_tag
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from math import floor
from psutil import cpu_count
from tqdm import tqdm

LEMMATIZER = WordNetLemmatizer()

@lru_cache(maxsize=50000)
def lemmatize(target: str) -> str:
    return LEMMATIZER.lemmatize(target)
lemmatize('')  # loads model when running for first time

stopwords = set(nltk_stops.words('english'))

WORD_TOKENIZER = TweetTokenizer(
    preserve_case = True,
    reduce_len = True,
    strip_handles = False,
    match_phone_numbers = False,
)

def word_tokenize(text: str) -> list[str]:
    return WORD_TOKENIZER.tokenize(text)


class EmbeddingProgress(CallbackAny2Vec):
    def __init__(self, max_epochs) -> None:
        super().__init__()
        self.epoch = 0
        self.pbar = tqdm(total = max_epochs, desc = 'generating embeddings')

    def on_epoch_end(self, _) -> None:
        self.pbar.update()
        self.epoch += 1
    
    def on_train_end(self, _) -> None:
        del self.pbar


def embeddings(sent_tokens: list[list[str]]) -> FastText:
    epochs = 10
    return FastText(
        sent_tokens,
        sample = 0.001,  # default
        vector_size = 100,  # default
        window = 5,  # default
        epochs = epochs,
        min_count = 5,  # default
        workers = floor(0.75 * cpu_count()),
        sg = 1,
        callbacks = (EmbeddingProgress(epochs),)
    )