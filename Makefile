.PHONY: models

all:
	python3 merge.py
	python3 clean.py
	python3 tokenizer.py
	python3 tagger.py 
	python3 ngram.py
models:
	python3 ngram.py 