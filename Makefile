all:
	python3 merge.py
	python3 clean.py
	python3 tokenizer.py
	python3 tagger.py