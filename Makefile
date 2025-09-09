.PHONY: install test run

install:
	pip install -r requirements.txt

test:
	pytest

run:
	python src/main_competitor.py
