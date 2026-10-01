.PHONY: install test lint format demo flagship clean

install:
	python -m pip install -r requirements.txt

test:
	pytest -q -W ignore::UserWarning

lint:
	ruff check fairforge tests examples
	ruff format --check fairforge tests examples

format:
	ruff format fairforge tests examples

demo:
	python examples/run_demo.py

flagship:
	python examples/flagship_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info examples/out
