.PHONY: install test lint format run baseline-phl baseline-gan clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

install:
	$(PIP) install -e .[dev]

test:
	$(PYTHON) -m unittest discover -s tests

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

run:
	$(PYTHON) Main/main.py

baseline-phl:
	$(PYTHON) Baselines/phl_segmentation/run_phl.py

baseline-gan:
	$(PYTHON) Baselines/gan_reconstruction/run_gan.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*$py.class" -delete
	rm -rf *.egg-info .eggs build dist .pytest_cache
