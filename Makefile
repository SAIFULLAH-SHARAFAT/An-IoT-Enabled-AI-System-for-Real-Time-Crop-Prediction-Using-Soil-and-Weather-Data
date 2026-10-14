PYTHON ?= python
EXTRAS ?= dev,explain,edge

.PHONY: install install-deep lint format typecheck test check reproduce docker clean

install:
	$(PYTHON) -m pip install -e ".[$(EXTRAS)]"

install-deep:
	$(PYTHON) -m pip install -e ".[$(EXTRAS),deep]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

reproduce:
	crop-rec reproduce

docker:
	docker build -t crop-rec .

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage models reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
