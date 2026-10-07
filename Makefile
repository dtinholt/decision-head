.PHONY: dev test lint typecheck build run clean

VENV ?= .venv
PYTHON ?= python3

dev:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -e ".[dev]"

test:
	$(PYTHON) -m pytest -q --cov=decisionhead --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy decisionhead

build:
	$(PYTHON) -m build

run:
	$(PYTHON) -m decisionhead serve --dry --port 8787

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name "__pycache__" -type d -exec rm -rf {} +
