PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: install lint type test reproduce-mock

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest

# Phase A: the mock-backend reproduction is the simulation test of the
# selective-risk guarantee (no GPU, no network).
reproduce-mock:
	$(PY) -m pytest tests/test_ltt_simulation.py -v
	bash benchmarks/reproduce_mock.sh
