PY := .venv/bin/python

.PHONY: data backtest predict simulate update test lint format typecheck cov

simulate:
	$(PY) -m mundial.models.simulate

update:
	$(PY) -m mundial.online.update_cycle

data:
	$(PY) -m mundial.ingest.results

backtest:
	$(PY) -m mundial.eval.backtest

predict:
	$(PY) scripts/predict_next.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

typecheck:
	$(PY) -m mypy

cov:
	$(PY) -m pytest --cov=mundial --cov-report=term-missing -q
