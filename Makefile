PY := .venv/bin/python

.PHONY: data backtest predict test

data:
	$(PY) -m mundial.ingest.results

backtest:
	$(PY) -m mundial.eval.backtest

predict:
	$(PY) scripts/predict_next.py

test:
	.venv/bin/pytest -q
