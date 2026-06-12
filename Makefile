PY := .venv/bin/python

.PHONY: data backtest predict simulate test

simulate:
	$(PY) -m mundial.models.simulate

data:
	$(PY) -m mundial.ingest.results

backtest:
	$(PY) -m mundial.eval.backtest

predict:
	$(PY) scripts/predict_next.py

test:
	.venv/bin/pytest -q
