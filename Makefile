PY := .venv/bin/python

.PHONY: data backtest predict simulate update test

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
	.venv/bin/pytest -q
