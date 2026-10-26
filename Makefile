PYTHON ?= python3
DATASET ?= chicago
RECSYS := PYTHONPATH=src $(PYTHON) -m restaurant_recs

.PHONY: install download data train report bench serve demo train-feedback test all docker diagram clean

install:  ## editable install with training + test extras
	$(PYTHON) -m pip install -e ".[dev]"

download: ## Google Local (2021) Illinois files -> data/raw/google_local/
	$(RECSYS) download --dataset chicago

data:     ## raw data -> validated interactions in data/processed/$(DATASET)
	$(RECSYS) prepare --dataset $(DATASET)

train:    ## train every retriever/ranker, select on validation, score test once -> artifacts/$(DATASET) (ARGS=--quick for a smoke run)
	$(RECSYS) train --dataset $(DATASET) $(ARGS)

report:   ## print the offline metrics table
	$(RECSYS) report --dataset $(DATASET)

bench:    ## p50/p95/p99 latency per served model, in-process and over HTTP
	$(RECSYS) bench --dataset $(DATASET)

serve:    ## API + web UI on http://127.0.0.1:8766 (add ARGS=--llm-rerank for the LLM arm)
	$(RECSYS) serve --dataset $(DATASET) $(ARGS)

demo:     ## prepare (if needed) + train + serve
	$(RECSYS) demo --dataset $(DATASET)

train-feedback: ## position-debiased multi-task ranker from the impression log (needs real traffic)
	$(RECSYS) train-feedback --dataset $(DATASET)

test:
	$(PYTHON) -m pytest -q

all: data train test

docker:
	docker build -t restaurant-recs .

diagram:  ## redraw docs/images/architecture.svg and the copy inlined in the web UI
	$(PYTHON) docs/architecture_diagram.py

clean:
	rm -rf artifacts data/processed data/logs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
