.PHONY: install test lint type-check security train serve docker-build quality drift-check drift-scheduler llm-judge llm-judge-live

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src --cov-fail-under=60 -v

lint:
	ruff check src monitoring tests

type-check:
	mypy src monitoring --explicit-package-bases

security:
	bandit -r src/ -c pyproject.toml

train:
	python -u src/train_model.py

serve:
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t btc-predictor:latest .

quality: lint type-check security test

drift-check:
	curl -X POST http://127.0.0.1:8000/admin/check-drift -H "Content-Type: application/json" -d "{\"ticker\":\"BTC-USD\"}"

drift-scheduler:
	python -u scripts/run_drift_scheduler.py

llm-judge:
	python -u evaluation/llm_judge.py --golden-set data/golden_set/btc_rag_golden_set.json --min-questions 21 --output evaluation/llm_judge_results.json

llm-judge-live:
	python -u evaluation/llm_judge.py --golden-set data/golden_set/btc_rag_golden_set.json --min-questions 21 --api-url http://127.0.0.1:8000 --output evaluation/llm_judge_results.json
