.PHONY: install test lint type-check security train serve docker-build quality

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
