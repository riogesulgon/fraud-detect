install:
	python -m pip install -e '.[dev]'
data:
	python scripts/generate_data.py
train:
	python scripts/train.py
test:
	ruff check app src tests scripts && pytest
run:
	uvicorn app.main:app --reload
docker-run:
	docker build -t mlops-risk-platform . && docker run --rm -p 8000:8000 mlops-risk-platform
