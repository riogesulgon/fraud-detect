install:
	python -m pip install -e '.[dev]'
data:
	python scripts/generate_data.py
train:
	python scripts/train.py
test:
	ruff check . && pytest --cov=app --cov=src --cov-report=term-missing --cov-fail-under=80
run:
	uvicorn app.main:app --reload

drift:
	python scripts/drift.py
