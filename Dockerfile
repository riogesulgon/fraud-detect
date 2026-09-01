FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir . 'scikit-learn==1.8.0'
# Train the deterministic v1 model from the committed fixture data so the
# image is self-sufficient even when gitignored artifacts are absent.
RUN python scripts/train.py
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
