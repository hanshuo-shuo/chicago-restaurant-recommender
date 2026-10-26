# Serving image. Inference is NumPy (+ LightGBM for the learned ranker); PyTorch is not installed here.
# Train on the host (make data train), then mount data/ and artifacts/:
#   docker build -t restaurant-recs .
#   docker run --rm -p 8766:8766 -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" restaurant-recs
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# LightGBM's wheels load OpenMP at import time; the slim image doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

EXPOSE 8766
CMD ["recsys", "serve", "--host", "0.0.0.0", "--port", "8766"]
