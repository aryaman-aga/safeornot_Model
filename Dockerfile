# Stage 1: Download model
FROM python:3.10-slim AS builder

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Download model in builder stage
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    AutoModelForSequenceClassification.from_pretrained('aryaman1222/safeornot-safety-model'); \
    AutoTokenizer.from_pretrained('aryaman1222/safeornot-safety-model')"

# Stage 2: Final slim image
FROM python:3.10-slim

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cached model from builder
COPY --from=builder /app/.cache /app/.cache

# Copy application code
COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]