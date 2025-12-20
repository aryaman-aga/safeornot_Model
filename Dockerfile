# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the specific ONNX model and tokenizer to a clean directory
RUN python -c "import os; \
    from transformers import AutoTokenizer; \
    from huggingface_hub import hf_hub_download; \
    import shutil; \
    repo = 'aryaman1222/safeornot-safety-model-onnx'; \
    dest = '/app/model'; \
    os.makedirs(dest, exist_ok=True); \
    print(f'Downloading tokenizer from {repo}...'); \
    tokenizer = AutoTokenizer.from_pretrained(repo); \
    tokenizer.save_pretrained(dest); \
    print(f'Downloading model.onnx from {repo}...'); \
    model_path = hf_hub_download(repo_id=repo, filename='model.onnx'); \
    shutil.copy(model_path, os.path.join(dest, 'model.onnx'));"

# Stage 2: Final slim image
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user for security
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only the prepared model directory
COPY --from=builder --chown=appuser:appuser /app/model /app/model

# Copy application code
COPY --chown=appuser:appuser server.py .

# Set environment variables
ENV MODEL_PATH=/app/model
ENV HF_HOME=/tmp/hf_cache
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]