# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set HF cache location BEFORE downloading
ENV HF_HOME=/app/.cache/huggingface

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
# --no-cache-dir reduces image size
# Note: If you don't need GPU support, ensure requirements.txt specifies 
# the CPU version of torch, or the image will be significantly larger.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Download model (now saved to /app/.cache/huggingface)
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    AutoModelForSequenceClassification.from_pretrained('aryaman1222/safeornot-safety-model'); \
    AutoTokenizer.from_pretrained('aryaman1222/safeornot-safety-model')"

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
# Host 0.0.0.0 is crucial for accessing the container from outside
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]