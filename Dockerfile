# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Extract ONNX models if onnx.zip exists
RUN if [ -f "onnx.zip" ]; then \
        echo "Extracting onnx.zip..." && \
        unzip -o onnx.zip && \
        rm onnx.zip; \
    else \
        echo "Warning: onnx.zip not found. Make sure ONNX models are available in onnx/ directory."; \
    fi

# Create onnx directory if it doesn't exist
RUN mkdir -p onnx

# Verify required files exist
RUN python -c "\
import os; \
import sys; \
required_files = ['onnx/piper_medium_male.onnx', 'onnx/model.config.json', 'onnx/phonikud-1.0.onnx']; \
missing = [f for f in required_files if not os.path.exists(f)]; \
print('ERROR: Missing required files: ' + str(missing)) if missing else print('✅ All required model files found'); \
sys.exit(1) if missing else sys.exit(0)"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "\
import requests; \
import sys; \
try: \
    response = requests.get('http://localhost:8000/health', timeout=10); \
    sys.exit(0) if response.status_code == 200 and response.json().get('status') == 'healthy' else sys.exit(1) \
except: \
    sys.exit(1)"

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
