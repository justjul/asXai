# Dockerfile
# FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:12.6.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/asXai/code:/app/asXai/code/src \
    PATH="/opt/venv/bin:$PATH"

# System setup, Python install, venv creation, pip upgrade, and requirements installation
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    curl wget unzip build-essential \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
WORKDIR /app/asXai/code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENTRYPOINT ["python", "src/services/database/database.py"]