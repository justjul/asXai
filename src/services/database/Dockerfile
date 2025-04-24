# Dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    curl \
    wget \
    unzip \
    chromium-browser \
    chromium-chromedriver \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default (3.11 doesn't seem to work with Dockerized Ubuntu 22.04)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Create and activate virtualenv
# -----------------------------
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pip (inside virtualenv)
RUN pip install --upgrade pip

# Create app directory
WORKDIR /app/asXai/code

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/asXai/code:/app/asXai/code/src

ENTRYPOINT ["python", "src/services/database/database.py"]