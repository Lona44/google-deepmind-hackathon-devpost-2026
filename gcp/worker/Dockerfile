# G1 Experiment Worker
# Runs MuJoCo simulations with AI and saves trajectories

FROM python:3.11-slim

# Install system dependencies for MuJoCo
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY gcp/worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY gcp/worker/ ./

# Copy source code needed for simulation
COPY src/ ./src/
COPY unitree_rl_gym/ ./unitree_rl_gym/
COPY inspect_eval/ ./inspect_eval/

# Set Python path
ENV PYTHONPATH=/app

# Run worker
CMD ["python", "main.py"]
