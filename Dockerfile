# G1 Alignment Eval Runner
# Runs MuJoCo simulations headless with Inspect AI evaluation

FROM python:3.11-slim

# Install system dependencies for MuJoCo offscreen rendering (OSMesa)
RUN apt-get update && apt-get install -y \
    libosmesa6 \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files needed for pip install
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies (includes inspect-ai via [dev] extras)
# openai is needed by inspect_eval/kimi_provider.py but not in pyproject.toml
RUN pip install --no-cache-dir ".[dev]" openai imageio imageio-ffmpeg

# Copy remaining source code
COPY inspect_eval/ ./inspect_eval/
COPY scripts/ ./scripts/
COPY unitree_rl_gym/ ./unitree_rl_gym/
COPY gcp/worker/trajectory_recorder.py ./gcp/worker/trajectory_recorder.py
COPY run_inspect_visual.py ./

# Set environment for headless MuJoCo (OSMesa = software rendering, no GPU needed)
ENV G1_HEADLESS=true
ENV MUJOCO_GL=osmesa
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "run_inspect_visual.py"]
