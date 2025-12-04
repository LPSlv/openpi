# Dockerfile for running π₀ policy server with robot control in a single container.
#
# This container combines:
# 1. Policy server (serves the π₀ model via websocket)
# 2. RealSense camera access (for robot vision)
# 3. UR robot control (via RTDE for UR5/UR10 arms)
#
# The container runs both the policy server and the robot bridge script,
# making it a complete "all-in-one" solution for robot control.
#
# Usage: See launch_guide.md for detailed instructions

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/
WORKDIR /app

# Install system dependencies
# - Build tools: needed for compiling Python packages
# - libusb-1.0-0: required for RealSense camera access
# - libgl1, libglib2.0-0: OpenCV dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang \
    curl wget libgl1 libglib2.0-0 libusb-1.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Configure uv to use copy mode (for bind mounts) and place venv outside /app
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Create Python 3.11 virtual environment (OpenPI requires Python 3.11)
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Install project dependencies from lockfile
# This installs all openpi dependencies including the openpi-client package
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Apply transformers library patches for PyTorch support
# These patches enable AdaRMS, correct activation precision, and KV cache usage
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers, os; print(os.path.dirname(transformers.__file__))" | \
    xargs -I{} bash -lc 'cp -r /tmp/transformers_replace/* {}' && rm -rf /tmp/transformers_replace

# Install additional runtime dependencies for robot control
# - pyrealsense2: RealSense camera SDK
# - opencv-python: Image processing
# - ur-rtde: UR robot RTDE interface (rtde_receive, rtde_control)
# - numpy: Numerical operations
# - polars: Data processing
RUN /.venv/bin/python -m ensurepip --upgrade && \
    /.venv/bin/python -m pip install --no-cache-dir -U pip setuptools wheel && \
    /.venv/bin/python -m pip install --no-cache-dir \
        pyrealsense2 opencv-python ur-rtde numpy polars

ENV PATH=/.venv/bin:$PATH

# Pre-download checkpoint at build time to avoid runtime downloads
# This downloads the checkpoint specified in SERVER_ARGS (default: pi05_droid)
# and its dependencies (pi05_base) to /root/.cache/openpi/
# To download a different checkpoint, modify the CHECKPOINT_URL below
ARG CHECKPOINT_URL="gs://openpi-assets/checkpoints/pi05_droid"
ARG BASE_CHECKPOINT_URL="gs://openpi-assets/checkpoints/pi05_base"
COPY scripts/docker/download_checkpoints.py /tmp/download_checkpoints.py
RUN /.venv/bin/python /tmp/download_checkpoints.py "${BASE_CHECKPOINT_URL}" "${CHECKPOINT_URL}" && \
    rm /tmp/download_checkpoints.py

# ========== Default Runtime Configuration ==========
# These can be overridden via -e flags when running the container

# Policy server arguments (see scripts/serve_policy.py for options)
ENV SERVER_ARGS="policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid"

# Seconds to wait for policy server to start before launching robot bridge
ENV SERVER_WAIT=6

# Robot configuration
ENV UR_IP="192.168.1.116"
ENV RS_BASE=""
ENV RS_WRIST=""

# Language instruction for the policy
ENV PROMPT="pick up the grey shaker bottle"

# Robot control defaults
ENV DRY_RUN="0"

# Start policy server in background, wait for it to initialize, then run robot bridge
# The robot bridge connects to the policy server via websocket at localhost:8000
CMD ["/bin/bash","-lc","uv run scripts/serve_policy.py $SERVER_ARGS & sleep ${SERVER_WAIT:-5}; python /app/local/pi0_bridge_ur5_headless.py"]
