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

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
# - Build tools: needed for compiling Python packages
# - pkg-config: required for building PyAV (av package)
# - ffmpeg: required for PyAV (av package) - need version 7
# - libusb-1.0-0: required for RealSense camera access
# - libgl1, libglib2.0-0: OpenCV dependencies
# - libxext6, libxrender1, libsm6: X11 display support for OpenCV imshow
# - libgtk-3-0, libgdk-pixbuf2.0-0: GTK+ runtime libraries for OpenCV GUI (cv2.imshow)
# - libxkbcommon-x11-0 + libxcb*: Qt/X11 runtime deps (opencv-python wheels often use Qt HighGUI)
# - libqt5gui5 ships Qt5 platform plugins on Ubuntu 22.04 (incl. platforms/libqxcb.so)
RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang \
    curl wget ca-certificates xz-utils pkg-config \
    nasm yasm autoconf automake libtool \
    libgl1 libglib2.0-0 libusb-1.0-0 \
    libxext6 libxrender1 libsm6 \
    libgtk-3-0 libgdk-pixbuf2.0-0 \
    libxkbcommon-x11-0 \
    libxcb1 libxcb-render0 libxcb-shm0 libxcb-randr0 libxcb-xfixes0 \
    libxcb-keysyms1 libxcb-icccm4 libxcb-image0 libxcb-util1 \
    libxcb-xinerama0 libxcb-cursor0 \
    libqt5gui5 libqt5widgets5 \
 && rm -rf /var/lib/apt/lists/*

# Build and install FFmpeg 7 from source (PyAV requires FFmpeg 7)
ARG FFMPEG_VERSION="7.1.3"
RUN mkdir -p /tmp/ffmpeg-src && \
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o /tmp/ffmpeg.tar.xz && \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg-src --strip-components=1 && \
    cd /tmp/ffmpeg-src && \
    ./configure \
      --prefix=/usr/local \
      --enable-shared \
      --disable-static \
      --disable-doc \
      --disable-debug \
      --disable-programs \
      --enable-pic && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf /tmp/ffmpeg-src /tmp/ffmpeg.tar.xz

# Ensure pkg-config can find FFmpeg's .pc files
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig

# Configure uv to use copy mode (for bind mounts) and place venv outside /app
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Create Python 3.11 virtual environment (OpenPI requires Python 3.11)
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Install Cython before uv sync (required for building av package)
RUN /.venv/bin/python -m ensurepip --upgrade && \
    /.venv/bin/python -m pip install --no-cache-dir Cython

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
# - opencv-python: OpenCV with GUI support (installed in venv, not system)
# - ur-rtde: UR robot RTDE interface (rtde_receive, rtde_control)
# - numpy: Numerical operations (installed in venv via uv sync, but ensure it's available)
# - polars: Data processing
# Note: We install opencv-python in venv instead of system python3-opencv
# to avoid conflicts with system numpy (Python 3.10) vs venv numpy (Python 3.11)

# Install pip tools first
RUN /.venv/bin/python -m ensurepip --upgrade && \
    /.venv/bin/python -m pip install --no-cache-dir -U pip setuptools wheel

# Install packages separately for better debugging and caching
# Install numpy first (dependency for others) - ensure it's installed even if uv sync missed it
RUN /.venv/bin/python -m pip install --no-cache-dir "numpy<2.0.0" && \
    /.venv/bin/python -c "import numpy; print(f'✓ numpy {numpy.__version__} installed')"

# Install opencv-python (GUI build).
# Note: our lockfile also includes opencv-python-headless; if both are installed, whichever was installed last
# typically "wins" the `cv2` module. We explicitly remove headless variants first so `cv2.imshow` works.
# IMPORTANT: pin opencv-python to a NumPy<2 compatible build. Newer opencv-python releases pull NumPy>=2,
# which conflicts with openpi-client's NumPy<2 requirement.
#
# Also: ensure a clean OpenCV install (uv sync may bring in opencv-python-headless via other deps).
RUN /.venv/bin/python -m pip uninstall -y \
      opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless \
    || true
RUN /.venv/bin/python -m pip install --no-cache-dir "numpy<2.0.0" "opencv-python==4.11.0.86" && \
    /.venv/bin/python - <<'PY'
import numpy
import cv2

print("✓ numpy", numpy.__version__)
print("✓ cv2 module", getattr(cv2, "__file__", None))

ver = getattr(cv2, "__version__", None)
if ver is None and hasattr(cv2, "getVersionString"):
    try:
        ver = cv2.getVersionString()
    except Exception:
        ver = None

print("✓ opencv version", ver)
print("✓ has imshow", hasattr(cv2, "imshow"))

if hasattr(cv2, "getBuildInformation"):
    bi = cv2.getBuildInformation()
    lines = [l for l in bi.split("\n") if "GTK:" in l or "QT:" in l]
    print(lines)
else:
    print("NOTE: cv2.getBuildInformation not available in this build")
PY

# Install ur-rtde (usually fast)
RUN /.venv/bin/python -m pip install --no-cache-dir ur-rtde && \
    echo "✓ ur-rtde installed"

# Install polars (usually fast)
RUN /.venv/bin/python -m pip install --no-cache-dir polars && \
    echo "✓ polars installed"

# Install pyrealsense2 last (this is often the slowest, may compile from source)
RUN echo "Installing pyrealsense2 (this may take several minutes)..." && \
    /.venv/bin/python -m pip install --no-cache-dir --verbose pyrealsense2 && \
    echo "✓ pyrealsense2 installed"

# Do NOT add system Python site-packages to PYTHONPATH
# This was causing conflicts between system numpy (Python 3.10) and venv numpy (Python 3.11)
# All packages are now installed in the venv, so PYTHONPATH should only include venv paths

ENV PATH=/.venv/bin:$PATH

# Safer defaults for OpenCV HighGUI inside Docker/X11:
# - disable MIT-SHM (often causes "X11 connection broke" in containers)
# - force XCB backend (more reliable than Wayland/other backends inside containers)
ENV QT_X11_NO_MITSHM=1
ENV QT_QPA_PLATFORM=xcb
# IMPORTANT: the `opencv-python` wheel bundles its own Qt plugins under site-packages/cv2/qt/plugins.
# Forcing Qt to use system plugin directories can cause "xcb found but could not load" due to ABI mismatches.
ENV QT_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins/platforms

# Checkpoint is provided via bind mount from the host (checkpoints/ directory)
# No need to download at build time since the workspace is mounted at runtime

# ========== Default Runtime Configuration ==========
# These can be overridden via -e flags when running the container

# Policy server arguments (see scripts/serve_policy.py for options)
# Default to UR5 environment mode for robot control with local checkpoint
ENV SERVER_ARGS="--env=UR5 policy:checkpoint --policy.config=pi0_ur5 --policy.dir=checkpoints/pi0_ur5/ur5_fourth_1/499"

# Seconds to wait for policy server to start before launching robot bridge
ENV SERVER_WAIT=6

# Robot configuration
ENV UR_IP="192.10.0.11"
ENV RS_BASE=""
ENV RS_WRIST=""

# Language instruction for the policy
ENV PROMPT="pick up the grey shaker bottle"

# Robot control defaults
ENV DRY_RUN="0"

# Start policy server in background, wait for it to initialize, then run robot bridge
# The robot bridge connects to the policy server via websocket at localhost:8000
CMD ["/bin/bash","-lc","uv run scripts/serve_policy.py $SERVER_ARGS & sleep ${SERVER_WAIT:-5}; python /app/local/utils/pi0_bridge_ur5_headless.py"]
