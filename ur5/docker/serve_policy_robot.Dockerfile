# All-in-one image: pi0 policy server + RealSense capture + UR5 RTDE bridge.
# See ur5/docs/deployment.md for setup and runtime instructions.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/
WORKDIR /app

# silence interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# system deps:
#   build tools + pkg-config: compile PyAV (needs FFmpeg 7, built below)
#   libusb-1.0-0: RealSense
#   libgl1 / libglib2.0-0 / libxext6 / libxrender1 / libsm6: OpenCV
#   libgtk-3-0 / libgdk-pixbuf2.0-0: cv2.imshow GTK backend
#   libxkbcommon-x11-0 + libxcb*: Qt/X11 runtime for the opencv-python Qt plugin
#   libqt5gui5: Qt5 platform plugins (incl. platforms/libqxcb.so)
RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang \
    curl wget ca-certificates xz-utils pkg-config \
    nasm yasm autoconf automake libtool \
    libgfortran5 libgl1 libglib2.0-0 libusb-1.0-0 \
    libxext6 libxrender1 libsm6 \
    libgtk-3-0 libgdk-pixbuf2.0-0 \
    libxkbcommon-x11-0 \
    libxcb1 libxcb-render0 libxcb-shm0 libxcb-randr0 libxcb-xfixes0 \
    libxcb-keysyms1 libxcb-icccm4 libxcb-image0 libxcb-util1 \
    libxcb-xinerama0 libxcb-cursor0 \
    libqt5gui5 libqt5widgets5 \
 && rm -rf /var/lib/apt/lists/*

# PyAV requires FFmpeg 7, build it from source
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

# point pkg-config at the freshly-built FFmpeg
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig

# uv: use copy mode for bind mounts, put the venv outside /app so it survives them
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# openpi requires Python 3.11
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Cython is needed to build the av package during uv sync below
RUN /.venv/bin/python -m ensurepip --upgrade && \
    /.venv/bin/python -m pip install --no-cache-dir Cython

# install project dependencies (incl. openpi-client) from lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# patch the transformers library: enables AdaRMS, fixes activation precision,
# and turns on KV cache usage for the openpi pytorch path
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers, os; print(os.path.dirname(transformers.__file__))" | \
    xargs -I{} bash -lc 'cp -r /tmp/transformers_replace/* {}' && rm -rf /tmp/transformers_replace

# extra runtime deps: pyrealsense2 (camera SDK), opencv-python (GUI build),
# ur-rtde (UR control), polars. installed in the venv so they don't collide
# with the host's system numpy/python
RUN /.venv/bin/python -m ensurepip --upgrade && \
    /.venv/bin/python -m pip install --no-cache-dir -U pip setuptools wheel

# the lockfile also brings opencv-python-headless; whichever opencv variant is
# installed last wins the `cv2` module, so we wipe the headless ones before
# pinning the GUI build, otherwise cv2.imshow silently no-ops
RUN /.venv/bin/python -m pip uninstall -y \
      opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless \
    || true
RUN /.venv/bin/python -m pip install --no-cache-dir "opencv-python==4.11.0.86" && \
    /.venv/bin/python - <<'PY'
import numpy
import cv2

print("OK numpy", numpy.__version__)
print("OK cv2 module", getattr(cv2, "__file__", None))

ver = getattr(cv2, "__version__", None)
if ver is None and hasattr(cv2, "getVersionString"):
    try:
        ver = cv2.getVersionString()
    except Exception:
        ver = None

print("OK opencv version", ver)
print("OK has imshow", hasattr(cv2, "imshow"))

if hasattr(cv2, "getBuildInformation"):
    bi = cv2.getBuildInformation()
    lines = [l for l in bi.split("\n") if "GTK:" in l or "QT:" in l]
    print(lines)
else:
    print("NOTE: cv2.getBuildInformation not available in this build")
PY

RUN /.venv/bin/python -m pip install --no-cache-dir ur-rtde && \
    echo "ur-rtde installed"

RUN /.venv/bin/python -m pip install --no-cache-dir polars && \
    echo "polars installed"

# pyrealsense2 last because it's the slowest and sometimes compiles from source
RUN echo "Installing pyrealsense2 (this may take several minutes)..." && \
    /.venv/bin/python -m pip install --no-cache-dir --verbose pyrealsense2 && \
    echo "pyrealsense2 installed"

# do NOT extend PYTHONPATH with system site-packages: that mixes host numpy
# (python 3.10) with venv numpy (python 3.11) and breaks at import
ENV PATH=/.venv/bin:$PATH
# /app gets bind-mounted at runtime, this lets `import ur5...` resolve
ENV PYTHONPATH=/app

# JAX picks CUDA automatically; force CPU at runtime with -e JAX_PLATFORMS=cpu

# OpenCV HighGUI defaults safer for X11-in-Docker: MIT-SHM off, XCB backend.
ENV QT_X11_NO_MITSHM=1
ENV QT_QPA_PLATFORM=xcb
# point Qt at the plugins shipped inside the opencv-python wheel; pointing it
# at system plugin dirs causes "xcb found but could not load" ABI mismatches
ENV QT_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins/platforms

# checkpoints come in via bind mount, no need to download at build time

# default runtime configuration; override with -e flags
ENV SERVER_ARGS="policy:checkpoint --policy.config=pi05_ur5_blueblock10 --policy.dir=checkpoints/pi05_ur5_blueblock10/pi05_ur5_blueblock10-1/150"
ENV SERVER_WAIT=6

# robot
ENV UR_IP="192.10.0.11"
ENV RS_BASE="137322074310"
ENV RS_WRIST="137322075008"

ENV PROMPT="Pick up the blue block and place it in the cardboard box"

# camera exposure pinned to match recording (auto off, fixed exposure)
ENV RS_AUTO_EXPOSURE="0"
ENV RS_EXPOSURE="100"
ENV RS_WRIST_EXPOSURE="150"

# inference timing and motion
ENV HOLD_PER_STEP="0.1"
ENV HORIZON_STEPS="6"
ENV MAX_STEP_DEG="3.0"
ENV DT="0.02"
ENV VEL="0.5"
ENV ACC="0.5"
ENV LOOKAHEAD="0.1"
ENV GAIN="300"

ENV DRY_RUN="0"

# launch the policy server in the background, give it SERVER_WAIT seconds to
# come up, then start the robot bridge
CMD ["/bin/bash","-lc","uv run scripts/serve_policy.py $SERVER_ARGS & sleep ${SERVER_WAIT:-6}; python /app/ur5/utils/pi0_bridge_ur5_headless.py"]
