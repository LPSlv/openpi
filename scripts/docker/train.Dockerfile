# Dockerfile for fine-tuning openpi models.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_train -f scripts/docker/train.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_train /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies including build tools for FFmpeg 7
RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang \
    curl wget ca-certificates xz-utils pkg-config \
    nasm yasm autoconf automake libtool \
    libgl1 libglib2.0-0 \
    libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Build and install FFmpeg 7 from source (PyAV/av package requires FFmpeg 7)
ARG FFMPEG_VERSION="7.1.3"
RUN mkdir -p /tmp/ffmpeg-src && \
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o /tmp/ffmpeg.tar.xz && \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg-src --strip-components=1 && \
    cd /tmp/ffmpeg-src && \
    ./configure \
        --prefix=/usr/local \
        --enable-shared \
        --enable-pic \
        --disable-static \
        --disable-debug \
        --disable-doc \
        --disable-programs \
    && make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/ffmpeg-src /tmp/ffmpeg.tar.xz

# Ensure pkg-config can find FFmpeg's .pc files
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Install Cython before uv sync (required for building av package)
RUN /.venv/bin/python -m ensurepip --upgrade && \
    /.venv/bin/python -m pip install --no-cache-dir Cython

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Install the project in editable mode
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

CMD ["/bin/bash"]
