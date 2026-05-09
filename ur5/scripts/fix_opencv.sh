#!/bin/bash
# Quick fix script to reinstall opencv-python in the Docker container

echo "=== Fixing OpenCV installation in Docker container ==="

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q '^openpi-robot$'; then
    echo "Error: Container 'openpi-robot' is not running."
    echo "Please start the container first, then run this script."
    exit 1
fi

echo "Container found. Reinstalling opencv-python..."

# Reinstall opencv-python in the container
docker exec openpi-robot /.venv/bin/python -m pip uninstall -y \
    opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless \
    || true

docker exec openpi-robot /.venv/bin/python -m pip install --no-cache-dir --force-reinstall \
    "numpy<2.0.0" "opencv-python==4.11.0.86"

echo ""
echo "Verifying installation..."
docker exec openpi-robot /.venv/bin/python -c "import cv2; print(f'✓ OpenCV {cv2.__version__} installed successfully')"

echo ""
echo "=== Fix complete! ==="
echo "Try running your script again."
