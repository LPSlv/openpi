# UR5 Scripts

## Setup

```bash
python3 -m venv venv_ur5
source venv_ur5/bin/activate
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

## Scripts

### Record Waypoints

```bash
python local/scripts/ur5_record_freedrive_waypoints.py
```

1. Robot moves to start position
2. Teach mode enables automatically
3. Guide the arm
4. Press Enter to finish
5. Waypoints saved to `raw_episodes/<episode_id>/waypoints.json`

### Replay Waypoints

```bash
python local/scripts/ur5_replay_and_record_raw.py \
  --waypoints_path raw_episodes/ur5_freedrive_20260113_140635/waypoints.json
```

### Convert to LeRobot Format

**Hugging Face Authentication:**

Get token from https://huggingface.co/settings/tokens
**Environment Variable**


```bash
export HF_TOKEN=your_token_here
```

Or add to `~/.bashrc`:
```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
```

**Convert:**
```bash
uv run python local/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes \
  --repo_id LPSlvlv/ur5_pickandplace_3 \
  --fps 10

# Skip pushing to hub:
uv run python openpi/local/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes \
  --repo_id your_hf_username/ur5_freedrive \
  --fps 10 \
  --no-push-to-hub
```

## Fine-Tuning

**Option 1: Docker (recommended for HPC/systems with FFmpeg issues)**

```bash
# Build and run training container
docker compose -f scripts/docker/compose_train.yml up --build -d

# Enter container
docker exec -it scripts-docker-openpi_train-1 /bin/bash

# Inside container, run:
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_low_mem_finetune

# Notes:
# - This writes `norm_stats.json` under: `assets/pi05_ur5_low_mem_finetune/ur5e/`
# - The finetune config `pi05_ur5_low_mem_finetune` is set up to LOAD stats from that path (fresh stats on your data).
# - If you want to instead reuse the pretrained UR5e stats, update the config to point `assets_dir` at:
#     gs://openpi-assets/checkpoints/pi05_base/assets (asset_id="ur5e")
#
# Inference:
# - If you trained after computing stats, the checkpoint will save the stats under `<checkpoint_dir>/assets/ur5e/`
#   and inference will automatically use them.
# - If you want to force a specific stats file at inference time (without retraining), pass:
#     uv run scripts/serve_policy.py --norm_stats_dir=assets/pi05_ur5_low_mem_finetune/ur5e ...
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5_low_mem_finetune --exp-name=my_experiment --overwrite
```

**Option 2: Local installation**

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"  # or ~/.local/bin depending on install location
```

**Commands:**
```bash
# Compute normalization stats
export LD_LIBRARY_PATH="$HOME/ffmpeg7/lib:$LD_LIBRARY_PATH"
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_low_mem_finetune

# Run training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5_low_mem_finetune --exp-name=my_experiment --overwrite

#To download the checkpoint from HPC
rsync -avhP \
  lenardspatriks@rocket.hpc.ut.ee:/gpfs/helios/home/lenardspatriks/openpi/checkpoints/pi05_ur5_low_mem_finetune/ur5_third/499/ \
  ./openpi_ckpt_3_99/


# Run inference
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur5_low_mem_finetune \
  --policy.dir=checkpoints/pi05_ur5_low_mem_finetune/my_experiment/20000
```

## Building FFmpeg 7 on HPC

**Goal:** Compile and install FFmpeg 7 into your home directory so Python packages like PyAV (av) can build against it (via pkg-config), without touching system libraries.

**Assumptions:**
- You are on an HPC login node
- You have: `gcc`, `make`, `pkg-config`, `python3`
- You do not have: `nasm`, `yasm`, `cmake`

### 1) Directory Layout and Environment

Use a predictable layout:

```bash
mkdir -p "$HOME/src" "$HOME/local" "$HOME/ffmpeg7"
```

### 2) Build NASM Locally (Required)

FFmpeg uses NASM for optimized x86 code. If nasm is missing, FFmpeg build will be slow or may fail depending on config.

**Download + compile NASM:**
```bash
cd "$HOME/src"

curl -L -o nasm-2.16.01.tar.xz \
  https://www.nasm.us/pub/nasm/releasebuilds/2.16.01/nasm-2.16.01.tar.xz

tar -xf nasm-2.16.01.tar.xz
cd nasm-2.16.01

./configure --prefix="$HOME/local"
make -j"$(nproc)"
make install
```

**Put NASM on PATH (current shell):**
```bash
export PATH="$HOME/local/bin:$PATH"
which nasm
nasm -v
```

If `which nasm` still fails, your `make install` did not land in `~/local/bin` and needs investigation.

### 3) Build FFmpeg 7 Locally

**Download source:**
```bash
export PREFIX="$HOME/ffmpeg7"

cd "$HOME/src"
curl -L -o ffmpeg-7.0.2.tar.xz https://ffmpeg.org/releases/ffmpeg-7.0.2.tar.xz
tar -xf ffmpeg-7.0.2.tar.xz
cd ffmpeg-7.0.2
```

**Configure:**

This is a "safe default" build for PyAV:
- shared libraries enabled (PyAV needs these)
- static disabled
- no docs/debug to reduce build time
- PIC enabled

```bash
./configure \
  --prefix="$PREFIX" \
  --enable-shared \
  --disable-static \
  --disable-debug \
  --disable-doc \
  --enable-pic
```

**Compile + install:**
```bash
make -j"$(nproc)"
make install
```

### 4) Activate FFmpeg 7 for Builds

Export these variables so compilers and pkg-config resolve your local FFmpeg first (must be in same shell as `uv sync`):

```bash
export PREFIX="$HOME/ffmpeg7"
export PATH="$PREFIX/bin:$HOME/local/bin:$PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
export CPATH="$PREFIX/include:$CPATH"
export LIBRARY_PATH="$PREFIX/lib:$LIBRARY_PATH"
```

### 5) Verification Checklist

**Check FFmpeg binary:**
```bash
ffmpeg -version | head -n 2
```

**Check pkg-config resolves your local install:**
```bash
pkg-config --variable=prefix libavformat
pkg-config --modversion libavformat
```

**Expected:**
- `prefix` prints `$HOME/ffmpeg7`
- `version` prints something like `61.x.y` (this is libavformat versioning and is normal)

If prefix is not `$HOME/ffmpeg7`, your `PKG_CONFIG_PATH` is wrong.

### 6) Build Python deps (PyAV via uv)

PyAV builds may be cached by uv. After changing FFmpeg/toolchain, wipe uv build cache:

```bash
rm -rf ~/.cache/uv/builds-v0
```

Then run from repo root:

```bash
cd ~/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 7) Runtime Note for Compute Jobs

If your Python code imports `av` on compute nodes, the shared libraries must be discoverable. Set `LD_LIBRARY_PATH` in the job script as well.

**Example snippet to include in Slurm script:**
```bash
export PREFIX="$HOME/ffmpeg7"
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
```

### 8) Optional: Make the Environment Reusable

Create `~/env_ffmpeg7.sh`:

```bash
cat > "$HOME/env_ffmpeg7.sh" << 'EOF'
export PREFIX="$HOME/ffmpeg7"
export PATH="$PREFIX/bin:$HOME/local/bin:$PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
export CPATH="$PREFIX/include:$CPATH"
export LIBRARY_PATH="$PREFIX/lib:$LIBRARY_PATH"
EOF
```

**Use it when needed:**
```bash
source "$HOME/env_ffmpeg7.sh"
```

## Defaults

- Robot IP: `192.10.0.11`
- Output: `raw_episodes`
- Base camera: `137322074310`
- Wrist camera: `137322075008`
- Start position: `(-90, -70, -120, -80, 90, 0)` degrees
- Replay speed: `0.2` rad/s

## Troubleshooting

**Missing modules:**
```bash
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

**No cameras:** Use `--fake_cam` flag

**Robot connection:** Ensure Remote Control mode, check `ping 192.10.0.11`, verify RTDE enabled

**Gripper:** Enabled by default. Disable with `--no-use-gripper`

Run scripts with `--help` for all options.
