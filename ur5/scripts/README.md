# UR5 Scripts

## Setup

### Full environment (inference, training, and data conversion)

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
```

### Lightweight environment (recording, replay, and hardware tests)

For tasks that don't need the ML stack:

```bash
python3 -m venv .venv-robot
source .venv-robot/bin/activate
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

## Experiment Pipeline

### Step 1: Record data (local, robot machine)

```bash
# Record waypoints via freedrive
python ur5/scripts/ur5_record_freedrive_waypoints.py

# Replay and record with cameras
python ur5/scripts/ur5_replay_and_record_raw.py \
  --waypoints_path raw_episodes/<episode_id>/waypoints.json
```

Repeat for as many episodes as needed. All episodes land in `raw_episodes/`.

### Step 2: Convert to LeRobot format + push to HuggingFace (local)

Get a token from https://huggingface.co/settings/tokens:

```bash
export HF_TOKEN=your_token_here
```

Convert and push:

```bash
uv run python ur5/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes \
  --repo_id YourUsername/ur5_dataset_name \
  --fps 10

# Skip pushing to hub:
uv run python ur5/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes \
  --repo_id YourUsername/ur5_dataset_name \
  --fps 10 \
  --no-push-to-hub
```

### Step 3: Compute norm stats + train (HPC)


```bash
# Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_low_mem_finetune

# Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5_low_mem_finetune \
  --exp-name=my_exp --overwrite
```

To use a different dataset without editing `config.py`, see [CLI Config Overrides](#cli-config-overrides) below.

Checkpoint output: `checkpoints/pi05_ur5_low_mem_finetune/my_exp/<step>/`

### Step 4: Download checkpoint (local)

Only `params/` and `assets/` are needed for inference (~12 GB). The `train_state/` (~31 GB) is only needed to resume training and can be skipped.

**Find the checkpoint path on HPC:**

```bash
# On HPC, from the openpi repo root:
find $PWD/checkpoints -name '_CHECKPOINT_METADATA' -path '*my_exp*'
# Example output: /gpfs/helios/home/user/openpi/checkpoints/pi05_ur5/my_exp/399/_CHECKPOINT_METADATA
# The checkpoint path is the directory containing _CHECKPOINT_METADATA (drop the filename).
```

**Download to local machine (run as one line):**

```bash
# Create local directory first
mkdir -p ./checkpoints/<config>/<exp_name>/<step>

# Download (must be a single line — line breaks will break the command)
rsync -avhP --include='params/***' --include='assets/***' --include='_CHECKPOINT_METADATA' --exclude='*' user@rocket.hpc.ut.ee:/gpfs/helios/home/user/openpi/checkpoints/<config>/<exp_name>/<step>/ ./checkpoints/<config>/<exp_name>/<step>/
```

**Example** for config `pi05_ur5`, experiment `ur5_fifth_2`, step `399`:

```bash
mkdir -p ./checkpoints/pi05_ur5/ur5_fifth_2/399
rsync -avhP --include='params/***' --include='assets/***' --include='_CHECKPOINT_METADATA' --exclude='*' lenardspatriks@rocket.hpc.ut.ee:/gpfs/helios/home/lenardspatriks/openpi/checkpoints/pi05_ur5/ur5_fifth_2/399/ ./checkpoints/pi05_ur5/ur5_fifth_2/399/
```

Note: the step number is `num_train_steps - 1` (0-indexed), e.g. 500 steps → step `499`, 400 steps → step `399`.

### Step 5: Run inference (local, robot machine)

**Docker (recommended):**

```bash
docker compose -f ur5/docker/compose.yml run \
  -e SERVER_ARGS="policy:checkpoint --policy.config=pi05_ur5_low_mem_finetune --policy.dir=checkpoints/pi05_ur5_low_mem_finetune/my_exp/499" \
  -e PROMPT="pick up the bottle" \
  openpi_serve
```

**Without Docker:**

```bash
# Terminal 1: start policy server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur5_low_mem_finetune \
  --policy.dir=checkpoints/pi05_ur5_low_mem_finetune/my_exp/499

# Terminal 2: start robot bridge
python ur5/utils/pi0_bridge_ur5_headless.py
```

## CLI Config Overrides

You don't need to edit `config.py` for every new dataset. The training CLI supports overriding any config field:

```bash
uv run scripts/train.py pi05_ur5_low_mem_finetune \
  --exp-name=my_exp \
  --data.repo-id=YourUsername/ur5_new_dataset \
  --num-train-steps=300 \
  --overwrite
```

Common overrides:

| Flag | Purpose |
|------|---------|
| `--data.repo-id=X` | Use a different HF dataset |
| `--num-train-steps=N` | Change training length |
| `--exp-name=X` | Experiment name (determines checkpoint path) |
| `--batch-size=N` | Override batch size |
| `--save-interval=N` | Checkpoint save frequency |
| `--overwrite` | Overwrite existing checkpoint dir |
| `--resume` | Resume training from last checkpoint |

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

All hardware defaults live in `ur5/defaults.py` and can be overridden via environment variables:

| Setting | Env var | Default |
|---------|---------|---------|
| Robot IP | `UR_IP` | `192.10.0.11` |
| Output dir | `OUT_DIR` | `raw_episodes` |
| Base camera | `RS_BASE` | `137322074310` |
| Wrist camera | `RS_WRIST` | `137322075008` |
| Gripper port | `ROBOTIQ_PORT` | `63352` |
| Start position | — | `(-90, -40, -140, -50, 90, 0)` degrees |

## Troubleshooting

**No cameras:** Use `--fake_cam` flag

**Robot connection:** Ensure Remote Control mode, check `ping 192.10.0.11`, verify RTDE enabled

**Gripper:** Enabled by default. Disable with `--no-use-gripper`

Run scripts with `--help` for all options.
