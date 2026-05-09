# UR5 Final Evaluations

Systematic evaluation of `pi05_ur5_blueblock10-1` across training checkpoints.

For each checkpoint, two evaluation conditions are run. The prompt is identical in both conditions; only the blue block's physical location changes.

> **Prompt (both conditions)**: `Pick up the blue block and place it in the cardboard box`

- **In-Distribution (ID)**: blue block placed **inside** the ~30 cm × 30 cm area covered by the training dataset.
- **Out-of-Distribution (OOD location)**: blue block placed **outside** that 30 cm training area (farther out on the table, off to a side, or at an angle the dataset did not cover). The cardboard box stays in its training location. This probes whether the policy generalizes spatially or only interpolates within the demonstrated workspace.

Each condition is run for **5 trials** per checkpoint.

---

## Training Run Under Evaluation

> **Experiment**: `pi05_ur5_blueblock10-1`
> **Dataset**: `LPSlvlv/ur5_blueblock_box_10` (10 episodes)
> **Config**: `pi05_ur5_blueblock10` | **Model**: Pi0.5 (full fine-tune) | **Checkpoint base**: `pi05_base`
> **Training**: 500 steps | warmup=50 | peak_lr=2.5e-5 | batch=16 | save_interval=30 | early_stop patience=20
> **Norm stats**: reloaded from `pi05_base/assets/ur5e`

**Checkpoints under evaluation**: 90, 120, 150, 180, 210 (add more as needed)

---

## Scoring System

Each trial is scored across 4 sequential stages. A stage scores **1** if achieved, **0** otherwise.
Normalized score = sum of stages × 0.25 (range: 0.00 – 1.00).
Time is recorded as `mm:ss` from trial start to stage completion (blank if stage not reached).

**Aggregate notation.** Avg rows and summary tables report the per-trial mean and (min, max) across the trials in that cell. For example, `0.80 (0.50–1.00)` means the mean trial score was 0.80 and observed scores ranged from 0.50 to 1.00. When all trials in a cell scored the same value, only the value is shown (e.g. `0.25`). Min/max is preferred over ±std at this small sample size because it directly reports the worst- and best-case observed performance without assuming a distribution.

| Stage | Criteria | Points |
|-------|----------|--------|
| **Reach Object** | End effector within 10 cm of target object | 0.25 |
| **Pick Up** | Object successfully gripped and lifted | 0.25 |
| **Reach Drop-off** | End effector within 10 cm of drop location | 0.25 |
| **Release** | Object released at target location | 0.25 |

**Target object**: the blue block in both ID and OOD conditions. Reach is scored against its actual position on the table for that trial, wherever it is placed.

---

## Trial Setup

> **Inference**: pi0_bridge_ur5_headless
> **Prompt (both conditions)**: `Pick up the blue block and place it in the cardboard box`
> **Reset pose**: default from config (`-90, -40, -140, -50, 90, 0` deg)
> **Cardboard box**: kept in its training location for all trials
> **Block placement**:
> - **ID**: anywhere inside the ~30 cm × 30 cm area covered by training demos
> - **OOD**: outside that area; record the approximate offset (e.g. `+15 cm beyond right edge`, `far corner`) per trial in Notes

### Dataset area reference

Training dataset (`LPSlvlv/ur5_blueblock_box_10`) covers an approximate **30 cm × 30 cm** patch in front of the robot. Before running OOD trials, mark the boundary on the table (tape / chalk) so placements are reproducible and classifying a trial as ID vs OOD is unambiguous.

Docker command (fill in the right checkpoint path before each batch):

```bash
docker run --rm -it --gpus=all --network=host --ipc=host \
  --device=/dev/bus/usb:/dev/bus/usb $RUN_DEVICES --group-add video \
  -v "$PWD":/app -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e QT_QPA_PLATFORM=xcb \
  -e QT_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins \
  -e QT_QPA_PLATFORM_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins/platforms \
  --name openpi-robot \
  -e RS_BASE=137322074310 -e RS_WRIST=137322075008 \
  -e PROMPT="Pick up the blue block and place it in the cardboard box" \
  -e HOLD_PER_STEP=0.15 -e HORIZON_STEPS=15 -e MAX_STEP_DEG=3.0 \
  -e DT=0.02 -e VEL=0.5 -e ACC=0.5 -e LOOKAHEAD=0.1 -e GAIN=300 \
  openpi_robot
```

---

## Checkpoint: step 90

> **Checkpoint path**: `checkpoints/pi05_ur5_blueblock10/pi05_ur5_blueblock10-1/90`
Loss: 0.0101

### ID: block inside training area

> **Block placement**: inside the ~30 cm × 30 cm training area

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 2   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 3   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 4   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 5   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| **Avg** | **0/5** | **0/5** | **0/5** | **0/5** | **0.00** |  |  |  |  | No stage reached on any trial |

### OOD: block outside training area

> **Block placement**: outside the ~30 cm × 30 cm training area (record offset per trial in Notes)

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 2   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 3   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 4   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| 5   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stalled ~60 cm from block, no approach |
| **Avg** | **0/5** | **0/5** | **0/5** | **0/5** | **0.00** |  |  |  |  | No stage reached on any trial |

**Observations (step 90)**: Robot does not approach the object. End effector stalls ~60 cm from the block across all ID and OOD trials; no stage reached. Model has not yet learned the reach behavior.

---

## Checkpoint: step 120

> **Checkpoint path**: `checkpoints/pi05_ur5_blueblock10/pi05_ur5_blueblock10-1/120`
Loss: 0.0105

### ID: block inside training area

> **Block placement**: inside the ~30 cm × 30 cm training area

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 2   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 3   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 4   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 5   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| **Avg** | **0/5** | **0/5** | **0/5** | **0/5** | **0.00** |  |  |  |  | No stage reached on any trial |

### OOD: block outside training area

> **Block placement**: outside the ~30 cm × 30 cm training area (record offset per trial in Notes)

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 2   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 3   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 4   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| 5   |   0   |  0   |   0  |    0    | 0.00  |         |        |        |           | Stopped ~15 cm from block, no closer approach |
| **Avg** | **0/5** | **0/5** | **0/5** | **0/5** | **0.00** |  |  |  |  | No stage reached on any trial |

**Observations (step 120)**: Robot orients toward the object but halts with ~15 cm of residual distance across all ID and OOD trials. Policy is closer than step 90 (60 cm) but still does not close the gap to within the 10 cm reach threshold.

---

## Checkpoint: step 150

> **Checkpoint path**: `checkpoints/pi05_ur5_blueblock10/pi05_ur5_blueblock10-1/150`
Loss: 0.0085

### ID: block inside training area

> **Block placement**: inside the ~30 cm × 30 cm training area

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   1   |  1   |   1  |    1    | 1.00  |   00:57 |  1:30  |  00:37 |  2:01     |       |
| 2   |   1   |  1   |   1  |    0    | 0.75  |   01:03 |  2:00  |  00:55 |           |    release reached 5min timeout   |
| 3   |   1   |  1   |   1  |    0    | 0.75  |   00:50 |  3:11  |  01:25 |           |     timeout release  |
| 4   |   1   |  1   |   1  |    1    | 1.00  |   00:37 |  3:25  |  01:03 |  02:14    |       |
| 5   |   1   |  1   |   0  |    0    | 0.5   |   01:18 |  3:01   |        |           |   Bad grasp resulted in drop while transporting  |
| **Avg** | **5/5** | **5/5** | **4/5** | **2/5** | **0.80 (0.50–1.00)** | **0:57** | **2:37** | **1:00** | **2:08** | Times averaged over trials reaching each stage (n = 5, 5, 4, 2) |

### OOD: block outside training area

> **Block placement**: outside the ~30 cm × 30 cm training area (record offset per trial in Notes)

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |  1    |  1   |  1   |    0    | 0.75  |  00:41  |  01:25 |  01:36 |           |   Started moving drop box around    |
| 2   |  1    |  1   |  1   |    0    | 0.75  |  00:52  |  01:31 |  01:02 |           |  Object fell out of gripper before release     |
| 3   |  1    |  1   |  0   |    0    | 0.50  |  00:58  |  02:03 |        |           |  Object fell out when moving to drop location   |
| 4   |  1    |  1   |  1   |    0    | 0.75  |  00:49  |  01:33 |  01:27 |           |       |
| 5   |  1    |  1   |  1   |    0    | 0.75  |  01:17  |  02:27 |  02:07 |           |  Moved object to bad position while trying to pick.      |
| **Avg** | **5/5** | **5/5** | **4/5** | **0/5** | **0.70 (0.50–0.75)** | **0:55** | **1:48** | **1:33** |  | Times averaged over trials reaching each stage (n = 5, 5, 4, 0) |

**Observations (step 150)**:

---

## Checkpoint: step 180

> **Checkpoint path**: `checkpoints/pi05_ur5_blueblock10/pi05_ur5_blueblock10-1/180`
Loss: 0.0071

### ID: block inside training area

> **Block placement**: inside the ~30 cm × 30 cm training area

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   1   |  1   |   1  |    0    | 0.75  |   0:32  |  0:22  |   0:40 |           | total 1:30 |
| 2   |   1   |  1   |   1  |    0    | 0.75  |   0:30  |  0:13  |   1:23 |           |       |
| 3   |   1   |  0   |   0  |    0    | 0.25  |   0:38  |        |        |           |  Crashed at gripping     |
| 4   |   1   |  1   |   1  |    0    | 0.75  |   0:39  |  1:20  |   0:46 |           |       |
| 5   |   1   |  1   |   1  |    0    | 0.75  |   0:42  |  0:50  |   1:01 |           |       |
| **Avg** | **5/5** | **4/5** | **4/5** | **0/5** | **0.65 (0.25–0.75)** | **0:36** | **0:41** | **0:58** |  | Times averaged over trials reaching each stage (n = 5, 4, 4, 0) |

### OOD: block outside training area

> **Block placement**: outside the ~30 cm × 30 cm training area (record offset per trial in Notes)

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   1   |  1   |   1  |     0   | 0.75  |   00:39 |  01:56 |  01:55 |           |       |
| 2   |   1   |  1   |   1  |     0   | 0.75  |   00:46 |  00:23 |  01:03 |           |       |
| 3   |   1   |  1   |   1  |     0   | 0.75  |   00:32 |  01:57 |  01:41 |           |       |
| 4   |   1   |  1   |   1  |     0   | 0.75  |   00:38 |  00:57 |  00:28 |           |       |
| 5   |   1   |  0   |   0  |     0   | 0.25  |   00:52 |  00:50 |        |           |    crashed at pick   |
| **Avg** | **5/5** | **4/5** | **4/5** | **0/5** | **0.65 (0.25–0.75)** | **0:41** | **1:18** | **1:17** |  | Times averaged over trials reaching each stage (n = 5, 4, 4, 0) |

**Observations (step 180)**:


---

## Checkpoint: step 210

> **Checkpoint path**: `checkpoints/pi05_ur5_blueblock10/pi05_ur5_blueblock10-1/210`
Loss: 0.0084

### ID: block inside training area

> **Block placement**: inside the ~30 cm × 30 cm training area

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |   1   |  1   |   1  |    0    | 0.75  |  00:40  |  00:16 | 00:59  |           | Crashed while releasing      |
| 2   |   1   |  1   |   1  |    0    | 0.75  |  00:31  |  00:26 | 01:23  |           |   Crashed while releasing    |
| 3   |   1   |  1   |   1  |    0    | 0.75  |  00:36  |  00:32 | 00:57  |           |   Crashed while releasing    |
| 4   |   1   |  1   |   1  |    0    | 0.75  |  00:48  |  00:24 | 01:02  |           |       |
| 5   |   1   |  1   |   1  |    0    | 0.75  |  00:35  |  00:31 | 01:06  |           |       |
| **Avg** | **5/5** | **5/5** | **5/5** | **0/5** | **0.75** | **0:38** | **0:26** | **1:05** |  | Times averaged over trials reaching each stage (n = 5, 5, 5, 0) |

### OOD: block outside training area

> **Block placement**: outside the ~30 cm × 30 cm training area (record offset per trial in Notes)

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   |  1    |  0   |   0  |    0    | 0.25  |   00:49 |        |        |           |  Crashed while trying to pick up the object     |
| 2   |  1    |  0   |   0  |    0    | 0.25  |   00:42 |        |        |           |  Crashed the same     |
| 3   |  1    |  0   |   0  |    0    | 0.25  |   00:53 |        |        |           |    Same crash   |
| 4   |  1    |  0   |   0  |    0    | 0.25  |   01:04 |        |        |           |       |
| 5   |  1    |  0   |   0  |    0    | 0.25  |   00:50 |        |        |           |       |
| **Avg** | **5/5** | **0/5** | **0/5** | **0/5** | **0.25** | **0:52** |  |  |  | Times averaged over trials reaching each stage (n = 5, 0, 0, 0) |

**Observations (step 210)**:

---

## Summary

Per-stage times are means over trials that reached each stage (blank if no trial reached that stage). Score format is `mean (min–max)`, with a single value when all trials scored the same.

| Ckpt | Cond | Score             | Reach | Pick | Drop | Release | t_reach | t_pick | t_drop | t_release |
|------|------|-------------------|-------|------|------|---------|---------|--------|--------|-----------|
| 90   | ID   | 0.00              | 0/5   | 0/5  | 0/5  | 0/5     |         |        |        |           |
| 90   | OOD  | 0.00              | 0/5   | 0/5  | 0/5  | 0/5     |         |        |        |           |
| 120  | ID   | 0.00              | 0/5   | 0/5  | 0/5  | 0/5     |         |        |        |           |
| 120  | OOD  | 0.00              | 0/5   | 0/5  | 0/5  | 0/5     |         |        |        |           |
| 150  | ID   | 0.80 (0.50–1.00)  | 5/5   | 5/5  | 4/5  | 2/5     | 0:57    | 2:37   | 1:00   | 2:08      |
| 150  | OOD  | 0.70 (0.50–0.75)  | 5/5   | 5/5  | 4/5  | 0/5     | 0:55    | 1:48   | 1:33   |           |
| 180  | ID   | 0.65 (0.25–0.75)  | 5/5   | 4/5  | 4/5  | 0/5     | 0:36    | 0:41   | 0:58   |           |
| 180  | OOD  | 0.65 (0.25–0.75)  | 5/5   | 4/5  | 4/5  | 0/5     | 0:41    | 1:18   | 1:17   |           |
| 210  | ID   | 0.75              | 5/5   | 5/5  | 5/5  | 0/5     | 0:38    | 0:26   | 1:05   |           |
| 210  | OOD  | 0.25              | 5/5   | 0/5  | 0/5  | 0/5     | 0:52    |        |        |           |

**Overall conclusions**: _fill in after evaluation complete: best checkpoint, ID vs OOD gap, failure modes, recommended checkpoint for deployment._

---

## Action Horizon Sweep (ID only)

Fixed-checkpoint sweep varying how many predicted action-chunk steps the bridge executes per policy query.

> **Checkpoint**: `pi05_ur5_blueblock10-1/150` (best from the checkpoint sweep above)
> **Varying**: `HORIZON_STEPS` ∈ `{3, 6, 9, 12, 15}` (15 = full action chunk; model was trained with `action_horizon=15`)
> **Condition**: ID only, blue block inside the ~30 cm × 30 cm training area
> **Trials per horizon**: 5
> **Scoring**: same 4-stage rubric (0.25 per stage); times not recorded for this sweep
> **All other settings**: match `ur5/docker/serve_policy_robot.Dockerfile` defaults (HOLD_PER_STEP=0.1, MAX_STEP_DEG=3.0, DT=0.02, VEL=0.5, ACC=0.5, LOOKAHEAD=0.1, GAIN=300); only `HORIZON_STEPS` changes.

To run, override `-e HORIZON_STEPS=<N>` on the docker command line. Example:

```bash
docker run ... -e HORIZON_STEPS=3 ... openpi_robot
```

---

### HORIZON_STEPS = 3

| Try | Reach | Pick | Drop | Release | Score | Notes |
|-----|-------|------|------|---------|-------|-------|
| 1   |   1   |   0  |   0  |    0    | 0.25  |   crashed on grasp    |
| 2   |   1   |   0  |   0  |    0    | 0.25  |   crashed on grasp    |
| 3   |   1   |   0  |   0  |    0    | 0.25  |   crashed on grasp   |
| 4   |   1   |   0  |   0  |    0    | 0.25  |   crashed on grasp   |
| 5   |   1   |   0  |   0  |    0    | 0.25  |   crashed on grasp   |
| **Avg** | **5/5** | **0/5** | **0/5** | **0/5** | **0.25** |  |

**Observations (H=3)**:

---

### HORIZON_STEPS = 6

Mirrors the step 150 ID checkpoint-sweep batch (same checkpoint, same conditions, same trials). Reproduced here so the horizon sweep is self-contained.

| Try | Reach | Pick | Drop | Release | Score | Notes |
|-----|-------|------|------|---------|-------|-------|
| 1   |   1   |  1   |   1  |    1    | 1.00  |       |
| 2   |   1   |  1   |   1  |    0    | 0.75  |    release reached 5min timeout   |
| 3   |   1   |  1   |   1  |    0    | 0.75  |     timeout release  |
| 4   |   1   |  1   |   1  |    1    | 1.00  |       |
| 5   |   1   |  1   |   0  |    0    | 0.50  |   Bad grasp resulted in drop while transporting   |
| **Avg** | **5/5** | **5/5** | **4/5** | **2/5** | **0.80 (0.50–1.00)** |  |

**Observations (H=6)**:

---

### HORIZON_STEPS = 9

| Try | Reach | Pick | Drop | Release | Score | Notes |
|-----|-------|------|------|---------|-------|-------|
| 1   |   1   |  0   |  0   |    0    | 0.25  |    Tried for a long time to pick it up, but failed   |
| 2   |   1   |  0   |  0   |    0    | 0.25  |    Tried for a long time to pick it up, but failed   |
| 3   |   1   |  1   |  1   |    1    | 1.00  |       |
| 4   |   1   |  0   |  0   |    0    | 0.25  |       |
| 5   |   1   |  1   |  1   |    0    | 0.75  |   Bad grasp caused it to drop the object, instead of releasing    |
| **Avg** | **5/5** | **2/5** | **2/5** | **1/5** | **0.50 (0.25–1.00)** |  |

**Observations (H=9)**:

---

### HORIZON_STEPS = 12

| Try | Reach | Pick | Drop | Release | Score | Notes |
|-----|-------|------|------|---------|-------|-------|
| 1   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 2   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 3   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 4   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 5   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| **Avg** | **5/5** | **0/5** | **0/5** | **0/5** | **0.25** |  |

**Observations (H=12)**:

---

### HORIZON_STEPS = 15

| Try | Reach | Pick | Drop | Release | Score | Notes |
|-----|-------|------|------|---------|-------|-------|
| 1   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 2   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 3   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 4   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| 5   |   1   |   0  |  0   |   0     | 0.25  |    Crashed while trying to grasp   |
| **Avg** | **5/5** | **0/5** | **0/5** | **0/5** | **0.25** |  |

**Observations (H=15)**:

---

### Summary: Horizon Sweep (ID, checkpoint 150)

H=6 reuses the step 150 ID checkpoint-sweep batch (same checkpoint, same conditions). Score format is `mean (min–max)`, with a single value when all trials scored the same.

| Horizon | Score             | Reach | Pick | Drop | Release |
|---------|-------------------|-------|------|------|---------|
| 3       | 0.25              | 5/5   | 0/5  | 0/5  | 0/5     |
| 6       | 0.80 (0.50–1.00)  | 5/5   | 5/5  | 4/5  | 2/5     |
| 9       | 0.50 (0.25–1.00)  | 5/5   | 2/5  | 2/5  | 1/5     |
| 12      | 0.25              | 5/5   | 0/5  | 0/5  | 0/5     |
| 15      | 0.25              | 5/5   | 0/5  | 0/5  | 0/5     |

**Horizon conclusions**: _fill in after all 25 trials complete: which horizon gives best score, any horizon-specific failure modes (drift / reactivity tradeoff), recommended HORIZON_STEPS for deployment._
