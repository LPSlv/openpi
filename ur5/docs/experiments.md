# UR5 Experiments Log

---

## Experiment: ur5_first (1 episode, bus the table)

> **Dataset**: LPSlvlv/ur5_busthetable_1 (1 episode, ~1000 frames) | **Config**: pi05_ur5_low_mem_finetune (LoRA) | **Checkpoint**: pi05_base | **Steps**: 1000 | **Actions**: delta | **GPU**: H200-141GB (~25 min training) | **Norm stats**: reloaded from pretrained

Experiments with the 1st pipeline testing dataset - 1 episode of prompt bus the table. Putting a white plastic tray and a orange mug in the plastic bin. (ur5_first in wandb)

This experiment showed, that the model was able to learn features. The is moving towards the first object, the plastic tray, but it has problem reaching it. With a external help from human, bringing it into the gripper, the robot grips it. And starts movign to the plastic bin. Although incorrectly, not moving up and then into the bin, but going straight into the bin from the side. But this shows that the robot has learned to activate the gripper when the object is inside the gripper claws. But there are problems with movement.

This showed that the robot control scripts are correct (robot is moving towards the object, joints are correct, movement is working). That is a good start, and is expected due to the only one episode of fine tuning.

It was seen that the robot manipulator has trouble moving below the imaginary ground plane of the robot base. It needs to be investigated if there is some safety mechanism interfiering, or its model issue.

It was also noticed, that the model parameters that were used with pi0.5 base didnt work with with the fine tuned version.

docker run --rm -it \
  --gpus=all \
  --network=host \
  --ipc=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --name openpi-robot \
  -e RS_BASE=137322074310 \
  -e RS_WRIST=137322075008 \
  -e PROMPT="bus the table" \
  -e INFER_PERIOD=0.6 \
  -e HORIZON_STEPS=8 \
  -e HOLD_PER_STEP=0.05 \
  -e MAX_STEP_DEG=0.10 \
  -e DT=0.05 \
  -e VEL=0.05 \
  -e ACC=0.10 \
  -e LOOKAHEAD=0.10 \
  -e GAIN=200 \
  openpi_robot

  These settings were too slow. Not enough future steps were inferenced, and the movements were incremental.

  With these it moves atleast somewhat correctly:

  docker run --rm -it \
  --gpus=all \
  --network=host \
  --ipc=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins \
  -e QT_QPA_PLATFORM_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins/platforms \
  --name openpi-robot \
  -e RS_BASE=137322074310 \
  -e RS_WRIST=137322075008 \
  -e PROMPT="bus the table" \
  -e INFER_PERIOD=2.0 \
  -e HORIZON_STEPS=20 \
  -e MAX_STEP_DEG=2.0 \
  -e DT=0.05 \
  -e VEL=0.05 \
  -e ACC=0.10 \
  -e LOOKAHEAD=0.20 \
  -e GAIN=150 \
  openpi_robot


  This was using 1st dataset created (test): LPSlvlv/ur5_busthetable_1. Consisted of just one episode of picking a plastic tray and mug from a table and putting in a bin. Fine tuning took around 25 minutes on HPC using h200-141gb. Interestingly on HPC with tesla it was expected to take 10 hours. Which could be maybe of the tesla gpu not having tensor core native support for bfloat16 data types. 1000 training steps were used, but noticed that with 500 the loss already stabilizes.

---

## Experiment: ur5_busthetable_2 (10 episodes, varied objects)

> **Dataset**: LPSlvlv/ur5_busthetable_2 (10 episodes, ~3000 frames) | **Config**: same LoRA config as ur5_first | **Actions**: delta | **Task**: pick varied objects (mugs, fork, spoon, tray, bowl) from table into bin | **Variation**: different objects, bin location changed between episodes

  2nd dataset created: LPSlvlv/ur5_busthetable_2. Consisted of 10 episodes of picking dishes from a table and putting in a bin. Different objects were used like red, yellow, white mug. Fork and a spoon, plastic tray and a white bowl. Different variations were created, but most consisted of 3 objects on the table and around 30 waypoints recorded. Fork and spoon was placed on a table, inside the mug and bowl. Cup was also placed in a bowl but picked up seperately. Bin location was changed, but mostly stayed in the right.

  Experiments with this, didnt show better results. The model sometimes moved towards the object, but couldnt reach it. If it was placed in the grippers, the robot gripped it, but sometimes dropped after a few seconds, sometimes started to move to drop of location, but at the end not reaching it fully. This could be because the environment was too varied for the dataset size, instead acting not like 10 episodes but like 10x1 single episodes.

---

## Experiment: ur5_third (LoRA, 10 episodes, blue triangle)

> **Dataset**: LPSlvlv/ur5_pickandplace_3 (10 episodes, 7577 frames) | **Config**: pi05_ur5_low_mem_finetune | **Model**: pi0.5 LoRA (gemma_2b_lora rank=16, gemma_300m_lora rank=32) | **Checkpoint**: pi05_base | **Steps**: 500 | **Actions**: delta | **Norm stats**: reloaded from pretrained | **EMA**: disabled

  UR5_third dataset: A simpler 10 episode was  created using 10 episodes of picking up a blue triangular object and placing in a cardboard box. The object location was varied a little bit, but not as much as before.

  Config:

     #
    # Fine-tuning UR5 configs.
    #
    TrainConfig(
        name="pi05_ur5_low_mem_finetune",
        # Pi0.5 LoRA fine-tuning (low memory).
        model=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=180,
        ),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_3",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the docs/norm_stats.md file for more details.
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. When True, LeRobot automatically converts task strings
                # to task_index, and PromptFromLeRobotTask creates "prompt" from task_index. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Load the pi0.5 base model checkpoint.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=500,
        log_interval=10,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        # Reset pose matches dataset recording start position: (-90.0, -40.0, -140.0, -50.0, 90.0, 0.0) degrees
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

  After should try versions without Lora, but with full fine tuning. And should try calculating norm statistics, not reloading.

  This config was moving quickly downards, and hitting joint limits, activating protective stop. And the motion was repeating.

---

## Experiment: ur5_third_2 (LoRA, computed norm stats)

> **Dataset**: LPSlvlv/ur5_pickandplace_3 | **Config**: pi05_ur5_low_mem_finetune (updated) | **Change vs ur5_third**: locally computed norm stats instead of reloaded pretrained stats, added explicit action_dim=32 and action_horizon=15 | **Norm stats**: computed from dataset via `compute_norm_stats.py`

    #
    # Fine-tuning UR5 configs.
    #
    TrainConfig(
        name="pi05_ur5_low_mem_finetune",
        # Pi0.5 LoRA fine-tuning (low memory).
        model=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=180,
            action_dim=32,  # Must match pi05_base checkpoint (UR5Outputs transform slices to 7 dims)
            action_horizon=15,  # Must match base pi05_ur5 config
        ),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_3",
            # Norm stats behavior:
            # - By default, this config expects you to compute stats for your dataset and store them locally under:
            #     assets/pi05_ur5_low_mem_finetune/ur5e/norm_stats.json
            #   via: `uv run scripts/compute_norm_stats.py --config-name pi05_ur5_low_mem_finetune`
            #
            # - If you instead want to *reload* the pretrained UR5e stats, set:
            #     assets=AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets", asset_id="ur5e")
            assets=AssetsConfig(asset_id="ur5e"),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. When True, LeRobot automatically converts task strings
                # to task_index, and PromptFromLeRobotTask creates "prompt" from task_index. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Load the pi0.5 base model checkpoint.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=500,
        log_interval=10,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=15,
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        # Reset pose matches dataset recording start position: (-90.0, -40.0, -140.0, -50.0, 90.0, 0.0) degrees
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

     Fix compute_norm_stats.py to save stats to asset_id path (not repo_id)
- Update pi05_ur5_low_mem_finetune config to use local computed stats

---

## Experiment: ur5_third_3 (pi0_ur5, full training, 1000 steps)

> **Dataset**: LPSlvlv/ur5_pickandplace_3 | **Config**: pi0_ur5 (full finetuning, NOT LoRA) | **Model**: pi0 (z-score norm, action_horizon=50) | **Checkpoint**: pi0_base | **Steps**: 1000 | **Actions**: delta | **Norm stats**: reloaded from pi0_base/assets/ur5e | **EMA**: 0.999

Took config from example with ur5, and reloading the norm statistics, with 50 action horizon. first run with full training (not LORA). Also overfitting with 1000 train steps.

    TrainConfig(
        name="pi0_ur5",
        model=pi0_config.Pi0Config(),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_3",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                # Recommended: load prompt from the LeRobot `task` field.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=1_000,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),


what is a good finetuning dataset amount?
https://github.com/Physical-Intelligence/openpi/issues/768 say 90minutes got acceptable results
https://github.com/Physical-Intelligence/openpi/issues/850 had one hour

This robot finally learned to move, moving quickly to the object, but failing to activate gripper on it. Possibly due to overfitting.

---

## Experiment: ur5_third_4 (pi0_ur5, 400 steps)

> **Dataset**: LPSlvlv/ur5_pickandplace_3 | **Config**: pi0_ur5 | **Change vs ur5_third_3**: reduced from 1000 to 400 steps to prevent overfitting | **All other params identical**

Previous config but changed train steps to 400 to stop overfitting,

    TrainConfig(
        name="pi0_ur5",
        model=pi0_config.Pi0Config(),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_3",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                # Recommended: load prompt from the LeRobot `task` field.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=400,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

## Experiment: ur5_third_5 (pi05_droid checkpoint)

> **Dataset**: LPSlvlv/ur5_pickandplace_3 | **Config**: pi05_ur5_droid | **Model**: pi0.5 (quantile norm, action_horizon=15) | **Checkpoint**: pi05_droid (pretrained on Franka/DROID data, different robot family) | **Steps**: 400 | **Actions**: delta | **Norm stats**: reloaded from pi05_base/assets/ur5e

Using droid pi05 checkpoint:

    TrainConfig(
        name="pi05_ur5_droid",
        # Fine-tune pi05_droid checkpoint on UR5 LeRobot dataset.
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=15,  # Must match pi05_droid checkpoint
            action_dim=32,  # pi05 is trained with 32-dim actions
            max_token_len=180,
        ),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_3",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                # Recommended: load prompt from the LeRobot `task` field.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=400,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

## Bug Fixes (applied between ur5_third experiments)

    These bugs were fixed:

    Bug 1 — Gripper proprioception is dead (FIXED)
In pi0_bridge_ur5_headless.py:683, the gripper state was hardcoded to 0.0. The model never sees the real gripper position.

During training, the dataset has state[6] = g_cmd matching action[6] = g_cmd. The model learns the shortcut action[6] ≈ state[6]. During inference with state[6] always 0.0, the model outputs action[6] ≈ 0.0 — keeping the gripper open forever.

Applied fix: Now reads the actual Robotiq position via gripper.get_position_normalized() and feeds it into state[6].


Bug 3 — Blocking gripper I/O freezes the arm (FIXED)
gripper.move_normalized() called move_and_wait() which polls the gripper for up to 12 seconds, halting all arm motion. Replaced with non-blocking send_normalized() that sets the target and returns immediately.



But for future also this needs to be fixed, but this requires dataset fixing and reworking:

Bug 2 — Action temporal misalignment at gripper transitions (dataset issue)
In ur5_replay_and_record_raw.py:1002-1007, the recording stores action[6] = g_cmd which is the same value as state[6] at the same timestep. The model never sees a training example where observation=gripper_open but action=gripper_close. The "close now" signal only exists implicitly within the 50-step action chunk.

Impact: The model learns to maintain the current gripper state rather than initiate transitions. This is partially mitigated by action chunking but weakens the transition signal significantly.

Fix for future datasets: Record action_t = [q_{t+1} - q_t, gripper_{t+1}] (forward-looking) instead of [q_t - q_{t-1}, gripper_t] (backward-echoing). Or better: record absolute joint positions as actions and use use_delta_action_transform=True in the config, which matches the pretrained model's convention.

---

## Retesting with bug fixes

Now I will try the pi0_ur5 config with training: UR5_THIRD_3 that was overfitted. It was moving good to object, but not gripping good.

Result: Didnt want to grip it.



Lets try with the same non overfit model. UR5_THIRD_4

Gripping now works. But still movemement is not good enough.


Lets try UR5_THIRD_5 that uses pi05 droid checkpoint. Loss showed a interesting curve, it started way higher (1.5 vs 0.08)

---

## Experiment: ur5_pickandplace_4 (combined dataset 2+3)

> **Dataset**: LPSlvlv/ur5_pickandplace_4 (20 episodes, ~10k frames — busthetable_2 + pickandplace_3 combined) | **Config**: pi0_ur5 (full finetuning) | **Steps**: 500 | **Actions**: delta | **Norm stats**: reloaded from pi0_base/assets/ur5e

Create new dataset LPSlvlv/ur5_pickandplace_4" which is combination of dataset 2 and 3. Will try it with pi0_ur5. Increased also to 500 training steps. Without reloading normalization statistics.

    TrainConfig(
        name="pi0_ur5",
        model=pi0_config.Pi0Config(),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_4",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                # Recommended: load prompt from the LeRobot `task` field.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=500,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

## Experiment: ur5_fifth_1 (absolute actions, pi05_ur5)

> **Dataset**: LPSlvlv/ur5_pickandplace_5 (10 episodes, pickandplace_3 converted to absolute) | **Config**: pi05_ur5 (full finetuning) | **Model**: pi0.5 (quantile norm, action_horizon=15) | **Checkpoint**: pi05_base | **Steps**: 500 | **Actions**: absolute (forward-looking: action[i]=state[i+1]), use_delta_action_transform=True | **Norm stats**: reloaded from pi05_base/assets/ur5e | **Key change**: first experiment with absolute actions + delta transform, fixing the backward-looking bug

LPSlvlv/ur5_pickandplace_5: Dataset pickandplace_3 converted to absolute actions using convert_raw_deltas_to_absolute.py. Fixes the backward-looking delta bug (old: action[i] = q[i] - q[i-1], new: action[i] = state[i+1], forward-looking absolute). Also enabled use_delta_action_transform=True in LeRobotUR5DataConfig so training applies DeltaActions (abs→delta) and inference applies AbsoluteActions (delta→abs). Bridge ACTION_MODE changed to absolute. FPS remains 10 (original recording rate).
Based on this run experiment ur5_fifth_1:
    TrainConfig(
        name="pi05_ur5",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, max_token_len=180),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_5",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=500,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

### Inference timing

                               Timing Statistics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┓
┃ Metric              ┃ Mean ┃ S… ┃  P25 ┃  P50 ┃   P75 ┃  P90 ┃   P95 ┃  P99 ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━┩
│ client_infer_ms     │ 232… │ 1… │ 231… │ 232… │ 233.5 │ 235… │ 235.5 │ 237… │
│ policy_infer_ms     │ 187… │ 0… │ 187… │ 187… │ 188.2 │ 188… │ 188.4 │ 189… │
│ server_infer_ms     │ 231… │ 1… │ 230… │ 231… │ 232.1 │ 233… │ 234.0 │ 235… │
│ server_prev_total_… │ 233… │ 3… │ 231… │ 232… │ 233.7 │ 235… │ 238.2 │ 245… │
└─────────────────────┴──────┴────┴──────┴──────┴───────┴──────┴───────┴──────┘

### Recommended bridge settings for pi05_ur5

From the policy:

action_horizon = 15 (15 action steps per inference)
Dataset recorded at 10Hz → each step = 0.1s of real time

```
# Action execution
ACTION_MODE=absolute          # matches AbsoluteActions output transform
HORIZON_STEPS=15              # use full action horizon from policy
HOLD_PER_STEP=0.1             # 0.1s per step = 10Hz, matches dataset recording rate

# servoJ parameters (VEL/ACC are ignored by UR firmware for servoJ)
DT=0.02                       # 50Hz servoJ command rate (smooth updates within each step)
VEL=0.5                       # not used by servoJ, but set non-zero
ACC=0.5                       # not used by servoJ, but set non-zero
LOOKAHEAD=0.1                 # moderate trajectory smoothing (range 0.03-0.2)
GAIN=300                      # position tracking stiffness (range 100-2000)
```

Timing per cycle:

Inference: ~232ms
Execution: 15 steps x 0.1s = 1.5s
Total cycle: ~1.73s

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| HORIZON_STEPS=15 | Matches action_horizon=15 from policy |
| HOLD_PER_STEP=0.1 | 1/10Hz = 0.1s, matches dataset timestep |
| DT=0.02 | 5 servoJ updates per action step (smooth) |
| LOOKAHEAD=0.1 | Balances smoothness vs responsiveness |
| GAIN=300 | Stiffer than 150 (previous) — absolute targets can handle it since the model predicts positions close to current state |

For docker:

```
-e ACTION_MODE=absolute \
-e HORIZON_STEPS=15 \
-e HOLD_PER_STEP=0.1 \
-e DT=0.02 \
-e VEL=0.5 \
-e ACC=0.5 \
-e LOOKAHEAD=0.1 \
-e GAIN=300
```

### Result

This experiment finally created good results. The robot was moving to the blue object, picking it up and going to the cardboard box to drop it. The only issue being that in the box, it doesnt want to drop it. But testing for generalization, it went to the box even if it was on the other side than in the training dataset (it was not moved in training dataset). Also to check prompt understanding it was asked to pick up the blue screwdriver, not block which was in the training set. And it did it.

---

## Experiment: ur5_sixth_1 (busthetable_6, absolute actions)

> **Dataset**: LPSlvlv/ur5_busthetable_6 (10 episodes — busthetable_2 converted to absolute actions) | **Config**: pi05_ur5 | **Model**: pi0.5 (quantile norm, action_horizon=15) | **Steps**: 500 | **Actions**: absolute | **Task**: pick varied objects from table into bin (same task as busthetable_2, fixed action format)

Created LPSlvlv/ur5_busthetable_6 dataset which is LPSlvlv/ur5_busthetable_2 datsaset but fixed and converted to absolute
Based on this run experiment ur5_sixsth_1:

    TrainConfig(
        name="pi05_ur5",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, max_token_len=180),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_busthetable_6",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=500,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

## Experiment: ur5_fifth_2 (lerobot v3 format conversion)

> **Dataset**: LPSlvlv/ur5_pickandplace_5 (same data as ur5_fifth_1, converted from LeRobot v2.1 → v3.0 format) | **Config**: pi05_ur5 (identical to ur5_fifth_1) | **Purpose**: validate lerobot v3 upgrade — format change only, no data or config differences

LPSlvlv/ur5_pickandplace_5 converted from LeRobot v2.1 to v3.0 dataset format using
`python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=LPSlvlv/ur5_pickandplace_5`.
Same data content as ur5_fifth_1 — only the dataset storage format changed to be compatible
with lerobot>=0.4.2 (lerobot-v3-upgrade branch). Training config is identical.
Based on this run experiment ur5_fifth_2:
    TrainConfig(
        name="pi05_ur5",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, max_token_len=180),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_pickandplace_5",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=500,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

## BLUEBLOCK_BOX scaling datasets (2026-03-18)

Created a family of nested datasets for scaling experiments on the single task:
"pick up the blue block and place it in the cardboard box".

Source data: 20 replay episodes total from two recording sessions.
- raw_episodes_blueblock2/ — 10 episodes recorded 2026-03-18 (4 at 20Hz, 6 at 10Hz). All absolute forward-looking actions (action[i] = state[i+1]).
- raw_episodes_blueblock_2/ — 10 episodes recorded 2026-02-01 (all 10Hz). Originally stored delta actions; converted in-place to absolute using convert_raw_deltas_to_absolute.py before combining.

The 4 episodes at 20Hz were downsampled to 10Hz by taking every 2nd step with action correction: action[2k] = state[2k+2] to preserve forward-looking semantics.

All 20 episodes shuffled with deterministic seed=42. Datasets are nested prefixes — the 1-episode set is a strict subset of the 5-episode set, and so on. This ensures fair scaling comparisons: adding episodes never removes previously seen data.

Conversion script: ur5/scripts/combine_and_split_ur5_datasets.py

| Dataset | Episodes | Frames | Duration |
|---------|----------|--------|----------|
| LPSlvlv/ur5_blueblock_box_1 | 1 | 838 | 1.4 min |
| LPSlvlv/ur5_blueblock_box_5 | 5 | 4,067 | 6.8 min |
| LPSlvlv/ur5_blueblock_box_10 | 10 | 8,078 | 13.5 min |
| LPSlvlv/ur5_blueblock_box_15 | 15 | 11,760 | 19.6 min |
| LPSlvlv/ur5_blueblock_box_20 | 20 | 15,546 | 25.9 min |

All datasets: 10Hz, absolute actions, forward-looking (action[i] = state[i+1]), 256x256 RGB base + wrist images, 7-dim state/action (6 joints + 1 gripper). Task prompt: "pick up the blue block and place it in the cardboard box". Robot: UR5e with Robotiq Hand-E gripper. LeRobot v3 format. Pushed to HuggingFace Hub under LPSlvlv org.

Training configs in config.py (pi0_ur5, pi05_ur5, pi05_ur5_low_mem_finetune, etc.) can use these by setting repo_id to any of the above dataset names. use_delta_action_transform=True converts absolute actions to deltas during training and back to absolute during inference.

---

### Experiment: ur5_blueblock_box_10-1 (pi05_ur5, full finetuning) (21.03.26)

- **Config**: `pi05_ur5` (full finetuning, default LR schedule)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_10` (10 episodes)
- **Checkpoint**: `checkpoints/pi05_ur5/ur5_blueblock_box_10-1/270` (early stopped at step 270/500)
- **Result**: Confirms pipeline works end-to-end with lerobot v3 upgrade. Performance similar to previous 10-episode configs. Loss converged around step 75-100 then plateaued at ~0.005. Grad norm dropped to ~0.2 by step 75.
- **Issue found**: Default `warmup_steps=1000` meant the LR never reached peak during the 500-step run. Fixed by setting `warmup_steps=50`.

---

### Experiment: ur5_blueblock_box_10-2 (pi05_ur5_lora, LoRA finetuning) (21.03.26)

- **Config**: `pi05_ur5_lora`
- **Dataset**: `LPSlvlv/ur5_blueblock_box_10` (10 episodes)
- **Changes vs full finetuning**:
  - LoRA adapters: `gemma_2b_lora` (rank 16) + `gemma_300m_lora` (rank 32), all other params frozen
  - `batch_size=16` (was 32) — more gradient updates per epoch, implicit regularization
  - `weight_decay=1e-4` (was 1e-10) — L2 regularization to reduce overfitting
  - `warmup_steps=50` — LR reaches peak early in the 500-step run
  - `ema_decay=None` — disabled for LoRA (standard practice)
  - Early stopping: `patience=10`, `min_delta=1e-4`
- **Rationale**: Full finetuning a 3B+ param model on 10 episodes risks overfitting. LoRA reduces trainable parameters by >99%, combined with smaller batch size and weight decay for additional regularization.
- **Checkpoint**: `checkpoints/pi05_ur5_lora/ur5_blueblock_box_10-2/260` (early stopped at step 260/500)
- **Result**: Very poor performance. Robot produced random jagged motion, not moving towards the object at all. Significantly worse than the full finetuning baseline (ur5_blueblock_box_10-1) which could reach and grip the object. Multiple variables changed simultaneously (LoRA, batch_size, weight_decay), making it hard to isolate the cause. Most likely culprit is LoRA — with only 10 episodes, the low-rank adapters may not have enough capacity to learn the task-specific motion patterns, while the frozen base model weights dominate with generic (non-task-relevant) behavior. The higher weight_decay (1e-4) may also be penalizing the small LoRA updates too aggressively. Next steps: try full finetuning with the warmup_steps=50 fix (isolate LR improvement from LoRA), or try LoRA without the weight_decay/batch_size changes.

---

### Experiment: ur5_blueblock_box_10-3 (pi05_ur5, full finetuning with warmup fix) (21.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_10 (10 episodes) | **Config**: pi05_ur5 (full finetuning) | **Model**: pi0.5 (quantile norm, action_horizon=15) | **Checkpoint**: pi05_base | **Steps**: 500 | **LR**: warmup_steps=50 (fixed from 1000 default) | **Batch size**: 32 | **Early stopping**: patience=10, min_delta=1e-4

- **Config**: `pi05_ur5` (full finetuning, warmup_steps=50 fix applied)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_10` (10 episodes)
- **Change vs ur5_blueblock_box_10-1**: only `warmup_steps` reduced from 1000 to 50, so LR actually reaches peak (2.5e-5) during training. All other params identical.
- **Purpose**: isolate the effect of the LR warmup fix from the LoRA/batch_size/weight_decay changes that failed in ur5_blueblock_box_10-2.
- **Checkpoint**: `checkpoints/pi05_ur5/ur5_blueblock_box_10-3/270` (early stopped at step 270/500)
- **Result**: Looks better than the LoRA experiment (ur5_blueblock_box_10-2). Robot motion is more purposeful. However, still possibly overfitting — early stopped at the same step (270) as ur5_blueblock_box_10-1 despite the warmup fix. May need more data or fewer training steps to improve further.

Config used:

    TrainConfig(
        name="pi05_ur5",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, max_token_len=180),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_blueblock_box_10",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=50),
        num_train_steps=500,
        early_stop_patience=10,
        early_stop_min_delta=1e-4,
        log_interval=10,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

### Experiment: ur5_blueblock_box_10-4 (pi05_ur5, batch_size=16, earlier stopping)

> **Dataset**: LPSlvlv/ur5_blueblock_box_10 (10 episodes) | **Config**: pi05_ur5 (full finetuning) | **Model**: pi0.5 (quantile norm, action_horizon=15) | **Checkpoint**: pi05_base | **Steps**: 500 | **LR**: warmup_steps=50 | **Batch size**: 16 | **Early stopping**: patience=5, min_delta=1e-4

- **Config**: `pi05_ur5` (full finetuning)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_10` (10 episodes)
- **Changes vs ur5_blueblock_box_10-3**:
  - `batch_size=16` (was 32) — doubles gradient updates per epoch, adds implicit regularization via gradient noise
  - `early_stop_patience=5` (was 10) — stops after 50 steps without improvement instead of 100, to catch overfitting earlier
- **Purpose**: address possible overfitting seen in ur5_blueblock_box_10-3 (early stopped at same step 270 as -1). Smaller batch + earlier stopping should produce a less overfit checkpoint.
- **Result**: TBD

Config used:

    TrainConfig(
        name="pi05_ur5",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, max_token_len=180),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_blueblock_box_10",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=50),
        batch_size=16,
        num_train_steps=500,
        early_stop_patience=5,
        early_stop_min_delta=1e-4,
        log_interval=10,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

### Experiment: ur5_blueblock_box_20-1 (pi05_ur5, 20 episodes)

> **Dataset**: LPSlvlv/ur5_blueblock_box_20 (20 episodes, 15,546 frames, 25.9 min) | **Config**: pi05_ur5 (full finetuning) | **Model**: pi0.5 (quantile norm, action_horizon=15) | **Checkpoint**: pi05_base | **Steps**: 500 | **LR**: warmup_steps=50 | **Batch size**: 16 | **Early stopping**: patience=5, min_delta=1e-4

- **Config**: `pi05_ur5` (full finetuning)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_20` (20 episodes, 15,546 frames — doubled from 10 episodes)
- **Change vs ur5_blueblock_box_10-4**: dataset doubled from 10 to 20 episodes. All training params identical.
- **Purpose**: test whether more data addresses the overfitting issue. 20 episodes = ~971 steps/epoch at batch_size=16, so 500 steps is less than 1 full epoch — should reduce overfitting significantly.
- **Checkpoint**: `checkpoints/pi05_ur5/ur5_blueblock_box_20-1/130` (early stopped at step 130/500)
- **Result**: Loss dropped to ~0.003 by step 80, then rose back to ~0.005 by step 130 — classic overfitting. Early stopping caught it. peak_lr=2.5e-5 was too aggressive, causing the model to overshoot after initial convergence. Only saw ~13% of one epoch before stopping.

Config used:

    TrainConfig(
        name="pi05_ur5",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, max_token_len=180),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_blueblock_box_20",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=50),
        batch_size=16,
        num_train_steps=500,
        early_stop_patience=5,
        early_stop_min_delta=1e-4,
        log_interval=10,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

### Experiment: ur5_blueblock_box_20-2 (pi05_ur5, lower LR) (22.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_20 (20 episodes) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 1000 | **LR**: peak_lr=1e-5, warmup_steps=50, decay_steps=30000 (default) | **Batch size**: 32 | **Early stopping**: patience=10

- **Config**: `pi05_ur5` (full finetuning)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_20` (20 episodes)
- **Changes vs ur5_blueblock_box_20-1**: peak_lr reduced from 2.5e-5 to 1e-5, batch_size increased from 16 to 32, num_train_steps increased from 500 to 1000, early_stop_patience increased from 5 to 10.
- **Checkpoint**: `checkpoints/pi05_ur5/ur5_blueblock_box_20-2/250` (early stopped at step 250/1000)
- **Result**: Best curve so far. Smooth descent from 0.015 to ~0.002-0.003. No loss overshoot. However, loss showed oscillations after step 50 and plateaued rather than continuing to decrease. Analysis: with decay_steps=30000, the cosine decay barely starts during 250 steps of training — the LR was essentially flat at peak (1e-5) for the entire run after warmup. This explains the oscillations: constant LR means constant update magnitude even as the model approaches convergence. Also, batch_size=32 with 15,546 frames means each batch is only 0.2% of data, adding natural variance. Model only saw ~51% of one epoch (250/486 steps per epoch) before early stopping.

---

### Experiment: ur5_blueblock_box_20-3 (pi05_ur5, cosine decay fix) (23.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_20 (20 episodes) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 1000 | **LR**: peak_lr=1e-5, warmup_steps=50, decay_steps=500 | **Batch size**: 64 | **Early stopping**: patience=10

- **Config**: `pi05_ur5` (full finetuning)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_20` (20 episodes)
- **Changes vs ur5_blueblock_box_20-2**:
  - `decay_steps=500` (was 30000) — the cosine decay now actually reduces LR during training. LR goes from 1e-5 (peak at step 50) down to 2.5e-6 by step 550. This should reduce oscillations in later training steps as updates get smaller near convergence.
  - `batch_size=64` (was 32) — smoother gradients, less per-batch variance. One epoch = ~243 steps, so 1000 steps = ~4 full epochs through the data.
- **Purpose**: address two issues from 20-2: (1) LR was flat at peak causing oscillations, (2) model only saw ~51% of data. With decay_steps=500 the LR properly decays, and with batch_size=64 + 1000 steps the model sees all data ~4 times.
- **Note**: batch_size=64 caused OOM on A100-80GB, was reduced to 32 mid-run.
- **Checkpoint**: `checkpoints/pi05_ur5/ur5_blueblock_box_20-3/430` (ran full 430 steps with batch_size=32)
- **Result**: Smoothest loss curve yet — steady descent from ~0.015 to ~0.003 by step 100, then plateau. No oscillations thanks to cosine decay. However, clear overfitting signs after step ~100-150: param_norm climbed steadily (1802.388 → 1802.396), grad_norm dropped to ~0.05 (near zero useful learning signal). The model memorized training data rather than continuing to learn. With batch_size=32 and 430 steps, model saw ~56% of one epoch (430/~486 steps/epoch). The cosine decay properly reduced LR but couldn't prevent overfitting — the model simply converged too fast on this dataset. Sweet spot was likely around step 100-150 before param_norm started climbing.

---

### Experiment: ur5_blueblock_box_20-4 (pi05_ur5, shorter training to prevent overfitting) (23.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_20 (20 episodes) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 250 | **LR**: peak_lr=1e-5, warmup_steps=50, decay_steps=250 | **Batch size**: 64 | **Early stopping**: patience=10

- **Config**: `pi05_ur5` (full finetuning)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_20` (20 episodes)
- **Changes vs ur5_blueblock_box_20-3**:
  - `num_train_steps=250` (was 1000) — stop before overfitting kicks in. Experiment 20-3 showed loss plateaued by step ~100 and param_norm climbed after, so 250 steps captures the useful learning window while the cosine decay brings LR down to minimum.
  - `decay_steps=250` (was 500) — match decay to training length so LR completes full cosine schedule. LR goes from 1e-5 (peak at step 50) down to 2.5e-6 by step 300.
  - `batch_size=64` (retry, was reduced to 32 in 20-3 due to OOM) — smoother gradients. One epoch = ~243 steps, so 250 steps ≈ 1 full epoch through the data.
- **Purpose**: address overfitting observed in 20-3. Research into OpenPI's training methodology (PI paper, community) confirms that for small datasets (10-20 episodes), short training (300-500 steps) is recommended. The PI team relies on training loss early stopping + real robot evaluation rather than validation loss (which LeRobot research showed is not predictive of robot success rate for behavior cloning). By limiting to 250 steps (~1 epoch), the model sees all data once without memorizing it.
- **Result**: TBD

Config used:

    TrainConfig(
        name="pi05_ur5",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, max_token_len=180),
        data=LeRobotUR5DataConfig(
            repo_id="LPSlvlv/ur5_blueblock_box_20",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=50, peak_lr=1e-5, decay_steps=250),
        batch_size=64,
        num_train_steps=250,
        early_stop_patience=10,
        early_stop_min_delta=1e-4,
        log_interval=10,
        policy_metadata={"reset_pose": [-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0]},
    ),

---

### Experiment: ur5_blueblock_box_20-5 (pi05_ur5, longer training) (23.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_20 (20 episodes) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 900 | **LR**: peak_lr=1e-5, warmup_steps=50, decay_steps=250 | **Batch size**: 64 | **Early stopping**: patience=20, min_delta=1e-5

- **Config**: `pi05_ur5` (full finetuning)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_20` (20 episodes)
- **Changes vs ur5_blueblock_box_20-4**:
  - `num_train_steps=900` (was 250) — train longer, ~3.7 epochs through data. Previous experiments showed the model approaches correctly but gripper never closes consistently; more training may strengthen the gripper close signal.
  - `early_stop_patience=20` (was 10) — more patience before stopping, allow the model to explore longer plateaus.
  - `early_stop_min_delta=1e-5` (was 1e-4) — detect smaller improvements, keep training through subtle loss decreases.
  - `decay_steps=250` unchanged — LR decays fully by step 300, then stays at minimum (2.5e-6) for remaining ~600 steps. This means most training happens at very low LR, which may help with fine-grained gripper learning without destabilizing joint actions.
- **Purpose**: investigate whether longer training improves gripper close consistency. Experiment 20-4 (250 steps, 1 epoch) showed the model approaches correctly but outputs weak/inconsistent gripper close signals (~0.4-0.5 instead of 1.0). More epochs may strengthen the bimodal gripper signal. Also testing with GRIPPER_THRESHOLD=0 (disabled) in bridge to let raw model output through.
- **Result**: No significant change. Model still approaches correctly but gripper close signal remains weak/inconsistent. Longer training did not improve gripper transitions — the fundamental issue is the binary gripper data, not training duration.

Config used (same as 20-4 except training params):

    num_train_steps=900, early_stop_patience=20, early_stop_min_delta=1e-5
    (all other params identical to 20-4)

---

### Experiment: ur5_blueblock_box_20-6 (pi05_ur5, smooth gripper dataset) (24.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_20_smooth (20 episodes, smooth gripper ramps) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 900 | **LR**: peak_lr=1e-5, warmup_steps=50, decay_steps=250 | **Batch size**: 64 | **Early stopping**: patience=20, min_delta=1e-5

- **Config**: `pi05_ur5` (full finetuning, CLI override: `--data.repo-id=LPSlvlv/ur5_blueblock_box_20_smooth`)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_20_smooth` (20 episodes, smooth gripper transitions)
- **Changes vs ur5_blueblock_box_20-5**:
  - **New dataset with smooth gripper ramps** — instead of instant binary 0→1 jumps, gripper transitions are linear ramps over 5 frames (0→0.2→0.4→0.6→0.8→1.0). This matches how DROID/ALOHA/LIBERO datasets record actual gripper position feedback.
  - Training params unchanged from 20-5.
- **Purpose**: fix the fundamental gripper training signal problem. Analysis showed:
  - Original dataset: 99.7% of frames have action[6]=state[6] (echo). Only 40 single-frame transitions.
  - Smooth dataset: 120 intermediate ramp frames + with action_horizon=15 overlap = ~3,000 transition-aware training samples (5x more than before).
  - Official OpenPI datasets (DROID, ALOHA, LIBERO) all use continuous gripper feedback, not binary.
- **Bridge changes for this test**: GRIPPER_THRESHOLD=0.005 (very low, lets weak signals through), state[6] uses commanded position (matches training binary/ramp format).
- **Result**: Model does not grip the object at all. Worse than the binary dataset — the smooth gripper ramps combined with the 20-episode dataset (which includes inconsistent newer episodes) produced no gripping behavior. The 20-episode mixed dataset appears to be the problem: the newer 10 episodes (2026-03-18) have different approach angles than the original 10 (2026-02-01), causing the model to average trajectories and hit the object with the gripper base instead of the fingers.

---

### Experiment: ur5_blueblock_box_20_smooth-1 (pi05_ur5, 20 episodes smooth, default params) (25.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_20_smooth (20 episodes, smooth gripper ramps) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 500 | **LR**: default (warmup_steps=1000, effectively low LR) | **Batch size**: 32 (default) | **Early stopping**: disabled

- **Config**: `pi05_ur5` (full finetuning, CLI override: `--data.repo-id=LPSlvlv/ur5_blueblock_box_20_smooth`)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_20_smooth` (20 episodes, smooth gripper ramps)
- **Changes vs 20-6**: reverted to all default training params (same as og_smooth-3's final config). warmup_steps=1000 (was 50), batch_size=32 (was 64), no early stopping (was patience=20), steps=500 (was 900).
- **Purpose**: test whether the default training params that worked well for og_smooth-3 (10 episodes) also improve the 20-episode smooth dataset. 20-6 failed with aggressive hyperparams; the slow warmup schedule may help the model handle the mixed approach angles.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi05_ur5 \
      --data.repo-id=LPSlvlv/ur5_blueblock_box_20_smooth \
      --exp-name=ur5_blueblock_box_20_smooth-1 \
      --overwrite

---

### Experiment: ur5_blueblock_box_og_smooth-1 (pi05_ur5, original 10 episodes + smooth gripper) (24.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_og_smooth (10 original episodes, smooth gripper ramps) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 300 | **LR**: peak_lr=1e-5, warmup_steps=50, decay_steps=250 | **Batch size**: 64 | **Early stopping**: patience=20, min_delta=1e-5

- **Config**: `pi05_ur5` (full finetuning, CLI override: `--data.repo-id=LPSlvlv/ur5_blueblock_box_og_smooth`)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_og_smooth` (10 original episodes from 2026-02-01 only, smooth gripper ramps)
- **Changes vs ur5_blueblock_box_20-6**:
  - **Original 10 episodes only** — dropped the newer 10 episodes (2026-03-18) that had inconsistent approach angles. The original 10 episodes had better results in earlier experiments.
  - `num_train_steps=300` (was 900) — ~2.5 epochs with 10 episodes at batch_size=64 (~120 steps/epoch).
  - Smooth gripper ramps retained (5-frame linear ramp per transition).
- **Purpose**: isolate whether the problem is the mixed dataset or the smooth gripper change. The original 10-episode dataset produced the best gripping behavior in earlier experiments. Combining with smooth gripper should give better transition signal without the trajectory averaging problem from inconsistent episodes.
- **Result**: No gripping. batch_size=64 + peak_lr=1e-5 + fast decay gave ~10x weaker gradient signal per gripper transition frame compared to the earlier working config (10-4: batch_size=16, peak_lr=2.5e-5). The transition frames (0.26% of data) were averaged out in the large batches.

---

### Experiment: ur5_blueblock_box_og_smooth-2 (pi05_ur5, reverted to working config) (24.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_og_smooth (10 original episodes, smooth gripper ramps) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 500 | **LR**: peak_lr=2.5e-5, warmup_steps=50, decay_steps=30000 (default) | **Batch size**: 16 | **Early stopping**: patience=5, min_delta=1e-4

- **Config**: `pi05_ur5` (full finetuning, CLI override: `--data.repo-id=LPSlvlv/ur5_blueblock_box_og_smooth`)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_og_smooth` (10 original episodes, smooth gripper ramps)
- **Changes vs og_smooth-1**:
  - `batch_size=16` (was 64) — 4x more gradient influence per transition frame
  - `peak_lr=2.5e-5` (was 1e-5) — 2.5x stronger updates, default decay_steps=30000 (LR stays near peak)
  - `num_train_steps=500` (was 300) — ~1 epoch at batch_size=16 (~474 steps/epoch)
  - `early_stop_patience=5` (was 20)
- **Purpose**: match the config from experiment 10-4 (which produced gripping behavior) but with the smooth gripper dataset. The only variable changed vs 10-4 is the dataset (smooth vs binary gripper).
- **Bridge**: GRIPPER_THRESHOLD=0.1, actual gripper position feedback, debug logging enabled.
- **Result**: Gripper is activated more than previous experiments — the smooth gripper ramps appear to be helping with transition signal. However, movements are a little bouncy and less precise than the binary gripper experiments (10-3, 10-4). The higher peak_lr=2.5e-5 with default decay (LR stays near peak) may be causing the jittery motion. The smooth gripper dataset changes the action distribution which the model may need different hyperparameters to learn well.

---

### Experiment: ur5_blueblock_box_og_smooth-3 (pi05_ur5, more patience) (24.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_og_smooth (10 original episodes, smooth gripper ramps) | **Config**: pi05_ur5 (full finetuning) | **Steps**: 500 | **LR**: peak_lr=2.5e-5, warmup_steps=50, decay_steps=30000 (default) | **Batch size**: 16 | **Early stopping**: patience=20, min_delta=1e-4

- **Config**: `pi05_ur5` (full finetuning, CLI override: `--data.repo-id=LPSlvlv/ur5_blueblock_box_og_smooth`)
- **Dataset**: `LPSlvlv/ur5_blueblock_box_og_smooth` (10 original episodes, smooth gripper ramps)
- **Changes vs og_smooth-2**:
  - `early_stop_patience=20` (was 5) — allow the model to train longer past loss plateaus. og_smooth-2 showed improved gripper activation but bouncy movements; more training may smooth out the motion.
- **Purpose**: same config as og_smooth-2 but with more patience to see if additional training steps improve precision while retaining the gripper activation gains.
- **Result**: OK results. Checkpoint `pi05_ur5/ur5_blueblock_box_og_smooth-2/120` (early stopped at step 120). Analysis revealed that ur5_fifth_1 (the experiment that worked well) used all default training params: warmup_steps=1000 (LR never reached peak during 500 steps = effectively low LR), batch_size=32, early_stop disabled. Reverted config to match those defaults. The smooth gripper dataset combined with the original slow-warmup training schedule produced the best results so far — gripper activates and movements are acceptable. Experiments concluded.

Launch command:

    docker run --rm -it --gpus=all --network=host --ipc=host --device=/dev/bus/usb:/dev/bus/usb $RUN_DEVICES --group-add video -v "$PWD":/app -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e QT_QPA_PLATFORM=xcb -e QT_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins -e QT_QPA_PLATFORM_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins/platforms --name openpi-robot -e RS_BASE=137322074310 -e RS_WRIST=137322075008 -e PROMPT="Pick up the blue block and place it in the cardboard box" -e HOLD_PER_STEP=0.15 -e HORIZON_STEPS=15 -e MAX_STEP_DEG=3.0 -e DT=0.02 -e VEL=0.5 -e ACC=0.5 -e LOOKAHEAD=0.1 -e GAIN=300 openpi_robot

---

## Evaluation: Dataset Size Scaling

Systematic comparison of robot performance across dataset sizes for the task:
**"Pick up the blue block and place it in the cardboard box"**

### Scoring System

Each trial is scored across 4 sequential stages. A stage scores **1** if achieved, **0** otherwise.
Normalized score = sum of stages × 0.25 (range: 0.00 – 1.00).
Time is measured in seconds from trial start to stage completion (blank if stage not reached).

| Stage | Criteria | Points |
|-------|----------|--------|
| **Reach Object** | End effector within 10 cm of object | 0.25 |
| **Pick Up** | Object successfully gripped and lifted | 0.25 |
| **Reach Drop-off** | End effector within 10 cm of drop location | 0.25 |
| **Release** | Object released at target location | 0.25 |

---

### 1 Episode — `LPSlvlv/ur5_blueblock_box_1`

> **Config**: `pi05_ur5` | **Experiment**: `ur5_first`

Qualitative results only (no structured trials). Robot was mostly just rotating in the general direction of the object — no meaningful reaching or grasping behavior. Useful as a sanity check that the configuration and learning pipeline are correct — just needs more data.

---

### 10 Episodes — `LPSlvlv/ur5_blueblock_box_og_smooth` (og_smooth-3)

> **Config**: `pi05_ur5` | **Checkpoint**: `pi05_ur5/ur5_blueblock_box_og_smooth-2/120`

| Try | Reach | Pick | Drop | Release | Score | t_reach | t_pick | t_drop | t_release | Notes |
|-----|-------|------|------|---------|-------|---------|--------|--------|-----------|-------|
| 1   | 1     |      |      |         | 0.25  | 2:03    |        |        |           |       |
| 2   | 1     |      |      |         | 0.25  | 2:10    |        |        |           | Crashed into object/ground, triggered safety stop |
| 3   | 1     | 1    |      |         | 0.50  | 2:06    | 0:40   |        |           | Dropped object midway between pick and drop |
| 4   | 1     |      |      |         | 0.25  | 2:18    |        |        |           | Crashed into object/ground, triggered safety stop |
| 5   | 1     | 1    | 1    | 1       | 1.00  | 2:30    | 1:59   | 1:34   | 2:20      | Full success |
| 6   | 1     |      |      |         | 0.25  | 1:55    |        |        |           | Crashed into object/ground, triggered safety stop |
| 7   | 1     | 1    | 1    | 0       | 0.75  | 2:20    | 0:20   | 1:49   |           | Reached drop-off but failed to release |
| 8   | 1     |      |      |         | 0.25  | 2:25    |        |        |           | Crashed into object/ground, triggered safety stop |
| 9   | 1     | 1    | 1    | 1       | 1.00  | 1:39    | 2:06   | 1:49   | 2:29      | Full success |
| 10  | 1     |      |      |         | 0.25  | 2:08    |        |        |           | Crashed into object/ground, triggered safety stop |
| 11  | 1     | 1    | 1    | 0       | 0.75  | 2:23    | 0:17   | 1:29   |           | Reached drop-off but failed to release |
| 12  | 1     |      |      |         | 0.25  | 2:30    |        |        |           | Crashed into object/ground, triggered safety stop |
| 13  | 1     |      |      |         | 0.25  | 2:00    |        |        |           | Crashed into object/ground, triggered safety stop |
| 14  | 1     | 1    | 1    | 0       | 0.75  | 2:40    | 2:17   | 1:21   |           | Reached drop-off but failed to release |
| 15  | 1     | 1    | 1    | 0       | 0.75  | 2:06    | 2:33   | 1:28   |           | Reached drop-off but failed to release |
| 16  | 1     |      |      |         | 0.25  | 2:15    |        |        |           | Crashed into object/ground, triggered safety stop |
| 17  | 1     | 1    | 1    | 0       | 0.75  | 2:06    | 1:27   | 1:21   |           | Reached drop-off but failed to release |
| 18  | 1     |      |      |         | 0.25  | 2:22    |        |        |           | Crashed into object/ground, triggered safety stop |
| 19  | 1     |      |      |         | 0.25  | 2:12    |        |        |           | Crashed into object/ground, triggered safety stop |
| 20  | 1     |      |      |         | 0.25  | 2:14    |        |        |           | Drove into the object and triggered protective stop |
| **Avg** | **20/20** | **8/20** | **7/20** | **2/20** | **0.463 ± 0.277** | **2:13** | **1:27** | **1:33** | **2:24** | Means: times averaged over trials that reached each stage (n = 20, 8, 7, 2). |

**Conclusion — 10 Episodes.** Reach is perfectly reliable (20/20, mean 2:13) — the policy consistently navigates to the object. However, **pick is now the clear bottleneck at 40% (8/20)**: 12 out of 20 trials failed at the grasp stage, with the dominant failure mode being the arm crashing into the object or ground and triggering a safety stop (trials 10–20), indicating the policy has not learned safe approach trajectories. Among the 8 successful picks, timing reveals two distinct modes: **fast picks** (0:17–0:40 in trials 2, 4, 6) where the grasp succeeded on the first attempt, and **slow picks** (1:27–2:33 in trials 3, 5, 7–9) indicating one or more failed grasp attempts before the object was secured. The transport stage remains the most consistent phase: once the object is held, the arm reliably reaches the drop-off in 1:21–1:49 (mean 1:33, range of only 28 s). Release is also poor (2/20, 10%) — in 5 out of 7 trials that reached the box the gripper failed to open, suggesting the policy has not learned a robust release behaviour. Overall, 10 demonstrations produce a policy that can reliably reach the object but fails more often than not at grasping (60% failure rate due to unsafe approach angles) and almost never completes the full task; both the pick and release stages need significantly more training signal.

---

### 20 Episodes — `LPSlvlv/ur5_blueblock_box_20_smooth` (20_smooth-2, 20_smooth-3)

> **Config**: `pi05_ur5` (og_smooth-3 params: batch_size=16, warmup_steps=50, early_stop_patience=20)

**20_smooth-2** — Checkpoint `pi05_ur5/ur5_blueblock_box_20_smooth-2/149`. Trained with default params (batch_size=32, warmup_steps=1000, num_train_steps=150). With warmup_steps=1000 and only 150 steps, the LR never reached peak — model was undertrained.

**20_smooth-3** — Checkpoint `pi05_ur5/ur5_blueblock_box_20_smooth-3/199`. Trained with og_smooth-3 params (batch_size=16, warmup_steps=50, num_train_steps=200, early_stop_patience=20).

**Result: FAIL.** Both 20-episode checkpoints hit safety limits — the robot drives too deep into the object and hits the ground, triggering protective stops. The 20-episode dataset introduces inconsistent approach angles that cause the policy to learn overly aggressive downward trajectories. The 10-episode original dataset (og_smooth-3) remains the best performing model.

---

### Experiment: ur5_blueblock_box_v2_20-1 (pi05_ur5, 20 new episodes, non-smooth) (30.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_20 (20 new episodes) | **Config**: pi05_ur5 | **Steps**: 110

- **Result**: FAIL. Good arm movement but model outputs gripper ~0.002 constantly — no gripper activation at all. The non-smooth (binary) gripper data provides even less transition signal than smooth ramps.

---

### Experiment: ur5_blueblock_box_v2_20_smooth-1 (pi05_ur5, 20 new episodes, smooth) (30.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_20_smooth (20 new episodes, smooth gripper) | **Config**: pi05_ur5 | **Steps**: 130

- **Result**: FAIL. Same as v2_20-1 — good movement, no gripper activation. Gripper output stays near 0.0.

---

### Experiment: ur5_blueblock_box_v2_30-1 (pi05_ur5, 30 episodes) (30.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_30 (30 episodes: new 20 + OG 10) | **Config**: pi05_ur5 | **Steps**: 199

- **Result**: FAIL. Good movement, no gripper activation. Adding the OG 10 episodes to the new 20 did not help recover gripper behavior.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-1 (pi05_ur5, 40 episodes, smooth, quantile norm) (30.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi05_ur5 (quantile normalization) | **Steps**: 199

- **Result**: FAIL. Good movement, no gripper activation. Gripper output ~0.002 throughout inference.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-2 (pi05_ur5, 40 episodes, smooth, quantile norm, long training) (31.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi05_ur5 (quantile normalization) | **Steps**: 9999

- **Result**: FAIL. Trained for 10k steps. Good movement but gripper output stays at ~0.002 — more training did not help. The gripper transition signal (~1% of frames) is too diluted for the model to learn.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-3 (pi05_ur5, 40 episodes, smooth, z-score norm) (31.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi05_ur5 (z-score normalization — disabled quantile norm per openpi#763) | **Steps**: 239

- **Changes**: Set `use_quantile_norm=False` in LeRobotUR5DataConfig to use z-score normalization instead of quantile. Community reports (openpi#763) indicate quantile norm is unreliable on small fine-tuning datasets.
- **Result**: FAIL. Disabling quantile normalization did not fix gripper activation. Model still outputs gripper ~0.0. The gripper problem appears to be fundamental to the data signal strength, not the normalization method.

---

### Analysis: Why gripper fails on larger datasets (31.03.26)

All V2 dataset experiments (20-40 episodes) show the same pattern: good arm movement but zero gripper activation. The model outputs gripper values of ~0.002 (effectively zero) throughout inference.

**Root cause**: Gripper transitions are ~1% of training frames. The model learns to always predict "keep gripper unchanged" because this is correct 99% of the time. The action chunking (action_horizon=15) provides some signal amplification but is insufficient.

**Key contrast with og_smooth-3** (which DOES grip): og_smooth-3 used only 10 consistent episodes with ~7,500 frames. The transition-to-total ratio is the same, but the simpler trajectory distribution (consistent approach angles) allowed the model to learn the full pick-and-place behavior including gripper transitions. With 40 varied episodes, the model spends its capacity learning diverse arm trajectories and the gripper signal is lost.

**Next steps**: Try pi0 model (instead of pi0.5) — community evidence suggests pi0 fine-tunes better on small datasets (openpi#692). Pi0 uses z-score normalization by default and has action_horizon=50 (vs 15), giving ~3x more gripper signal per training sample.

---

## Alternative Models (2026-04-01)

### Experiment: ur5_blueblock_box_v2_40_smooth-4 (pi0_ur5, pi0 instead of pi0.5) (31.03.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5 (pi0, NOT pi0.5) | **Steps**: 390 (early stopped from 500) | **action_horizon**: 50 | **Norm**: z-score (pi0 default)

- **Config**: `pi0_ur5` — switched from pi0.5 to pi0 based on community evidence that pi0 fine-tunes better on small datasets (openpi#692). Pi0 uses z-score normalization by default and action_horizon=50 (vs 15 for pi0.5), providing ~3x more gripper signal per training sample.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi0_ur5 --exp-name=ur5_blueblock_box_v2_40_smooth-4 --overwrite

---

### Experiment: ur5_blueblock_box_v2_40-1 (pi0_fast_ur5, FAST model, BROKEN normalization) (01.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40 (40 episodes, binary gripper — non-smooth) | **Config**: pi0_fast_ur5 | **Steps**: 499 | **action_horizon**: 10 | **Model**: Pi0 FAST

- **Config**: `pi0_fast_ur5` — Pi0 FAST uses discrete action tokens with cross-entropy loss instead of continuous flow matching with MSE.
- **Bug**: Trained with `use_quantile_norm=False` (z-score), but FAST tokenizer discretizes state into 256 bins in [-1, 1] range which REQUIRES quantile normalization. Z-score values go outside [-1, 1], causing the tokenizer to clip all state values to bin 0 or 255. Model saw garbage state input during training.
- **Additional bug**: Unnormalize didn't handle 32-dim norm stats from pi0_fast_base with 7-dim UR5 actions, causing ValueError at inference. Fixed by slicing stats to match action dims.
- **Result**: FAIL — robot doesn't move, gripper outputs constant -0.006. Model learned nothing useful due to broken normalization.

---

### Experiment: ur5_blueblock_box_v2_40-1 (pi0_fast_ur5, FAST model, fixed normalization but wrong norm stats) (02.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40 (40 episodes, binary gripper — non-smooth) | **Config**: pi0_fast_ur5 | **Steps**: 249 | **action_horizon**: 10 | **Model**: Pi0 FAST | **Norm**: quantile (fixed, but using pretrained stats)

- **Config**: `pi0_fast_ur5` — retrained after fixing normalization: FAST now correctly uses quantile normalization (required by tokenizer), while pi0/pi05 use z-score.
- **Fixes applied**:
  - `use_quantile_norm=True` for PI0_FAST model type (was False for all UR5)
  - Unnormalize slices 32-dim norm stats to match 7-dim actions (pi0_fast_base stores padded stats)
- **Bug discovered during inference**: Config still loaded pretrained norm stats from `gs://openpi-assets/checkpoints/pi0_fast_base/assets/ur5e/norm_stats.json`. These pretrained quantile ranges (from OXE UR5e data) don't cover this robot's joint angles. Joint 0: pretrained q01=+1.03 but actual value is -1.57 → quantile normalization maps to -2.88, completely outside [-1, 1] range. FAST tokenizer discretizes state into 256 bins in [-1, 1], so joints 0 and 2 clamp to bin 0, destroying state information.
- **Result**: FAIL — robot moved but incorrectly (straight down). FAST BPE decode errors: `cannot reshape array of size 64 into shape (7)` because model generated wrong tokens due to corrupted state input. Fallback produced gripper=0.500 (quantile unnorm midpoint of zeros). When decode succeeded, gripper output was -0.006 (same near-zero as pi0.5).

---

### Experiment: ur5_blueblock_box_v2_40-2 (pi0_fast_ur5, FAST model, dataset-specific norm stats) (02.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40 (40 episodes, binary gripper — non-smooth) | **Config**: pi0_fast_ur5 | **Steps**: 250 | **action_horizon**: 10 | **Model**: Pi0 FAST | **Norm**: quantile (dataset-specific)

- **Config**: `pi0_fast_ur5` — switched from pretrained to dataset-specific norm stats. Config changed `AssetsConfig(assets_dir="gs://...", asset_id="ur5e")` → `AssetsConfig(asset_id="ur5e")` to use locally computed stats.
- **Norm stats**: Computed fresh from training dataset via `compute_norm_stats.py`. New stats have state q01/q99 ranges that cover this robot's actual joint angles (e.g. joint 0: q01=-1.85, q99=-0.88, covering reset position -1.57). All joints normalize within [-1, 1] at inference.
- **Note**: This is also a known community issue — openpi#912 and #692 report the same gripper failure pattern on pi0.5 single-arm fine-tuning. Pi0_FAST with cross-entropy loss may handle the binary gripper better than flow matching MSE.
- **Result**: FAIL — 250 steps produced near-zero actions (same fallback midpoint as before). Model learned correct token structure (no more reshape errors) but token VALUES still encoded zero deltas. Insufficient training.

---

### Experiment: ur5_blueblock_box_v2_40-3 (pi0_fast_ur5, 5K steps) (02.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40 (40 episodes, binary gripper) | **Config**: pi0_fast_ur5 | **Steps**: 4350 (early stopped from 5000) | **action_horizon**: 10 | **Model**: Pi0 FAST | **Norm**: quantile (dataset-specific)

- **Config**: `pi0_fast_ur5` with 5K steps (up from 250). Same dataset-specific norm stats.
- **Progress**: No more FAST decode errors — model learned to generate correct token count (70 chars instead of 1023). Some joint movement appeared in later chunks (deltas 0.01-0.02 rad for joints 3-4).
- **But**: Most chunks still near-zero. Gripper constant at -0.006. Then FAST decode errors returned with `cannot reshape array of size 1023 into shape (7)` — model generating pretrained-length token sequences (1024 chars) instead of fine-tuned (70 chars). Token 571 repeated ~18 times (pretrained padding pattern).
- **Root cause analysis**: FAST tokenizer encodes gripper entangled with joints via DCT+BPE. Binary gripper = worst case for DCT compression (step function needs many high-frequency coefficients that get quantized away). Community confirms: openpi#585 (FAST gripper stays open), #782 (same reshape error), #843 (identical outputs). The FAST paper only validated on continuous action spaces.
- **Conclusion**: Pi0_FAST is architecturally unsuitable for binary gripper tasks. Cross-entropy on BPE tokens doesn't help when the gripper is entangled with joints in the DCT representation. Community consensus (openpi#692, #766, #414): pi0 (flow matching) outperforms pi0_fast and pi0.5 on single-arm grasping.
- **Result**: FAIL — abandoned pi0_fast approach.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-5 (pi0_ur5, flow matching with gripper loss weighting) (03.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5 | **Steps**: 2000 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Norm**: z-score (pretrained pi0_base) | **Custom**: 10× gripper loss weight

- **Config**: `pi0_ur5` — switched from pi0.5/pi0_fast to pi0 based on community evidence that pi0 fine-tunes best for single-arm grasping (openpi#692: pi0 ~100% success vs pi0.5 total failure; #414: pi0_base ~100% with 100 demos).
- **Key changes**:
  - **Per-dimension loss weighting**: Added `action_dim_weights=((6, 10.0),)` to Pi0Config. Flow matching loss `MSE(v_t, u_t)` now applies 10× weight to gripper (dim 6), giving it 62.5% of total loss signal (was 3.1%). Padded dims 7-31 get weight 0 to avoid dilution.
  - **Fixed LR schedule**: Previous pi0_ur5 had `warmup_steps=1000` (default) > `num_train_steps=500` — LR never reached peak! Now `warmup_steps=50`, `decay_steps=2000` for proper cosine annealing.
  - **batch_size=16**: Matches og_smooth-3 (best-performing config). Smaller batches give gripper transitions more gradient influence per update (og_smooth-1 with bs=64 failed: "~10x weaker gradient signal per gripper transition frame").
  - **Smooth dataset**: 5-frame linear ramps on gripper transitions provide more DCT-friendly gradient signal than binary jumps.
  - **save_interval=500**: Produces checkpoints at 500/1000/1500/1999 for real-world evaluation (PI team says real-world eval is the only reliable metric for behavior cloning).
- **Training analysis**: batch_size=16, ~30K frames → 1875 steps/epoch. 2000 steps ≈ 1.07 epochs. ~27 batches containing gripper transitions per epoch. With 10× weight, gripper contributes 62.5% of loss.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi0_ur5 --exp-name=ur5_blueblock_box_v2_40_smooth-5 --overwrite

**Result**: FAIL — good arm movement but gripper still outputs near-zero (-0.002 to 0.007). Loss trained smoothly from 0.18 → 0.02 over 2000 steps (no oscillation with LR=1e-5). But 5× gripper weight wasn't enough to overcome the data imbalance. The model perfectly predicts "maintain gripper" (98% of frames) and ignores rare transitions. Per-dimension weighting amplifies all gripper frames equally — doesn't change the relative proportion of transition vs maintain frames.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-6 (pi0_ur5, 10× gripper weight, LR=1e-5) (03.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5 | **Steps**: 2000 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Norm**: z-score (pretrained pi0_base) | **Custom**: 10× gripper loss weight, LR=1e-5

- **Config**: `pi0_ur5` — increased gripper weight from 5× to 10× while keeping LR=1e-5.
- **Rationale**: 5× trained smoothly but gripper didn't activate. Previous 10× with LR=2.5e-5 oscillated (CV=26%) — but that was the LR, not the weight. With LR=1e-5, variance is only 1.9× higher than the stable 5× run.
- **Analysis**: 90% of batches already contain ≥1 transition frame (13% of chunks overlap a transition due to action_horizon=50). Gripper gets 63% of total loss at 10×. The lower LR should keep training stable.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi0_ur5 --exp-name=ur5_blueblock_box_v2_40_smooth-6 --overwrite

---

### Experiment: ur5_blueblock_box_v2_40_smooth-7 (pi0_ur5, gripper transition oversampling) (03.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5 | **Steps**: 2000 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Norm**: z-score (pretrained pi0_base) | **Custom**: 5× gripper transition oversampling (no dim weighting)

- **Config**: `pi0_ur5` — replaced per-dimension loss weighting with per-FRAME oversampling.
- **Key insight**: dimension weighting (5× and 10×) amplifies ALL gripper frames equally — including the 87% where gripper is constant and already easy to predict. The model minimizes loss by perfecting the "maintain" prediction, never learning transitions.
- **New approach**: `gripper_oversample_factor=5.0` uses `WeightedRandomSampler` to sample transition frames 5× more often. Pre-scan (0.9s) identifies 4,000/29,966 frames (13.3%) whose 50-step action chunk contains a gripper transition. With 5× oversampling, transitions appear in 43.5% of training samples (up from 13.3%). The model sees "arm near object → close gripper" examples 3× more often per epoch.
- **No loss changes**: standard unweighted MSE flow matching loss. Only the data sampling distribution is modified.
- **Params**: batch_size=16, LR=1e-5, warmup=50, decay_steps=2000, early_stop disabled, save_interval=500.
- **Result**: FAIL — good arm movement but gripper still outputs near-zero (range -0.006 to 0.007 across all chunks). Training loss went from ~0.85 → ~0.78 over 2000 steps — most of this is the irreducible noise floor from 25 padded dimensions (78% of loss). The oversampling increased transition frame frequency in batches, but without `action_dim_weights` the gripper signal is only 1/32 = 3.1% of total loss, completely drowned by padded-dim noise.

Training command:

    uv run scripts/train.py pi0_ur5 --exp-name=ur5_blueblock_box_v2_40_smooth-7 --overwrite

---

### Analysis: Why all smooth-5/6/7 failed (03.04.26)

Deep codebase audit revealed that each experiment addressed ONE factor but never the combination:

| Experiment | Dim Weights | Oversampling | Steps | peak_lr | Result |
|---|---|---|---|---|---|
| smooth-5 | 10× gripper, pad=0 | none | 2000 (1 epoch) | 2.5e-5 | FAIL |
| smooth-6 | 10× gripper, pad=0 | none | 2000 (1 epoch) | 1e-5 | FAIL |
| smooth-7 | **none (all 32 uniform)** | 5× | 2000 (1 epoch) | 1e-5 | FAIL |

**Key findings from the audit:**

1. **smooth-7 loss dilution**: With `action_dim_weights=None`, the loss is `jnp.mean(sq_err, axis=-1)` over all 32 dims. 25 padded dims create an irreducible noise floor (~78% of total loss). Gripper is 1/32 = 3.1% of loss — even 5× oversampling can't overcome this. (smooth-5/6 had weights that zeroed padded dims.)

2. **1 epoch is insufficient**: 2000 steps × batch_size 16 = 32K samples ÷ 30K frames = 1.07 epochs. Reference configs (Libero, ALOHA) use 20K-30K steps. The model learns "always predict gripper=open" (correct for 87% of frames) and never gets enough gradient updates to learn the conditional transition behavior.

3. **No per-dimension visibility**: Only total loss was logged — impossible to know if gripper was improving. The loss dropping from 0.85→0.78 looked "good" but was entirely joints improving while gripper stayed flat.

4. **No other openpi config needs special gripper handling**: ALOHA, Libero, DROID configs use default `Pi0Config()` with no dim weights or oversampling — but they train for 20K-30K steps on much larger datasets with more transitions.

5. **No actual bugs found**: Gradients flow correctly, no frozen layers, no normalization bugs, DeltaActions correctly leaves gripper absolute, EMA (0.99) is standard. The issue is purely signal-to-noise ratio.

**Fix: combine ALL levers (never tried together):**
- `action_dim_weights=((6, 10.0),)` — zeros padded dims + 10× gripper weight
- `gripper_oversample_factor=5.0` — more transition frames per batch
- `num_train_steps=5000` — 2.7 epochs instead of 1
- `peak_lr=2.5e-5` — matches og_smooth-3 (only success used this LR)

**Diagnostic instrumentation added** (commit `0a736be`):
- Per-dim loss logged to wandb: `loss_joints_mean`, `loss_gripper`, `loss_padded_mean`
- Per-dim gradient norms: `gnorm_joints`, `gnorm_gripper`, `gnorm_padded`
- Batch transition percentage: `batch_transition_pct`
- New script `ur5/scripts/verify_data_pipeline.py` traces gripper through each transform stage

---

### Experiment: ur5_blueblock_box_v2_40_smooth-8 (pi0_ur5, combined dim weights + oversampling + diagnostics) (03.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5 | **Steps**: 5000 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Norm**: z-score (pretrained pi0_base) | **Custom**: 10× gripper dim weight + 5× oversampling + per-dim diagnostics

- **Config**: `pi0_ur5` — the untried combination of ALL known good practices.
- **Key changes vs smooth-7**:
  - `action_dim_weights=((6, 10.0),)` — zeros padded dims (no more noise floor), 10× weight on gripper (62.5% of total loss)
  - `gripper_oversample_factor=5.0` — kept from smooth-7 (43% of batches contain transitions)
  - `num_train_steps=5000` — 2.7 epochs (was 2000 = 1 epoch)
  - `peak_lr=2.5e-5` — matches og_smooth-3 (was 1e-5)
  - `warmup_steps=100` (was 50)
  - `decay_steps=5000` (was 2000)
- **Diagnostics**: Per-dim loss, gradient norms, and batch transition % logged to wandb for the first time. This will reveal whether gripper loss actually decreases during training.
- **Params**: batch_size=16, early_stop disabled, save_interval=500, ema_decay=0.99 (default).
- **Result**: TBD

Training command:

    uv run scripts/train.py pi0_ur5 \
      --exp-name=ur5_blueblock_box_v2_40_smooth-8 \
      --overwrite \
      --model.action-dim-weights '((6,10.0),)' \
      --num-train-steps 5000 \
      --lr-schedule.peak-lr 2.5e-5 \
      --lr-schedule.warmup-steps 100 \
      --lr-schedule.decay-steps 5000

**Result**: Model WORKS in offline evaluation — **gripper has learned visual conditioning**.

Diagnostic test (`ur5/scripts/test_gripper_diagnostics.py`) with real dataset images:

| Image | Gripper output (mean across 50 steps) | First 10 steps |
|-------|---------------------------------------|----------------|
| **Pre-grip** (frame 403, arm near object) | **0.86–0.93** (ramps 0→1) | Ramps to 1.0 within 5-10 steps |
| **Far** (frame 350, arm far from object) | **0.001–0.003** (stays open) | Flat near 0.0 |

The model correctly distinguishes "near object → close gripper" from "far from object → keep open." The 0.001–0.004 values seen on the robot match the "far" pattern — meaning **the arm never reached a position/visual context where the model would trigger gripper close**.

Pipeline verification:
- Data pipeline round-trip: error = 0.0 (PASS)
- DeltaActions leaves gripper unchanged: True (PASS)
- Norm stats match between og_smooth-3 and smooth-8: True (PASS)
- Gripper normalization: 0→-1.12, 1→+1.22 (correct range)

**Next investigation**: The model is trained correctly. The issue is on the robot side — either the arm trajectory doesn't bring the end effector close enough, or there's a visual domain gap between training images and live camera images.

---

### Analysis: Image Overfitting Discovery (04.04.26)

Hybrid inference test (swap images vs joints between dataset and robot observations) proved **100% image overfitting**:

| Test | Gripper output |
|------|---------------|
| Dataset images + dataset joints | **0.863** (close) |
| Robot images + dataset joints | **-0.003** (zero) |
| Dataset images + robot joints | **0.874** (close) |
| Robot images + robot joints | **-0.002** (zero) |

Joints have zero effect — it's entirely the images. The model memorized training pixel patterns and can't generalize to live camera images, even though brightness and resolution now match.

**Infrastructure fixes applied:**
- Fixed `RS_FPS`: was hardcoded 60 in bridge vs 30 in recording → now matches (30)
- Fixed `_process_bgr`: resize to 256×256 (matching training) instead of direct 224×224
- Added camera intrinsics/exposure logging on startup
- Added `RECORD_DIR` env var for saving inference observations as .npz files
- Added `RS_AUTO_EXPOSURE` and `RS_EXPOSURE` env vars for manual camera exposure control
- New scripts: `replay_inference_recording.py`, `compare_training_vs_inference_images.py`
- New eval: `eval_generalization.py` — tests each checkpoint on dataset, perturbed, and robot images

**Generalization eval tool** (`ur5/scripts/eval_generalization.py`): runs inference on dataset pre-grip images, brightness-perturbed images, and recorded robot images for each checkpoint step. Reports UNDERFIT/GOOD/OVERFIT per step, identifying the best generalizing checkpoint.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-9 (pi0_ur5, short training + generalization eval) (04.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5 | **Steps**: 200 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Custom**: 5× oversampling, save_interval=30, keep_period=30

- **Config**: `pi0_ur5` — short training (200 steps) with frequent checkpoints (every 30 steps) to find generalization sweet spot.
- **Params**: batch_size=16, peak_lr=2.5e-5, warmup=50, no action_dim_weights.

Generalization eval results:

| Step | Dataset(orig) | Dataset(perturb) | Robot | Overfit | Verdict |
|------|--------------|-----------------|-------|---------|---------|
| 30 | +0.37 | +0.52 | **+0.44** | -0.08 | **GOOD** |
| 60 | +0.46 | +0.49 | +0.21 | +0.25 | GOOD |
| 90 | +0.49 | +0.53 | +0.19 | +0.30 | GOOD |
| 120 | +0.40 | +0.50 | +0.11 | +0.29 | GOOD |
| 150 | +0.42 | +0.49 | +0.04 | +0.38 | OVERFIT |
| 180 | +0.47 | +0.47 | +0.01 | +0.46 | OVERFIT |
| 199 | +0.52 | +0.49 | +0.01 | +0.51 | OVERFIT |

**Best checkpoint: step 30** (robot gripper=0.44). The model overfits rapidly — by step 150 the robot gripper score is dead. Step 30 was tested on the robot: gripper DID activate (0.14–0.47 in some chunks) but arm trajectory was chaotic (too few training steps for joints).

**Result**: PARTIAL — gripper activates for the first time, but arm undertrained at step 30.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-10 (pi0_ur5_frozen_vision, frozen SigLIP + weight_decay) (04.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5_frozen_vision | **Steps**: 200 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Custom**: Frozen SigLIP vision encoder, weight_decay=1e-4, 5× oversampling, peak_lr=2.5e-5

- **Config**: `pi0_ur5_frozen_vision` — freezes the SigLIP vision encoder (23 params frozen, 27 trainable) to prevent image memorization. Only the Gemma LLM and action projection layers are trained.
- **Hypothesis**: Freezing SigLIP prevents the vision backbone from overfitting to training image pixels, extending the generalization window to later training steps where arm trajectories are better.

Generalization eval results:

| Step | Dataset(orig) | Dataset(perturb) | Robot | Overfit | Verdict |
|------|--------------|-----------------|-------|---------|---------|
| 30 | +0.08 | +0.31 | **+0.55** | -0.46 | UNDERFIT |
| 60 | +0.32 | +0.42 | +0.24 | +0.08 | **GOOD** |
| 90 | +0.50 | +0.47 | +0.18 | +0.32 | GOOD |
| 120 | +0.56 | +0.54 | +0.12 | +0.44 | GOOD |
| 150 | +0.53 | +0.55 | +0.02 | +0.50 | OVERFIT |
| 180 | +0.54 | +0.54 | +0.01 | +0.53 | OVERFIT |
| 199 | +0.57 | +0.55 | +0.01 | +0.57 | OVERFIT |

**Result**: Same overfitting pattern as unfrozen — SigLIP freeze didn't help significantly. The overfitting occurs in the **LLM layers** (how Gemma interprets vision features), not in SigLIP itself. Step 30 robot=0.55 has massive variance (±0.47), step 60 is more reliable (robot=0.24).

**Key insight**: The visual domain gap exists at the SigLIP feature level. Pretrained SigLIP encodes training images and robot images into different enough features that the LLM can distinguish them and overfit.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-11 (pi0_ur5_frozen_vision, lower LR=5e-6) (04.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5_frozen_vision | **Steps**: 200 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Custom**: Frozen SigLIP, weight_decay=1e-4, 5× oversampling, peak_lr=5e-6 (5× lower)

- **Config**: `pi0_ur5_frozen_vision` with `peak_lr=5e-6` (was 2.5e-5).
- **Hypothesis**: 5× lower LR slows LLM overfitting, moving the generalization sweet spot from step 30-60 to step 90-150, giving arm trajectories more time to learn.

Generalization eval results:

| Step | Dataset(orig) | Dataset(perturb) | Robot | Overfit | Verdict |
|------|--------------|-----------------|-------|---------|---------|
| 30 | +0.08 | +0.31 | **+0.55** | -0.46 | UNDERFIT |
| 60 | +0.32 | +0.42 | +0.24 | +0.08 | **GOOD** |
| 90 | +0.50 | +0.47 | +0.18 | +0.32 | GOOD |
| 120 | +0.56 | +0.54 | +0.12 | +0.44 | GOOD |
| 150 | +0.53 | +0.55 | +0.02 | +0.50 | OVERFIT |
| 180 | +0.54 | +0.54 | +0.01 | +0.53 | OVERFIT |
| 199 | +0.57 | +0.55 | +0.01 | +0.57 | OVERFIT |

**Result**: Lower LR improved robot scores at later steps vs smooth-10 (step 150 robot=0.15 vs 0.02), but overfitting still occurs. Sweet spot remains at step 60 (robot=0.24). Arm trajectory quality at step 60 with LR=5e-6 is similar to step 60 with LR=2.5e-5 — lower LR doesn't trade off arm quality.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-12 (strong augmentation + LR=2e-6 + 500 steps) (05.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5_frozen_vision | **Steps**: 500 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Custom**: Strong augmentation, full fine-tuning (unfrozen), weight_decay=1e-4, 5× oversampling, peak_lr=2e-6

- **Config**: `pi0_ur5_frozen_vision` — full fine-tuning (no frozen layers) with very low LR=2e-6 and strong image augmentation.
- **Key change: stronger augmentation** in `model.py` `preprocess_observation()`:
  - RandomCrop: 95% → **90%** (more aggressive crop)
  - Rotate: ±5° → **±10°** (more rotation)
  - brightness: 0.3 → **0.5** (stronger jitter)
  - contrast: 0.4 → **0.5** (stronger jitter)
- **Hypothesis**: Aggressive augmentation prevents image memorization at the pixel level, allowing both arm and gripper to train for 500 steps without overfitting. Low LR (2e-6) further slows overfitting.

Generalization eval results:

| Step | Dataset(orig) | Dataset(perturb) | Robot | Overfit | Verdict |
|------|--------------|-----------------|-------|---------|---------|
| 30 | +0.08 | +0.22 | **+0.67** | -0.59 | UNDERFIT |
| 60 | +0.20 | +0.42 | **+0.54** | -0.34 | partial |
| 90 | +0.28 | +0.46 | **+0.42** | -0.14 | partial |
| **120** | **+0.30** | **+0.46** | **+0.35** | **-0.05** | **GOOD** |
| **150** | **+0.44** | **+0.51** | **+0.24** | **+0.20** | **GOOD** |
| 180 | +0.58 | +0.57 | +0.15 | +0.43 | GOOD |
| 210 | +0.71 | +0.61 | +0.10 | +0.60 | GOOD |
| 240 | +0.68 | +0.60 | +0.06 | +0.62 | OVERFIT |
| 270+ | ~0.66 | ~0.59 | <0.04 | >0.63 | OVERFIT |

**Result**: Best experiment so far. Strong augmentation significantly extended the generalization window:
- Robot score stays above 0.3 through step 120 (vs step 30 in smooth-9, step 60 in smooth-11)
- **Step 120 is the first checkpoint with BOTH dataset > 0.3 AND robot > 0.3** — both arm and gripper are adequately trained
- Step 150: dataset=0.44 (good arm), robot=0.24 (marginal gripper) — also viable
- Overfitting onset shifted from step 60 (smooth-9) → step 90 (smooth-11) → **step 210 (smooth-12)**

**Comparison across all overfitting experiments:**

| Experiment | Augmentation | Frozen SigLIP | peak_lr | Steps | Robot@120 | Robot@150 | Overfit onset |
|-----------|-------------|---------------|---------|-------|-----------|-----------|---------------|
| smooth-9 | default | no | 2.5e-5 | 200 | 0.11 | 0.04 | ~step 60 |
| smooth-10 | default | **yes** | 2.5e-5 | 200 | 0.12 | 0.02 | ~step 90 |
| smooth-11 | default | **yes** | 5e-6 | 200 | 0.17 | 0.15 | ~step 150 |
| **smooth-12** | **strong** | no | **2e-6** | **500** | **0.35** | **0.24** | **~step 210** |

**Key insight**: Strong image augmentation is the most effective anti-overfitting technique — more impactful than freezing SigLIP or lowering LR alone. The combination of augmentation + low LR gives the best results.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-13 (ultra-low LR=5e-7 + strong augmentation + 1000 steps) (06.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5_frozen_vision | **Steps**: 1000 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Custom**: Strong augmentation, full fine-tuning, weight_decay=1e-4, 5× oversampling, peak_lr=5e-7

- **Config**: `pi0_ur5_frozen_vision` with `peak_lr=5e-7` (was 2e-6), `num_train_steps=1000` (was 500), `save_interval=50`, `warmup_steps=100`.
- **Hypothesis**: Each ~4x LR reduction doubles the overfit onset step (2.5e-5→60, 5e-6→150, 2e-6→210). At 5e-7 (4x lower), expect onset ~step 400+, giving arm 400+ steps to converge while gripper generalization persists. 1000 steps ensures enough training time at this very low LR.
- **Risk**: At 50x below default LR, updates may be too small for the flow matching head to adapt to UR5 actions at all.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi0_ur5_frozen_vision --exp-name=ur5_blueblock_box_v2_40_smooth-13 --overwrite

**Result**: Ultra-low LR (5e-7) did not help — model learns too slowly for both arm and gripper. LR reduction has diminishing returns below 2e-6.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-14 (pi0_droid checkpoint + smooth-12 config) (06.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5_frozen_vision | **Steps**: 200 | **action_horizon**: 50 | **Model**: Pi0 (flow matching) | **Custom**: Strong augmentation, weight_decay=1e-4, 5× oversampling, peak_lr=2e-6, **pi0_droid checkpoint** (not pi0_base)

- **Config**: `pi0_ur5_frozen_vision` — same as smooth-12 (best result) but starting from `gs://openpi-assets/checkpoints/pi0_droid/params` instead of `pi0_base`.
- **Hypothesis**: DROID checkpoint was fine-tuned on diverse real-world manipulation scenes with many different cameras, lighting conditions, and environments. Its visual features should be more robust to domain shift than pi0_base, reducing image overfitting.
- **Research backing**: CoRL 2025 paper showed flow matching policies memorize training images as nearest-neighbor lookup. Starting from DROID (which has seen diverse scenes) means the visual feature space better covers real-world variation. DROID paper shows scene diversity is the key factor for OOD generalization.
- **Norm stats**: Still loaded from `pi0_base/assets/ur5e` (DROID checkpoint doesn't include UR5 norm stats).
Generalization eval results:

| Step | Dataset(orig) | Dataset(perturb) | Robot | Overfit | Verdict |
|------|--------------|-----------------|-------|---------|---------|
| **30** | **+0.61** | **+0.58** | **+0.33** | **+0.28** | **GOOD** |
| 60 | +0.41 | +0.35 | +0.12 | +0.29 | GOOD |
| 90 | +0.38 | +0.33 | +0.09 | +0.30 | OVERFIT |
| 120+ | <0.31 | <0.30 | <0.05 | — | OVERFIT |
| 150-299 | ~0.23 | ~0.17 | <0.04 | — | partial (stuck) |

**Result**: DROID checkpoint is worse than pi0_base. Only step 30 is viable (robot=0.33), weaker than smooth-12's step 30 (robot=0.67). Model plateaus at dataset≈0.23 after step 150 — DROID's action_horizon=10 dynamics don't transfer well to our action_horizon=50 task. The visual diversity advantage of DROID is negated by the temporal dynamics mismatch.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-15 (pi05_ur5, clean config) (06.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi05_ur5 | **Steps**: 500 | **action_horizon**: 15 | **Model**: Pi0.5 | **Checkpoint**: pi05_base | **No modifications**: no freezing, no oversampling, no dim weights, no custom optimizer, default augmentation

- **Config**: `pi05_ur5` — clean config matching the og_smooth-3 experiment (the only one that worked on the robot), now on the 40-episode dataset.
- **Settings**: `pi05_base` checkpoint, `action_horizon=15`, `warmup=50`, `peak_lr=2.5e-5`, `decay=30000`, `batch_size=16`, `early_stop_patience=20`, `save_interval=50`.
- **No modifications**: No frozen layers, no gripper oversampling, no action_dim_weights, no custom weight_decay, default augmentation (crop 95%, rotate ±5°, brightness 0.3, contrast 0.4).
- **Purpose**: Establish a clean pi0.5 baseline on the 40-episode dataset. og_smooth-3 worked on 10 episodes with these exact settings — 500 steps with frequent saves will show if it scales to 40 episodes.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi05_ur5 --exp-name=ur5_blueblock_box_v2_40_smooth-15 --overwrite

---

### Experiment: ur5_blueblock_box_v2_40_smooth-16 (pi05_ur5, og_smooth-3 matched for 40 episodes) (08.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi05_ur5 | **Steps**: 500 | **action_horizon**: 15 | **Model**: Pi0.5 | **Checkpoint**: pi05_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=20

- **Config**: `pi05_ur5` — exact og_smooth-3 settings (the only experiment that worked on the robot), scaled to 40 episodes.
- **Matching data exposure**: og_smooth-3 trained 120 steps × batch 16 = 1,920 samples from 7,577 frames = 0.25 epochs. On 40-episode dataset (29,966 frames), 0.25 epochs = 468 steps. With `num_train_steps=500` and `early_stop_patience=20`, the model will see roughly the same fraction of data before stopping.
- **Settings**: Identical to og_smooth-3 — no oversampling, no dim weights, no frozen layers, no custom optimizer, default augmentation, `pi05_base` checkpoint.
- **Purpose**: Test whether og_smooth-3's success (72% task score on 10 episodes) can transfer to the 40-episode dataset with proportionally matched training.

**Result**: Good arm movement quality but **no gripper activation**. Same pattern as all pi0.5 full fine-tuning attempts on 40 episodes — pi0.5 cannot learn gripper from 40 diverse episodes (vs. 10 consistent episodes that worked in og_smooth-3).

Training command:

    uv run scripts/train.py pi05_ur5 --exp-name=ur5_blueblock_box_v2_40_smooth-16 --overwrite

---

### Experiment: ur5_blueblock_box_og_smooth-lora-1 (pi05_ur5_lora, LoRA on 10 episodes, 2-3 epochs) (08.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_og_smooth (10 episodes, smooth gripper) | **Config**: pi05_ur5_lora | **Steps**: 1400 (~3 epochs) | **action_horizon**: 15 | **Model**: Pi0.5 LoRA (gemma_2b_lora + gemma_300m_lora) | **Checkpoint**: pi05_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=20 | **EMA**: disabled

- **Config**: `pi05_ur5_lora` — og_smooth-3 settings with LoRA on the 10-episode dataset that previously worked.
- **Hypothesis**: og_smooth-3 (full fine-tuning) worked at step 120 (0.25 epochs) but overfit to training images on larger datasets. LoRA constrains weight updates to low-rank subspaces, preventing image memorization while allowing 2-3 epochs of training. This should give the model enough time to learn both arm trajectories AND gripper without overfitting.
- **Data exposure**: 1400 steps × batch 16 = 22,400 samples ÷ 7,577 frames = 2.95 epochs. og_smooth-3 only saw 0.25 epochs — LoRA allows ~12x more training without overfitting.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi05_ur5_lora --exp-name=ur5_blueblock_box_og_smooth-lora-1 --overwrite

**Results (robot tests):**
- **Step 700 (~1.5 epochs)**: Arm movement is good quality but doesn't go to the object correctly. **Some gripper activation happening** — first time gripper works with LoRA. Vision appears degraded.
- **Step 350**: Not tested — earlier checkpoint would have even worse vision (arm misses object more at later steps, so earlier would be worse).
- **Conclusion**: LoRA on 10 episodes shows gripper learning but arm can't learn diverse trajectories from only 10 episodes. Need more episodes for arm, LoRA for gripper overfitting prevention.

---

### Experiment: ur5_blueblock_box_v2_40_smooth-lora-3 (pi05_ur5_lora on 40 episodes, 2000 steps) (08.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi05_ur5_lora | **Steps**: 2000 (~1 epoch) | **action_horizon**: 15 | **Model**: Pi0.5 LoRA (gemma_2b_lora + gemma_300m_lora) | **Checkpoint**: pi05_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=20 | **EMA**: disabled

- **Config**: `pi05_ur5_lora` — now configured for 40-episode dataset with 2000 steps.
- **Hypothesis**: LoRA on 10 episodes (lora-1) showed gripper activation but arm missed the object (too few episodes for arm diversity). 40 episodes should give enough arm trajectories. LoRA prevents the gripper overfitting that killed full fine-tuning on 40 episodes (smooth-15 had zero gripper across all 500 steps). 2000 steps = ~1 epoch on 29,966 frames.
- **Checkpoints saved**: every 100 steps (100, 200, ..., 1900, 1999).

**Result (robot tests, steps 100-800):** Performance improves consistently with more training steps. Arm trajectories get progressively better and more precise at each checkpoint. At step 800, the robot had some gripping attempts — it tried to close the gripper at roughly the right time but didn't complete the grasp successfully. This is the most promising LoRA result so far: both arm and gripper are learning, but need more training.

**Conclusion**: LoRA on 40 episodes is working — performance scales with steps. Need more training to reach a usable checkpoint. Unlike full fine-tuning (which overfit by step 60-120), LoRA shows no overfitting through step 800.

Training command:

    uv run scripts/train.py pi05_ur5_lora --exp-name=ur5_blueblock_box_v2_40_smooth-lora-3 --overwrite

---

### Experiment: ur5_blueblock_box_v2_40_smooth-lora-4 (pi05_ur5_lora on 40 episodes, 1500 steps) (09.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi05_ur5_lora | **Steps**: 1500 (~0.8 epochs) | **action_horizon**: 15 | **Model**: Pi0.5 LoRA | **Checkpoint**: pi05_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=100 | **EMA**: disabled

- **Config**: `pi05_ur5_lora` — same as lora-3 but with 1500 steps and early_stop_patience=100 (was 20).
- **Rationale**: lora-3 showed performance scaling with training steps through step 800, with gripper attempts starting. More steps should improve both arm precision and gripper completion. Higher patience allows training to continue past loss plateaus.
- **Checkpoints saved**: every 100 steps.
- **Result**: TBD

Training command:

    uv run scripts/train.py pi05_ur5_lora --exp-name=ur5_blueblock_box_v2_40_smooth-lora-4 --overwrite

**Results (robot tests):**
- **Step 600**: Gripper IS gripping, but unreliable — crashing into objects, inconsistent grasps.
- **Step 700**: Same as 600 — gripper attempts but unreliable, crashing.
- **Step 900**: Gripper overfit — gripper closes at wrong times or stays closed.

**Conclusion**: The "best" window is around step 600-700 where gripper actually activates but execution quality is poor. Beyond step 800 the gripper starts overfitting (vs lora-3 which had no overfitting through step 800 — possibly because lora-3 stopped earlier or the higher patience in lora-4 caused different training dynamics). Need a checkpoint between 600 and 800 with stable gripper without overfitting.

---

## Mixed Multi-Task Dataset (2026-04-09)

### `LPSlvlv/ur5_mixed_51_smooth` — 51 episodes, 41,160 frames

Combined dataset with two tasks:
- **40 episodes** "pick up the blue block and place it in the cardboard box" (from v2_40_smooth)
- **11 episodes** "bus the table" (from raw_episodes_ur5_second, originally delta-action format, converted to absolute via `convert_raw_deltas_to_absolute.py`)

Tasks are stored as separate `task_index` values (0=bus_the_table, 1=blueblock). Model learns both behaviors and uses prompt to switch.

Created via:
```bash
# Convert delta to absolute
uv run python ur5/scripts/convert_raw_deltas_to_absolute.py --raw-dir raw_episodes_bus_the_table

# Combine all 51 episodes (after symlinking into raw_episodes_mixed_all)
uv run python ur5/scripts/combine_and_split_ur5_datasets.py \
  --raw-dirs raw_episodes raw_episodes_blueblock_2 raw_episodes_blueblock2 raw_episodes_bus_the_table \
  --repo-id-prefix LPSlvlv/ur5_mixed --splits 51

# Smooth gripper version
uv run python ur5/scripts/convert_ur5_smooth_gripper.py \
  --raw-dir raw_episodes_mixed_all --repo-id LPSlvlv/ur5_mixed_51_smooth --ramp-frames 5
```

---

### Experiment: ur5_blueblock_box_v2_40_smooth-frozen-1 (pi0_ur5_frozen_vision, frozen SigLIP + Gemma LLM) (09.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_v2_40_smooth (40 episodes, smooth gripper) | **Config**: pi0_ur5_frozen_vision | **Steps**: 2000 | **action_horizon**: 50 | **Model**: Pi0 (full fine-tuning, but with frozen backbone) | **Checkpoint**: pi0_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=100

- **Config**: `pi0_ur5_frozen_vision` — Pi0 with maximum freeze: SigLIP vision encoder + main Gemma LLM frozen, only action expert (llm_1) and action projections train.
- **Frozen params**: 32 (23 SigLIP + 9 main Gemma)
- **Trainable params**: 18 (8 action expert + 10 action projections)
- **Freeze filter**:
  ```python
  nnx.Any(
      nnx_utils.PathRegex(".*img.*"),
      nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
  )
  ```
- **Hypothesis**: This mirrors pi0.5's "Knowledge Insulation" approach (stop_gradient between action expert and VLM). The action expert has 300M params — 30x more capacity than LoRA's deltas — but cannot corrupt the pretrained vision/language understanding. The cleanest separation of "what is in the scene" (frozen VLM) vs "what to do" (trainable action prediction).
- **Comparison to previous experiments**:
  - smooth-10 (frozen SigLIP only): LLM still memorized SigLIP features → still overfit
  - lora-3/4 (LoRA on LLM + action expert): showed gripper learning but limited capacity
  - **frozen-1**: full action expert capacity + VLM completely locked
- **Result**: TBD

Training command:

    uv run scripts/train.py pi0_ur5_frozen_vision --exp-name=ur5_blueblock_box_v2_40_smooth-frozen-1 --overwrite

---

---

## V2 Datasets (2026-03-30)

20 new episodes recorded on 2026-03-30 with consistent approach depth (J1 > -2.48, J2 < -1.71), matching the og_smooth range. The previous 20-episode dataset (`ur5_blueblock_box_20_smooth`, episodes from 2026-03-18) failed due to episodes 10-19 having much deeper joint ranges (J1 down to -2.71, J2 up to -0.85), causing the robot to hit the ground.

### New datasets

| Dataset | Episodes | Frames | Source |
|---------|----------|--------|--------|
| `LPSlvlv/ur5_blueblock_box_v2_20` | 20 | 11,503 | New 20 episodes (2026-03-30) |
| `LPSlvlv/ur5_blueblock_box_v2_20_smooth` | 20 | 11,503 | New 20 episodes, smooth gripper (ramp_frames=5) |
| `LPSlvlv/ur5_blueblock_box_v2_30` | 30 | 19,080 | New 20 + OG 10 (2026-02-01) |
| `LPSlvlv/ur5_blueblock_box_v2_30_smooth` | 30 | 19,080 | New 20 + OG 10, smooth gripper |
| `LPSlvlv/ur5_blueblock_box_v2_40` | 40 | 29,966 | New 20 + OG 10 + Old 10 (2026-03-18) |
| `LPSlvlv/ur5_blueblock_box_v2_40_smooth` | 40 | 29,966 | New 20 + OG 10 + Old 10, smooth gripper |

Episode sources:
- **New 20** (2026-03-30): `raw_episodes/ur5_replay_20260330_*` — consistent depth, 48-69s per episode
- **OG 10** (2026-02-01): `raw_episodes_blueblock_2/ur5_replay_*` — best performing recordings
- **Old 10** (2026-03-18): `raw_episodes_blueblock2/ur5_replay_*` — deeper/wider joint ranges, caused safety limit issues in earlier training

---

### Summary

| Episodes | Dataset | Config | Tries | Score (avg ± std) | Reach % | Pick % | Drop % | Release % |
|----------|---------|--------|-------|-------------------|---------|--------|--------|-----------|
| 1  | `ur5_blueblock_box_1`  | `pi05_ur5` | — | qualitative only | — | — | — | — |
| 10 | `ur5_blueblock_box_og_smooth` | `pi05_ur5` | 9 | 0.72 ± 0.20 | 100% | 89% | 78% | 22% |
| 20 | `ur5_blueblock_box_20_smooth` | `pi05_ur5` | — | FAIL — safety limits (hits ground) | — | — | — | — |
| 20 | `ur5_blueblock_box_v2_20` | `pi05_ur5` | — | FAIL — good movement, no gripper activation | — | — | — | — |
| 20 | `ur5_blueblock_box_v2_20_smooth` | `pi05_ur5` | — | FAIL — good movement, no gripper activation | — | — | — | — |
| 30 | `ur5_blueblock_box_v2_30` | `pi05_ur5` | — | FAIL — good movement, no gripper activation | — | — | — | — |
| 40 | `ur5_blueblock_box_v2_40_smooth` (quantile norm) | `pi05_ur5` | — | FAIL — good movement, no gripper activation | — | — | — | — |
| 40 | `ur5_blueblock_box_v2_40_smooth` (z-score norm) | `pi05_ur5` | — | FAIL — good movement, no gripper activation | — | — | — | — |
| 40 | `ur5_blueblock_box_v2_40_smooth` | `pi0_ur5_frozen_vision` (frozen-1, step 1999) | — | **FAIL — random positions, not reaching object** | 0% | 0% | 0% | 0% |

---

### Experiment: ur5_blueblock_box_v2_40_smooth-frozen-1 — RESULT (12.04.26)

> **Status**: FAILED. Robot goes to random positions, does not approach the object at all.

- **Checkpoint tested**: `pi0_ur5_frozen_vision` step 1999
- **Behavior**: Robot moves to arbitrary joint positions unrelated to the task. No directed reaching toward the blue block.
- **Diagnosis**: Freezing SigLIP + main Gemma LLM while training only the action expert (llm_1) for 2000 steps on 40 episodes was insufficient. The action expert alone cannot learn meaningful manipulation from scratch in so few steps — it needs either (a) more data, (b) more steps, or (c) the full backbone unfrozen to leverage pretrained representations. The frozen backbone locks out the visual grounding that pi0_base provides, and the action expert has no way to bridge the gap in 2000 steps.
- **Next**: Train on F-Fer's 800-episode dataset (`pi0_ur5_ffer_merged-1`) with no freezing.

---

### Experiment: pi0_ur5_ffer_merged-1 (F-Fer's 800-episode dataset, full fine-tune) (12.04.26)

> **Dataset**: F-Fer/ur-tasks-merged (800 episodes, 1.18M frames, 60 Hz, 4 tasks, 30.9 GB) | **Config**: pi0_ur5_ffer_merged | **Steps**: 10000 | **action_horizon**: 30 | **Model**: Pi0 (full fine-tune, no freezing) | **Checkpoint**: pi0_base | **LR**: peak_lr=2.5e-5, warmup=100, decay=10000 | **Batch**: 16 | **Save interval**: 2000

- **Config**: `pi0_ur5_ffer_merged` — Pi0 base with no freezing, no oversampling, no early stopping.
- **Dataset**: F-Fer's merged UR5e dataset from https://huggingface.co/datasets/F-Fer/ur-tasks-merged. Recorded at 60 Hz with GELLO teleop, 4 ZED stereo cameras, actual gripper sensor feedback (not commands). 4 tasks across 800 episodes.
- **Key differences from previous experiments**:
  - 40x more data (1.18M frames vs 30k)
  - 6x higher recording FPS (60 Hz vs 10 Hz)
  - Actual gripper sensor readings (not command echo)
  - 4 tasks instead of 1
  - Different workstation — F-Fer's robot has different joint pose distribution than ours
  - action_horizon=30 (matching F-Fer's configs) vs our usual 50
  - No freezing — full fine-tune of all parameters
- **Norm stats**: Computed fresh on the merged dataset (`asset_id="F-Fer/ur-tasks-merged"`).
- **Inference note**: At inference time, set `HOLD_PER_STEP=0.0167` (60 Hz) and `HORIZON_STEPS=30`.
- **Result**: TBD

Training commands:
```bash
# 1. Compute norm stats (downloads ~31 GB on first run)
uv run scripts/compute_norm_stats.py --config-name pi0_ur5_ffer_merged

# 2. Train
uv run scripts/train.py --config-name pi0_ur5_ffer_merged --exp-name=pi0_ur5_ffer_merged-1
```

---

### Experiment: pi05_ur5_og_smooth_short-1 (replicate og_smooth-3, 300 steps, save every 30) (12.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_og_smooth (10 original episodes, smooth gripper ramps) | **Config**: pi05_ur5_og_smooth_short | **Steps**: 300 | **action_horizon**: 15 | **Model**: Pi0.5 (full fine-tune) | **Checkpoint**: pi05_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=20, min_delta=1e-4 | **Save interval**: 30

- **Config**: `pi05_ur5_og_smooth_short` — exact same recipe as og_smooth-3 (the only experiment that worked on the real robot) but capped at 300 steps with checkpoints every 30 steps.
- **Purpose**: Reproduce og_smooth-3's success. og_smooth-3 early-stopped at step 120 and produced the best real-robot results (0.72 avg score, 89% pick rate). Saving every 30 steps gives 10 checkpoints (30, 60, 90, 120, ..., 300) so we can find the exact sweet spot without overshooting.
- **Same as og_smooth-3**: dataset, model (Pi0.5), LR schedule (warmup=50, peak=2.5e-5, decay=30k), batch=16, early_stop_patience=20, pretrained ur5e norm stats from pi05_base.
- **Different from og_smooth-3**: save_interval=30 (was 100), num_train_steps=300 (was 500), keep_period=30 (was 100).
- **Result**: TBD

Training command:
```bash
uv run scripts/train.py --config-name pi05_ur5_og_smooth_short --exp-name=pi05_ur5_og_smooth_short-1
```

---

### Experiment: pi05_ur5-short-2 (replicate og_smooth-3 on og_smooth dataset, 200 steps) (13.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_og_smooth (10 original episodes, smooth gripper ramps) | **Config**: pi05_ur5 | **Steps**: 200 | **action_horizon**: 15 | **Model**: Pi0.5 (full fine-tune) | **Checkpoint**: pi05_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=10, min_delta=1e-4 | **Save interval**: 30

- **Config**: `pi05_ur5` (og_smooth dataset, Pi0.5 full fine-tune). Replica of the `ur5_blueblock_box_og_smooth-3` checkpoint that worked on the real robot (the only successful run so far — 0.72 avg score, 89% pick rate, early-stopped at step 120).
- **Goal**: **try different checkpoint steps to find the best one**. og_smooth-3 early-stopped at step 120 but we never systematically checked steps around it. This run saves every 30 steps (30, 60, 90, 120, 150, 180, 199) so each can be tested individually on the robot to find the sweet spot between underfitting and overfitting.
- **Caveat — smooth dataset may be broken**: the `og_smooth` dataset uses the OLD `convert_ur5_smooth_gripper.py` which placed gripper ramps **BEFORE** the transition frame (causally impossible — state shows the gripper moving before the command fires). The ramp-direction bug was only found and fixed on 2026-04-12. The model trained on this data is learning a signal the real robot cannot reproduce at inference time. Even if the checkpoint works, it's working despite the data issue, not because of it. Re-recording / re-converting with the fixed ramp direction is a separate follow-up.
- **Result**: TBD — each saved checkpoint will be tested individually.

Training command:
```bash
uv run scripts/train.py --config-name pi05_ur5 --exp-name=pi05_ur5-short-2
```

---

### Experiment: pi0_ur5_ffer_merged (train on F-Fer's 800-episode dataset, then fine-tune on ours)

> **Dataset**: F-Fer/ur-tasks-merged (800 episodes, 1.18M frames, 60 Hz, 4 tasks) | **Config**: pi0_ur5_ffer_merged | **Steps**: 10000 | **action_horizon**: 30 | **Model**: Pi0 (full fine-tune) | **Stage**: 1 of 2

- **Config**: `pi0_ur5_ffer_merged` — Pi0 base, no freezing, no oversampling, no early stopping. See pi0_ur5_ffer_merged-1 entry above for full details.
- **Goal**: **build a UR5e foundation model** on a much larger, higher-quality dataset than anything we can record ourselves. F-Fer's data is 40× bigger, recorded at 60 Hz with actual Robotiq gripper sensor readings (not command echo), and covers 4 tasks on a different workstation.
- **Why this might help**: our own datasets have known issues (gripper command-echo instead of actual sensor readings; 10 Hz vs pi0_base's 20 Hz expectation; synthetic ramps with wrong direction). F-Fer's data avoids all three. A model pretrained on F-Fer's data sees real gripper trajectories and fine-grained motion that our data cannot provide.
- **Next step — stage 2 fine-tune**: after this run finishes, pick the best F-Fer checkpoint step from wandb and plug it into `pi0_ur5_ffer_then_og_smooth`. That config loads the F-Fer checkpoint as the starting weights, then runs the og_smooth-3 recipe on our 10-episode og_smooth dataset to adapt to our specific task + workstation. This two-stage training should combine F-Fer's strong gripper foundation with our task-specific data without needing to re-record our episodes.
- **Inference at stage 1**: `HOLD_PER_STEP=0.0167` (60 Hz), `HORIZON_STEPS=30`.
- **Inference at stage 2**: revert to our usual `HOLD_PER_STEP=0.1` (10 Hz) — the stage-2 model learns our 10 Hz distribution from the og_smooth data.
- **Result**: TBD

Training commands:
```bash
# Stage 1: train on F-Fer's dataset
uv run scripts/compute_norm_stats.py --config-name pi0_ur5_ffer_merged
uv run scripts/train.py --config-name pi0_ur5_ffer_merged --exp-name=pi0_ur5_ffer_merged-1

# Stage 2: after stage 1 finishes, update BEST_STEP in pi0_ur5_ffer_then_og_smooth config,
# then fine-tune on our og_smooth dataset
uv run scripts/train.py --config-name pi0_ur5_ffer_then_og_smooth --exp-name=pi0_ur5_ffer_then_og_smooth-1
```

---

### Experiment: pi05_ur5 on mixed_51 non-smooth (fresh norm stats) (13.04.26)

> **Dataset**: LPSlvlv/ur5_mixed_51 (51 episodes, NO smoothing — raw binary gripper transitions) | **Config**: pi05_ur5 | **Steps**: 200 | **action_horizon**: 15 | **Model**: Pi0.5 (full fine-tune) | **Checkpoint**: pi05_base | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Early stop**: patience=10, min_delta=1e-4 | **Save interval**: 30

- **Config**: `pi05_ur5` — updated to use `LPSlvlv/ur5_mixed_51` (51 episodes, no smoothing).
- **Why no smoothing**: the `convert_ur5_smooth_gripper.py` ramp direction was wrong until 2026-04-12 (ramps placed BEFORE the transition, causally impossible — state showed the gripper moving before the command fired). Skipping smoothing avoids poisoning the data with that bad signal. The raw binary gripper has its own problem (single-frame jumps instead of smooth ramps), but at least it's causally consistent.
- **Why 51 episodes**: more data than the 10-episode og_smooth dataset. Mixed dataset spans multiple recording sessions (blueblock + bus-the-table + older batches), so gives the model broader task coverage at the cost of distribution shift within the dataset.
- **Fresh norm stats** (key difference from previous pi05_ur5 runs): previously this config reused the pretrained `asset_id="ur5e"` stats from `gs://openpi-assets/checkpoints/pi05_base/assets`. Now `asset_id="LPSlvlv/ur5_mixed_51"` — `compute_norm_stats.py` runs on the actual 51-episode dataset and writes to `assets/pi05_ur5/LPSlvlv/ur5_mixed_51/norm_stats.json`. Training then loads from there. This means the model's normalization matches OUR data distribution, not PI's pretrained ur5e distribution.
- **Hypothesis**: PI's pretrained `ur5e` norm stats were computed on a different workstation with a different pose manifold (confirmed in `ur5_comparison_report.md` §3.3.3 — F-Fer's state means are completely different from ours). Using those stats on our data means our observations get normalized to a distribution the pretrained model never saw. Fresh stats on our dataset should produce a more in-distribution input signal.
- **Caveat**: re-computing stats breaks the warm-start assumption that "the model already understands ur5e stats." The model now sees a different normalization than it was pretrained on. In theory pi0_base should adapt quickly during fine-tuning, but if it doesn't, reverting to pretrained stats is the fallback.
- **Result**: TBC

Training commands:
```bash
# 1. Compute fresh norm stats on the 51-episode mixed dataset
uv run scripts/compute_norm_stats.py --config-name pi05_ur5

# 2. Train
uv run scripts/train.py --config-name pi05_ur5 --exp-name=pi05_ur5-mixed-51-raw-1
```

---

### Experiment: pi0_ur5_ffer_then_mine-1 (stage 2: F-Fer checkpoint → mixed_51 raw fine-tune) (13.04.26)

> **Dataset**: LPSlvlv/ur5_mixed_51 (51 episodes, NO smoothing — raw binary gripper) | **Config**: pi0_ur5_ffer_then_mine | **Steps**: 300 | **action_horizon**: 15 | **Model**: Pi0.5 (full fine-tune) | **Warm-start**: pi0_ur5_ffer_merged-1 step 9999 | **LR**: peak_lr=2.5e-5, warmup=50, decay=30000 | **Batch**: 16 | **Save interval**: 30

- **Config**: `pi0_ur5_ffer_then_mine` — Pi0.5, og_smooth-3 recipe, trains on our 51-episode raw mixed dataset starting from the F-Fer stage-1 checkpoint.
- **Stage of a two-stage training**:
  - **Stage 1** = `pi0_ur5_ffer_merged-1` (Pi0, 10k steps on F-Fer's 800-episode dataset at 60 Hz). Builds a strong UR5e foundation with real gripper sensor readings and fine-grained motion.
  - **Stage 2** (this experiment) = `pi0_ur5_ffer_then_mine-1` (Pi0.5, 300 steps on our data at 10 Hz). Adapts the F-Fer foundation to our specific task + workstation + camera setup.
- **Warm-start checkpoint**: `checkpoints/pi0_ur5_ffer_merged/pi0_ur5_ffer_merged-1/9999/params`. Assumes stage 1 ran to at least step 9999 (within the 10k-step `num_train_steps`). If the best stage-1 checkpoint turns out to be an earlier step per wandb, update the path in the config.
- **Model family switch**: stage 1 uses **Pi0 base** (`action_horizon=30`, matching F-Fer's recipe). Stage 2 uses **Pi0.5** (`action_horizon=15, pi05=True, max_token_len=180`, matching the og_smooth-3 recipe that previously worked on our robot). This is a deliberate architecture change between stages — we leverage the F-Fer stage-1 weights but tokenize/serve via Pi0.5 because that's what our successful robot runs used.
- **Dataset**: `LPSlvlv/ur5_mixed_51` (raw 51 episodes). No smoothing — avoids the causally-broken ramp direction bug in the smooth datasets.
- **Norm stats**: `asset_id="LPSlvlv/ur5_mixed_51"`. Fresh stats computed on our dataset (same approach as `pi05_ur5-mixed-51-raw-1` above). Writes to `assets/pi0_ur5_ffer_then_mine/LPSlvlv/ur5_mixed_51/norm_stats.json` — note the `pi0_ur5_ffer_then_mine` prefix: each config gets its own assets directory, so the stats from `pi05_ur5` cannot be auto-shared.
- **Hypothesis**: the F-Fer foundation should give the action expert a much stronger prior for "how Robotiq grippers actually move" (real sensor data) and "what UR5e joint trajectories look like" (800 episodes worth of diverse motion), while our stage-2 data teaches the specific task + workstation + camera setup. Best of both worlds: F-Fer's scale + our task fidelity.
- **Inference**: at inference time, use our usual timing — `HOLD_PER_STEP=0.1` (10 Hz) and `HORIZON_STEPS=15` — because stage 2 adapted the model to our 10 Hz distribution. Do NOT use the stage-1 60 Hz timing.
- **Result**: TBC — blocked until `pi0_ur5_ffer_merged-1` reaches step 9999.

Training commands:
```bash
# Prerequisite: pi0_ur5_ffer_merged-1 must have saved at step 9999
ls checkpoints/pi0_ur5_ffer_merged/pi0_ur5_ffer_merged-1/9999/params  # must exist

# 1. Compute fresh norm stats for this config (separate assets dir, cannot reuse pi05_ur5 stats)
uv run scripts/compute_norm_stats.py --config-name pi0_ur5_ffer_then_mine

# 2. Train
uv run scripts/train.py --config-name pi0_ur5_ffer_then_mine --exp-name=pi0_ur5_ffer_then_mine-1
```

Alternative to step 1 — if `pi05_ur5` stats are already computed, copy them to save time:
```bash
mkdir -p assets/pi0_ur5_ffer_then_mine/LPSlvlv
cp -r assets/pi05_ur5/LPSlvlv/ur5_mixed_51 assets/pi0_ur5_ffer_then_mine/LPSlvlv/
```

---

### Experiment: pi0_ur5_ffer_merged-1 robot test (stage 1 only, step 9999) (18.04.26)

> **Status**: FAILED. No meaningful behavior on robot.

- **Checkpoint tested**: `pi0_ur5_ffer_merged-1` step 9999 (10k steps on F-Fer's 800-episode dataset)
- **Behavior**: Robot did not show any patterns or directed movement toward objects. No task-relevant behavior observed.
- **Diagnosis**: The stage-1 model learned F-Fer's task distribution (bolt insertion, bearing pressing) on F-Fer's workstation with different cameras and joint pose manifold. Without stage-2 adaptation to our cameras/task/workstation, the model cannot generalize to our setup.

---

### Experiment: pi0_ur5_ffer_then_mine-2 (stage 2 from step 9999, blueblock_box_10) (18.04.26)

> **Status**: FAILED — all checkpoints overfit. | **Dataset**: LPSlvlv/ur5_blueblock_box_10 (10 episodes) | **Config**: pi0_ur5_ffer_then_mine | **Steps**: 300 | **action_horizon**: 30 | **Model**: Pi0 (full fine-tune) | **Warm-start**: pi0_ur5_ffer_merged-1 step 9999 | **Save interval**: 100

- **Checkpoints tested**: steps 100, 200, 299
- **Behavior**: All three checkpoints showed overfitting — robot goes to the same fixed position regardless of object placement or task prompt. No adaptive behavior.
- **Diagnosis**: Starting from a 10k-step stage-1 checkpoint that already overfit to F-Fer's distribution, then fine-tuning on only 10 episodes, causes the model to collapse onto a narrow output mode. The stage-1 model may have been too far into F-Fer's distribution to recover meaningful generalization in 300 steps on our small dataset.

---

### Experiment: pi0_ur5_ffer_then_mine-3 (stage 2 from step 2000, blueblock_box_10) (18.04.26)

> **Status**: FAILED. | **Dataset**: LPSlvlv/ur5_blueblock_box_10 (10 episodes) | **Config**: pi0_ur5_ffer_then_mine | **Steps**: 300 | **action_horizon**: 30 | **Model**: Pi0 (full fine-tune) | **Warm-start**: pi0_ur5_ffer_merged-1 step 2000 | **Save interval**: 100

- **Checkpoints tested**: steps 100, 200, 299
- **Step 100**: Random movements — model has not yet learned the task from 10 episodes. Insufficient training.
- **Steps 200, 299**: Overfit — same fixed-position behavior as experiment -2. Robot goes to the same place regardless of scene.
- **Diagnosis**: Using an earlier stage-1 checkpoint (2000 vs 9999) did not solve the core problem. The model either undertrained (step 100, random) or overfit (steps 200+, fixed position). The 10-episode dataset may be too small for stage-2 fine-tuning to find a useful operating point between underfitting and overfitting.
- **Conclusion**: The two-stage F-Fer approach has not produced usable results with our current 10-episode dataset. The narrow window between underfitting and overfitting suggests either (a) more data is needed for stage 2, (b) a lower learning rate or regularization (e.g., LoRA) to slow down overfitting, or (c) the domain gap between F-Fer's setup and ours is too large for this transfer strategy.

---

### Experiment: pi05_ur5_blueblock10-1 (back to basics with new dataset) (18.04.26)

> **Dataset**: LPSlvlv/ur5_blueblock_box_10 (10 episodes) | **Config**: pi05_ur5_blueblock10 | **Steps**: 500 | **action_horizon**: 15 | **Model**: Pi0.5 (full fine-tune) | **Checkpoint**: pi05_base | **LR**: warmup=50, peak_lr=2.5e-5, decay=30000 | **Batch**: 16 | **Early stopping**: patience=20 | **Save interval**: 30 | **Norm stats**: reloaded from pi05_base/assets/ur5e

- **Config**: `pi05_ur5_blueblock10` — new dedicated config based on the og_smooth-3 recipe.
- **Motivation**: After the two-stage F-Fer approach failed (experiments -2 and -3), return to the simpler single-stage approach that worked in og_smooth-3: fine-tune directly from `pi05_base` on our own data.
- **Dataset**: `LPSlvlv/ur5_blueblock_box_10` — 10 episodes of blue block pick-and-place, raw (no smooth gripper ramps).
- **Norm stats**: Reloaded from pretrained `pi05_base/assets/ur5e` (not computed fresh).
- **Key differences vs ur5_fifth_1** (the experiment that previously worked on robot):
  - `warmup_steps=50` (ur5_fifth_1 used default 1000 — LR never reached peak in 500 steps, acting as implicit regularizer)
  - `batch_size=16` (ur5_fifth_1 used default 32)
  - `early_stop_patience=20` with `log_interval=10` (ur5_fifth_1 used default patience=10, log_interval=100 — early stopping checked 10x less frequently)
  - Different dataset (`ur5_blueblock_box_10` vs `ur5_pickandplace_5`)
- **Risk**: The faster warmup (50 vs 1000) means the model hits full LR much earlier, which may cause overfitting on 10 episodes — the same failure mode seen in many previous experiments. The slow warmup in ur5_fifth_1 was identified as a key factor in its success.
- **Result**: TBD

Training commands:
```bash
uv run scripts/train.py --config-name pi05_ur5_blueblock10 --exp-name=pi05_ur5_blueblock10-1
```

---

## What it takes to get a working policy

Synthesis of the 87-run sweep (Jan – Apr 2026, 33.8 GPU-hours) plus the on-robot evaluations. Captures the configuration that actually produces a deployable policy, which knobs were load-bearing, common failure modes, and approaches that were tried and abandoned.

### 1. Minimum viable recipe (the one config that worked)

| Knob | Value | Where it lives |
|---|---|---|
| Model | pi0.5 full fine-tune (gemma_2b + gemma_300m) | `pi05_ur5_blueblock10` |
| Base checkpoint | `pi05_base` | `weight_loader` |
| Action dim / horizon (train) | 32 / 15 | `Pi0Config` |
| State input | continuous (`discrete_state_input=False`) | `Pi0Config` |
| Norm stats | **reload** from `pi05_base/assets/ur5e` (don't compute fresh) | `AssetsConfig` |
| Action transform | delta (`use_delta_action_transform=True`) | `LeRobotUR5DataConfig` |
| Quantile norm | off (z-score) for pi0/pi0.5; on for pi0_FAST | `use_quantile_norm` |
| Dataset | 10 episodes, ~7 500 frames, 10 Hz, raw gripper | `ur5_blueblock_box_10` |
| Optimizer | peak_lr=2.5e-5, warmup=50, decay=30 000 | `lr_schedule` |
| Batch size | 16 | — |
| EMA decay | 0.99 | — |
| Early stopping | patience=20, min_delta=1e-4, log_interval=10 | — |
| Train budget | 500 steps; **best checkpoint = ~150** | — |
| Inference horizon | `HORIZON_STEPS=6` | dockerfile |
| Inference smoothing | `HOLD_PER_STEP=0.1`, `MAX_STEP_DEG=3.0` | dockerfile |

### 2. Critical vs non-critical hyperparameters

What broke things vs what was forgiving, based on the 87-run sweep.

| Knob | Status | Evidence |
|---|---|---|
| Model family | **CRITICAL** | pi0_FAST: 5/5 FAIL on binary gripper; pi0.5 LoRA: undertested; pi0.5 full: only family with deployable policies |
| Dataset size | **CRITICAL** | 10 ep works, 20–40 ep loses the gripper signal — non-monotonic |
| Norm-stat source | **CRITICAL** | reload pretrained: works; compute fresh on small data: state ranges wrong |
| `warmup_steps` | **CRITICAL** | 50 → reaches LR; 1000 → never reaches peak in 500-step budget |
| Inference `HORIZON_STEPS` | **CRITICAL** | 6 → 0.90; 3 → 0.25; ≥12 → 0.25 (inverted-U) |
| Checkpoint pick | **CRITICAL** | 90 → 0%; 150 → 75% ID; 210 → OOD collapse |
| `batch_size` | non-critical | 16 vs 32 indistinguishable in loss; 64 only used for LoRA |
| `peak_lr` (within 1e-5 – 2.5e-5) | non-critical | 12 runs at 1e-5 reached comparable losses |
| `ema_decay` | not tested | always 0.99 |
| `seed` | not tested | single seed throughout — biggest unmeasured risk |
| `action_horizon` at training | family-locked | 15 for pi0.5, 50 for pi0; not independently swept |

### 3. Failure mode → root cause → fix

Tactical lookup table for someone reproducing the work.

| Symptom on robot | Root cause | Fix | Source |
|---|---|---|---|
| Robot doesn't move toward object | Undertrained (< ~150 steps on 10 ep) | Train longer | step 90 / 120 ID rows |
| Stalls 10–15 cm short | Still undertrained at the boundary | Wait for ≥ step 150 | step 120 obs |
| Crashes into ground/object | 20+ ep dataset with varied approach angles | Use 10 consistent episodes | 20_smooth-2/3 |
| Gripper output ~0.0 (never closes) | Gripper transitions too rare in larger dataset | Smaller dataset OR pi0 with H=50 | v2_40 cluster |
| Drops object mid-transport | Bad grasp pose | More demos OR accept as failure mode | 150 OOD trials 2/3 |
| Reaches drop, doesn't release | Release stage learned only briefly (~step 150) | Stop at 150; 180–210 lose it | step 180/210 release rows |
| Drives straight down regardless of scene | FAST tokenizer + z-score norm mismatch | Use quantile norm (FAST) or switch to pi0.5 | pi0_fast section |
| Goes to fixed pose ignoring objects | Overfit in stage 2 of two-stage transfer | Abandon two-stage | F-Fer experiments |

### 4. Dead ends (what was tried, didn't work, and why we stopped)

| Approach | Compute spent | Why it failed | Reference |
|---|---|---|---|
| Two-stage F-Fer transfer (10k steps + finetune) | ~5 h | Stage-2 overfit on 10 ep regardless of stage-1 step (2k or 9k) | `pi0_ur5_ffer_then_mine-2/-3` |
| pi0_FAST with binary gripper | ~6 h, 5 runs | DCT+BPE entangles gripper with joints; binary = worst-case for DCT | v2_40 FAST cluster |
| 20–40 ep datasets | ~12 h, 35+ runs | Variation outpaced consistency; gripper signal diluted | v2_20 / v2_30 / v2_40 |
| Smooth gripper ramps on v2 datasets | overlap | Didn't recover gripper activation | v2_*_smooth |
| Compute fresh quantile norm on small data | ~1 h | Joint-range coverage too narrow → out-of-range at inference | norm-stats discussion |

### 5. Baseline configuration: what openpi provides vs what needs configuring

OpenPI is a model and training framework, not a turnkey deployment system. Understanding which side of that line each component sits on saved a lot of confusion later.

**What openpi provides out of the box.** The framework supplies three model families — pi0 (continuous flow-matching action head, 50-step action horizon), pi0.5 (continuous head with VLM grounding via PaliGemma, 15-step horizon), and pi0_FAST (discrete-token action head with cross-entropy loss, 10-step horizon). For each family there are pretrained checkpoints stored on `gs://openpi-assets/checkpoints/`: `pi0_base` and `pi05_base` are trained on the OXE/internal robot mix that includes some UR5e data; `pi0_fast_base` is the FAST-tokenized variant; `pi05_droid` and `pi0_droid` are Franka-only priors. Each checkpoint ships alongside an `assets/` directory containing per-robot normalization statistics (mean/std for z-score, q01/q99 percentiles for quantile) keyed by `asset_id` — `ur5e`, `franka`, `droid`, etc. The training entry point is `scripts/train.py` (JAX, FSDP across visible devices, automatic checkpointing, wandb integration), and the deployment entry point is `scripts/serve_policy.py`, which exposes the policy as a websocket endpoint a robot client can connect to.

**What does not exist out of the box for a UR5.** The repo's UR5e support is a thin wrapper — there are normalization stats (because UR5e was in the OXE pretraining mix) but no robot driver, no camera handling, no recording loop, no dataset converter, and no application-side policy code that knows about UR5 joint conventions or Robotiq grippers. Everything that actually moves a real arm had to be built. Concretely:

- **Data config** (`LeRobotUR5DataConfig` in `src/openpi/training/config.py`). This class binds a Hugging Face LeRobot dataset to the openpi training pipeline. Its fields control: `repo_id` (which dataset to pull), `assets` (where to find norm stats — either a remote pretrained location or a local `assets/<config_name>/<asset_id>/` directory populated by `compute_norm_stats.py`), `use_delta_action_transform` (whether the data pipeline subtracts state from action to convert absolute targets into deltas), `use_quantile_norm` (which normalization scheme to apply — automatically forced `True` for `PI0_FAST` per openpi#763), `gripper_oversample_factor` (a sampler weight that increases the probability of gripper-transition frames showing up in batches; not used in the main runs but available), and the `repack_structure` mapping that tells the dataloader which dataset fields populate which model inputs (`base_rgb`, `wrist_rgb`, `joints`, `prompt`, `actions`).
- **Input/output transforms** (`src/openpi/policies/ur5_policy.py`). `UR5Inputs` constructs the model's state vector at inference: 6 joint positions in radians + 1 gripper value normalized to [0, 1] (Robotiq raw 0–255 divided by 255), padded to the model's full state width. It also packages the dual RealSense streams as `base_image` and `wrist_image` after resizing/encoding. `UR5Outputs` does the inverse on the action side: pi0/pi0.5 produce a 32-dimensional action vector per chunk step (padded to the foundation model's action width), but the UR5 only consumes 7 dimensions (6 joints + gripper), so the output transform slices `action[:, :7]`. Without these transforms the model would receive the wrong-shape state and produce unusable actions.
- **Robot bridge** (`ur5/utils/pi0_bridge_ur5_headless.py`). The runtime that actually drives the robot. Opens an RTDE control + receive connection to the UR5 controller (uses `rtde_control.ServoJ` for streaming joint targets), opens a TCP socket to the Robotiq URCap on port 63352 (custom protocol — not RTDE), opens two RealSense pipelines for the base and wrist cameras (with a fixed exposure that matches the recording), runs the inference loop (collect state → call policy server → walk the action chunk → repeat), and applies safety clipping (`MAX_STEP_DEG`) before each commanded joint move.
- **Recording and conversion** (`ur5/scripts/ur5_replay_and_record_raw.py`, `ur5/scripts/convert_ur5_raw_to_lerobot.py`). The recorder captures synchronized joint state, gripper state, and RGB at 10 Hz into a raw Parquet/PNG layout. The converter packages those raw recordings into the LeRobot v3 schema (episodes table, frames table with `task_index`, video files), uploads to Hugging Face, and writes the per-task `tasks.jsonl` that pi0.5 reads to map `task_index → prompt string`. Both scripts had to be written from scratch — there is no UR5 recorder in mainline LeRobot.
- **Norm-stats decision.** The data config's `assets` field controls whether normalization statistics come from a remote pretrained source (e.g., `gs://openpi-assets/checkpoints/pi05_base/assets/ur5e`) or from a locally computed file (`assets/<config>/<asset_id>/norm_stats.json` produced by `scripts/compute_norm_stats.py`). For small fine-tuning datasets, pretrained stats are strictly better — they cover a wider joint manifold than a 10-episode dataset can, so values at inference stay safely inside the normalization range. Computing fresh on small data narrows the q01–q99 envelope to whatever appeared in those 10 episodes; any joint pose slightly outside that envelope at inference clips and breaks the policy. This matters more for FAST (which discretizes state into 256 bins inside [-1, 1]) than for pi0/pi0.5 (which tolerate moderately out-of-range z-scores), but the principle is the same.
- **Reset pose and prompt metadata.** `policy_metadata.reset_pose` in the training config is a 6-vector of joint angles in radians that the bridge moves to before each trial begins. It must equal (within a few degrees) the joint configuration of the first frame of every training episode — otherwise the policy is asked to extrapolate from an unfamiliar starting state, and on a 10-episode fine-tune that extrapolation usually collapses the entire trajectory. The prompt strings are similarly load-bearing: `prompt_from_task=True` causes pi0.5 to read the prompt from the dataset's `tasks.jsonl`, and at inference the bridge passes `PROMPT="..."` to the policy server. The string must lexically match a task label seen in training; a synonym like "grab" instead of "pick up" measurably degrades behavior.

**The takeaway.** OpenPI gives you the brain. You have to build the body, the eyes, the hands, the muscle memory of where to start, and the language model knows you've started. Calendar-wise this project spent at least as much time on the bridge, recorder, converter, and per-robot config as on training — and the bridge bugs documented in §6 below were where most early "policy doesn't work" reports actually originated.

### 6. What openpi assumes but doesn't document well

A handful of openpi behaviors and conventions cost real experiment time before being understood. None of them are bugs — they are design choices that the documentation either glosses over or assumes the reader already knows. Documenting them here as the lessons that would have saved the most calendar time on a fresh integration. (Bridge-side bugs in our own UR5 integration code — gripper state desync, blocking gripper commands, action timing — are not included; those were our implementation errors, not openpi-level concerns.)

**Norm-stat source is a load-bearing decision, not a knob.** The data config's `assets` field controls whether normalization stats come from a remote pretrained location (`gs://openpi-assets/checkpoints/<base>/assets/<asset_id>`) or from a local file produced by `compute_norm_stats.py`. The default convention is "compute fresh from your dataset," which is correct for medium-to-large datasets but wrong for small fine-tunes. Computing fresh stats from 10 episodes narrows the q01–q99 envelope (or the ±2σ envelope for z-score) to whatever joint poses appeared in those 10 episodes; any pose slightly outside that envelope at inference clips into the saturated tails of the normalization and breaks the policy. Reloading the pretrained stats from `pi05_base/assets/ur5e` gives a much wider envelope (collected across many UR5e robots and tasks) that comfortably covers any pose your fine-tune dataset can reach. The README mentions the existence of both options but doesn't flag this as a non-obvious decision; it cost several weeks of "policy diverges at edges of workspace" debugging before the pattern was identified.

**`use_delta_action_transform` should be on (per openpi maintainers).** When recorded actions are absolute targets — i.e., `action[t]` is the desired joint configuration at step t+1 (which is how every dataset in this project was captured) — there are two ways to feed them to the model: as-is (absolute prediction), or with the transform that subtracts current state (delta prediction). Maintainer guidance in the issue tracker recommends always enabling `use_delta_action_transform=True` even when the dataset is absolute — the data pipeline computes deltas internally and the inference pipeline reverses the transform automatically. Delta prediction is generally easier for the model to learn (smaller dynamic range, more uniformly distributed targets) and more robust to small calibration mismatches between training data and the deployed robot. The default does what you want here; the load-bearing fact is "don't override it to False thinking your absolute dataset needs absolute targets" — that was tried in 10 runs in this project and none deployed.

**`use_quantile_norm` is model-family-locked, not user choice.** pi0/pi0.5 expect z-score normalized inputs and quietly tolerate values outside their training distribution. pi0_FAST's tokenizer discretizes state and action into 256 BPE bins inside [-1, 1] and absolutely requires quantile normalization — anything else clips most values into bin 0 or bin 255 and feeds garbage to the model. The correct policy is "quantile norm if-and-only-if FAST," implemented in `LeRobotUR5DataConfig.create()` as `use_quantile_norm = (model_type == PI0_FAST)` per openpi#763. Forcing quantile norm on pi0/pi0.5 is the more common mistake (some early UR5 examples in the repo defaulted to it) and silently narrows the joint coverage on small data — the model trains successfully but underperforms at inference. This is a one-line override but the asymmetry across model families is not flagged anywhere in the main docs.

**`save_interval` default is calibrated for long runs.** The default is 1000 steps. That is appropriate for the 30 000-step pretraining runs the repo's defaults assume but useless for a 500-step fine-tune — you get exactly one checkpoint at the end and have no way to evaluate intermediate checkpoints to find the early-stopping sweet spot. Set this to 30 (or smaller) for any fine-tune budget under 1000 steps. Without this change the entire per-checkpoint evaluation methodology would have been impossible.

**`warmup_steps` default also assumes long runs.** Default `warmup_steps=1000` ramps the LR over the first 1000 steps. With a 500-step training budget the LR never reaches `peak_lr` — the entire run executes at a fraction of the configured value. Loss still drops (so wandb looks fine) but the policy underfits relative to what the same compute could have achieved. This is the single highest-impact unflagged default in the trainer; setting `warmup_steps=50` for short fine-tunes was the change that unblocked the `pi05_ur5_blueblock10-1` recipe.

**Rare-event behaviors require thinking about action-horizon × signal density.** OpenPI's three model families differ in their action horizon: pi0_FAST H=10, pi0.5 H=15, pi0 H=50. This is presented as an architectural detail, but it directly determines how often a rare event (e.g., a binary gripper toggle that only happens in ~1% of frames) appears inside a single training sample. With H=15 most batches contain zero gripper transitions; with H=50 transitions show up frequently enough to matter. This is why pi0 sometimes succeeds where pi0.5 fails on sparse-event datasets, and why the gripper completely collapsed on the v2_40 datasets across both pi0.5 and pi0 — at some point the dataset variance overwhelms even the H=50 advantage. The framework provides `gripper_oversample_factor` in `LeRobotUR5DataConfig` as a workaround, but its existence and intended use are not surfaced anywhere obvious; it was found by reading the data config source.

**FAST is structurally unfit for binary gripper signals.** pi0_FAST's tokenizer applies DCT compression to action chunks before BPE encoding — this is great for continuous, smooth trajectories (most joint motion) and terrible for step functions (binary gripper toggles). DCT represents a step function with many high-frequency coefficients, all of which get quantized away during BPE encoding, so the gripper toggle is effectively lost from the action representation. The FAST paper validates only on continuous action spaces; the openpi README does not mention this restriction. Five training runs (~6 GPU-hours) were spent confirming this empirically before community evidence (openpi#585, #692, #766) crystallized the conclusion: don't use FAST for tasks with binary or near-binary gripper signals.

**`reset_pose` and prompt phrasing are part of the policy.** Two pieces of the deployment configuration that look like infrastructure but actually belong to the policy: the reset pose (a 6-vector of joint angles in `policy_metadata.reset_pose`) must match the dataset's first-frame pose to within a few degrees, and the prompt string passed at inference must lexically match a task label that appeared during training. Off-distribution start poses cause systematic trajectory failures that look like model bugs but are initialization bugs; prompt synonyms (e.g., "grab" for "pick up") measurably degrade behavior on small fine-tunes because pi0.5's PaliGemma encoder binds the entire trained behavior to the exact training string. Neither sensitivity is documented in the model README, and both surface only on robot — never in offline metrics.

**The common thread.** Every item above is a case where a sensible-looking default or convention silently degrades a small-data fine-tune. None of them produce a wandb-visible signal: the loss curves look fine, training completes, the model serves cleanly, and the robot quietly underperforms or fails outright. The only diagnostic loop that catches them is on-robot evaluation, which is exactly what the per-checkpoint × per-condition evaluation methodology was built for. If you take one thing away from this section: don't trust offline metrics to validate a fine-tune for deployment.

### 7. Data quantity: zero-shot, minimal data, and the surprise of "more"

This was the most counterintuitive thread of the project. Conventional ML wisdom says "more data → better model," but for foundation-model fine-tuning on sparse-event behaviors the relationship is non-monotonic. Here is the full progression as we walked it.

**Zero-shot (`pi05_base` only, no fine-tuning).** The base checkpoint includes UR5e data in its training mix, so we expected at least some sensible behavior. In practice it produced none. The arm moved erratically with no goal-directed structure; the gripper occasionally twitched. The base model has knowledge of UR5e *kinematics* but no notion of *this* workstation, *these* cameras, *this* table-mounted bin, or the specific blue-block task. Foundation policies trained on heterogeneous robot mixes are not zero-shot deployable in the same sense that, say, a base LLM is zero-shot deployable for QA. They need at least minimal task adaptation on the target robot.

**1 episode (`ur5_busthetable_1`, ~1 000 frames, 1 000 training steps).** The first real signal of learning. The arm clearly moved toward the table and toward the object; the gripper closed when an object was placed in its claws (showing it learned "object near gripper → close"); but it could not reliably reach the object on its own and did not generalize to even small position changes. Useful as a smoke test that the entire pipeline (recording → conversion → training → serving → bridge) is wired correctly — if the policy doesn't move at all after 1 episode, something deeper is broken. This is the recommended first run for any new robot/task combination: check the wires before scaling the data.

**10 episodes (`ur5_blueblock_box_og_smooth`, `ur5_blueblock_box_10`, ~7 500 frames).** The smallest dataset that reliably produced a deployable policy across multiple training runs. Single task variant ("Pick up the blue block and place it in the cardboard box"); blue block placed within a roughly 30 cm × 30 cm patch on the table; cardboard box in a fixed location on the right; consistent approach trajectory across episodes (operator started from the same reset pose, moved to the block in roughly the same arc, returned via roughly the same path). After fine-tuning with the recipe in §1, evaluation at checkpoint 150 showed 75% ID success (5/5 reach, 4/5 pick, 4/5 drop, 2/5 release). The narrowness of this dataset is the critical feature, not a limitation: every gripper transition lands in the same general region, so the model's MSE loss has a chance to actually constrain the gripper output.

**20 episodes (`ur5_blueblock_box_20`, `ur5_blueblock_box_20_smooth`).** Performance got worse, not better. The 20-episode dataset was assembled by re-recording with more positional variance, hoping to teach better generalization. Instead the model learned overly-aggressive downward trajectories (apparently because some episodes recorded steeper approach angles), causing the arm to drive into the table and trigger the controller's protective stop. The dominant failure mode shifted from "doesn't reach" (fixable with more training) to "crashes into ground" (not fixable without better data). On both `_20` and `_20_smooth` variants, the 10-episode model was strictly better.

**40 episodes (`v2_40`, `v2_40_smooth`).** The gripper failed completely. Across more than 25 runs spanning 199 to 9 999 training steps, both binary and ramped gripper signals, both quantile and z-score normalization, both pi0.5 and pi0 model families — the policy emitted gripper output values of ~0.002 (functionally zero) throughout inference, regardless of object proximity or task progress. The arm moved cleanly toward the object and even to the drop location, but never gripped anything. This is the most stark counter-evidence to the "more data is better" assumption in the project. The mechanism is the gripper-rarity argument from §6: gripper transitions are ~1% of frames, the MSE objective optimizes for the 99% of "no transition" frames, and with 4× more demonstrations the trajectory diversity drowns out the gripper signal entirely.

**Memorization signals — two distinct patterns.** Even at the optimal 10-episode setting, training too long produced position-specific overfitting visible in the OOD evaluation:

- *Gradual memorization* (`pi05_ur5_blueblock10-1`). At checkpoint 150, ID and OOD scores were 0.75 and 0.70 — essentially the same; the policy generalized to block positions outside the training area. By checkpoint 210, ID was still 0.75 but OOD had collapsed to 0.25 (every trial crashed at pick). The policy hadn't gotten better at the trained positions; it had gotten worse at the untrained ones. This is the classic widening-generalization-gap signature of overfitting: early in training the model learns the *task structure* (reach + grasp + transport + release); late in training it starts memorizing *exact joint trajectories* that work on the in-distribution positions and fail when those trajectories don't apply. The training loss continues to fall throughout — there is no loss-side signal that this is happening; only on-robot OOD evaluation reveals it.
- *Catastrophic memorization* (two-stage F-Fer transfer). Trained 10 000 steps on F-Fer's 800-episode dataset (heterogeneous bolt insertion / bearing pressing tasks on a different workstation), then fine-tuned 300 steps on our 10-episode blueblock dataset. The result was a fixed-pose policy that drove the arm to the same approximate joint configuration on every trial regardless of where the object was placed or what the prompt said. This is the failure mode at the limit: the model collapsed onto a single output mode that happened to be the safest interpolation between F-Fer's distribution and ours. Two earlier-stage-1 attempts (warm-starting from step 2 000 instead of 9 999) showed the same collapse, so the cause was the small stage-2 dataset, not over-baking stage 1.

**What this says about adaptation complexity.** Three useful observations crystallized from these experiments:

1. *Foundation-policy fine-tuning has a sweet-spot in data quantity, not a monotonic improvement.* For a single-task pick-and-place on a known robot, that sweet spot is around 10 well-rehearsed episodes. Below 10, behaviors aren't reliable; above 20, the trajectory variance starts overwhelming sparse-event signals.
2. *Demonstration consistency matters more than demonstration count.* Ten episodes recorded by one operator following the same approach pattern outperform forty episodes recorded with deliberate variance. This is the opposite of what supervised image classification teaches; for behavior cloning, variance is noise unless the model has the capacity to disentangle it.
3. *The signal density of rare events caps what data quantity can achieve.* If your task includes a behavior that occurs in 1% of frames (gripper toggle, mode switch, recovery from slip), no amount of additional balanced data will teach it; you need either targeted oversampling, a longer action horizon, or an architecture-level change. The gripper-oversample mechanism in `LeRobotUR5DataConfig.gripper_oversample_factor` exists exactly for this purpose but was not validated in the deployment runs.

These observations together explain why the hyperparameter sweep didn't follow the "throw more compute at it" pattern: the binding constraint was data design, not optimization budget.

### 8. Transfer to a more complex task

The recipe in §1 was validated on a single task (single-object pick-and-place from a 30 cm patch into a fixed bin). Whether it transfers to harder, longer tasks is the most important open question, and this project has only weak evidence either way.

**The closest data point** is `ur5_busthetable_2` (10 episodes, varied objects: red mug, yellow mug, white mug, fork, spoon, plastic tray, white bowl; bin location varied between episodes; some episodes had compound steps like "fork inside mug"). Training behaved normally — loss curves looked similar to the blueblock runs — but on-robot behavior was poor: the arm moved toward objects but couldn't reliably reach them, and when objects were manually placed in the gripper the arm started moving toward the bin but stopped or dropped midway. This pre-dates the §1 recipe (it used the older `ur5_busthetable` config with default warmup=1000 and computed-fresh norm stats), so it isn't a clean test of whether the recipe transfers. The qualitative conclusion was that 10 episodes spread across 7 distinct objects effectively gives 1.4 demonstrations per object — well below the consistent-narrow-distribution sweet spot the recipe assumes.

**What a clean transfer test would look like.** The right next experiment is to take the working recipe (pi0.5, 10 episodes, reload norm stats, warmup=50, full FT, early stop at ~150) and apply it to a *single* harder task — same data quantity, more task complexity, not more task variants. Two reasonable candidates:

- *Two-step manipulation*: 10 episodes of "pick up the blue block, place it in the bowl, pick up the bowl, place the bowl in the bin." This roughly doubles action horizon coverage and introduces a sequential dependency (the bowl pick is conditioned on the block being in it). Tests whether the recipe can learn temporally extended sub-goals.
- *Wider workspace*: 10 episodes of "pick up the blue block and place it in the cardboard box," but with the block placed across a 60 cm × 60 cm region instead of 30 × 30. Tests whether spatial generalization scales with training distribution width without breaking the gripper-signal density.

Neither has been run. The strong prior from this project's results is that *consistency-within-task* is the critical axis — so the two-step task (10 episodes of *the same* two-step) should work; the wider-workspace task may suffer from the same diluted-signal problem the v2_40 runs hit unless the per-position frame count stays high.

**Why the recipe might not transfer cleanly.** Two specific risks. First, the action horizon ceiling: pi0.5's 15-step chunk corresponds to 1.5 seconds of trajectory at 10 Hz; tasks that require longer-than-1.5-second open-loop commitment will need more re-planning, exposing weaknesses in the inference-time `HORIZON_STEPS` decision (see §10 — the inverted-U around H=6 may shift for longer tasks). Second, the visual encoder's grounding: pi0.5's PaliGemma is text-conditioned and may handle multi-step prompts ("pick up X, then put it in Y") differently from single-clause prompts; if the demonstrations are recorded with a single compound prompt, the encoder may bind the entire behavior to that string and fail to decompose it.

**The honest conclusion.** This project demonstrated *that a working policy is achievable* on a UR5 with 10 episodes and the right recipe; it did not demonstrate *that the recipe scales to harder tasks*. The transfer question deserves at least one focused experiment before treating §1 as a general procedure.

### 9. The hyperparameters that matter (training)

Listed in roughly decreasing order of effect on the deployed policy. For each, we cover what it is, what was tried, and the mechanism by which it affects behavior.

**Base checkpoint** (`weight_loader.params_path`). The pretrained weights you start from. Five distinct base checkpoints were used across the 87 runs:

| Base | Runs | Notes |
|---|---|---|
| `pi05_base` | 51 | The only base that produced a deployable policy. Pretrained on heterogeneous robot mix including UR5e. |
| `pi0_base` | 22 | Trained successfully but undertested at evaluation time. action_horizon=50 gives more gripper signal per training sample. |
| `pi0_fast_base` | 5 | Failed structurally on binary gripper (FAST tokenizer + DCT issue). |
| `pi05_droid` | 1 | Franka prior; high initial loss (1.5 vs 0.08 for pi05_base) suggests poor transfer. |
| `pi0_droid` | 2 | Similar concerns to pi05_droid. |

The base checkpoint matters more than any other single decision because it determines the inductive biases the model brings to fine-tuning — what kinds of behaviors are easy to learn, what action distributions feel "natural" to the model, which visual features are pre-extracted. `pi05_base` worked because it was already in the right neighborhood for our task; the others had pathways to working but were not pursued to deployment.

**Fine-tuning method.** Two methods were tried:

- *Full fine-tuning.* Updates all model parameters. Used in 72 runs. The only method confirmed to deploy. Memory cost: ~80 GB at batch 16 on H200; runs in ~25 min for 500 steps.
- *LoRA* (`gemma_2b_lora` with rank 16, `gemma_300m_lora` with rank 32). Updates low-rank adapters around the linear layers, freezing the base weights. Used in 10 runs. Trained successfully (losses comparable to full FT) but no LoRA checkpoint was ever evaluated on the robot. Memory cost is roughly 4× lower; training time is similar.

The fact that LoRA was never validated on robot is a real coverage gap — community evidence (openpi#692, #763) suggests LoRA can match or exceed full FT for small-data fine-tuning by acting as an implicit regularizer. This would be a useful follow-up.

**Normalization choices.** Two independent decisions controlled by the data config:

- `use_delta_action_transform` (boolean). When `True`, the dataloader subtracts the current state from the recorded action during training to produce a delta-action target; the inverse happens at inference. Maintainer guidance in the issue tracker recommends keeping this on as the default — delta prediction is generally easier for the model to learn and more robust to small calibration mismatches. Turning it off was tried in 10 runs in this project (an early hypothesis that absolute datasets needed absolute prediction) and none deployed; subsequent runs all kept it on.
- `use_quantile_norm` (boolean). Selects between z-score normalization (mean / std, used for pi0/pi0.5) and quantile normalization (q01 / q99, required for pi0_FAST whose tokenizer assumes [-1, 1] inputs). The data config in this project automatically sets it via `use_quantile_norm = (model_type == PI0_FAST)` per openpi#763. Earlier experiments that forced quantile norm on pi0/pi0.5 unintentionally narrowed the joint coverage; the auto-selection prevents that mistake.

**`warmup_steps`.** The number of steps over which the LR ramps from 0 to `peak_lr` at the start of training. Tested at 50 (59 runs), 100 (2 runs), 1000 (23 runs), 2000 (3 runs). With a 500-step training budget and warmup=1000, the LR never reaches peak — the entire training run executes at a fraction of the configured LR, acting as an implicit regularizer. This was the load-bearing factor in why early experiments with default settings failed: warmup=1000 wasn't a deliberate choice, it was the openpi default inherited from runs that train for 30 000 steps. Setting warmup=50 was the single most impactful config change in the project.

**`peak_lr`.** The LR after warmup, before decay begins. Tested values: 5e-7, 2e-6, 5e-6, 1e-5, 2.5e-5. The vast majority (71 runs) used 2.5e-5; 12 runs used 1e-5; the very low values (5e-7, 2e-6) were attempts at fine-grained recovery from overfit checkpoints and did not produce notable improvements. Within 1e-5 to 2.5e-5 the difference is small; below 1e-5 training visibly slows; above 2.5e-5 was not tried but pi0.5's documentation suggests it would be unstable with the gemma-2b backbone.

**`decay_steps`.** How many steps the cosine LR schedule decays over after warmup. Default is 30 000 (much longer than any of our actual training budgets), which means the LR stays close to peak throughout a 500-step run. A few experiments used shorter decay (200, 250, 500, 2000) to deliberately drop the LR mid-run for fine-tuning; results were inconclusive.

**`batch_size`.** Per-step batch. Tested at 16 (53 runs), 32 (28 runs), 64 (6 runs, only LoRA). 16 vs 32 produces indistinguishable loss curves on this dataset and similar runtime per epoch. 16 is preferred because it leaves headroom for FSDP across smaller GPUs and allows running multiple experiments concurrently. Larger batches (64) were only feasible with LoRA's reduced memory footprint.

**`num_train_steps` and `early_stop_patience`.** The maximum step count and the early-stopping condition. Tested step ranges: 110 to 10 000. The functional rule that emerged: train with `num_train_steps=500`, `early_stop_patience=20`, `early_stop_min_delta=1e-4`, `log_interval=10`, and pick a checkpoint between steps 130 and 200. Beyond ~210 steps the policy starts position-memorizing (see §7); below ~120 it hasn't learned to reach. The 20-step patience with 10-step logging means early stopping evaluates 200 windows of "no loss improvement" before terminating — generous enough to avoid premature stops but tight enough to catch flat plateaus.

**`save_interval`.** How often to write a checkpoint. Default is 1000 — useless on a 500-step budget. Set to 30 throughout the production runs, which produces ~17 checkpoints per 500-step run; combined with the every-30-step granularity, this is what enabled the per-checkpoint evaluation sweep.

**Model architecture dimensions.** `model.action_dim` (32 for pi0/pi0.5, 7 for pi0_FAST), `model.action_horizon` (15 for pi0.5, 50 for pi0, 10 for pi0_FAST, 30 for the F-Fer pi0 variant), `model.max_token_len` (180 throughout). These are not free hyperparameters in practice — each base checkpoint expects specific values, and mismatches at load time produce shape errors. The interesting choice is action_horizon: pi0's 50-step horizon gives ~3× more gripper-signal exposure per training sample than pi0.5's 15-step, which is one reason pi0 sometimes succeeds where pi0.5 fails on sparse-event datasets.

**`ema_decay`.** Exponential moving average over training weights. Always 0.99 in full-FT runs, always disabled in LoRA. Never independently varied. EMA generally stabilizes training and provides a smoother set of "soft" weights for inference; we never tested whether 0.999 or 0.95 would change behavior, which is an unmeasured knob.

**`freeze_filter`.** A pattern that controls which parameters get frozen during training. Used only to enable LoRA (freezing all base weights so only the adapters update); for full FT it's left unset.

**Less-consequential parameters within the explored ranges**: `optimizer.b1=0.9`, `b2=0.95`, `eps=1e-8`, `weight_decay=1e-10`, `clip_gradient_norm=1.0` — defaults throughout, never varied. `seed` — single seed throughout the entire 87-run sweep, which is the largest unmeasured risk in the project (we have no per-run variance estimate on training, so any single result could be a lucky/unlucky seed). `pytorch_training_precision` — defaulted to bfloat16; one-off experiments at fp32 didn't change behavior visibly.

### 10. The hyperparameters that matter (inference)

Inference settings live in `ur5/docker/serve_policy_robot.Dockerfile` as environment variables and can be overridden per launch with `-e VAR=value`. The settings below are the ones that change real-world success rate; other knobs (servo control parameters, camera identifiers, exposure values) are bridge-side details that, once set correctly to match the recording, do not need per-policy tuning.

**Action-chunk playback.**

- `HORIZON_STEPS` (integer, default 6 in dockerfile, model trained with 15). The number of predicted action-chunk steps the bridge executes before re-querying the policy. The model returns a chunk of length 15 (pi0.5); `HORIZON_STEPS=6` means we play the first 6 steps, then drop the remaining 9 and ask for a new chunk based on updated observations. This is the most consequential single inference setting. The horizon sweep at checkpoint 150 showed:
  - H=3: 0.25 — re-queries before grasp completes; gripper command gets overwritten before it actuates; failure is "crashed on grasp."
  - H=6: 0.90 — sweet spot. Long enough for the gripper to close in one chunk; short enough to react to errors.
  - H=9: 0.50 — intermediate; some trials commit too long to a stale plan.
  - H=12 / H=15: 0.25 — commits to the full predicted trajectory regardless of what's happening; same crashed-on-grasp failure mode as H=3 but for a different reason (no closed-loop correction).
  This is an inverted-U with the optimum at a fraction of the training horizon, not the training horizon itself. The intuition: training optimizes the model to predict 15 coherent action steps from a single observation, but at inference the model gets a *new* observation every chunk-replan, and the value of incorporating that new observation outweighs the cost of replanning before the chunk completes.

- `HOLD_PER_STEP` (seconds, default 0.1). The wall-clock interval between consecutive action-chunk steps. Set to 0.1 to match the 10 Hz recording rate. The dataset captures one frame every 100 ms; the model learned that "one action step = 100 ms of motion." At inference, if `HOLD_PER_STEP=0.05` the bridge plays the chunk at 2× real-time and joints accelerate beyond what the recorded motion implied; if `HOLD_PER_STEP=0.2` it plays at half speed and grasps that should occur before the gripper actuates instead happen during it. Match the recording rate exactly.

**State and prompt.**

- `policy_metadata.reset_pose` (radians, baked into training config). The 6-vector of joint angles the bridge commands at the start of each trial. Must equal the dataset's first-frame pose to within a few degrees. For our config: `(-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0)` corresponding to `(-90°, -40°, -140°, -50°, 90°, 0°)`. Off-distribution start poses propagate through the action chunk and produce systematic trajectory failures that look like model bugs.
- `PROMPT` (string). The natural-language task instruction passed to pi0.5's PaliGemma encoder. Must lexically match a task label seen in training; pi0.5 is sensitive enough on small fine-tunes that "Pick up the blue block and place it in the cardboard box" works while "Grab the blue block and put it in the box" measurably degrades behavior. The dataset's `tasks.jsonl` is the source of truth for valid prompts.

**Server lifecycle.**

- `SERVER_ARGS` (string passed to `serve_policy.py`). Specifies the policy class and checkpoint path. The current default points at `pi05_ur5_blueblock10-1/150`; changing this is how you swap deployed checkpoints.
- `SERVER_WAIT` (seconds, default 6). How long the bridge waits after launching the policy server before connecting. 6 s is enough for JAX to JIT-compile the model on first call; bumps up to 60+ s if you forget to pre-compile.
- `DRY_RUN` (0/1). When 1, the bridge logs intended joint commands without sending them to the robot. Useful for verifying camera + policy response without risking damage; we used it after every config change before authorizing real motion.

**Why these matter.** Inference settings have zero effect on training loss, so a wandb dashboard cannot help you tune them. They are tuned by on-robot evaluation only. The horizon sweep shows just how big the effect can be: same model checkpoint, same dataset, same training config — `HORIZON_STEPS=6` gives 90% success, `HORIZON_STEPS=15` gives 25%. Practically: log the deployment-relevant env vars (`HORIZON_STEPS`, `HOLD_PER_STEP`, `PROMPT`, `SERVER_ARGS`) alongside every evaluation result, because the inference config is part of the model.
