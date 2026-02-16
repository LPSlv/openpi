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

  2nd dataset created: LPSlvlv/ur5_busthetable_2. Consisted of 10 episodes of picking dishes from a table and putting in a bin. Different objects were used like red, yellow, white mug. Fork and a spoon, plastic tray and a white bowl. Different variations were created, but most consisted of 3 objects on the table and around 30 waypoints recorded. Fork and spoon was placed on a table, inside the mug and bowl. Cup was also placed in a bowl but picked up seperately. Bin location was changed, but mostly stayed in the right. 

  Experiments with this, didnt show better results. The model sometimes moved towards the object, but couldnt reach it. If it was placed in the grippers, the robot gripped it, but sometimes dropped after a few seconds, sometimes started to move to drop of location, but at the end not reaching it fully. This could be because the environment was too varied for the dataset size, instead acting not like 10 episodes but like 10x1 single episodes.

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


  UR5_third_2:

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



UR5_THIRD_3:

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


UR5_THIRD_4:

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

UR5_THIRD_5:

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


Now I will try the pi0_ur5 config with training: UR5_THIRD_3 that was overfitted. It was moving good to object, but not gripping good.

Result: Didnt want to grip it.



Lets try with the same non overfit model. UR5_THIRD_4

Gripping now works. But still movemement is not good enough. 


Lets try UR5_THIRD_5 that uses pi05 droid checkpoint. Loss showed a interesting curve, it started way higher (1.5 vs 0.08)



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





                               Timing Statistics                               
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┓
┃ Metric              ┃ Mean ┃ S… ┃  P25 ┃  P50 ┃   P75 ┃  P90 ┃   P95 ┃  P99 ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━┩
│ client_infer_ms     │ 232… │ 1… │ 231… │ 232… │ 233.5 │ 235… │ 235.5 │ 237… │
│ policy_infer_ms     │ 187… │ 0… │ 187… │ 187… │ 188.2 │ 188… │ 188.4 │ 189… │
│ server_infer_ms     │ 231… │ 1… │ 230… │ 231… │ 232.1 │ 233… │ 234.0 │ 235… │
│ server_prev_total_… │ 233… │ 3… │ 231… │ 232… │ 233.7 │ 235… │ 238.2 │ 245… │
└─────────────────────┴──────┴────┴──────┴──────┴───────┴──────┴───────┴──────┘

From the policy:

action_horizon = 15 (15 action steps per inference)
Dataset recorded at 10Hz → each step = 0.1s of real time
Recommended bridge settings for pi05_ur5:


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
Timing per cycle:

Inference: ~232ms
Execution: 15 steps x 0.1s = 1.5s
Total cycle: ~1.73s
Why these values:

Parameter	Value	Reasoning
HORIZON_STEPS=15	Matches action_horizon=15 from policy	
HOLD_PER_STEP=0.1	1/10Hz = 0.1s, matches dataset timestep	
DT=0.02	5 servoJ updates per action step (smooth)	
LOOKAHEAD=0.1	Balances smoothness vs responsiveness	
GAIN=300	Stiffer than 150 (previous) — absolute targets can handle it since the model predicts positions close to current state	
For docker, that would be:


-e ACTION_MODE=absolute \
-e HORIZON_STEPS=15 \
-e HOLD_PER_STEP=0.1 \
-e DT=0.02 \
-e VEL=0.5 \
-e ACC=0.5 \
-e LOOKAHEAD=0.1 \
-e GAIN=300
