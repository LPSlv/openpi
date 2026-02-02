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


  

