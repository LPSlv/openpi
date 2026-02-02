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

  