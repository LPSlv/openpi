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


  