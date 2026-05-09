# UR5 Integration for OpenPI

UR5-specific code: robot control, data collection, Docker deployment,
diagnostics, and the docs that go with them.

```
ur5/
  defaults.py                       hardware defaults (UR_IP, camera serials, gripper port)
  docker/
    serve_policy_robot.Dockerfile   inference image (policy server + robot bridge)
  utils/
    pi0_bridge_ur5_headless.py      inference bridge (RealSense + RTDE + websocket)
    rtde_utils.py                   RTDE helpers shared by recording + bridge
    robotiq_gripper.py              Robotiq URCap socket client (port 63352)
  scripts/
    ur5_record_freedrive_waypoints.py    1. record sparse waypoints in freedrive
    ur5_replay_and_record_raw.py         2. replay waypoints + record episode at 10 Hz
    convert_ur5_raw_to_lerobot.py        3. convert raw episodes to LeRobot format
    convert_raw_deltas_to_absolute.py    one-shot: migrate old delta datasets
    combine_and_split_ur5_datasets.py    build nested ablation datasets
    download_checkpoints.py              docker build-time checkpoint helper
  test/
    rs_list.py                      list connected RealSense cameras
    camera_test.py                  dual-camera live preview
    README.md                       hardware test setup
  charts/                           evaluation figures (PDF + PNG, eval_charts.ipynb)
  docs/
    quickstart.md                   end-to-end workflow (start here)
    training.md                     training configs, commands, lessons
    deployment.md                   docker bridge + env var reference
    experiments.md                  lab notebook
    final_evaluations.md            formal evaluation results
    papers/                         pi0.pdf, pi05.pdf
```

## UR5 files that live in the upstream openpi tree

These follow the upstream project's conventions:

- [`src/openpi/policies/ur5_policy.py`](../src/openpi/policies/ur5_policy.py): UR5Inputs / UR5Outputs transforms
- [`src/openpi/training/config.py`](../src/openpi/training/config.py): UR5 train configs (search for `ur5`)
- [`examples/ur5/`](../examples/ur5/): upstream-style runtime example and API tutorial

## Workflow

1. **record** sparse waypoints in freedrive mode
2. **replay** the waypoints while capturing images and proprio
3. **convert** to LeRobot format and (optionally) push to Hugging Face
4. **train** on the resulting dataset
5. **deploy** the policy server + bridge on the robot workstation

Start at [quickstart.md](docs/quickstart.md).
