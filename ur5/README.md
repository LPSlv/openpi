# UR5 Integration for OpenPI

This directory contains all UR5-specific code for the OpenPI fork: robot control,
data collection, Docker deployment, hardware diagnostics, and HPC training scripts.

## Directory Structure

```
ur5/
  docker/
    serve_policy_robot.Dockerfile     GPU Docker image: policy server + robot bridge
  scripts/
    ur5_record_freedrive_waypoints.py Step 1: record sparse waypoints in freedrive
    ur5_replay_and_record_raw.py      Step 2: replay waypoints, record dataset at 10 Hz
    convert_ur5_raw_to_lerobot.py     Step 3: convert raw episodes to LeRobot format
    convert_raw_deltas_to_absolute.py Utility: migrate old delta actions to absolute
    download_checkpoints.py           Build-time checkpoint download helper
    ur5e_diagnostics.py               Hardware diagnostics and norm stats verification
    fix_opencv.sh                     Docker OpenCV reinstall helper
    README.md                         Operational notes, training commands, FFmpeg guide
    hpc/                              HPC (Slurm + Singularity) job scripts
  utils/
    pi0_bridge_ur5_headless.py        Main inference bridge (RealSense + RTDE + websocket)
  test/
    rs_list.py                        List connected RealSense cameras
    camera_test.py                    Dual-camera live preview
    ur_read_state.py                  Read UR5 joint state via RTDE
    ur_test_movement.py               Safe base joint nudge + gripper test
    README.md                         Test scripts setup guide
  docs/
    quickstart.md                     Getting started from scratch
    data_pipeline.md                  Full data recording and conversion workflow
    training.md                       Training guide (norm stats, configs, HPC)
    deployment.md                     Docker build, run, and env var reference
    experiments.md                    Lab notebook / experiment log
    research_report.md                System research report
    ur5e_boot_failure_report.md       Hardware troubleshooting
    ur5e_boot_failure_diagnostic_D7.md
    papers/
      pi0.pdf                         Pi0 model paper
      pi05.pdf                        Pi0.5 model paper
```

## Related Files in the Main OpenPI Tree

These files follow the upstream project's conventions and live alongside other robot
policies/configs:

- `src/openpi/policies/ur5_policy.py` -- model I/O transforms (UR5Inputs, UR5Outputs)
- `src/openpi/training/config.py` -- training configs (search for `ur5`)
- `examples/ur5/` -- example environment, inference entry point, and README

## Workflow Overview

1. **Record** sparse waypoints in freedrive mode ([data_pipeline.md](docs/data_pipeline.md))
2. **Replay** waypoints while recording images + proprio at 10 Hz
3. **Convert** raw episodes to LeRobot format, push to Hugging Face
4. **Train** on HPC or locally ([training.md](docs/training.md))
5. **Deploy** via Docker on the robot workstation ([deployment.md](docs/deployment.md))

See [quickstart.md](docs/quickstart.md) to get started from scratch.
