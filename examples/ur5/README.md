# UR5 Example

Below we provide an outline of how to implement the key components mentioned in the "Finetune on your data" section of the [README](../README.md) for finetuning on UR5 datasets.

## Collecting a UR5 freedrive dataset (raw episodes)

This repo includes simple, standalone scripts to:
- record **sparse joint-space waypoints** in freedrive/teach mode (no cameras required), and
- replay those waypoints with **UR motion commands** (`movej` + blending) while recording a dataset at fixed FPS (default **10 Hz**).

### 1) Record sparse freedrive waypoints (no cameras)

Run:

```bash
uv run python openpi/local/scripts/ur5_record_freedrive_waypoints.py \
  --ur_ip 192.168.1.116 \
  --prompt "do something" \
  --out_dir raw_episodes
```

This will create:
- `raw_episodes/<episode_id>/meta.json`
- `raw_episodes/<episode_id>/waypoints.json`

### 2) Replay and record a raw episode (10 Hz, images + proprio + actions)

You need two RealSense serials (external + wrist). You can list them with:

```bash
uv run python openpi/local/test/rs_list.py
```

Then run:

```bash
uv run python openpi/local/scripts/ur5_replay_and_record_raw.py \
  --ur_ip 192.168.1.116 \
  --waypoints_path raw_episodes/<episode_id>/waypoints.json \
  --rs_base_serial <RS_SERIAL_BASE> \
  --rs_wrist_serial <RS_SERIAL_WRIST> \
  --prompt "do something" \
  --out_dir raw_episodes \
  --fps 10
```

Raw episode format:

```
raw_episodes/<episode_id>/
  meta.json
  waypoints.json
  steps.jsonl
  images/
    base/000000.jpg
    wrist/000000.jpg
```

Each line in `steps.jsonl` includes:
- `image_path` / `wrist_image_path` (relative paths to JPEGs)
- `state` (float32, shape `(7,)`): `actual_q(6) + gripper_cmd(1)`
- `actions` (float32, shape `(7,)`): `delta_q(6) + absolute gripper_cmd(1)`
- `task`: the language instruction / prompt

First, we will define the `UR5Inputs` and `UR5Outputs` classes, which map the UR5 environment to the model and vice versa. Check the corresponding files in `src/openpi/policies/libero_policy.py` for comments explaining each line.

```python

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        state = np.concatenate([data["joints"], data["gripper"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}

```

Next, we will define the `UR5DataConfig` class, which defines how to process raw UR5 data from LeRobot dataset for training. For a full example, see the `LeRobotLiberoDataConfig` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

```python

@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):
    """Data pipeline config for training on LeRobot-formatted UR5 datasets."""

    # If true, interpret dataset actions as absolute (joint targets) and convert to deltas.
    # If your dataset already stores delta actions (as in openpi/local/scripts/ur5_replay_and_record_raw.py),
    # leave this as False.
    use_delta_action_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # Optionally convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        # Note: The raw recording script (ur5_replay_and_record_raw.py) already records delta actions,
        # so use_delta_action_transform should be False for datasets created with that script.
        if self.use_delta_action_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

```

Finally, we define the TrainConfig for our UR5 dataset. Here, we define a config for fine-tuning pi0 on our UR5 dataset. See the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py) for more examples, e.g. for pi0-FAST or for LoRA fine-tuning.

```python
TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_dataset",
        # This config lets us reload the UR5 normalization stats from the base model checkpoint.
        # Reloading normalization stats can help transfer pre-trained models to new environments.
        # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. The recommended setting is True.
            prompt_from_task=True,
        ),
    ),
    # Load the pi0 base model checkpoint.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```





