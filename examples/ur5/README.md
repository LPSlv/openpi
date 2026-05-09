# UR5 Example

API tutorial for the UR5 transforms and training config. For the end-to-end
workflow (record, train, deploy) see [`ur5/docs/quickstart.md`](../../ur5/docs/quickstart.md).

## UR5Inputs / UR5Outputs

The transforms map a raw UR5 sample (joints + gripper + 2 RGB images) into the
keys the model expects, and slice the model's 7-dim output back out. The full
implementation lives in `src/openpi/policies/ur5_policy.py`.

```python
@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # joints (6) + gripper (1) -> state (7,)
        state = np.concatenate([data["joints"], data["gripper"]])

        # LeRobot stores images as float32 (C,H,W); the model wants uint8 (H,W,C)
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # image-key conventions differ between pi0/pi05 and pi0_fast
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # pi0_fast wants (base_0_rgb, base_1_rgb, wrist_0_rgb), no masking
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 6 DoF + gripper, drop padding dims
        return {"actions": np.asarray(data["actions"][:, :7])}
```

## LeRobotUR5DataConfig

Binds a HuggingFace LeRobot dataset to the openpi training pipeline. Full
implementation (with gripper oversampling, etc.) in `src/openpi/training/config.py`.

```python
@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):
    """Data pipeline config for training on LeRobot-formatted UR5 datasets."""

    use_delta_action_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "base_rgb": "image",
                    "wrist_rgb": "wrist_image",
                    "joints": "joints",
                    "gripper": "gripper",
                    "actions": "actions",
                    "prompt": "prompt",
                })
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # convert absolute actions to deltas for training; gripper (dim 6) stays absolute
        if self.use_delta_action_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

## TrainConfig

A minimal pi0 fine-tune config. See `src/openpi/training/config.py` for pi0-FAST,
pi0.5, LoRA, and gripper-oversampling variants.

```python
TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="<hf_username>/ur5_dataset",
        # reload UR5e norm stats from the base checkpoint instead of computing fresh ones;
        # see docs/norm_stats.md for when this helps
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```
