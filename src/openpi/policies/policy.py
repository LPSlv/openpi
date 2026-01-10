from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def model_type(self) -> _model.ModelType:
        """Get the model type from the policy metadata."""
        if "model_type" in self._metadata:
            # Convert string value back to enum if needed
            model_type_val = self._metadata["model_type"]
            if isinstance(model_type_val, str):
                return _model.ModelType(model_type_val)
            return model_type_val
        # Fallback: try to get from model config if available
        # This shouldn't happen in normal usage, but provides a fallback
        raise AttributeError("model_type not found in policy metadata. This should not happen.")

    def with_io(
        self,
        input_transform: _transforms.DataTransformFn | None = None,
        output_transform: _transforms.DataTransformFn | None = None,
    ) -> "Policy":
        """Create a new Policy with swapped input/output transforms.

        The new transforms are prepended to input transforms (applied first) and
        prepended to output transforms (applied first to outputs).

        Args:
            input_transform: Optional new input transform to prepend. If None, keeps existing transforms.
            output_transform: Optional new output transform to prepend. If None, keeps existing transforms.

        Returns:
            A new Policy instance with the specified transforms.
        """
        # Extract existing transforms from the composed transform
        existing_input_transforms = (
            list(self._input_transform.transforms) if isinstance(self._input_transform, _transforms.CompositeTransform) else []
        )
        existing_output_transforms = (
            list(self._output_transform.transforms)
            if isinstance(self._output_transform, _transforms.CompositeTransform)
            else []
        )

        # Replace the first input transform (repack transform) and last output transform
        # This swaps the environment-specific transforms while keeping normalization, etc.
        new_input_transforms = existing_input_transforms
        if input_transform is not None:
            if existing_input_transforms:
                # Replace first transform (repack transform) with new one
                new_input_transforms = [input_transform] + existing_input_transforms[1:]
            else:
                new_input_transforms = [input_transform]

        new_output_transforms = existing_output_transforms
        if output_transform is not None:
            if existing_output_transforms:
                # Replace last transform (repack transform) with new one
                new_output_transforms = existing_output_transforms[:-1] + [output_transform]
            else:
                new_output_transforms = [output_transform]

        # Don't pass rng - let the Policy constructor create a new one
        # This avoids issues with reusing JAX keys and works for both JAX and PyTorch models
        return Policy(
            self._model,
            rng=None,  # Will be created by Policy.__init__ if needed
            transforms=new_input_transforms,
            output_transforms=new_output_transforms,
            sample_kwargs=self._sample_kwargs,
            metadata=self._metadata,
            pytorch_device=self._pytorch_device,
            is_pytorch=self._is_pytorch_model,
        )


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
