import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import ur5_policy


def test_ur5_inputs_pi05():
    ex = ur5_policy.make_ur5_example()
    out = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI05)(ex)
    assert out["state"].shape == (7,)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_ur5_inputs_pi0_fast():
    ex = ur5_policy.make_ur5_example()
    out = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0_FAST)(ex)
    assert out["state"].shape == (7,)
    assert set(out["image"]) == {"base_0_rgb", "base_1_rgb", "wrist_0_rgb"}


def test_ur5_inputs_validates_state_shape():
    ex = ur5_policy.make_ur5_example()
    ex["observation/state"] = np.random.randn(5).astype(np.float32)

    with pytest.raises(ValueError, match="Expected observation/state shape"):
        ur5_policy.UR5Inputs(model_type=_model.ModelType.PI05)(ex)


def test_ur5_outputs_slices():
    out = ur5_policy.UR5Outputs()({"actions": np.random.randn(5, 12).astype(np.float32)})
    assert out["actions"].shape == (5, 7)

