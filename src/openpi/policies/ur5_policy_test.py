import numpy as np
import pytest

from openpi.policies import ur5_policy
from openpi.models import model as _model


def test_ur5_inputs_outputs():
    """Test that UR5Inputs and UR5Outputs correctly transform data shapes and keys."""
    ex = ur5_policy.make_ur5_example()
    # add dummy actions
    ex["actions"] = np.random.randn(5, 12).astype(np.float32)

    # Test PI05 model type
    inp = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI05)(ex)
    assert inp["state"].shape == (7,), f"Expected state shape (7,), got {inp['state'].shape}"
    assert set(inp["image"].keys()) >= {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert inp["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert inp["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert inp["image"]["right_wrist_0_rgb"].shape == (224, 224, 3)

    # Test output transform
    out = ur5_policy.UR5Outputs()({"actions": ex["actions"]})
    assert out["actions"].shape == (5, 7), f"Expected actions shape (5, 7), got {out['actions'].shape}"

    # Test PI0_FAST model type
    inp_fast = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0_FAST)(ex)
    assert inp_fast["state"].shape == (7,)
    assert set(inp_fast["image"].keys()) >= {"base_0_rgb", "base_1_rgb", "wrist_0_rgb"}


def test_ur5_inputs_joint_validation():
    """Test that UR5Inputs validates state size."""
    ex = ur5_policy.make_ur5_example()
    ex["observation/state"] = np.random.randn(5).astype(np.float32)

    with pytest.raises(ValueError, match="UR5 expects state of size 7"):
        ur5_policy.UR5Inputs(model_type=_model.ModelType.PI05)(ex)


def test_ur5_outputs_validation():
    """Test that UR5Outputs validates action dimensions."""
    # Test with too few dimensions
    with pytest.raises(ValueError, match="need at least 7"):
        ur5_policy.UR5Outputs()({"actions": np.random.randn(5, 5).astype(np.float32)})

    # Test with 1D array (should work)
    out = ur5_policy.UR5Outputs()({"actions": np.random.randn(10).astype(np.float32)})
    assert out["actions"].shape == (1, 7)

