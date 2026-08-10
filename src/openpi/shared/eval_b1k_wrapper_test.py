import numpy as np

from openpi.models.model import ModelType
from openpi.policies.b1k_policy import B1KInputs
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.shared.eval_b1k_wrapper import metric_depth_to_uint8


class _FakePolicy:
    def reset(self) -> None:
        pass


def test_metric_depth_to_uint8_with_fixed_bounds() -> None:
    depth = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]], dtype=np.float32)

    actual = metric_depth_to_uint8(depth, metric_bounds=(1.0, 4.0))

    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, [[0, 255, 60], [22, 0, 0]])


def test_metric_depth_to_uint8_is_inverse_depth_and_handles_invalid_frame() -> None:
    depth = np.linspace(0.5, 5.0, 100, dtype=np.float32).reshape(10, 10)

    actual = metric_depth_to_uint8(depth)

    assert actual.dtype == np.uint8
    assert actual[0, 0] == 255
    assert actual[-1, -1] == 0
    assert np.all(np.diff(actual.diagonal().astype(np.int16)) <= 0)
    np.testing.assert_array_equal(metric_depth_to_uint8(np.zeros((2, 3), dtype=np.float32)), 0)


def test_depth_wrapper_converts_metric_depth_to_model_images() -> None:
    wrapper = B1KPolicyWrapper(
        policy=_FakePolicy(),
        robot="s2rg/sim_r1pro_depth",
        text_prompt="Pick up the radio on the table.",
        control_mode="receding_horizon",
        action_horizon=15,
        max_len=32,
        obs_size=(4, 5),
        depth_bounds=(0.5, 5.0),
    )
    depth = np.linspace(0.5, 5.0, 20, dtype=np.float32).reshape(4, 5)
    obs = {
        "robot::proprio": np.zeros(50, dtype=np.float32),
        "external::zed_camera::depth_linear": depth,
        "external::left_realsense_camera::depth_linear": depth[..., None],
        "external::right_realsense_camera::depth_linear": np.repeat(depth[..., None], 3, axis=-1),
    }

    processed = wrapper.process_input(obs)

    assert len(processed) == 1
    for camera_id in range(3):
        image = processed[0][f"observation/image_{camera_id}"]
        assert image.shape == (4, 5, 3)
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image[..., 0], image[..., 1])
        np.testing.assert_array_equal(image[..., 1], image[..., 2])

    model_input = B1KInputs(robot_config=wrapper.robot, model_type=ModelType.PI05)(processed[0])
    assert model_input["state"].shape == (20,)
    assert tuple(model_input["image"]) == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    for image in model_input["image"].values():
        assert image.shape == (4, 5, 3)
        assert image.dtype == np.uint8


def test_depth_wrapper_preserves_preprocessed_uint8_depth() -> None:
    wrapper = B1KPolicyWrapper(
        policy=_FakePolicy(),
        robot="s2rg/sim_r1pro_depth",
        text_prompt="Pick up the radio on the table.",
        control_mode="receding_horizon",
        action_horizon=15,
        max_len=32,
        obs_size=(4, 5),
    )
    gray = np.arange(20, dtype=np.uint8).reshape(4, 5)
    obs = {
        "robot::proprio": np.zeros(50, dtype=np.float32),
        "external::zed_camera::depth_linear": gray,
        "external::left_realsense_camera::depth_linear": gray,
        "external::right_realsense_camera::depth_linear": gray,
    }

    processed = wrapper.process_input(obs)

    for camera_id in range(3):
        image = processed[0][f"observation/image_{camera_id}"]
        np.testing.assert_array_equal(image[..., 0], gray)


def test_depth_wrapper_supports_batched_channel_less_depth() -> None:
    wrapper = B1KPolicyWrapper(
        policy=_FakePolicy(),
        robot="s2rg/sim_r1pro_depth",
        text_prompt="Pick up the radio on the table.",
        control_mode="receding_horizon",
        action_horizon=15,
        max_len=32,
        obs_size=(4, 5),
    )
    depth = np.stack(
        [
            np.linspace(0.5, 5.0, 20, dtype=np.float32).reshape(4, 5),
            np.linspace(1.0, 3.0, 20, dtype=np.float32).reshape(4, 5),
        ]
    )
    obs = {
        "robot::proprio": np.zeros((2, 50), dtype=np.float32),
        "external::zed_camera::depth_linear": depth,
        "external::left_realsense_camera::depth_linear": depth[..., None],
        "external::right_realsense_camera::depth_linear": depth,
    }

    processed = wrapper.process_input(obs)

    assert len(processed) == 2
    assert processed[0]["observation/image_0"].shape == (4, 5, 3)
    assert processed[1]["observation/image_2"].dtype == np.uint8


def test_rgb_wrapper_path_is_unchanged() -> None:
    wrapper = B1KPolicyWrapper(
        policy=_FakePolicy(),
        robot="s2rg/sim_r1pro",
        text_prompt="Pick up the radio on the table.",
        control_mode="receding_horizon",
        action_horizon=15,
        max_len=32,
        obs_size=(4, 5),
    )
    rgb = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    obs = {
        "robot::proprio": np.zeros(50, dtype=np.float32),
        "external::zed_camera::rgb": rgb,
        "external::left_realsense_camera::rgb": rgb,
        "external::right_realsense_camera::rgb": rgb,
    }

    processed = wrapper.process_input(obs)

    np.testing.assert_array_equal(processed[0]["observation/image_0"], rgb)
