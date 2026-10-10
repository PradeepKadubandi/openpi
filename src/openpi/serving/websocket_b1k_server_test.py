"""No GPU, checkpoint, simulator, or real sockets required."""

import asyncio
from types import SimpleNamespace

import numpy as np
import torch

from openpi.serving import websocket_b1k_server as server
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.shared.performance import KEY
from openpi.shared.performance import Capture


class Policy:
    _is_pytorch_model = False

    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": np.arange(48, dtype=np.float32).reshape(16, 3)}

    def reset(self):
        pass


def wrapper():
    value = B1KPolicyWrapper.__new__(B1KPolicyWrapper)
    value.policy = Policy()
    value.robot = SimpleNamespace(name="robot", observations={})
    value.obs_size = (2, 2)
    value.text_prompt = "test"
    value.control_mode = "receding_horizon"
    value.action_horizon = 16
    value.max_len = 16
    value.temporal_ensemble_max = 5
    value.reset()
    return value


def test_receding_actions_unchanged_by_capture(tmp_path):
    baseline, instrumented = wrapper(), wrapper()
    capture = Capture(tmp_path, backend="none")
    instrumented.performance = capture
    flags = []
    for step in range(1, 66):
        if step == 33:
            capture.start()
        capture.step = step
        obs = {"robot::proprio": np.zeros((1, 3), dtype=np.float32)}
        torch.testing.assert_close(baseline.act(obs), instrumented.act(obs))
        if capture.active:
            flags.append(instrumented.performance_inference)
        if step == 48:
            capture.finish(complete=True)
    assert flags == [True] + [False] * 15
    assert baseline.policy.calls == instrumented.policy.calls == 5
    baseline.reset()
    instrumented.reset()
    torch.testing.assert_close(baseline.act(obs), instrumented.act(obs))
    assert capture.done
    assert not capture.active


class DisconnectError(Exception):
    pass


class Socket:
    remote_address = "test"

    def __init__(self, requests):
        self.requests = iter(requests)
        self.responses = []

    async def recv(self):
        try:
            return server.packb(next(self.requests))
        except StopIteration:
            raise DisconnectError() from None

    async def send(self, payload):
        self.responses.append(server.unpackb(payload))


def test_control_metadata_removed_and_disconnect_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(server.websockets, "ConnectionClosed", DisconnectError)
    monkeypatch.setenv("B1K_PERFORMANCE_DIR", str(tmp_path))
    captures = []

    def capture_factory(directory, **kwargs):
        kwargs["backend"] = "none"
        capture = Capture(directory, **kwargs)
        captures.append(capture)
        return capture

    monkeypatch.setattr(server, "Capture", capture_factory)
    policy = wrapper()
    original = policy.act
    seen = []

    def act(obs):
        assert KEY not in obs
        seen.append(obs)
        return original(obs)

    policy.act = act
    session = "a" * 32
    obs = {"robot::proprio": np.zeros((1, 3), dtype=np.float32)}
    requests = [{KEY: {"command": "start", "session": session}}]
    requests += [dict(obs, **{KEY: {"session": session, "step": i}}) for i in range(33, 49)]
    requests += [{KEY: {"command": "stop", "session": session}}, obs, {"reset": True}]
    socket = Socket(requests)
    instance = server.WebsocketPolicyServer(policy)
    asyncio.run(instance._handler(socket))  # noqa: SLF001 - test protocol boundary
    assert len(seen) == 17
    assert socket.responses[0][KEY]["cycle_length"] == 16
    assert socket.responses[1][KEY]["session"] == session
    sampled = socket.responses[2:18]
    assert [r[KEY]["inference"] for r in sampled] == [True] + [False] * 15
    assert socket.responses[18][KEY]["session"] == session
    assert KEY not in socket.responses[19]
    assert captures[0].done
    assert not captures[0].active
    import json

    assert json.loads((tmp_path / session / "summary.json").read_text())["complete"]


def test_disconnect_during_capture_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(server.websockets, "ConnectionClosed", DisconnectError)
    monkeypatch.setenv("B1K_PERFORMANCE_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "Capture", lambda directory, **kw: Capture(directory, backend="none", session=kw["session"])
    )
    instance = server.WebsocketPolicyServer(wrapper())
    socket = Socket([{KEY: {"command": "start", "session": "b" * 32}}])
    asyncio.run(instance._handler(socket))  # noqa: SLF001 - test protocol boundary
    import json

    assert not json.loads((tmp_path / ("b" * 32) / "summary.json").read_text())["complete"]
