import asyncio
from copy import deepcopy
import functools
import http
import logging
import os
from pathlib import Path
import time
import traceback
from typing import Any, Optional

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server

from openpi.shared.performance import KEY
from openpi.shared.performance import Capture
from openpi.shared.performance import measured

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["WebsocketPolicyServer"]


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        cycle = self._policy.action_horizon if self._policy.control_mode == "receding_horizon" else 1
        self._metadata[KEY] = {"version": 1, "cycle_length": cycle}
        self.performance = None
        self._captured_requests = 0

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}...")
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    @measured("server.decode")
    def _decode(self, raw):
        return unpackb(raw, strict_map_key=False)

    @measured("server.observation_copy")
    def _copy(self, obs):
        return deepcopy(obs)

    @measured("server.policy_act")
    def _act(self, obs):
        return self._policy.act(obs)

    @measured("server.pack")
    def _pack(self, packer, response):
        return packer.pack(response)

    @measured("server.action_to_numpy")
    def _action_to_numpy(self, action):
        return action.cpu().numpy()

    def _start_capture(self, session):
        if not isinstance(session, str) or len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
            raise ValueError("Invalid performance session")
        if self.performance is not None and not self.performance.done:
            self.performance.finish()
        base_policy = getattr(self._policy.policy, "_policy", self._policy.policy)
        backend = "torch" if getattr(base_policy, "_is_pytorch_model", False) else "jax"
        directory = Path(os.environ.get("B1K_PERFORMANCE_DIR", "performance")) / session
        self.performance = Capture(
            directory,
            backend=backend,
            session=session,
            metadata={"control_mode": self._policy.control_mode, "horizon": self._policy.action_horizon},
        )
        self._policy.performance = self.performance
        self._policy.policy.performance = self.performance
        base_policy.performance = self.performance
        self._captured_requests = 0
        self.performance.start()
        return {"session": session, "directory": str(directory), "backend": backend}

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        try:
            prev_total_time = None
            while True:
                try:
                    start_time = time.monotonic()
                    raw = await websocket.recv()
                    # Decode belongs to this request; the step ID is known only after unpacking.
                    record_start = len(self.performance.records) if self.performance and self.performance.active else 0
                    result = self._decode(raw)
                    profile_request = result.pop(KEY, None)
                    if profile_request and "command" in profile_request:
                        session = profile_request["session"]
                        if self.performance is not None and self.performance.active:
                            # Control traffic is not a sampled action request.
                            del self.performance.records[record_start:]
                        if profile_request["command"] == "start":
                            reply = self._start_capture(session)
                        elif profile_request["command"] == "stop":
                            if self.performance is not None and self.performance.session == session:
                                self.performance.finish(
                                    complete=self._captured_requests == self._metadata[KEY]["cycle_length"]
                                )
                            reply = {"session": session}
                        else:
                            raise ValueError("Unknown profiling command")
                        await websocket.send(packer.pack({KEY: reply}))
                        continue
                    if self.performance is not None and self.performance.active:
                        if (
                            profile_request
                            and profile_request.get("session") == self.performance.session
                            and self._captured_requests < self._metadata[KEY]["cycle_length"]
                        ):
                            self._captured_requests += 1
                            self.performance.step = profile_request["step"]
                            for record in self.performance.records[record_start:]:
                                record["step"] = self.performance.step
                        else:
                            self.performance.finish()

                    if "reset" in result:
                        self._policy.reset()
                        continue

                    obs = self._copy(result)

                    infer_time = time.monotonic()
                    action = self._act(obs)
                    infer_time = time.monotonic() - infer_time

                    action = {
                        "action": self._action_to_numpy(action),
                    }
                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    if self.performance is not None and self.performance.active:
                        action[KEY] = {
                            "session": self.performance.session,
                            "step": self.performance.step,
                            "inference": self._policy.performance_inference,
                        }
                        payload = self._pack(packer, action)
                        with self.performance.span("server.send"):
                            await websocket.send(payload)
                    else:
                        await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                    try:
                        # Try new websockets API first
                        await websocket.close(
                            code=websockets.frames.CloseCode.INTERNAL_ERROR,
                            reason="Internal server error. Traceback included in previous frame.",
                        )
                    except AttributeError:
                        # Fallback for older websockets versions
                        await websocket.close(code=1011, reason="Internal server error")
                    raise
        finally:
            if self.performance is not None:
                self.performance.finish()


def _health_check(connection, request) -> Optional[Any]:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        else:
            # For older websockets versions, return a simple response
            return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    # Continue with the normal request handling.
    return None


"""
Adds NumPy array and PyTorch tensor support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

This module supports serializing both NumPy arrays and PyTorch tensors. PyTorch tensors are converted to
NumPy arrays (zero-copy when possible) before serialization. On deserialization, arrays are returned as NumPy arrays.

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""


def pack_data(obj):
    if isinstance(obj, torch.Tensor):
        data = obj.detach().cpu().numpy()
        return {
            b"__ndarray__": True,
            b"data": data.tobytes(),
            b"dtype": data.dtype.str,
            b"shape": data.shape,
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
packb = functools.partial(msgpack.packb, default=pack_data)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)
