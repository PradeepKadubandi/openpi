"""Bounded host spans plus native profiler traces. No simulator imports.

The OpenPI copy intentionally has the same format without a cross-repository dependency.
Host spans are inclusive and must not be summed with their children or GPU events.
"""

from contextlib import contextmanager
from contextlib import suppress
from functools import wraps
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

logger = logging.getLogger(__name__)
KEY = "__b1k_performance__"


def measured(name):
    """After capture this decorator only checks an attribute and a boolean."""

    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            capture = getattr(self, "performance", None)
            if capture is None or not capture.active:
                return fn(self, *args, **kwargs)
            with capture.span(name):
                return fn(self, *args, **kwargs)

        return call

    return decorate


class Capture:
    def __init__(self, directory, *, backend="torch", session=None, metadata=None):
        self.directory = Path(directory)
        self.session = session or uuid.uuid4().hex
        self.backend = backend
        self.metadata = dict(metadata or {})
        self.records = []
        self.errors = []
        self.active = False
        self.done = False
        self.step = 0
        self.profiler = None
        self.origin = 0
        self.clock = time.perf_counter_ns

    def error(self, exc):
        message = str(exc)
        if not self.errors:
            logger.warning("Performance capture: %s", message)
        if message not in self.errors and len(self.errors) < 10:
            self.errors.append(message)

    def start(self):
        if self.active or self.done:
            return
        begin = self.clock()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.metadata["python"] = platform.python_version()
            self.metadata["checkpoint"] = os.environ.get("B1K_POLICY_CHECKPOINT")
            for label, args in [
                ("revision", ["git", "rev-parse", "HEAD"]),
                ("tracked_diff_stat", ["git", "diff", "--stat"]),
            ]:
                try:
                    self.metadata[label] = subprocess.check_output(args, text=True, timeout=5).strip()
                except (OSError, subprocess.SubprocessError):
                    self.metadata[label] = "unavailable"
            if self.backend == "jax":
                import jax

                self.metadata.update(framework_version=jax.__version__, devices=[str(d) for d in jax.devices()])
                jax.profiler.start_trace(str(self.directory), create_perfetto_trace=True)
                self.profiler = jax.profiler
            elif self.backend == "torch":
                import torch

                activities = [torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                self.metadata.update(
                    framework_version=torch.__version__,
                    cuda_available=torch.cuda.is_available(),
                    devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                )
                profiler = torch.profiler.profile(
                    activities=activities, record_shapes=False, profile_memory=False, with_stack=False
                )
                self.profiler = profiler
                profiler.start()
        except Exception as exc:
            self.error(exc)
            if self.profiler is not None:
                with suppress(Exception):
                    self.profiler.stop()
                self.profiler = None
        self.metadata["profiler_start_ms"] = (self.clock() - begin) / 1e6
        self.origin = self.clock()
        self.active = True

    @contextmanager
    def span(self, name):
        begin = self.clock()
        annotation = None
        if self.profiler is not None:
            try:
                if self.backend == "jax":
                    annotation = self.profiler.TraceAnnotation(name, step=self.step)
                else:
                    import torch

                    annotation = torch.profiler.record_function(name)
                annotation.__enter__()
            except Exception as exc:
                self.error(exc)
                annotation = None
        try:
            yield
        finally:
            if annotation is not None:
                try:
                    annotation.__exit__(None, None, None)
                except Exception as exc:
                    self.error(exc)
            end = self.clock()
            self.records.append(
                {"step": self.step, "name": name, "start_ns": begin - self.origin, "duration_ns": end - begin}
            )

    def stop(self):
        """Stop native collection before remote control/export waits."""
        self.active = False
        if self.profiler is None:
            return
        begin = self.clock()
        try:
            if self.backend == "jax":
                self.profiler.stop_trace()
                self.profiler = None  # JAX exports as part of stop_trace.
            else:
                self.profiler.stop()
        except Exception as exc:
            self.error(exc)
            self.profiler = None
        self.metadata["profiler_stop_ms"] = (self.clock() - begin) / 1e6

    def finish(self, *, complete=False):
        if self.done:
            return
        if "profiler_stop_ms" not in self.metadata:
            self.stop()
        self.active = False
        self.done = True
        begin = self.clock()
        try:
            if self.profiler is not None:
                self.profiler.export_chrome_trace(str(self.directory / "trace.json"))
        except Exception as exc:
            self.error(exc)
        finally:
            self.profiler = None
        self.metadata["profiler_stop_export_ms"] = (self.clock() - begin) / 1e6
        totals = {}
        for row in self.records:
            totals[row["name"]] = totals.get(row["name"], 0) + row["duration_ns"] / 1e6
        classifications = {
            row["step"]: ("replan" if row["details"].get("inference") else "buffered")
            for row in self.records
            if row["name"] == "server"
        }
        groups = {}
        for row in self.records:
            if row["name"] == "end_to_end_step":
                group = classifications.get(row["step"], "unclassified")
                entry = groups.setdefault(group, {"steps": 0, "total_ms": 0})
                entry["steps"] += 1
                entry["total_ms"] += row["duration_ns"] / 1e6
        self.metadata["sampled_step_groups"] = groups
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "summary.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "session": self.session,
                        "backend": self.backend,
                        "complete": complete,
                        "metadata": self.metadata,
                        "errors": self.errors,
                        "inclusive_phase_ms": totals,
                        "records": self.records,
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            self.error(exc)
