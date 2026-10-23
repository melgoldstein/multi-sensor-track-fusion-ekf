import os
import json
import platform
import statistics
import time
import numpy as np
from track_fusion.native_core import NativeCore, build_native

core = NativeCore(build_native(os.environ.get("CC", "gcc")))
state, covariance = np.zeros(4), np.eye(4)
jacobian, innovation, variance = np.eye(2, 4), np.ones(2), np.eye(2)


def numpy_update():
    gain = covariance @ jacobian.T @ np.linalg.inv(jacobian @ covariance @ jacobian.T + variance)
    correction = np.eye(4) - gain @ jacobian
    return (
        state + gain @ innovation,
        correction @ covariance @ correction.T + gain @ variance @ gain.T,
    )


results = {}
for label, operation in (
    ("native_ctypes", lambda: core.update(state, covariance, jacobian, innovation, variance)),
    ("numpy", numpy_update),
):
    durations = []
    for _ in range(11):
        started = time.perf_counter_ns()
        for _ in range(10000):
            operation()
        durations.append((time.perf_counter_ns() - started) / 1e7)
    results[label] = {"median_us": statistics.median(durations[1:])}
print(
    json.dumps(
        {"platform": platform.platform(), "python": platform.python_version(), "results": results},
        indent=2,
    )
)
