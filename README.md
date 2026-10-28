# Multi-sensor track fusion EKF

Four-state east/north position and velocity tracking with a C99 numerical core and
Python association layer. Radar range/bearing and passive optical bearing reports
arrive asynchronously in a common local metric coordinate frame.

```mermaid
flowchart LR
  R[Radar polar reports] --> V[Validated measurement]
  O[Optical bearings] --> V
  V --> P[Predict each track to report time]
  P --> G[Mahalanobis gate and nearest neighbor]
  G --> J[C99 Joseph covariance update]
  J --> T[Track state and covariance]
```

## Quickstart (30 Seconds)

```sh
pip install -e '.[test]'
track-fusion --cc gcc
pytest -q
python benchmarks/benchmark.py
```

The CLI explicitly compiles the shared library into a source-hashed user cache.
The wheel includes C source; GCC/Clang must be installed. Library callers construct
`NativeCore(build_native())`, seed a `TrackManager`, and ingest validated
`Measurement` instances. State order is east, north, east velocity, north velocity.
Bearings are radians counterclockwise from east; sensors share the origin.

| Parameter | Convention |
|---|---|
| Process noise | White acceleration spectral density, m squared/s cubed |
| Radar gate | Chi-square squared distance 9.21034, 2 DOF, 99 percent |
| Optical gate | Chi-square squared distance 6.63490, 1 DOF, 99 percent |
| Ordering | Nondecreasing timestamps, seconds |
| Covariance | Symmetric positive definite seed, Joseph measurement update |

Reports outside all gates return None. Track birth is explicit because a bearing
alone cannot determine range. Association is sequential nearest neighbor, not joint
assignment: dense scans should use an external scan-level assignment stage. Track
deletion, sensor extrinsics, missed-detection likelihoods and out-of-sequence replay
are application responsibilities. Near-origin linearizations are rejected.

## Benchmark comparison

`benchmarks/results.json` compares the same two-dimensional Joseph update in NumPy
and the native core including ctypes and copy overhead. Ten timed batches follow
one warmup; each batch performs 10,000 updates. No association or I/O is included.

| Implementation | Numerical method | Measured median, microseconds/update |
|---|---|---:|
| Validated C99 + ctypes | Fixed 4x4 loops, analytic 1x1/2x2 inverse | 51.71 |
| NumPy algebra baseline | Matrix multiplication and inversion | 15.15 |

Measured on Windows x86-64 with Python 3.12.14 and GCC -O3. The native wrapper
validates shapes, finite values and positive covariances; the NumPy baseline
assumes valid inputs. For these tiny matrices, validation and boundary costs exceed
the arithmetic savings. These results do not establish an end-to-end speedup.

Tests compare 40 random native updates against NumPy, verify analytic prediction,
association, gating, angle wrap and source/package consistency. MIT.
