import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import platform
import uuid
import numpy as np


def build_native(compiler: str = "gcc") -> Path:
    source = Path(__file__).parent / "native" / "ekf_step.c"
    fingerprint = (
        source.read_bytes()
        + f"{compiler}:{sys.platform}:{platform.machine()}:{ctypes.sizeof(ctypes.c_void_p)}".encode()
    )
    identity = hashlib.sha256(fingerprint).hexdigest()[:16]
    destination = Path.home() / ".cache" / "track-fusion" / identity
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / ("ekf.dll" if sys.platform == "win32" else "libekf.so")
    if library.is_file():
        return library
    candidate = destination / f"{uuid.uuid4().hex}{library.suffix}"
    try:
        subprocess.run(
            [
                compiler,
                "-std=c99",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
                str(source),
                "-lm",
                "-o",
                str(candidate),
            ],
            check=True,
        )
        try:
            os.replace(candidate, library)
        except OSError:
            if not library.is_file():
                raise
    finally:
        candidate.unlink(missing_ok=True)
    return library


class NativeCore:
    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path.resolve()))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.library.ekf_predict.argtypes = [
            pointer,
            pointer,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self.library.ekf_predict.restype = ctypes.c_int
        self.library.ekf_update.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
        ]
        self.library.ekf_update.restype = ctypes.c_int

    @staticmethod
    def _state(
        state: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        vector = np.array(state, dtype=np.float64, order="C", copy=True)
        matrix = np.array(covariance, dtype=np.float64, order="C", copy=True)
        if (
            vector.shape != (4,)
            or matrix.shape != (4, 4)
            or not np.isfinite(vector).all()
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, atol=1e-12)
        ):
            raise ValueError(
                "finite four-state vector and symmetric 4x4 covariance required"
            )
        np.linalg.cholesky(matrix)
        return vector, matrix

    def predict(
        self, state: np.ndarray, covariance: np.ndarray, dt: float, noise: float
    ) -> tuple[np.ndarray, np.ndarray]:
        predicted, uncertainty = self._state(state, covariance)
        if not self.library.ekf_predict(predicted, uncertainty, dt, noise):
            raise ValueError("native prediction rejected")
        if not np.isfinite(predicted).all() or not np.isfinite(uncertainty).all():
            raise ValueError("prediction overflow")
        return predicted, uncertainty

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        jacobian: np.ndarray,
        innovation: np.ndarray,
        variance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        posterior, uncertainty = self._state(state, covariance)
        residual = np.ascontiguousarray(innovation, dtype=np.float64)
        observation_matrix = np.ascontiguousarray(jacobian, dtype=np.float64)
        measurement_covariance = np.ascontiguousarray(variance, dtype=np.float64)
        dimensions = residual.size
        if (
            dimensions not in (1, 2)
            or residual.shape != (dimensions,)
            or observation_matrix.shape != (dimensions, 4)
            or measurement_covariance.shape != (dimensions, dimensions)
            or not all(
                np.isfinite(array).all()
                for array in (residual, observation_matrix, measurement_covariance)
            )
            or not np.allclose(
                measurement_covariance, measurement_covariance.T, atol=1e-12
            )
        ):
            raise ValueError("invalid measurement dimensions, values or covariance")
        np.linalg.cholesky(measurement_covariance)
        if not self.library.ekf_update(
            posterior,
            uncertainty,
            observation_matrix,
            residual,
            measurement_covariance,
            dimensions,
        ):
            raise ValueError("singular innovation covariance")
        if not np.isfinite(posterior).all() or not np.isfinite(uncertainty).all():
            raise ValueError("update overflow")
        return posterior, uncertainty
