import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .native_core import NativeCore, build_native
from .maha_association import linearize, mahalanobis_squared


class Measurement(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, strict=True
    )
    timestamp: float = Field(ge=0)
    sensor: Literal["radar", "optical"]
    bearing_rad: float
    range_m: float | None = Field(default=None, gt=0)
    bearing_std: float = Field(default=0.005, gt=0)
    range_std: float = Field(default=10, gt=0)

    @model_validator(mode="after")
    def radar_range(self) -> "Measurement":
        if (self.sensor == "radar") != (self.range_m is not None):
            raise ValueError("range is required only for radar")
        return self


@dataclass
class Track:
    state: np.ndarray
    covariance: np.ndarray
    timestamp: float


class TrackManager:
    def __init__(self, core: NativeCore, acceleration_psd: float = 1):
        if not np.isfinite(acceleration_psd) or acceleration_psd < 0:
            raise ValueError("invalid acceleration PSD")
        self.core = core
        self.acceleration_psd = acceleration_psd
        self.tracks: dict[str, Track] = {}
        self.last_timestamp = 0.0

    def seed(
        self,
        identifier: str,
        state: np.ndarray,
        covariance: np.ndarray,
        timestamp: float = 0,
    ) -> None:
        vector, matrix = (
            np.asarray(state, dtype=float),
            np.asarray(covariance, dtype=float),
        )
        if (
            not identifier
            or identifier in self.tracks
            or vector.shape != (4,)
            or matrix.shape != (4, 4)
            or not np.isfinite(vector).all()
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T)
            or not np.isfinite(timestamp)
            or timestamp < self.last_timestamp
        ):
            raise ValueError("invalid or duplicate seed")
        np.linalg.cholesky(matrix)
        self.tracks[identifier] = Track(
            np.array(vector, order="C", copy=True),
            np.array(matrix, order="C", copy=True),
            timestamp,
        )

    def ingest(self, report: Measurement) -> str | None:
        if report.timestamp < self.last_timestamp:
            raise ValueError("out-of-order measurement")
        radar = report.sensor == "radar"
        observation = np.array(
            [report.range_m, report.bearing_rad] if radar else [report.bearing_rad],
            dtype=float,
        )
        variance = np.diag(
            [report.range_std**2, report.bearing_std**2]
            if radar
            else [report.bearing_std**2]
        )
        candidates = []
        for identifier, track in sorted(self.tracks.items()):
            if report.timestamp < track.timestamp:
                continue
            state, covariance = self.core.predict(
                track.state,
                track.covariance,
                report.timestamp - track.timestamp,
                self.acceleration_psd,
            )
            try:
                innovation, jacobian = linearize(state, observation, radar)
            except ValueError:
                continue
            distance = mahalanobis_squared(
                innovation, jacobian @ covariance @ jacobian.T + variance
            )
            if distance <= (9.21034 if radar else 6.63490):
                candidates.append(
                    (distance, identifier, state, covariance, innovation, jacobian)
                )
        self.last_timestamp = report.timestamp
        if not candidates:
            return None
        _, identifier, state, covariance, innovation, jacobian = min(
            candidates, key=lambda candidate: (candidate[0], candidate[1])
        )
        posterior, uncertainty = self.core.update(
            state, covariance, jacobian, innovation, variance
        )
        self.tracks[identifier] = Track(posterior, uncertainty, report.timestamp)
        return identifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the native EKF core and run a radar/optical example"
    )
    parser.add_argument("--cc", default="gcc")
    arguments = parser.parse_args()
    manager = TrackManager(NativeCore(build_native(arguments.cc)))
    manager.seed(
        "track-1",
        np.array([1000.0, 100.0, 5.0, 0.0]),
        np.diag([100.0, 100.0, 25.0, 25.0]),
    )
    for timestamp in range(1, 11):
        position = np.array([1000 + 5 * timestamp, 100.0])
        radar = timestamp % 2 == 1
        manager.ingest(
            Measurement(
                timestamp=timestamp,
                sensor="radar" if radar else "optical",
                range_m=float(np.linalg.norm(position)) if radar else None,
                bearing_rad=float(np.arctan2(position[1], position[0])),
            )
        )
    track = manager.tracks["track-1"]
    print(
        json.dumps(
            {
                "track_id": "track-1",
                "state": track.state.tolist(),
                "covariance": track.covariance.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
