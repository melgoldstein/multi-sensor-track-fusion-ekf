import os
from pathlib import Path
import numpy as np
import pytest
from track_fusion.native_core import NativeCore, build_native
from track_fusion.track_manager import TrackManager, Measurement
from track_fusion.maha_association import wrap_angle


@pytest.fixture(scope="session")
def core():
    return NativeCore(build_native(os.environ.get("CC", "gcc")))


def test_native_matches_joseph(core):
    rng = np.random.default_rng(9)
    for dimensions in (1, 2):
        for _ in range(20):
            state = rng.normal(size=4)
            factor = rng.normal(size=(4, 4))
            covariance = factor @ factor.T + np.eye(4)
            jacobian = rng.normal(size=(dimensions, 4))
            innovation = rng.normal(size=dimensions)
            variance = np.eye(dimensions) * 0.2
            gain = (
                covariance
                @ jacobian.T
                @ np.linalg.inv(jacobian @ covariance @ jacobian.T + variance)
            )
            correction = np.eye(4) - gain @ jacobian
            expected = correction @ covariance @ correction.T + gain @ variance @ gain.T
            posterior, uncertainty = core.update(
                state, covariance, jacobian, innovation, variance
            )
            np.testing.assert_allclose(posterior, state + gain @ innovation, atol=1e-12)
            np.testing.assert_allclose(uncertainty, expected, atol=1e-12)


def test_prediction(core):
    state, covariance = core.predict(np.array([1.0, 2.0, 3.0, 4.0]), np.eye(4), 2, 0)
    np.testing.assert_allclose(state, [7, 10, 3, 4])
    transition = np.eye(4)
    transition[0, 2] = transition[1, 3] = 2
    np.testing.assert_allclose(covariance, transition @ transition.T)


def test_association_and_rejection(core):
    manager = TrackManager(core)
    manager.seed("near", np.array([1000.0, 100.0, 0.0, 0.0]), np.eye(4) * 10)
    manager.seed("far", np.array([2000.0, 500.0, 0.0, 0.0]), np.eye(4) * 10)
    report = Measurement(
        timestamp=1,
        sensor="radar",
        range_m=np.hypot(1000, 100),
        bearing_rad=np.arctan2(100, 1000),
    )
    assert manager.ingest(report) == "near"
    assert (
        manager.ingest(report.model_copy(update={"timestamp": 2, "range_m": 100000.0}))
        is None
    )
    with pytest.raises(ValueError, match="out-of-order"):
        manager.ingest(report)
    assert wrap_angle(-np.pi - 0.01) == pytest.approx(np.pi - 0.01)


def test_native_sources_match():
    assert (
        Path("csrc/ekf_step.c").read_bytes()
        == Path("src/native/ekf_step.c").read_bytes()
    )


def test_reuses_loaded_library(core, monkeypatch):
    def unexpected_compile(*args, **kwargs):
        raise AssertionError("cached library must not be overwritten while loaded")

    monkeypatch.setattr("track_fusion.native_core.subprocess.run", unexpected_compile)
    assert build_native(os.environ.get("CC", "gcc")).is_file()


def test_overflow_preserves_input(core):
    state, covariance = np.ones(4), np.eye(4)
    with pytest.raises(ValueError, match="overflow"):
        core.predict(state, covariance, 1e200, 1)
    np.testing.assert_array_equal(state, np.ones(4))
    np.testing.assert_array_equal(covariance, np.eye(4))


def test_rejects_unsafe_native_shapes(core):
    with pytest.raises(ValueError):
        core.predict(np.zeros(3), np.eye(4), 1, 1)
    with pytest.raises(ValueError):
        core.update(np.zeros(4), np.eye(4), np.eye(2, 4), np.zeros(1), np.eye(1))
    with pytest.raises(ValueError):
        core.predict(np.full(4, np.nan), np.eye(4), 1, 1)
    with pytest.raises(np.linalg.LinAlgError):
        core.predict(np.zeros(4), -np.eye(4), 1, 1)


def test_async_convergence_and_covariance(core):
    rng = np.random.default_rng(15)
    manager = TrackManager(core, acceleration_psd=0.01)
    manager.seed(
        "vessel",
        np.array([1010.0, 190.0, 4.0, 1.0]),
        np.diag([400.0, 400.0, 25.0, 25.0]),
    )
    errors = []
    accepted = 0
    for index in range(1, 201):
        timestamp = index * 0.5
        truth = np.array([1000 + 5 * timestamp, 200 + 2 * timestamp])
        radar = index % 3 == 0
        report = Measurement(
            timestamp=timestamp,
            sensor="radar" if radar else "optical",
            range_m=float(np.linalg.norm(truth) + rng.normal(0, 5)) if radar else None,
            bearing_rad=float(np.arctan2(truth[1], truth[0]) + rng.normal(0, 0.002)),
            range_std=5.0,
            bearing_std=0.002,
        )
        association = manager.ingest(report)
        assert association in (None, "vessel")
        accepted += association is not None
        track = manager.tracks["vessel"]
        assert np.linalg.eigvalsh(track.covariance).min() > 0
        np.testing.assert_allclose(track.covariance, track.covariance.T, atol=1e-10)
        errors.append(np.linalg.norm(track.state[:2] - truth))
    assert np.mean(errors[-50:]) < 5
    assert accepted >= 190
