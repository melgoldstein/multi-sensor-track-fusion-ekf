import numpy as np


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def linearize(
    state: np.ndarray, measurement: np.ndarray, radar: bool
) -> tuple[np.ndarray, np.ndarray]:
    east, north = state[:2]
    squared_range = east * east + north * north
    if squared_range < 1e-6:
        raise ValueError("sensor-origin singularity")
    distance = np.sqrt(squared_range)
    bearing = np.arctan2(north, east)
    angle_jacobian = [-north / squared_range, east / squared_range, 0, 0]
    if radar:
        return np.array(
            [measurement[0] - distance, wrap_angle(measurement[1] - bearing)]
        ), np.array([[east / distance, north / distance, 0, 0], angle_jacobian])
    return np.array([wrap_angle(measurement[0] - bearing)]), np.array([angle_jacobian])


def mahalanobis_squared(innovation: np.ndarray, covariance: np.ndarray) -> float:
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), innovation)
    return float(whitened @ whitened)
