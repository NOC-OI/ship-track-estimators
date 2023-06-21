import numpy as np
import pytest
from track_estimators.unscented_kf import UnscentedKalmanFilter


@pytest.fixture
def ukf():
    """Fixture to define a basic Unscented Kalman Filter."""
    # Define system dynamics
    H = np.diag([1, 1])

    # Define initial state
    x = np.array([0.5, 0.8])
    x = x.reshape(-1, 1)

    # Instantiate the UKF
    ukf = UnscentedKalmanFilter(H=H, x0=x)

    return ukf


def test_sigma_points(ukf):
    """
    Test the calculation of sigma points.

    Notes
    -----
    The sigma points should have the same mean and covariance as the state.
    """
    # Compute sigma points
    ukf.compute_sigma_points()

    # Check that the sigma points have the same mean
    assert np.all(np.isclose(ukf.x[:, 0], np.mean(ukf.sigma_points, axis=1)))

    # TODO: check that the weighted sigma points have the same covariance


def test_weights(ukf):
    """
    Test the calculation of weights.

    Notes
    -----
    Weights sum to 1 and they can be positive or negative.
    """
    # Compute weights
    ukf.compute_weights()

    # Check that weights sum to 1 (i.e., they are normalized)
    assert np.isclose(np.sum(ukf.weights), 1.0)

    # Check that the matrix is diagonal
    assert np.count_nonzero(ukf.weights - np.diag(np.diagonal(ukf.weights))) == 0


def test_weighted_sigma_points(ukf):
    """
    Test the calculation of weighted sigma points.

    Notes
    -----
    The weighted sigma points should have the same mean and covariance as the state.
    """
    # Compute sigma points
    ukf.compute_sigma_points()

    # Compute weights
    ukf.compute_weights()

    # Compute weighted sigma points
    weighted_sigma = np.dot(ukf.sigma_points, ukf.weights)

    # Check that the weighted sigma points have the same mean
    assert np.all(np.isclose(ukf.x[:, 0], np.sum(weighted_sigma, axis=1)))

    # TODO: check that the weighted sigma points have the same covariance
