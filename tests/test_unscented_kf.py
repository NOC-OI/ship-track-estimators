import numpy as np
import pytest
from track_estimators.kalman_filters.unscented import UnscentedKalmanFilter


@pytest.fixture
def ukf():
    """Fixture to define a basic Unscented Kalman Filter."""
    # Define system dynamics
    H = np.diag([1, 1])

    # Estimate Covariance
    P = np.diag([np.random.uniform(0, 1), np.random.uniform(0, 1)])

    # Define initial state
    x = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)]).reshape(-1, 1)

    # Instantiate the UKF
    ukf = UnscentedKalmanFilter(H=H, P=P, x0=x)

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

    # Check that the average of the sigma points is equal to the mean
    assert np.all(np.isclose(ukf.x[:, 0], np.mean(ukf.sigma_points, axis=1)))

    # Check that the sigma points have the same covariance as the estimate covariance matrix
    assert np.all(np.isclose(ukf.P, np.cov(ukf.sigma_points)))


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
    assert np.isclose(np.trace(ukf.weights), 1.0)

    # Check that weight0 is in range[-1, 1]
    weight0 = ukf.weights[0, 0]
    assert weight0 < 1.0 and weight0 > -1.0

    # Check that the matrix is diagonal
    assert np.count_nonzero(ukf.weights - np.diag(np.diagonal(ukf.weights))) == 0


def test_weighted_sigma_points(ukf):
    """
    Test the calculation of weighted sigma points.

    Notes
    -----
    The weighted sigma points should have the same mean as the state.
    """
    # Compute weights
    ukf.compute_weights()

    # Compute sigma points
    ukf.compute_sigma_points()

    # Check that the weighted average of the sigma points is equal to the mean
    assert np.all(
        np.isclose(ukf.x[:, 0], np.sum(np.dot(ukf.sigma_points, ukf.weights), axis=1))
    )

    # Check that the sigma points have the same covariance as the estimate covariance matrix
    residuals = ukf.sigma_points - ukf.x
    assert np.all(
        np.isclose(ukf.P, np.dot(np.dot(residuals, ukf.weights), residuals.T))
    )
