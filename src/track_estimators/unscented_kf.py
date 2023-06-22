"""
Unscented Kalman Filter module.

(1) Cole, B.; Schamberg, G.
Unscented Kalman Filter for Long-Distance Vessel Tracking in Geodetic Coordinates.
Applied Ocean Research 2022, 124, 103205.
https://doi.org/10.1016/j.apor.2022.103205.
"""
import logging
from typing import Callable

import numpy as np
import scipy.linalg


class UnscentedKalmanFilter:
    def __init__(
        self,
        H=None,
        Q=None,
        R=None,
        P=None,
        x0=None,
        non_linear_process: Callable | None = None,
    ):
        """
        Initialize the Unscented Kalman Filter.

        Parameters
        ----------
        H: np.ndarray
            State transition matrix.
        Q: np.ndarray
            Process noise covariance matrix. Defaults to identity matrix.
        R: np.ndarray
            Measurement noise covariance matrix. Defaults to identity matrix.
            P: np.ndarray
                Covariance matrix of the state estimate. Defaults to identity matrix.
        x0: np.ndarray
            Initial state estimate. Defaults to zero matrix
        non_linear_process
            Non-linear process model.
        """
        if H is None:
            raise ValueError("Set proper system dynamics.")

        self.H = H
        self.n = H.shape[1]

        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P_orig = np.eye(self.n) if P is None else P
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

        # Sigma points and weights
        self.n_sigma_points = 2 * self.n + 1
        self.sigma_points = np.zeros((self.n, self.n_sigma_points))
        self.weights = np.zeros((self.n_sigma_points, self.n_sigma_points))

        # Non-linear process model
        self.non_linear_process = non_linear_process

    def compute_sigma_points(self) -> np.ndarray:
        """
        Compute the sigma points the UKF using the prior state estimate and covariance matrix.

        Returns
        -------
        self.sigma_points
            Sigma points of the unscented kalman filter.
        """
        term = self.n / (1 - self.weights[0, 0])
        term = term * self.P
        term = scipy.linalg.sqrtm(term)

        # Eq. (4a)
        self.sigma_points[:, 0] = self.x[:, 0]

        # Eq. (4c, 4d)
        for i in range(self.n):
            self.sigma_points[:, i + 1] = self.x[:, 0] + term[:, i]
            self.sigma_points[:, i + 1 + self.n] = self.x[:, 0] - term[:, i]

        return self.sigma_points

    def compute_weights(self, weight0: float | None = None) -> np.ndarray:
        """
        Compute weights of the unscented kalman filter.

        Parameters
        ----------
        weight0
            Weight of the mean.

        Returns
        -------
        self.weights(self.n_sigma_points, self.n_sigma_points)
            Weights of sigma points given as a diagonal matrix.
        """
        # W0
        if weight0 is None:
            weight0 = 1 - self.n / 3.0

        assert (
            weight0 < 1.0 and weight0 > -1.0
        ), "Weight0 value ({}) is outside [-1, 1] range.".format(weight0)

        # W1, W2, ..., Wn
        weightn = (1 - weight0) / (2 * self.n)

        # Eq. (4e)
        np.fill_diagonal(self.weights, weightn)

        # Eq. (4b)
        self.weights[0, 0] = weight0

        logging.debug("Weights\n\n", self.weights)
        return self.weights

    def predict(
        self,
        non_linear_process: Callable | None = None,
        **non_linear_process_kwargs,
    ) -> None:
        """
        Predict the state and covariance matrix.

        Parameters
        ----------
        non_linear_process
            Non-linear process model.
        non_linear_process_kwargs
            Non-linear process model parameters.

        Notes
        -----
        Matrix dimensions:

        - sigma_points(n, nsigma)
        - weights(nsigma, nsigma)

        where nsigma = 2n + 1
        """
        # Set the non-linear process model
        if non_linear_process is None:
            assert self.non_linear_process is not None, "Non-linear process is not set."
            non_linear_process = self.non_linear_process

        assert callable(
            non_linear_process
        ), "Non-linear process model must be callable."

        # Ensure correct shape of state vector
        self.x = self.x.reshape(-1, 1)

        # 1a. Use the prior state estimate and covariance matrix to compute sigma points and weights
        # Eq. (4)
        self.compute_weights()
        self.sigma_points = self.compute_sigma_points()

        # 1b. Pass sigma points through the non linear process model
        # Eq. (5)
        for sigmap in range(self.n_sigma_points):
            self.sigma_points[:, sigmap] = non_linear_process(
                self.sigma_points[:, sigmap], **non_linear_process_kwargs
            )

        # 1c. Compute the a priori state estimate and covariance matrix using the transformed sigma points
        # Eq. (6)
        self.x = np.sum(np.dot(self.sigma_points, self.weights), axis=1, keepdims=True)

        # Eq. (7)
        S = self.sigma_points - self.x
        self.P = np.dot(np.dot(S, self.weights), S.T) + self.Q

    def update(self, z: np.ndarray) -> None:
        # Ensure correct shape of observation vector
        z = z.reshape(-1, 1)

        # 2a. Compute the Kalman gain using the a priori covariance
        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Eq. (8) Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))

        # 2b. Map the state prediction into the measurement space, and compute the residual error
        # Eq. (9), Innovation
        y = z - np.dot(self.H, self.x)

        # 2c. Compute the a posteriori state estimate and covariance matrix
        # Eq. (10), update (correct) the state
        self.x = self.x + np.dot(K, y)

        # Eq. (11), update (correct) the covariance
        identity = np.eye(self.n)

        self.P = np.dot(
            np.dot(identity - np.dot(K, self.H), self.P),
            (identity - np.dot(K, self.H)).T,
        ) + np.dot(np.dot(K, self.R), K.T)
