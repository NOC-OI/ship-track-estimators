"""
Unscented Kalman Filter module.

(1) Cole, B.; Schamberg, G.
Unscented Kalman Filter for Long-Distance Vessel Tracking in Geodetic Coordinates.
Applied Ocean Research 2022, 124, 103205.
https://doi.org/10.1016/j.apor.2022.103205.
"""
import logging
from typing import Callable, Optional

import numpy as np
import scipy.linalg
from track_estimators.ship_track import ShipTrack

from .kalman_filter import KalmanFilterBase


class UnscentedKalmanFilter(KalmanFilterBase):
    def __init__(
        self,
        H=None,
        Q=None,
        R=None,
        P=None,
        x0=None,
        non_linear_process: Optional[Callable] = None,
        measurement_model: Optional[Callable] = None,
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
        measurement_model
            Measurement model.
        """
        super().__init__()

        if H is None:
            raise ValueError("Set proper system dynamics.")

        self.H = H
        self.n = H.shape[1]

        self.Q = np.eye(self.n) if Q is None else np.asarray(Q)
        self.R = np.eye(self.n) if R is None else np.asarray(R)
        self.P_orig = np.eye(self.n) if P is None else np.asarray(P)
        self.P = np.eye(self.n) if P is None else np.asarray(P)
        self.x = np.zeros((self.n, 1)) if x0 is None else np.asarray(x0).reshape(-1, 1)

        # Sigma points and weights
        self.n_sigma_points = 2 * self.n + 1
        self.sigma_points = np.zeros((self.n, self.n_sigma_points))
        self.sigma_points_orig = None
        self.weights = np.zeros((self.n_sigma_points, self.n_sigma_points))

        # Non-linear process model
        self.non_linear_process = non_linear_process

        # Measurement model
        self.measurement_model = measurement_model

    def compute_sigma_points(
        self, x: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the sigma points the UKF using the prior state estimate and covariance matrix.

        Returns
        -------
        self.sigma_points
            Sigma points of the unscented kalman filter.
        """
        if x is None:
            assert self.x is not None, "Set proper initial state estimate."
            x = self.x

        if P is None:
            assert self.P is not None, "Set proper initial state covariance matrix."
            P = self.P

        term = self.n / (1 - self.weights[0, 0])
        term = term * P
        term = scipy.linalg.sqrtm(term)

        # Eq. (4a)
        self.sigma_points[:, 0] = x[:, 0]

        # Eq. (4c, 4d)
        for i in range(self.n):
            self.sigma_points[:, i + 1] = x[:, 0] + term[:, i]
            self.sigma_points[:, i + 1 + self.n] = x[:, 0] - term[:, i]

        return self.sigma_points

    def compute_weights(self, weight0: Optional[float] = None) -> np.ndarray:
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
        non_linear_process: Optional[Callable] = None,
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
        self.sigma_points_orig = self.sigma_points.copy()

        # 1b. Pass sigma points through the non linear process model
        # Eq. (5)
        for sigmap in range(self.n_sigma_points):
            self.sigma_points[:, sigmap] = non_linear_process(
                self.sigma_points[:, sigmap], **non_linear_process_kwargs
            )

        # 1c. Compute the a priori state estimate and covariance matrix using the transformed sigma points
        # Eq. (6)
        self.x = np.sum(np.dot(self.sigma_points, self.weights), axis=1, keepdims=True)

        # Eq. (1) process model is affected by additive, zero-mean Gaussian noise
        process_white_noise = np.random.normal(
            scale=np.sqrt(np.diag(self.Q)), size=(self.n)
        ).reshape(-1, 1)

        self.x += process_white_noise

        # Eq. (7)
        S = self.sigma_points - self.x

        self.P = np.dot(np.dot(S, self.weights), S.T) + self.Q

    def update(self, z: np.ndarray) -> None:
        """
        Update the state.

        Parameters
        ----------
        z
            Measurement vector.
        """
        # Ensure correct shape of observation vector
        z = z.reshape(-1, 1)

        if self.measurement_model is not None:
            assert callable(
                self.measurement_model
            ), "Measurement model must be callable."
            z = self.measurement_model(z)

        # Kalman filter robustification
        # self.check_robustness(z, self.P, self.R)
        R = self.R

        # Add measurement noise
        measurement_white_noise = np.random.normal(
            scale=np.sqrt(np.diag(R)), size=(self.n)
        ).reshape(-1, 1)

        z += measurement_white_noise

        # 2a. Compute the Kalman gain using the a priori covariance
        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + R  # self.R

        # Eq. (8) Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))

        # 2b. Map the state prediction into the measurement space, and compute the residual error
        # Eq. (9), Innovation
        y = z - np.dot(self.H, self.x)

        # Eq. (43)
        y[3, 0] = (y[3, 0] + 180.0) % 360.0 - 180.0

        # 2c. Compute the a posteriori state estimate and covariance matrix
        # Eq. (10), update (correct) the state
        self.x = self.x + np.dot(K, y)

        # Eq. (44)
        self.x[3, 0] = self.x[3, 0] % 360.0

        # Eq. (11), update (correct) the covariance
        identity = np.eye(self.n)

        self.P = np.dot(
            np.dot(identity - np.dot(K, self.H), self.P),
            (identity - np.dot(K, self.H)).T,
        ) + np.dot(np.dot(K, R), K.T)

    def rts_step(self, fwd_means, fwd_vars, ship_track: ShipTrack, *args, **kwargs):
        """
        Run the unscented Rauch-Tung-Striebel (RTS) smoother.

        Parameters
        ----------
        fwd_means
            Forward state means.
        fwd_vars
            Forward state variances.
        ship_track
            The ship's track data, given as a ShipTrack object.

        Returns
        -------
        x_bwd, P_bwd
            Smoothed state estimate and covariance.
        """
        nsteps = fwd_means.shape[0]

        ship_track.sog_rate = np.repeat(
            ship_track.sog_rate, int(nsteps / len(ship_track.dts))
        )
        ship_track.cog_rate = np.repeat(
            ship_track.cog_rate, int(nsteps / len(ship_track.dts))
        )

        # Copy the state estimates and covariances
        x_fwd, P_fwd = fwd_means.copy(), fwd_vars.copy()

        for step in range(nsteps - 2, -1, -1):
            # Create sigma points from state estimate
            self.compute_weights()

            self.sigma_points = self.compute_sigma_points(x_fwd[step], P_fwd[step])
            self.sigma_points_orig = self.sigma_points.copy()

            # Pass sigma points through the non linear process model
            for sigmap in range(self.n_sigma_points):
                self.sigma_points[:, sigmap] = self.non_linear_process(
                    self.sigma_points[:, sigmap],
                    c=None,
                    dt=self.dt[step],
                    sog_rate=ship_track.sog_rate[step],
                    cog_rate=ship_track.cog_rate[step],
                )

            # Compute the backward state estimate and covariance
            x_bwd = np.sum(
                np.dot(self.sigma_points, self.weights), axis=1, keepdims=True
            )

            # Add the white noise
            process_white_noise = np.random.normal(
                scale=np.sqrt(np.diag(self.Q)), size=(self.n)
            ).reshape(-1, 1)
            x_bwd += process_white_noise
            S = self.sigma_points - x_fwd[step]
            P_bwd = np.dot(np.dot(S, self.weights), S.T) + self.Q

            # Compute cross-variance
            S = self.sigma_points - x_bwd
            S_orig = self.sigma_points_orig - x_fwd[step]
            D = np.dot(np.dot(S_orig, self.weights), S.T)

            #  Kalman gain
            K = np.dot(D, np.linalg.pinv(P_bwd))

            # 2b. Map the state prediction into the measurement space, and compute the residual error
            # Eq. (9), Innovation
            y = x_fwd[step + 1] - x_bwd

            # Eq. (43)
            y[3, 0] = (y[3, 0] + 180.0) % 360.0 - 180.0

            # Smoothed mean
            x_fwd[step] += np.dot(K, y)

            # Eq. (44)
            x_fwd[step][3, 0] = x_fwd[step][3, 0] % 360.0

            # Smoothed covariance
            P_fwd[step] += np.dot(np.dot(K, P_fwd[step + 1] - P_bwd), K.T)

        return x_fwd, P_fwd

    def check_robustness(
        self, z: np.ndarray, P: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        lambda_factor = 1
        chi_alpha = 50

        measurement_white_noise = np.random.normal(
            scale=np.sqrt(np.diag(R)), size=(self.n)
        ).reshape(-1, 1)
        zn = z + measurement_white_noise

        judging_index = self.criterion_index(zn, P, R)

        print(judging_index, lambda_factor)

        while judging_index > chi_alpha:
            # Update the lambda factor
            measurement_white_noise = np.random.normal(
                scale=np.sqrt(np.diag(R)), size=(self.n)
            ).reshape(-1, 1)
            zn = z + measurement_white_noise

            lambda_factor = self.update_lambda_factor(
                lambda_factor, judging_index, chi_alpha, zn, P, R
            )

            # Scale measurement uncertainty
            R = self.scale_measurement_uncertainty(R, lambda_factor)

            # Update the judging index
            judging_index = self.criterion_index(zn, P, R)

            print(judging_index, lambda_factor)

        return R

    def criterion_index(self, z: np.ndarray, P: np.ndarray, R: np.ndarray) -> float:
        """
        Calculate the criterion index.

        The criterion index is based on the Mahalanobis distance
        between the current state estimate and the measurement.

        Parameters
        ----------
        z
            Measurement vector.
        P
            Covariance matrix.
        R
            Measurement covariance.

        Returns
        -------
        criterion_index
            Criterion index.

        References
        ----------
        Chang, G.
        Robust Kalman Filtering Based on Mahalanobis Distance as Outlier Judging Criterion.
        J Geod 2014, 88 (4), 391-401. https://doi.org/10.1007/s00190-013-0690-8.
        """
        # Ensure correct shape of observation vector
        z = z.reshape(-1, 1)

        # Eq. (14) / Eq. (11)
        y = z - self.x
        P = np.dot(self.H, np.dot(P, self.H.T)) + R
        Pinv = np.linalg.pinv(P)

        # Calculate criterion index
        criterion_index = np.dot(np.dot(y.T, Pinv), y)
        criterion_index = np.abs(criterion_index)

        return criterion_index.item()

    def update_lambda_factor(
        self,
        lambda_factor: float,
        criterion_index: float,
        chi_alpha: float,
        z: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """
        Update the lambda factor.

        Parameters
        ----------
        lambda_factor
            Lambda factor.
        criterion_index
            Criterion index.
        chi_alpha
            Chi alpha.
        z
            Measurement vector.
        P
            Covariance matrix.
        R
            Measurement covariance.

        Returns
        -------
        lambda_factor
            Updated lambda factor.

        References
        ----------
        Chang, G.
        Robust Kalman Filtering Based on Mahalanobis Distance as Outlier Judging Criterion.
        J Geod 2014, 88 (4), 391-401. https://doi.org/10.1007/s00190-013-0690-8.
        """
        # Eq. (18)
        # Numerator
        numerator = criterion_index - chi_alpha

        # Denominator
        y = z - self.x
        P = np.dot(self.H, np.dot(P, self.H.T)) + R
        Pinv = np.linalg.pinv(P)

        denominator = np.dot(np.dot(Pinv, R), Pinv)
        denominator = np.dot(np.dot(y.T, denominator), y)

        # Propagate the lambda factor
        lambda_factor = lambda_factor + numerator / denominator

        return lambda_factor.item()

    def scale_measurement_uncertainty(
        self, R: np.ndarray, lambda_factor: float
    ) -> np.ndarray:
        """
        Scale the measurement uncertainty by a factor lambda.

        Parameters
        ----------
        R
            Measurement covariance.
        lambda_factor
            Scaling factor.

        Returns
        -------
        R
            Scaled measurement covariance.

        References
        ----------
        Chang, G.
        Robust Kalman Filtering Based on Mahalanobis Distance as Outlier Judging Criterion.
        J Geod 2014, 88 (4), 391-401. https://doi.org/10.1007/s00190-013-0690-8.
        """
        # Eq. (13)
        R = R * lambda_factor
        return R
