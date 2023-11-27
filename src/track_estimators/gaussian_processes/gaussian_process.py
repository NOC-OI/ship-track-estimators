"""Gaussian Process Regression model for joint modeling of latitude and longitude."""
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from track_estimators.ship_track import ShipTrack


class GPRegression:
    """
    Gaussian Process Regression model for joint modeling of latitude and longitude.

    Parameters
    ----------
    kernel
        The kernel for joint modeling of latitude and longitude.
    gpr
        The Gaussian Process Regression model.
    """

    def __init__(self, kernel, gpr=GaussianProcessRegressor, *args, **kwargs):
        """Initialse the GPRegression model."""
        self._kernel = kernel
        self._gpr = gpr
        self._model = None

    def fit(
        self,
        ship_track: ShipTrack,
        gpr_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the Gaussian Process Regression model to the joint data.

        Parameters
        ----------
        ship_track
            The ship's track data, given as a ShipTrack object.
        gpr_kwargs
            Keyword arguments for the Gaussian Process Regression model.

        Returns
        -------
        self._model
            The fitted Gaussian Process Regression model.
        """
        gpr_kwargs = gpr_kwargs or {"n_restarts_optimizer": 50}

        # Generate times
        times = np.cumsum(ship_track.dts)
        times = np.insert(times, 0, 0)

        # Combine latitude and longitude into a single array for joint modeling
        # Each data point is represented as [time, latitude, longitude]
        X = times.reshape(-1, 1)

        assert self._kernel is not None, "Kernel must be specified."

        # Create Gaussian Process Regression model for joint modeling of latitude and longitude
        self._model = self._gpr(kernel=self._kernel, **gpr_kwargs)

        # Fit the model to the joint data (latitude and longitude)
        self._model.fit(X, np.column_stack((ship_track.lon, ship_track.lat)))

        return self._model

    def predict(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the latitude and longitude for the given times.

        Parameters
        ----------
        times
            The times for which to predict the latitude and longitude.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The predicted latitude and longitude for the given times.
        """
        assert self._model is not None, "Model has not been fit yet."

        # Predict latitude and longitude for the new time
        predicted, std = self._model.predict(times.reshape(-1, 1), return_std=True)

        return predicted, std
