from typing import List, Tuple, Union

import numpy as np

from ..ship_track import ShipTrack


class KalmanFilterBase:
    """
    Kalman filter base class.

    Attributes
    ----------
    time: float
        Current time.
    predictions: list
        Kalman filter predictions.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the Kalman filter base class."""
        self.time = 0

        # Control input vector
        self.c = None

        # Kalman filter predictions
        self.means = []
        self.covariances = []
        self.means_smoothed = []
        self.covariances_smoothed = []

        self.dt = None
        self.nsteps = None

    def run(
        self,
        nsteps: int,
        dt: int | float | List[Union[int, float]] | np.ndarray,
        ship_track: ShipTrack,
        *args,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the algorithm to predict the ship's position based on the given ship_track.

        Parameters
        ----------
        nsteps
            The number of steps to perform.
        dt
            The time step.
        ship_track
            The ship's track data, given as a ShipTrack object.

        Returns
        -------
        means, covariances
            The mean and covariance of the predicted state.
        """
        if isinstance(dt, (list, np.ndarray)):
            assert len(dt) == nsteps, "dt must be the same length as nsteps"
        else:
            dt = np.ones(nsteps) * dt

        self.dt = dt
        self.nsteps = nsteps

        # Initialize the update index
        update_index = 0

        # Calculate the cumulative time
        time_cumsum = np.cumsum(ship_track.dts)

        # Save the initial state
        self.means.append(self.x)
        self.covariances.append(self.P)

        # Perform the update step to incorporate any
        # available initial measurements or information
        self.update(ship_track.z[:, update_index].copy())

        self.c = np.asarray(
            [ship_track.sog[update_index], ship_track.cog[update_index]]
        )

        # Run the Kalman filter
        for step, dt in enumerate(dt):
            # Predict the next state
            self.predict(
                dt=dt,
                c=None,  # self.c,
                sog_rate=ship_track.sog_rate[update_index],
                cog_rate=ship_track.cog_rate[update_index],
            )

            # Increment time
            self.time += dt

            # Update step
            if self.time in time_cumsum:
                update_index += 1

                self.c = np.asarray(
                    [ship_track.sog[update_index], ship_track.cog[update_index]]
                )

                self.update(ship_track.z[:, update_index].copy())

            # Store the predictions
            self.means.append(self.x)
            self.covariances.append(self.P)

        return (
            np.asarray(self.means).squeeze(),
            np.asarray(self.covariances).squeeze(),
        )

    def run_rts_smoother(self, ship_track: ShipTrack) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Rauch-Tung-Striebel (RTS) smoother.

        Parameters
        ----------
        ship_track
            The ship's track data, given as a ShipTrack object.

        Returns
        -------
        x, p
            Smoothed state estimate and covariance.
        """
        x, P = self.rts_step(
            np.asarray(self.means), np.asarray(self.covariances), ship_track
        )

        return x, P

    def predict(self, *args, **kwargs):
        """Predict the state."""
        raise NotImplementedError("Predict not implemented.")

    def update(self, *args, **kwargs):
        """Update the state."""
        raise NotImplementedError("Update not implemented.")
