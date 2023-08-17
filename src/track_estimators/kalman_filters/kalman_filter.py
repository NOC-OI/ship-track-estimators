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
        self.predictions = []

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
        predictions
            A list of predictions.
        """
        if isinstance(dt, (list, np.ndarray)):
            assert len(dt) == nsteps, "dt must be the same length as nsteps"
        else:
            dt = np.ones(nsteps) * dt

        # Calculate the cumulative time
        time_cumsum = np.cumsum(ship_track.dts)

        update_index = 0

        # Save the initial state
        self.predictions.append(self.x)
        # self.update(ship_track.z[:, update_index])

        estimate_variance = []

        self.c = np.asarray(
            [ship_track.sog[update_index], ship_track.cog[update_index]]
        )

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

                self.update(ship_track.z[:, update_index])

            # Save the predictions
            self.predictions.append(self.x)
            estimate_variance.append(np.diag(self.P))

        return (
            np.asarray(self.predictions).squeeze(),
            np.asarray(estimate_variance).squeeze(),
        )

    def predict(self, *args, **kwargs):
        """Predict the state."""
        raise NotImplementedError("Predict not implemented.")

    def update(self, *args, **kwargs):
        """Update the state."""
        raise NotImplementedError("Update not implemented.")
