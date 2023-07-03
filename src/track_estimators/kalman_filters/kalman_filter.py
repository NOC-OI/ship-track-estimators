import numpy as np

from ..ship_track import ShipTrack


class KalmanFilterBase:
    """Kalman filter base class."""

    def __init__(self, *args, **kwargs):
        self.time = 0
        self.predictions = []

    def run(
        self, nsteps: int, dt: int | float, ship_track: ShipTrack, *args, **kwargs
    ) -> list:
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
        # Calculate the cumulative time
        time_cumsum = np.cumsum(ship_track.dts)

        major_index = 0

        for step in range(nsteps):
            # Predict the next state
            self.predict(
                dt=dt,
                sog_rate=ship_track.sog_rate[major_index],
                cog_rate=ship_track.cog_rate[major_index],
            )

            # Increment time
            self.time += dt

            # Update step
            if self.time in time_cumsum:
                major_index += 1

                self.update(ship_track.z[:, major_index])

            # Save the predictions
            self.predictions.append(self.x)

        return self.predictions

    def predict(self, *args, **kwargs):
        """Predict the state."""
        raise NotImplementedError("Predict not implemented.")

    def update(self, *args, **kwargs):
        """Update the state."""
        raise NotImplementedError("Update not implemented.")
