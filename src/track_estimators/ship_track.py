import numpy as np
import pandas as pd

from .utils import haversine_formula, heading


class ShipTrack:
    def __init__(self, csv_file: str | None = None) -> None:
        self.lat = None
        self.lon = None
        self.cog = None
        self.sog = None
        self.dt = None

        # Pandas DataFrame
        self.df = None

        if csv_file is not None:
            self.read_csv(csv_file=csv_file)

        # Rates
        self.sog_rate = self.calculate_sog_rate()
        self.cog_rate = self.calculate_cog_rate()

    def read_csv(self, csv_file: str, ship_id: str | None = None) -> pd.DataFrame:
        # Read the csv file
        self.df = pd.read_csv(csv_file)

        # Get the ship course and sort by date
        if ship_id is not None:
            self.df = self.df.loc[self.df["id"] == ship_id]

        # Handle time
        # "yr","mo","dy","hr",
        # Datetype format: 2005-02-25T03:30'
        self.df["date"] = (
            self.df["yr"].astype(str)
            + "-"
            + self.df["mo"].astype(str)
            + "-"
            + self.df["dy"].astype(str)
        )
        self.df["date"] += "T" + self.df["hr"].astype(str).str.zfill(2) + ":00:00"

        # Create deltatimes
        dates = pd.to_datetime(self.df.date)
        self.dts = []
        for i in range(len(dates) - 1):
            dt = pd.Timedelta(dates[i + 1] - dates[i]).total_seconds()
            self.dts.append(dt / 3600.0)  # seconds to hours

        self.dts = np.asarray(self.dts)

        # Sort values by date
        self.df = self.df.sort_values(by="date")

        # Extract lat, lon
        self.lat = pd.to_numeric(self.df.lat).values
        self.lon = pd.to_numeric(self.df.lon).values

        return self.lat, self.lon, self.dt

    def calculate_sog(self) -> np.ndarray:
        """
        Calculate the speed over ground.

        Notes
        -----
        This function calculates the speed over ground (SOG) given a series of longitude,
        latitude, and time intervals.

        Parameters
        ----------
        lon
            A list or array floats representing the longitude coordinates
        lat
            A list or array floats representing the latitude coordinates
        dts
            A list or array of floats representing the time intervals

        Returns
        -------
        sog
            A list of floats representing the speed over ground
        """
        self.sog = []

        for i in range(1, len(self.lon)):
            dist = haversine_formula(
                self.lon[i - 1], self.lat[i - 1], self.lon[i], self.lat[i]
            )
            self.sog.append(dist / self.dts[i - 1])

        # Assume stationary trajectory from the end point onwards
        self.sog.append(0)

        self.sog = np.asarray(self.sog)

        return self.sog

    def calculate_sog_rate(self):
        if self.sog is None:
            self.calculate_sog()

        self.sog_rate = []

        for i in range(1, len(self.sog)):
            self.sog_rate.append((self.sog[i] - self.sog[i - 1]) / self.dts[i - 1])

        self.sog_rate = np.asarray(self.sog_rate)

        return self.sog_rate

    def calculate_cog(self):
        self.cog = []

        for i in range(1, len(self.lon)):
            cog = heading(self.lon[i - 1], self.lat[i - 1], self.lon[i], self.lat[i])
            self.cog.append(cog)

        # Assume stationary trajectory from the end point onwards
        self.cog.append(0.0)
        self.cog = np.asarray(self.cog)

        return self.cog

    def calculate_cog_rate(self):
        if self.cog is None:
            self.calculate_cog()

        self.cog_rate = []

        for i in range(1, len(self.cog)):
            print(self.cog[i], self.cog[i - 1])
            self.cog_rate.append((self.cog[i] - self.cog[i - 1]) / self.dts[i - 1])

        self.cog_rate = np.asarray(self.cog_rate)

        return self.cog_rate

    def get_measurements(self) -> np.ndarray:
        """
        Get the measurement matrix z.

        Returns
        -------
        z
            The measurement matrix.
        """
        if self.sog is None:
            self.calculate_sog()

        if self.cog is None:
            self.calculate_cog()

        # z = np.vstack((self.lon, self.lat, self.sog, self.cog))
        z = np.vstack((self.lon, self.lat, self.sog, self.cog))

        return z
