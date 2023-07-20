import numpy as np
import pandas as pd

from .utils import haversine_formula, heading


class ShipTrack:
    """
    ShipTrack class representing the data for a single ship track.

    Attributes
    ----------
    lat
        A list or array of floats representing the latitude.
    lon
        A list or array of floats representing the longitude.
    cog
        A list or array of floats representing the course over ground.
    sog
        A list or array of floats representing the speed over ground.
    dt
        A list or array of floats representing the time difference between measurements.
    df
        The dataframe containing the ship track data.
    sog_rate
        A list or array of floats representing the speed over ground rate.
    cog_rate
        A list or array of floats representing the course over ground rate.
    z
        A list or array of floats representing the measurement.
    """

    def __init__(self, csv_file: str | None = None) -> None:
        self.lat = None
        self.lon = None
        self.cog = None
        self.sog = None
        self.dt = None

        # Pandas DataFrame
        self.df = None

        # Rates
        self.sog_rate = None
        self.cog_rate = None
        self.z = None

        if csv_file is not None:
            # Read the csv file
            self.read_csv(csv_file=csv_file)

            # Rates
            self.sog_rate = self.calculate_sog_rate()
            self.cog_rate = self.calculate_cog_rate()

            # Measurements
            self.z = self.get_measurements()

    def read_csv(
        self,
        csv_file: str,
        ship_id: str | None = None,
        lat_col: str = "lat",
        lon_col: str = "lon",
        reverse: bool = False,
    ) -> pd.DataFrame:
        """
        Read a csv file containing ship track data.

        Parameters
        ----------
        csv_file
            The path to the csv file.
        ship_id
            The id of the ship.
        lat_col
            The name of the column containing the latitude.
        lon_col
            The name of the column containing the longitude.
        reverse
            Whether to reverse the order of measurements.

        Returns
        -------
        self.df
            The dataframe containing the ship track data.
        """
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
        self.lat = pd.to_numeric(self.df[lat_col]).values
        self.lon = pd.to_numeric(self.df[lon_col]).values

        if reverse:
            # Reverse the order of the lat, lon, dts values
            self.dts = self.dts[::-1]
            self.lat = self.lat[::-1]
            self.lon = self.lon[::-1]

        return self.lat, self.lon, self.dt

    def calculate_sog(self) -> np.ndarray:
        """
        Calculate the speed over ground.

        Returns
        -------
        self.sog
            A list of floats representing the speed over ground

        Notes
        -----
        This function calculates the speed over ground (SOG) given a series of longitude,
        latitude, and time intervals.
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

    def calculate_sog_rate(self) -> np.ndarray:
        """
        Calculate the speed over ground (SOG) rate.

        Returns
        -------
        self.sog_rate
            An array representing the speed over ground rate.

        Notes
        -----
        The backward difference formula is used to calculate the SOG rate.
        """
        if self.sog is None:
            self.calculate_sog()

        self.sog_rate = []

        for i in range(1, len(self.sog)):
            self.sog_rate.append((self.sog[i] - self.sog[i - 1]) / self.dts[i - 1])

        self.sog_rate = np.asarray(self.sog_rate)

        return self.sog_rate

    def calculate_cog(self):
        """
        Calculate the course over ground (COG).

        Returns
        -------
        self.cog
            A list of floats representing the course over ground

        Notes
        -----
        This function calculates the speed over ground (COG) given a series of longitude,
        latitude, and time intervals.
        """
        self.cog = []

        for i in range(1, len(self.lon)):
            cog = heading(self.lon[i - 1], self.lat[i - 1], self.lon[i], self.lat[i])
            self.cog.append(cog)

        # Assume stationary trajectory from the end point onwards
        self.cog.append(0.0)
        self.cog = np.asarray(self.cog)

        return self.cog

    def calculate_cog_rate(self) -> np.ndarray:
        """
        Calculate the course over ground (COG) rate.

        Returns
        -------
        self.cog_rate
            An array representing the course over ground rate.

        Notes
        -----
        The backward difference is used to calculate the COG rate.
        """
        if self.cog is None:
            self.calculate_cog()

        self.cog_rate = []

        for i in range(1, len(self.cog)):
            self.cog_rate.append((self.cog[i] - self.cog[i - 1]) / self.dts[i - 1])

        self.cog_rate = np.asarray(self.cog_rate)

        return self.cog_rate

    def get_measurements(self) -> np.ndarray:
        """
        Get the measurement matrix z.

        Returns
        -------
        self.z
            The measurement matrix.
        """
        if self.sog is None:
            self.calculate_sog()

        if self.cog is None:
            self.calculate_cog()

        # z = np.vstack((self.lon, self.lat, self.sog, self.cog))
        self.z = np.vstack((self.lon, self.lat, self.sog, self.cog))

        return self.z
