from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .utils import geographiclib_distance, geographiclib_heading


class ShipTrack:
    """
    ShipTrack class representing the data for a single ship track.

    Parameters
    ----------
    csv_file
        Name of csv file.
    ship_id
        The id of the ship.
    estimate_cog
        Whether to estimate the course over ground between successive measurements.
    estimate_sog
        Whether to estimate the speed over ground between successive measurements.
    estimate_cog_rate
        Whether to estimate the course over ground rate using numerical derivatives upon initialisation.
    estimate_sog_rate
        Whether to estimate the speed over ground rate using numerical derivatives upon initialisation.
    calc_distance_func
        A function to calculate the distance between two points.
    calc_heading_func
        A function to calculate the heading between two points.

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
    dts
        A list or array of floats representing the time difference between measurements.
    dates
        A list or array of datetime objects representing the dates of the measurements.
    df
        The dataframe containing the ship track data.
    sog_rate
        A list or array of floats representing the speed over ground rate.
    cog_rate
        A list or array of floats representing the course over ground rate.
    z
        A list or array of floats representing the measurement.
    calc_distance_func
        A function to calculate the distance between two points.
    calc_heading_func
        A function to calculate the heading between two points.
    """

    def __init__(
        self,
        csv_file: Optional[str] = None,
        estimate_cog: bool = False,
        estimate_sog: bool = False,
        estimate_sog_rate: bool = False,
        estimate_cog_rate: bool = False,
        calc_distance_func: Callable = geographiclib_distance,
        calc_heading_func: Callable = geographiclib_heading,
    ) -> None:
        self.lat = None
        self.lon = None
        self.cog = None
        self.sog = None
        self.dts = None
        self.dates = None

        # Pandas DataFrame
        self.df = None

        # Rates
        self.sog_rate = None
        self.cog_rate = None
        self.z = None

        # Heading and distance functions
        self.calc_distance_func = calc_distance_func
        self.calc_heading_func = calc_heading_func

        if csv_file is not None:
            # Read the csv file
            self.read_csv(csv_file=csv_file)

            # Estimate sog and cog and respective rates
            if estimate_sog_rate:
                self.calculate_sog_rate()
            elif estimate_sog:
                self.calculate_sog()

            if estimate_cog_rate:
                self.calculate_cog_rate()
            elif estimate_cog:
                self.calculate_cog()

            # Measurements
            self.z = self.get_measurements()

    def read_csv(
        self,
        csv_file: str,
        ship_id: Optional[Union[str, int]] = None,
        id_col: str = "id",
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
        id_col
            The name of the column containing the ship id.
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

        # Convert ship id to str
        self.df[id_col] = self.df[id_col].astype(str)
        ship_id = str(ship_id)

        # Get the ship course and sort by date
        if ship_id is not None:
            self.df = self.df.loc[self.df[id_col] == ship_id]

        # Sort index
        self.df.sort_index(axis=0, inplace=True, ignore_index=True)

        if self.df.empty:
            raise ValueError(f"No data found for ship '{ship_id}' in '{csv_file}'.")

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
        self.dates = pd.to_datetime(self.df.date).to_list()

        self.dts = []
        for i in range(len(self.dates) - 1):
            dt = pd.Timedelta(self.dates[i + 1] - self.dates[i]).total_seconds()
            self.dts.append(dt / 3600.0)  # seconds to hours

        self.dts = np.asarray(self.dts)

        # Sort values by date
        # self.df = self.df.sort_values(by="date")
        # Extract lat, lon
        self.lat = pd.to_numeric(self.df[lat_col]).values
        self.lon = pd.to_numeric(self.df[lon_col]).values

        # Make sure lat and lon are the same length, and they are not empty
        assert len(self.lon) > 0, f"Longitude list is empty for column '{lon_col}'."
        assert len(self.lat) > 0, f"Latitude list is empty for column '{lat_col}'."
        assert len(self.lat) == len(self.lon)

        if reverse:
            # Reverse the order of the lat, lon, dts values
            self.dts = self.dts[::-1]
            self.lat = self.lat[::-1]
            self.lon = self.lon[::-1]

        return self.lat, self.lon, self.dts

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
            dist = self.calc_distance_func(
                self.lon[i - 1], self.lat[i - 1], self.lon[i], self.lat[i]
            )
            self.sog.append(dist / self.dts[i - 1])

        # Assume stationary trajectory from the end point onwards
        self.sog.append(self.sog[-1])

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
        self.sog_rate.append(0)

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
            cog = self.calc_heading_func(
                self.lon[i - 1], self.lat[i - 1], self.lon[i], self.lat[i]
            )
            self.cog.append(cog)

        # Assume stationary trajectory from the end point onwards
        self.cog.append(self.cog[-1])
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
        self.cog_rate.append(0)

        for i in range(1, len(self.cog)):
            self.cog_rate.append((self.cog[i] - self.cog[i - 1]) / self.dts[i - 1])

        self.cog_rate = np.asarray(self.cog_rate)

        return self.cog_rate

    def get_measurements(
        self, include_sog: bool = False, include_cog: bool = False
    ) -> np.ndarray:
        """
        Get the measurement matrix z.

        Parameters
        ----------
        include_sog
            Whether to estimate the speed over ground and include it in the measurement matrix.
        include_cog
            Whether to estimate the course over ground and include it in the measurement matrix.

        Returns
        -------
        self.z
            The measurement matrix.
        """
        self.z = np.vstack((self.lon, self.lat))

        if include_sog:
            if self.sog is None:
                self.calculate_sog()

            self.z = np.vstack((self.z, self.sog))

        if include_cog:
            if self.cog is None:
                self.calculate_cog()

            self.z = np.vstack((self.z, self.cog))

        return self.z

    def plot_trajectory(
        self,
        figsize: tuple = (20, 15),
        scatter_kwargs: dict = {"s": 10, "color": "red"},
        savefig: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot the trajectory.

        Parameters
        ----------
        figsize
            The size of the figure.
        scatter_kwargs
            Keyword arguments for the scatter plot.
        savefig
            The path to save the figure. If None, no figure is saved.
        show
            Whether to show the figure.
        """
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        assert self.lat is not None, "Latitude is not set."
        assert self.lon is not None, "Longitude is not set."

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=figsize,
        )
        ax.stock_img()
        ax.coastlines()
        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.6,
            color="gray",
            alpha=0.5,
            linestyle="-.",
        )
        # Downsampled trajectory
        ax.scatter(self.lon, self.lat, transform=ccrs.PlateCarree(), **scatter_kwargs)

        if savefig:
            plt.savefig(savefig, bbox_inches="tight", dpi=300)

        if show:
            plt.show()

        return fig, ax
