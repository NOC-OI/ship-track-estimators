from typing import List

import numpy as np
import pandas as pd


def get_historical_ship_data(csv_file: str, primary_id: str):
    """
    Get historical ship data.

    Parameters
    ----------
    csv_file
        Name of csv file..
    primary_id
        Primary id of the course.

    Returns
    -------
    lon, lat, dts
        Tuple of lists of longitude, latitude and time intervals (in hours)
    """
    df = pd.read_csv(csv_file)

    # Get the ship course and sort by date
    df_id = df.loc[df["primary.id"] == primary_id]
    df_id = df_id.sort_values(by="date")

    # Get the lon and lat
    lon = df_id.loc[df_id["primary.id"] == primary_id].lon2.values
    lat = df_id.loc[df_id["primary.id"] == primary_id].lat.values
    lon = pd.to_numeric(lon).tolist()
    lat = pd.to_numeric(lat).tolist()

    # Get the dates and calculate the time intervals
    dates = pd.to_datetime(df_id.date)
    dts = []
    for i in range(len(dates) - 1):
        dt = pd.Timedelta(dates[i + 1] - dates[i]).total_seconds()
        dts.append(dt / 3600.0)

    return lon, lat, dts


def haversine_formula(lon1, lat1, lon2, lat2):
    """
    Calculate the haversine formula.

    https://www.movable-type.co.uk/scripts/latlong.html
    https://en.wikipedia.org/wiki/Haversine_formula.
    """
    r = 6378.137
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c * r


def calculate_cog(lon1, lat1, lon2, lat2):
    # calculate COG
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    cog = np.arctan2(y, x)
    return cog


def state_mean(sigmas, Wm):
    x = np.zeros(4)

    for i in [0, 1, 3]:
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, i]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, i]), Wm))
        x[i] = np.arctan2(sum_sin, sum_cos)

    x[2] = np.sum(np.dot(sigmas[:, 1], Wm))

    return x


def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for i in [0, 1]:
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, i]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, i]), Wm))
        x[i] = np.arctan2(sum_sin, sum_cos)

    return x


def get_sog(lon: List[float], lat: List[float], dts: List[float]) -> List[float]:
    """
    Calculate the speed over ground.

    Notes
    -----
    This function calculates the speed over ground (SOG) given a series of longitude,
    latitude, and time intervals.

    Parameters
    ----------
    lon
        A list of floats representing the longitude coordinates
    lat
        A list of floats representing the latitude coordinates
    dts
        A list of floats representing the time intervals

    Returns
    -------
    sog
        A list of floats representing the speed over ground
    """
    sog = []

    for i in range(1, len(lon)):
        dist = haversine_formula(lon[i - 1], lat[i - 1], lon[i], lat[i])
        sog.append(dist / dts[i - 1])

    return sog


def get_cog(lon: List[float], lat: List[float]) -> List[float]:
    """
    Calculate the course over ground (COG).

    Notes
    -----
    This function calculates the course over ground given a series of longitude and latitude.

    Parameters
    ----------
    lon
        A list of floats representing the longitude coordinates
    lat
        A list of floats representing the latitude coordinates

    Returns
    -------
    cog
        A list of floats representing the course over ground
    """
    cog = []
    for i in range(1, len(lon)):
        cog.append(calculate_cog(lon[i - 1], lat[i - 1], lon[i], lat[i]))

    return cog


def get_cog_rate(cog, dts):
    """
    Estimate the COG rate using numerical differentiation.

    Parameters
    ----------
    cog
        A list of floats representing the course over ground
    dts
        A list of floats representing the time intervals

    Returns
    -------
    cog_rate
        A list of floats representing the course over ground rate
    """
    cog_rate = []  # we assume in the first interval there is no change
    for i in range(0, len(cog) - 1):
        cog_rate.append((cog[i] - cog[i - 1]) / dts[i])

    cog_rate.append(0)

    return cog_rate


def get_sog_rate(sog, dts):
    """
    Estimate the SOG rate using numerical differentiation.

    Parameters
    ----------
    sog
        A list of floats representing the speed over ground
    dts
        A list of floats representing the time intervals

    Returns
    -------
    sog_rate
        A list of floats representing the speed over ground rate
    """
    sog_rate = []  # we assume in the first interval there is no change
    for i in range(0, len(sog) - 1):
        sog_rate.append((sog[i] - sog[i - 1]) / dts[i])

    sog_rate.append(0)

    return sog_rate


"""
sog = []
cog = []
for i in range(1, len(zs)):
    cog.append(calculate_cog(zs[i-1][0], zs[i-1][1], zs[i][0], zs[i][1]))

# Calculate SOG (speed over ground) rate
dt_measurement = 4.
sog = np.asarray(sog) / dt
sog_rate = (sog[1:]-sog[:-1]) / dt

# calculate COG (course over ground) rate
cog = np.asarray(cog)
cog_rate = (cog[1:]-cog[:-1]) / dt
"""
