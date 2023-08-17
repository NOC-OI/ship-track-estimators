from typing import List, Union

import numpy as np
from geographiclib.geodesic import Geodesic

from .constants import EARTH_RADIUS


def geographiclib_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the distance between two points using the geographiclib library.

    Parameters
    ----------
    lon1
        The longitude of the first point.
    lat1
        The latitude of the first point.
    lon2
        The longitude of the second point.
    lat2
        The latitude of the second point.

    Returns
    -------
    dist
        The distance between the two points in kilometers.

    References
    ----------
    1. https://geographiclib.sourceforge.io/html/python/geodesics.html#introduction
    """
    if np.abs(lat1 - lat2) < 1e-8 and np.abs(lon1 - lon2) < 1e-8:
        dist = 0.0
    else:
        res = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        dist = res["s12"] * 1e-3  # convert from meters to kilometers
    return dist


def geographiclib_heading(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the heading between two points using the geographiclib library.

    Parameters
    ----------
    lon1
        The longitude of the first point.
    lat1
        The latitude of the first point.
    lon2
        The longitude of the second point.
    lat2
        The latitude of the second point.

    Returns
    -------
    heading
        The heading between the two points.

    References
    ----------
    1. https://geographiclib.sourceforge.io/html/python/geodesics.html#introduction
    """
    if np.abs(lat1 - lat2) < 1e-8 and np.abs(lon1 - lon2) < 1e-8:
        heading = 0.0
    else:
        res = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        heading = res["azi1"]
        heading = (heading + 360) % 360

    return heading


def haversine_formula(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the distance between two points on Earth using the Haversine formula.

    Parameters
    ----------
    lon1
        The longitude of the first point.
    lat1
        The latitude of the first point.
    lon2
        The longitude of the second point.
    lat2
        The latitude of the second point.

    Returns
    -------
    dist
        The distance between the two points in kilometers.

    References
    ----------
    1. https://www.movable-type.co.uk/scripts/latlong.html
    2. https://en.wikipedia.org/wiki/Haversine_formula
    3. https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    dist = c * EARTH_RADIUS

    return dist


def heading(lon1, lat1, lon2, lat2):
    """
    Calculate the heading between two points on Earth.

    Parameters
    ----------
    lon1
        The longitude of the first point.
    lat1
        The latitude of the first point.
    lon2
        The longitude of the second point.
    lat2
        The latitude of the second point.

    Returns
    -------
    heading
        The heading between the two points in degrees.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # calculate COG
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    heading = np.arctan2(x, y)
    heading = np.degrees(heading)
    heading = (heading + 360) % 360

    return heading


def smooth(y: np.ndarray, box_pts: int) -> np.ndarray:
    """
    Smooth the input array by applying a moving average filter.

    Parameters
    ----------
    y
        The input array to be smoothed.
    box_pts
        The number of points to use for the moving average.

    Returns
    -------
    y_smooth
        The smoothed array.
    """
    # Create a box filter with ones / box_pts of size box_pts
    box = np.ones(box_pts) / box_pts
    # Apply the box filter to the input array using convolution
    y_smooth = np.convolve(y, box, mode="same")
    # Return the smoothed array

    return y_smooth


def generate_dts(
    dts: np.ndarray | List[Union[int, float]], substeps: int
) -> np.ndarray:
    """
    Generate an array of time steps based on the desired number of sub-steps.

    Parameters
    ----------
    dts
        The original time steps.
        Typically these are the time difference between measurements.
    substeps
        The number of sub-steps.

    Returns
    -------
    dt
        The time steps.
    """
    dt = []
    for time in dts:
        for i in range(substeps):
            dt.append(time / substeps)
    dt = np.asarray(dt)
    return dt
