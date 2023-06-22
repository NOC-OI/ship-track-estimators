import numpy as np

from .constants import EARTH_RADIUS


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
    - https://www.movable-type.co.uk/scripts/latlong.html
    - https://en.wikipedia.org/wiki/Haversine_formula
    - https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
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
