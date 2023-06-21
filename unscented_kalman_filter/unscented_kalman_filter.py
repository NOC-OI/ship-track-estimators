import numpy as np

# Constants
EARTH_RADIUS = 6378.137  # in km


def hx(x):
    """TODO write docstring."""
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    return np.array([x[0], x[1]])


def fx(x, dt, sog_rate=0.0, cog_rate=0):
    """
    TODO write docstring.

    Notes
    -----
    Dynamics based on the method described in:

    Cole, B.; Schamberg, G.
    Unscented Kalman Filter for Long-Distance Vessel Tracking in Geodetic Coordinates.
    Applied Ocean Research 2022, 124, 103205.
    https://doi.org/10.1016/j.apor.2022.103205.

    Parameters
    ----------
    x
        _description_
    dt
        _description_
    sog_rate, optional
        _description_, by default 0.
    cog_rate, optional
        _description_, by default 0

    Returns
    -------
        _description_
    """
    lon = x[0]
    lat = x[1]
    u = x[2]
    alpha = x[3]

    # Precompute sin and cos of udt_R
    udt_R = u * dt / EARTH_RADIUS
    udt_R_sin = np.sin(udt_R)
    udt_R_cos = np.cos(udt_R)

    # Eq. (35), lon
    arctan_term_a = udt_R_sin * np.sin(alpha)
    arctan_term_b = np.cos(lat) * udt_R_cos - np.sin(lat) * udt_R_sin * np.cos(alpha)
    transformed_lon = lon + np.arctan2(arctan_term_a, arctan_term_b)

    # Eq. (35), lat
    transformed_lat = np.sin(lat) * udt_R_cos + np.cos(lat) * udt_R_sin * np.cos(alpha)
    transformed_lat = np.arcsin(transformed_lat)

    # Eq. (35), u
    transformed_u = u + sog_rate * dt

    # Eq. (35), alpha
    transformed_alpha = alpha + cog_rate * dt

    transformed_x = np.array(
        [transformed_lon, transformed_lat, transformed_u, transformed_alpha]
    )

    return transformed_x
