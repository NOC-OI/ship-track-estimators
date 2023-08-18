import numpy as np

from ..constants import EARTH_RADIUS


def geodetic_dynamics(
    x: np.ndarray,
    c: np.ndarray,
    dt: float,
    sog_rate: float = 0.0,
    cog_rate: float = 0.0,
):
    """
    Update step of the geodetic process model.

    Parameters
    ----------
    x
        State vector typically but not necessarily
          representing [longitude, latitude, velocity, heading].
        Longitide, latitude and heading are assumed to be in degrees.
    c
        Control input vector.
    dt
        Time step.
    sog_rate
        Speed over ground rate, by default 0.0.
    cog_rate
        Course over ground rate, by default 0.0.

    Returns
    -------
    numpy.ndarray, shape=(4,)
        Transformed state vector representing [longitude, latitude, velocity, heading].
        Longitide, latitude and heading are given in degrees to be in degrees.

    Notes
    -----
    Equations taken from Section 2 of the following paper:

    Cole, B.; Schamberg, G.
    Unscented Kalman Filter for Long-Distance Vessel Tracking in Geodetic Coordinates.
    Applied Ocean Research 2022, 124, 103205.
    https://doi.org/10.1016/j.apor.2022.103205.
    """
    # Latitude, longitude and heading are converted to radians
    if c is None:
        c = np.asarray([])

    # Concatenate the state and control input vector
    xconcat = np.concatenate((x, c))

    # Extract lat, lon, sog, and cog
    lon = np.radians(xconcat[0])
    lat = np.radians(xconcat[1])
    u = xconcat[2]
    alpha = np.radians(xconcat[3])

    # Precompute sin and cos of udt_R
    udt_R = u * dt / EARTH_RADIUS
    udt_R_sin = np.sin(udt_R)
    udt_R_cos = np.cos(udt_R)

    # Eq. (35), lon
    arctan_term_a = udt_R_sin * np.sin(alpha)
    arctan_term_b = np.cos(lat) * udt_R_cos - np.sin(lat) * udt_R_sin * np.cos(alpha)
    transformed_lon = lon + np.arctan2(arctan_term_a, arctan_term_b)
    transformed_lon = np.degrees(transformed_lon)  # convert to degrees

    # Eq. (35), lat
    transformed_lat = np.sin(lat) * udt_R_cos + np.cos(lat) * udt_R_sin * np.cos(alpha)
    transformed_lat = np.degrees(np.arcsin(transformed_lat))  # convert to degrees

    # Eq. (35), u
    transformed_u = u + sog_rate * dt

    # Eq. (35), alpha
    transformed_alpha = np.degrees(alpha) + cog_rate * dt

    # Create the final state vector
    transformed_x = np.array(
        [transformed_lon, transformed_lat, transformed_u, transformed_alpha]
    )

    return transformed_x[: x.shape[0]]
