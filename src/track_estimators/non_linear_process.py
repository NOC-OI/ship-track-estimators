import numpy as np

from .constants import EARTH_RADIUS


def fx(x: np.ndarray, dt: float, sog_rate: float = 0.0, cog_rate: float = 0.0):
    """
    Summary.

    Parameters
    ----------
    x : numpy.ndarray, shape=(4,)
        State vector representing [longitude, latitude, velocity, heading].
    dt : float
        Time step.
    sog_rate : float, optional
        Speed over ground rate, by default 0.0.
    cog_rate : float, optional
        Course over ground rate, by default 0.0.

    Returns
    -------
    numpy.ndarray, shape=(4,)
        Transformed state vector representing [longitude, latitude, velocity, heading].
    """
    lon = np.radians(x[0])
    lat = np.radians(x[1])
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
    transformed_lon = np.degrees(transformed_lon)  # convert to degrees

    # Eq. (35), lat
    transformed_lat = np.sin(lat) * udt_R_cos + np.cos(lat) * udt_R_sin * np.cos(alpha)
    transformed_lat = np.degrees(np.arcsin(transformed_lat))  # convert to degrees

    # Eq. (35), u
    transformed_u = u + sog_rate * dt

    # Eq. (35), alpha
    transformed_alpha = alpha + cog_rate * dt

    transformed_x = np.array(
        [transformed_lon, transformed_lat, transformed_u, transformed_alpha]
    )

    return transformed_x
