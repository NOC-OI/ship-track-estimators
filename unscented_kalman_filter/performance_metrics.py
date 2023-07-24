import numpy as np


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the root mean square error.

    Parameters
    ----------
    x
        The first array.
    y
        The second array.

    Returns
    -------
    rmse
        The root mean square error.
    """
    return np.sqrt(np.mean((x - y) ** 2))


def cum_abs_diff(x: np.ndarray, xref: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative absolute difference between the two arrays.

    Parameters
    ----------
    x
        The first array.
    xref
        The second array.

    Returns
    -------
    cum_abs_diff
        The cumulative absolute difference.
    """
    return np.cumsum(np.abs(x - xref))


def abs_diff(x: np.ndarray, xref: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute difference between the two arrays.

    Parameters
    ----------
    x
        The first array.
    xref
        The second array.

    Returns
    -------
    abs_diff
        The absolute difference.
    """
    return np.abs(x - xref)
