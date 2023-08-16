import numpy as np
from track_estimators.utils import (
    geographiclib_distance,
    geographiclib_heading,
    haversine_formula,
    heading,
)


def test_haversine_formula():
    """Test the haversine formula."""
    # Check distance between New York City and Los Angeles
    lat1 = 40.7128
    lon1 = -74.0060
    lat2 = 34.0522
    lon2 = -118.2437

    assert np.isclose(haversine_formula(lon1, lat1, lon2, lat2), 3933.96, rtol=1e-2)

    # Check the distance between Porto and Lisbon
    lat1 = 38.7167
    lon1 = -9.13333
    lat2 = 41.1579
    lon2 = -8.6291

    assert np.isclose(haversine_formula(lon1, lat1, lon2, lat2), 273.59, rtol=1e-2)

    # Check distance when two points are identical
    lat = np.random.uniform(-180, 180)
    lon = np.random.uniform(-90, 90)

    assert np.isclose(haversine_formula(lon, lat, lon, lat), 0.0)


def test_geographiclib_distance():
    """Test the geographiclib_distance function."""
    # Check distance between New York City and Los Angeles
    lat1 = 40.7128
    lon1 = -74.0060
    lat2 = 34.0522
    lon2 = -118.2437

    assert np.isclose(
        geographiclib_distance(lon1, lat1, lon2, lat2), 3933.96, rtol=1e-2
    )

    # Check the distance between Porto and Lisbon
    lat1 = 38.7167
    lon1 = -9.13333
    lat2 = 41.1579
    lon2 = -8.6291

    assert np.isclose(geographiclib_distance(lon1, lat1, lon2, lat2), 273.59, rtol=1e-2)

    # Check distance when two points are identical
    lat = np.random.uniform(-180, 180)
    lon = np.random.uniform(-90, 90)

    assert np.isclose(geographiclib_distance(lon, lat, lon, lat), 0.0)


def test_heading():
    """
    Test the heading function.

    References
    ----------
    - https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/

    """
    # Check heading between Kansas City (P1) and St. Louis (P2)
    lat1 = 39.099912
    lon1 = -94.581213
    lat2 = 38.627089
    lon2 = -90.200203

    assert np.isclose(heading(lon1, lat1, lon2, lat2), 96.51, rtol=1e-3)

    # Check heading when two points are identical
    lat = np.random.uniform(-180, 180)
    lon = np.random.uniform(-90, 90)

    assert np.isclose(heading(lon, lat, lon, lat), 0.0)


def test_geographiclib_heading():
    """Test the geographiclib_heading function."""
    # Check heading between Kansas City (P1) and St. Louis (P2)
    lat1 = 39.099912
    lon1 = -94.581213
    lat2 = 38.627089
    lon2 = -90.200203

    assert np.isclose(geographiclib_heading(lon1, lat1, lon2, lat2), 96.51, rtol=1e-3)

    # Check heading when two points are identical
    lat = np.random.uniform(-180, 180)
    lon = np.random.uniform(-90, 90)

    assert np.isclose(geographiclib_heading(lon, lat, lon, lat), 0.0)
