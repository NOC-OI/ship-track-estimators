import numpy as np
from track_estimators.utils import haversine_formula


def test_haversine_formula():
    """Test the haversine formula."""
    # Check distance between New York City and Los Angeles
    lat1 = 40.7128
    lon1 = -74.0060
    lat2 = 34.0522
    lon2 = -118.2437
    expected_distance = 3933.96  # Distance in kilometers

    distance = haversine_formula(lon1, lat1, lon2, lat2)
    assert np.isclose(distance, expected_distance, rtol=1e-2)

    # Check the distance between Porto and Lisbon
    # https://www.prokerala.com/travel/distance/from-porto/to-lisbon/
    lat1 = 38.7167
    lon1 = -9.13333
    lat2 = 41.1579
    lon2 = -8.6291
    expected_distance = 273.59  # Distance in kilometers

    distance = haversine_formula(lon1, lat1, lon2, lat2)
    assert np.isclose(distance, expected_distance, rtol=1e-2)

    # Check distance when two points are identical
    lat = np.random.uniform(-180, 180)
    lon = np.random.uniform(-90, 90)

    distance = haversine_formula(lon, lat, lon, lat)

    assert np.isclose(distance, 0.0)
