"""
Example on how to use the ShipTrack class.

We are going to use the following ship track as examples and analyse some of their features:
- 01203823 (historical ship track)
- WCE5063 (modern ship track)
"""
import matplotlib.pyplot as plt
from track_estimators.ship_track import ShipTrack

# ------------------------------------------------------ #
#               Historical Ship Track                    #
# ------------------------------------------------------ #
csv_file = "data/historical_ships/historical_ship_data.csv"
ship_track = ShipTrack()
ship_track.read_csv(
    csv_file, ship_id="01203823", id_col="primary.id", lat_col="lat", lon_col="lon"
)

fig, ax = ship_track.plot_trajectory()

# To get the measurement matrix we need to call the get_measurements method
z = ship_track.get_measurements(include_sog=False, include_cog=False)
print("Measurements shape {}".format(z.shape))

# Notice we did not estimate the course over ground and speed over ground
# We can do so however, if we want
z = ship_track.get_measurements(include_sog=True, include_cog=True)
print("Measurements shape {}".format(z.shape))

# We can also plot time series of the latitude and longitude
plt.figure(figsize=(10, 10))
plt.plot(ship_track.dates, ship_track.lon, label="Longitude")
plt.plot(ship_track.dates, ship_track.lat, label="Latitude")
plt.grid()
plt.legend()
plt.show()

# Or the speed over ground and course over ground
# Which can also be calculated using the calculate_cog and calculate_sog methods
ship_track.calculate_cog()
ship_track.calculate_sog()
plt.figure(figsize=(10, 10))
plt.plot(ship_track.dates, ship_track.sog, label="Speed over ground")
plt.plot(ship_track.dates, ship_track.cog, label="Course over ground")
plt.grid()
plt.legend()
plt.show()

# ------------------------------------------------------ #
#                   Modern Ship Track                    #
# ------------------------------------------------------ #
# Modern ShipTrack
csv_file = "data/modern_ships/modern_ship_data.csv"
ship_track = ShipTrack()
ship_track.read_csv(
    csv_file, ship_id="WCE5063", id_col="id", lat_col="lat", lon_col="lon"
)

ship_track.get_measurements(include_sog=False, include_cog=False)
fig, ax = ship_track.plot_trajectory()
