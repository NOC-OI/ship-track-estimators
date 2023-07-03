import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from track_estimators.kalman_filters.non_linear_process import geodetic_dynamics
from track_estimators.kalman_filters.unscented import UnscentedKalmanFilter
from track_estimators.ship_track import ShipTrack

csv_file = "data/modern_ships/WCE5063_subset_downsampled.csv"
ship_track = ShipTrack(csv_file)

ship_track_original = ShipTrack("data/modern_ships/WCE5063_subset.csv")

# Create measurament matrix
z = ship_track.get_measurements()

# -------------------------------------------------------------- #
# Observation matrix
H = np.diag([1, 1, 0, 0])

# Measurement Uncertainty
corr_coff = 0.0
err_x, err_y = (0.5, 0.5)

R = np.array(
    [
        [err_x * err_x, err_x * err_y * corr_coff, 0.0, 0.0],
        [err_x * err_y * corr_coff, err_y * err_y, 0.0, 0.0],
        [0.0, 0.0, 10, 0.0],
        [0.0, 0.0, 0.0, 0.5],
    ]
)


# Process noise covariance
Q = np.diag(R**2)

# Estimate covariance matrix
# https://www.researchgate.net/post/Kalman_filter_how_do_I_choose_initial_P_0
P = np.diag([err_x, err_y, R[2, 2], R[3, 3]])
print(P)
# Initial state
x0 = z[:, 0].reshape(-1, 1)
# -------------------------------------------------------------- #

# Create the unscented kalman filter
ukf = UnscentedKalmanFilter(
    H=H, Q=Q, R=R, P=P, x0=x0, non_linear_process=geodetic_dynamics
)

predictions = []


ms = 10
alpha = 0.4

ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.scatter(
    predictions[:, 0, 0],
    predictions[:, 1, 0],
    label="Kalman Filter Prediction",
    s=ms,
    color="green",
    alpha=alpha,
)
ax.scatter(ship_track.lon, ship_track.lat, label="Measurements", s=ms, alpha=alpha)

# ax.plot(predictions[:, 0, 0], predictions[:, 1, 0],  alpha=alpha)
# ax.plot(lon, lat, alpha=alpha)
# ax.set_extent([-40, -20, -40, -10])
plt.xlim(-85, -70)
plt.ylim(25, 45)
plt.legend()
plt.show()


plt.plot(ship_track_original.lon - predictions[: len(ship_track_original.lon), 0, 0])
plt.plot(ship_track_original.lat - predictions[: len(ship_track_original.lat), 1, 0])
plt.show()
