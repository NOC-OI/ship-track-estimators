import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from track_estimators.kalman_filters.non_linear_process import geodetic_dynamics
from track_estimators.kalman_filters.unscented import UnscentedKalmanFilter
from track_estimators.ship_track import ShipTrack

# Read the subsampled data
csv_file = "data/modern_ships/WCE5063_subset_downsampled_rounded_noisy.csv"
ship_track = ShipTrack()
ship_track.read_csv(csv_file, lat_col="lat_round", lon_col="lon_round", reverse=False)

# Calculate the sog, cog, and respective rates
ship_track.calculate_cog()
ship_track.calculate_sog()

ship_track.calculate_cog_rate()
ship_track.calculate_sog_rate()

# Get the measurements
ship_track.get_measurements()


# Read the original data
ship_track_original = ShipTrack("data/modern_ships/WCE5063_subset.csv")


# -------------------------------------------------------------- #
#                     Define the KF matrices                     #
# -------------------------------------------------------------- #
# Observation matrix
H = np.diag([1, 1, 0, 0])

# Measurement noise covariance
corr_coff = 0.0
std_x, std_y = (0.1, 0.1)

R = np.array(
    [
        [std_x * std_x, std_x * std_y * corr_coff, 0.0, 0.0],
        [std_x * std_y * corr_coff, std_y * std_y, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)


# Process noise covariance
Q = np.diag([0.001, 0.001, 0.1, 0.5])

# Initial state
x0 = ship_track.z[:, 0].reshape(-1, 1)


# Estimate covariance matrix
# https://www.researchgate.net/post/Kalman_filter_how_do_I_choose_initial_P_0
# This does not seem to affect uch the output
P = np.diag([1, 1, 1, 1]) * 1e-3
# P = np.diag(x0[:, 0]**2)


# -------------------------------------------------------------- #
#                     Unscented Kalman Filter                    #
# -------------------------------------------------------------- #
# Create and run the unscented kalman filter
ukf = UnscentedKalmanFilter(
    H=H, Q=Q, R=R, P=P, x0=x0, non_linear_process=geodetic_dynamics
)

dt = 1
predictions, estimate_vars = ukf.run(int(204 / dt), dt, ship_track)
predictions = np.asarray(predictions)
estimate_vars = np.asarray(estimate_vars)


# -------------------------------------------------------------- #
#                         Plot the results                       #
# -------------------------------------------------------------- #
ms = 20
alpha = 0.4


plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()

ax.scatter(
    savgol_filter(predictions[:, 0, 0], 12, 1, mode="interp"),
    savgol_filter(predictions[:, 1, 0], 12, 1, mode="interp"),
    label="Kalman Filter + Savgol Filter",
    s=ms,
    color="green",
    alpha=alpha,
)


ax.scatter(
    ship_track.lon,
    ship_track.lat,
    label="Subsampled",
    s=ms,
    alpha=alpha,
    marker="*",
    color="red",
)
ax.scatter(
    ship_track_original.lon,
    ship_track_original.lat,
    label="Original",
    s=ms,
    alpha=alpha,
)


# ax.scatter(
#      predictions[:, 0, 0],
#      predictions[:, 1, 0],
#      label="Kalman Filter Prediction",
#      s=ms,
#      color="yellow",
#      alpha=alpha,
# )

# ax.plot(predictions[:, 0, 0], predictions[:, 1, 0],  alpha=alpha)
# ax.plot(lon, lat, alpha=alpha)
# ax.set_extent([-40, -20, -40, -10])
plt.xlim(-85, -70)
plt.ylim(25, 45)
plt.legend()
plt.savefig("kf_results.png", bbox_inches="tight", dpi=1200)
plt.show()


plt.plot(ship_track_original.lon[:170] - predictions[:170, 0, 0], label=r"$d\phi$")
plt.plot(ship_track_original.lat[:170] - predictions[:170, 1, 0], label=r"$d\lambda$")

# plt.scatter(estimate_vars[:170, 0], ship_track_original.lon[: 170] - predictions[:170, 0, 0])

plt.plot(estimate_vars[:, 0], label=r"Varlon")
plt.plot(estimate_vars[:, 1], label=r"Varlat")
plt.legend()
plt.ylabel("difference")
plt.savefig("diff.png", bbox_inches="tight", dpi=600)
plt.show()


# plt.plot(ship_track_original.lon - predictions[: len(ship_track_original.lon), 0, 0])
# plt.plot(ship_track_original.lat - predictions[: len(ship_track_original.lat), 1, 0])
# plt.show()