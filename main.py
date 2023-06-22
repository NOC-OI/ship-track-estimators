import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from track_estimators.non_linear_process import fx
from track_estimators.ship_track import ShipTrack
from track_estimators.unscented_kf import UnscentedKalmanFilter

csv_file = "data/modern_ships/WCE5063_subset_downsampled.csv"
ship_track = ShipTrack(csv_file)


# Create measurament matrix
z = ship_track.get_measurements()

# -------------------------------------------------------------- #
# Observation matrix
H = np.diag([1, 1, 0.0, 0.0])

# Measurement Uncertainty
corr_coff = 0.0
err_x = err_y = 0.5

R = np.array(
    [
        [err_x * err_x, err_x * err_y * corr_coff, 0.0, 0.0],
        [err_x * err_y * corr_coff, err_y * err_y, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)


# Process noise covariance
Q = np.diag([err_x, err_y, 0.0, 0.0])

# Estimate covariance matrix
# https://www.researchgate.net/post/Kalman_filter_how_do_I_choose_initial_P_0
P = np.diag([0.5, 0.5, 0.0, 0.0])

# Initial state
x0 = z[:, 0].reshape(-1, 1)
# -------------------------------------------------------------- #

# Create the unscented kalman filter
ukf = UnscentedKalmanFilter(H=H, Q=Q, R=R, P=P, x0=x0, non_linear_process=fx)


predictions = []

ukf.update(z[:, 0])

substeps = 4
for step in range(1, len(ship_track.lat)):
    for i in range(int(substeps)):
        ukf.predict(
            dt=ship_track.dts[step - 1] / substeps,
            sog_rate=ship_track.sog_rate[step - 1],
            cog_rate=ship_track.cog_rate[step - 1],
        )

        if i < substeps - 1:
            print(i, ukf.x)
            predictions.append(ukf.x)

    ukf.update(z[:, step])
    predictions.append(ukf.x)
    print(ukf.x)

predictions = np.asarray(predictions)
print(predictions.shape)


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
