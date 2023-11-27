import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from track_estimators.gaussian_processes.gp import GPRegression
from track_estimators.ship_track import ShipTrack
from track_estimators.utils import generate_dts

# -------------------------------------------------------------- #
#                                                                #
#                             INPUT                              #
#                                                                #
# -------------------------------------------------------------- #
ship_track = ShipTrack()
ship_track.read_csv(
    csv_file="../data/historical_ships/historical_ship_data.csv",
    ship_id="01203792",
    id_col="primary.id",
    lat_col="lat",
    lon_col="lon2",
    reverse=False,
)

# -------------------------------------------------------------- #
#                                                                #
#                 GAUSSIAN PROCESS REGRESSION                    #
#                                                                #
# -------------------------------------------------------------- #
# Define the kernel for joint modeling of latitude and longitude
kernel = 1.0 * RBF() + WhiteKernel(noise_level=0.5)

# Create Gaussian Process Regression model and do the fitting
gpr = GPRegression(kernel=kernel)
gpr.fit(ship_track)

# Generate new times for prediction
times_predict = generate_dts(ship_track.dts, substeps=1)
times_predict = np.cumsum(times_predict)
times_predict = np.insert(times_predict, 0, 0)

# Predict latitude and longitude on the new time points
predicted, std = gpr.predict(times=times_predict)
predicted_lon, predicted_lat = predicted[:, 0], predicted[:, 1]
std_lon, std_lat = std[:, 0], std[:, 1]

# -------------------------------------------------------------- #
#                                                                #
#                       ANALYSE THE RESULTS                      #
#                                                                #
# -------------------------------------------------------------- #
s = 30
alpha = 0.6

fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(20, 15),
)
ax.stock_img()
ax.coastlines()
ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.6,
    color="gray",
    alpha=0.5,
    linestyle="-.",
)


ax.scatter(
    ship_track.lon,
    ship_track.lat,
    transform=ccrs.PlateCarree(),
    marker="*",
    s=s * 3,
    label="Historical Track",
    alpha=alpha,
)


ax.scatter(
    predicted_lon,
    predicted_lat,
    transform=ccrs.PlateCarree(),
    marker="o",
    label="Gaussian Process regression",
    alpha=alpha,
    s=s,
    color="green",
)

plt.xlim(ship_track.lon.min() - 5, ship_track.lon.max() + 5)
plt.ylim(ship_track.lat.min() - 5, ship_track.lat.max() + 5)
plt.legend()
plt.show()
