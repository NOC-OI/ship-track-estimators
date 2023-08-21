import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------- #
#                                                                #
#                             INPUT                              #
#                                                                #
# -------------------------------------------------------------- #
predictions = np.loadtxt("output_01203823_predictions.txt")
predictions_smoothed = np.loadtxt("output_01203823_predictions_smoothed.txt")
original = np.loadtxt("original_01203823_track.txt")

# -------------------------------------------------------------- #
#                                                                #
#                       ANALYSE THE RESULTS                      #
#                                                                #
# -------------------------------------------------------------- #
s = 40
alpha = 0.7

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
    predictions[:, 0],
    predictions[:, 1],
    transform=ccrs.PlateCarree(),
    marker="*",
    s=s,
    label="Kalman filter - Forward",
    alpha=alpha,
)

ax.scatter(
    predictions_smoothed[:, 0],
    predictions_smoothed[:, 1],
    transform=ccrs.PlateCarree(),
    marker="o",
    s=s,
    label="Kalman filter - RTS",
    alpha=alpha,
)

ax.scatter(
    original[:, 0],
    original[:, 1],
    transform=ccrs.PlateCarree(),
    marker="^",
    s=s,
    label="Original Ship Track",
    alpha=alpha,
)


plt.xlim(original[:, 0].min() - 5, original[:, 0].max() + 5)
plt.ylim(original[:, 1].min() - 5, original[:, 1].max() + 5)
plt.legend()
plt.show()
