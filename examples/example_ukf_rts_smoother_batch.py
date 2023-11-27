import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from track_estimators.kalman_filters.non_linear_process import geodetic_dynamics
from track_estimators.kalman_filters.unscented import UnscentedKalmanFilter
from track_estimators.ship_track import ShipTrack
from track_estimators.utils import generate_dts

# create a PdfPages object
pdf = PdfPages("results.pdf")

# Historical data
df = pd.read_csv("../data/historical_ships/historical_ship_data.csv")
ids = df["primary.id"].unique().tolist()
ids.pop(1)

for id in ids:
    print("Doing", id)
    # -------------------------------------------------------------- #
    #                                                                #
    #                             INPUT                              #
    #                                                                #
    # -------------------------------------------------------------- #
    ship_track = ShipTrack()
    ship_track.read_csv(
        csv_file="../data/historical_ships/historical_ship_data.csv",
        ship_id=id,
        id_col="primary.id",
        lat_col="lat",
        lon_col="lon2",
        reverse=False,
    )

    z = ship_track.get_measurements(include_sog=True, include_cog=True)

    # Calculate cog and sog rates
    ship_track.calculate_cog_rate()
    ship_track.calculate_sog_rate()

    # Observation matrix
    H = np.diag([1, 1, 0, 0])

    # Measurement Uncertainty
    R = np.diag([0.25, 0.25, 0, 0])

    # Process noise covariance
    Q = np.diag([1e-4, 1e-4, 1e-6, 1e-6])

    # Estimate covariance matrix
    P = np.diag([1.0, 1.0, 1.0, 1.0])

    # -------------------------------------------------------------- #
    #                                                                #
    #                   UNSCENTED KALMAN FILTER                      #
    #                                                                #
    # -------------------------------------------------------------- #
    # Initial state
    x0 = z[:, 0].reshape(-1, 1).copy()

    # Create the unscented kalman filter instance
    ukf = UnscentedKalmanFilter(
        H=H, Q=Q, R=R, P=P, x0=x0, non_linear_process=geodetic_dynamics
    )

    # Generate dt array
    dt_array = generate_dts(ship_track.dts, 2)

    if dt_array.max() > 48:
        print("Skipping becasue dt > 48", id, dt_array.max())
        continue
    try:
        # Run the UKF
        predictions, estimate_vars = ukf.run(
            nsteps=len(dt_array), dt=dt_array, ship_track=ship_track
        )

        # -------------------------------------------------------------- #
        #                                                                #
        #                         RTS SMOOTHER                           #
        #                                                                #
        # -------------------------------------------------------------- #
        # Run the RTS smoother
        predictions_smoothed, estimate_vars_smoothed = ukf.run_rts_smoother(
            ship_track=ship_track
        )
    except Exception:
        print("Error in ", id)
        continue

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
        z[0, :],
        z[1, :],
        transform=ccrs.PlateCarree(),
        marker="^",
        s=s + 20,
        label="Original Track",
        alpha=alpha,
    )
    plt.title(id)
    plt.xlim(z[0, :].min() - 5, z[0, :].max() + 5)
    plt.ylim(z[1, :].min() - 5, z[1, :].max() + 5)
    plt.legend()
    pdf.savefig(fig)

    plt.clf()
    print("********************************************************************")

pdf.close()
