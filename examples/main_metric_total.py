import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from track_estimators.kalman_filters.non_linear_process import geodetic_dynamics
from track_estimators.kalman_filters.unscented import UnscentedKalmanFilter
from track_estimators.ship_track import ShipTrack
from track_estimators.utils import geographiclib_distance, geographiclib_heading

# Read the subsampled data
csv_file = "data/modern_ships/WCE5063_subset_downsampled_rounded_noisy.csv"
ship_track = ShipTrack()
ship_track.read_csv(csv_file, lat_col="lat_round", lon_col="lon_round", reverse=False)

# Calculate the sog, cog, and respective rates
# lontmp = np.copy(ship_track.lon)
# lattmp = np.copy(ship_track.lat)

# ship_track.lon = savgol_filter(ship_track.lon, 2, 1)
# ship_track.lat = savgol_filter(ship_track.lat, 2, 1)


ship_track.calculate_cog()
ship_track.calculate_sog()

ship_track.calculate_cog_rate()
ship_track.calculate_sog_rate()

# ship_track.lon = lontmp
# ship_track.lat = lattmp


# Get the measurements
ship_track.get_measurements()


xxx = ship_track.cog

csv_file = "data/modern_ships/WCE5063_subset_downsampled_rounded_noisy.csv"
ship_track = ShipTrack(
    calc_distance_func=geographiclib_distance, calc_heading_func=geographiclib_heading
)
ship_track.read_csv(csv_file, lat_col="lat_round", lon_col="lon_round", reverse=False)

# Calculate the sog, cog, and respective rates
ship_track.calculate_cog()
ship_track.calculate_sog()

ship_track.calculate_cog_rate()
ship_track.calculate_sog_rate()

# Get the measurements
ship_track.get_measurements()

print(xxx)
print(ship_track.cog)


# Read the original data
ship_track_original = ShipTrack("data/modern_ships/WCE5063_subset.csv")


# Process noise covariance
df = pd.DataFrame(columns=["q", "r", "rmse_lat", "rmse_lon", "rmse"])

q_tests = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
r_tests = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]

n = 4

counter = 0
for r in r_tests:
    for q in q_tests:
        Q = np.eye(n) * q
        R = np.eye(n) * r

        R = np.diag([1, 1, 0.0, 0.0]) * 0.25
        Q = np.eye(n) * 0.01

        # -------------------------------------------------------------- #
        #                     Define the KF matrices                     #
        # -------------------------------------------------------------- #
        # Observation matrix
        H = np.diag([1, 1, 0, 0])

        x0 = ship_track.z[:n, 0].reshape(-1, 1)

        # Estimate covariance matrix
        # https://www.researchgate.net/post/Kalman_filter_how_do_I_choose_initial_P_0
        # This does not seem to affect uch the output
        P = np.eye(n) * 1

        # -------------------------------------------------------------- #
        #                     Unscented Kalman Filter                    #
        # -------------------------------------------------------------- #
        # Create and run the unscented kalman filter
        ukf = UnscentedKalmanFilter(
            H=H, Q=Q, R=R, P=P, x0=x0, non_linear_process=geodetic_dynamics
        )

        dt = 4
        predictions, estimate_vars = ukf.run(int(120 / dt), dt, ship_track)
        predictions = np.asarray(predictions)
        estimate_vars = np.asarray(estimate_vars)

        # -------------------------------------------------------------- #
        #                Calculate the performance metrics               #
        # -------------------------------------------------------------- #
        length = 120
        y = np.copy(predictions[:length])
        y_true = ship_track_original.z[:, :length].T
        y_true = y_true.reshape(120, 4, 1)

        # y[:, 0, 0] = savgol_filter(y[:, 0, 0], 12, 1, mode="interp")
        # y[:, 1, 0] = savgol_filter(y[:, 1, 0], 12, 1, mode="interp")

        # rmse_res_lon = rmse(y[:, 0, 0], y_true[:, 0, 0])
        # rmse_res_lat = rmse(y[:, 1, 0], y_true[:, 1, 0])
        # rmse_res = rmse(y[:, :2, 0], y_true[:, :2, 0])

        # cum_abs_diff_res = cum_abs_diff(y, y_true)
        # abs_diff_res = abs_diff(y, y_true)

        # print("CUM ABS DIFF: ", cum_abs_diff_res[-1])
        # print("ABS DIFF: ", abs_diff_res)

        # Append to dataframe
        # df_to_concat = pd.DataFrame({'q': q, 'r': r, 'rmse_lat': rmse_res_lat,
        # 'rmse_lon': rmse_res_lon, 'rmse': rmse_res}, index=[counter])
        # df = pd.concat([df, pd.DataFrame(df_to_concat)], ignore_index=True)
        # print(df)

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
            predictions[:, 0, 0],
            predictions[:, 1, 0],
            label="Kalman Filter Prediction",
            s=ms,
            color="yellow",
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

        # ax.plot(predictions[:, 0, 0], predictions[:, 1, 0],  alpha=alpha)
        # ax.plot(lon, lat, alpha=alpha)
        # ax.set_extent([-40, -20, -40, -10])
        plt.xlim(-85, -70)
        plt.ylim(25, 45)
        plt.legend()
        # plt.savefig(f"benchmark_results/4n_kf_results_q{q}_r{r}.png", bbox_inches="tight", dpi=300)
        # plt.close()
        plt.show()

        # plt.plot(estimate_vars[:, 0], label="dlon")
        # plt.plot(estimate_vars[:, 1], label="dlat")
        # plt.legend()
        # plt.show()

        counter += 1

df.to_csv("benchmark_results/4n_kf_results.csv", index=False)
