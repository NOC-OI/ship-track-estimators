# Load the reticulate package to interface with Python
library(reticulate)

# Use the correct conda environment
use_condaenv("shiptrack-estimators", required=TRUE)

# Use reticulate to import necessary Python libraries
np <- import("numpy")
geodetic_dynamics_module <- import("track_estimators.kalman_filters.non_linear_process")
unscented_kalman_filter_module  <- import("track_estimators.kalman_filters.unscented")
ship_track_module  <- import("track_estimators.ship_track")
utils_module  <- import("track_estimators.utils")

# -------------------------------------------------------------- #
#                                                                #
#                             INPUT                              #
#                                                                #
# -------------------------------------------------------------- #

# Create a ShipTrack instance
ship_track  <- ship_track_module$ShipTrack()
ship_track$read_csv(
  csv_file="/home/joaomorado/git_repos/shiptrackers_repositores/clean/data/historical_ships/historical_ship_data.csv",
  ship_id="01203792",
  id_col="primary.id",
  lat_col="lat",
  lon_col="lon2",
  reverse=FALSE
)

z <- ship_track$get_measurements(include_sog=TRUE, include_cog=TRUE)

# Calculate cog and sog rates
ship_track$calculate_cog_rate()
ship_track$calculate_sog_rate()

# Observation matrix
H <- np$diag(c(1, 1, 0, 0))

# Measurement Uncertainty
R <- np$diag(c(0.25, 0.25, 0, 0))

# Process noise covariance
Q <- np$diag(c(1e-4, 1e-4, 1e-6, 1e-6))

# Estimate covariance matrix
P <- np$diag(c(1.0, 1.0, 1.0, 1.0))

# -------------------------------------------------------------- #
#                                                                #
#                   UNSCENTED KALMAN FILTER                      #
#                                                                #
# -------------------------------------------------------------- #
# Initial state
x0 <- z[, 1]

# Create the Unscented Kalman Filter instance
geodetic_dynamics <- geodetic_dynamics_module$geodetic_dynamics

ukf <- unscented_kalman_filter_module$UnscentedKalmanFilter(
  H=H, Q=Q, R=R, P=P, x0=x0, non_linear_process=geodetic_dynamics
)

# Generate dt array
dt_array <- utils_module$generate_dts(ship_track$dts, as.integer(1))

# Run the UKF
result <- ukf$run(nsteps=length(dt_array), dt=dt_array, ship_track=ship_track)
predictions <- result[[1]]
estimate_vars <- result[[2]]

# -------------------------------------------------------------- #
#                                                                #
#                         RTS SMOOTHER                           #
#                                                                #
# -------------------------------------------------------------- #
# Run the RTS smoother
result_smoothed <- ukf$run_rts_smoother(ship_track=ship_track)
predictions_smoothed <- result_smoothed[[1]]
estimate_vars_smoothed <- result_smoothed[[2]]
