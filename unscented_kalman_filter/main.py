import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from utils import get_historical_ship_data, z_mean, state_mean, get_cog, get_sog, get_cog_rate, get_sog_rate
from unscented_kalman_filter import hx, fx

dim_x = 4
dim_z = 2

# Get the historical ship data
lon, lat, dts = get_historical_ship_data("historical_ship_data.csv", "01203823")

# Define the vector of measurements (zs)
zs = np.array([np.radians(lon), np.radians(lat)]).T

# Get estimations of the course over ground at each time step
cog = get_cog(np.radians(lon), np.radians(lat))
sog = get_sog(np.radians(lon), np.radians(lat), dts)
sog_rate = get_sog_rate(sog, dts)
cog_rate = get_cog_rate(cog, dts)

# Create sigma points
sigma_points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

# Create the Unscented Kalman Filter instance
ukf = UnscentedKalmanFilter(dim_x=dim_x, 
                            dim_z=dim_z, 
                            dt=dts[0], 
                            fx=fx, hx=hx, 
                            points=sigma_points,
                            x_mean_fn=state_mean,
                            z_mean_fn=z_mean)

# Define the covariance matrices
x_std = np.radians(0.5)
ukf.P = np.diag([x_std**2, x_std**2, 10.**2, np.radians(0.5)**2])
z_std = np.radians(0.5)
ukf.R = np.diag([z_std**2, z_std**2]) 

# Define the initial state
ukf.x = np.array([np.radians(lon[0]), np.radians(lat[0]), 0., 0.])


# For batch processing
# xs, covs = ukf.batch_filter(zs[1:], dts=dts)

# Run the Kalman filter iteratively
xs = []
covs = []
dt_split = 2

for i, z in enumerate(zs[:-1]):
    # Get time step for this iteration
    dt_step = dts[i]

    # Split the time step into smaller intervals
    substep, remainder = np.divmod(dt_step, dt_split)
    dts_substep = np.full(dt_split, substep)
    dts_substep[-1] += remainder

    for dt_substep in dts_substep[:-1]:
        ukf.Q = Q_discrete_white_noise(dim=dim_z, dt=dt_substep, var=0.001**2, block_size=2)
        ukf.predict(dt=dt_substep)
        xs.append(ukf.x)
        covs.append(ukf.P)

    ukf.Q = Q_discrete_white_noise(dim=dim_z, dt=dts_substep[-1], var=0.001**2, block_size=2)
    ukf.predict(dt=dts_substep[-1])
    ukf.update(z)
    xs.append(ukf.x)
    covs.append(ukf.P)
    print(ukf.x, 'log-likelihood', ukf.log_likelihood)


xs = np.asarray(xs)
covs = np.asarray(covs)
xs, covs, K = ukf.rts_smoother(xs, covs)

# -------------------------------------------------------------------------------- #
#                                Plot the results                                  #
# -------------------------------------------------------------------------------- #
# Extract variances from predicted state covariance matrix
std_devs = []
for i in covs:
    variances = np.diag(i)
    # Compute standard deviations
    std_deviations = np.sqrt(variances)
    std_devs.append(std_deviations)

std_devs = np.asarray(std_devs)
plt.plot(std_devs[:, 0], label="Longitude standard deviation")
plt.plot(std_devs[:, 1], label="Latitude standard deviation")
plt.plot(std_devs[:, 2], label="COG standard deviation")
plt.plot(std_devs[:, 3], label="SOG standard deviation")
plt.legend()
plt.show()


ms = 10
alpha = 0.4
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
xs = np.degrees(np.asarray(xs))
ax.scatter(xs[:, 0], xs[:, 1], label='Kalman Filter Prediction', s=5, alpha=alpha)
ax.scatter(lon, lat, label="Measurements", s=5, alpha=alpha)
ax.plot(xs[:, 0], xs[:, 1],  alpha=alpha)
ax.plot(lon, lat, alpha=alpha)
#ax.set_extent([-40, -20, -40, -10])
plt.legend()
plt.show()
