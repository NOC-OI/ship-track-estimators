# Estimators of ship tracks

Python library that implements an Unscented Kalman Filter with geodetic dynamics (see refs. [1] and [2]).

## Conda environment

To utilise this package, it should be installed within a dedicated Conda environment. You can create this environment using the following command:

```bash
conda env create -n shiptrack-estimators -f environment.yml python=3.10
```

To activate the conda environment use:

```bash
conda activate shiptrack-estimators
```

## Install the package

To install this library, execute the following command within the source directory:

```bash
pip install -e .
```

## Examples

## Python Scripts

Numerous examples of Python scripts explaining how to use this package can be found in the examples directory.

### Command line interface (CLI)

The CLI provided by this package allows you to execute the Unscented Kalman Filter; however, it offers less flexibility compared to using the Python scripts. To run the track estimator in the terminal, type, e.g., the following command:

```bash
track_estimator -i matrices.json -o trial -t data/historical_ships/historical_ship_data.csv -s 01203823 -ic "primary.id" -lat "lat" -lon "lon"
```

#### Flags description

- `-i` or `--input`: Filepath to the input JSON file containing the Unscented Kalman Filter configurations (by default `input.json`)
- `-o` or `--output`: Output file prefix (by default `output`)
- `-t` or `--track-file`: Filepath to the ship track data (mandatory)
- `-s` or `--ship-id`: ID of the ship (mandatory)
- `-lat` or `--latitude-id`: Name of the latitude column (mandatory)
- `-lon` or `--longitude-id`: Name of the longitude column (mandatory)
- `-lat` or `--latitude-id`: Name of the latitude column (mandatory)
- `-ic` or `--id-col`: Name of the ship ID column (mandatory)

#### Example of `matrices.json`

```json
{
  "dim": 4,
  "H": [1, 1, 0, 0],
  "R": [0.01, 0.01, 0, 0],
  "Q": [1e-4, 1e-4, 1e-4, 1e-4],
  "P": [1.0, 1.0, 1.0, 1.0],
  "dt": 1,
  "nsteps": 500
}
```

Here, `dim` represents the dimensions of matrices, `H` is the measurement matrix, `R` is the measurement noise covariance matrix, `Q` the process noise covariance, and `P` is the estimate error covariance matrix. Furthermore, `dt` is the time step (which can be either a constant time step or a list of numbers), and `nsteps` indicates the number of estimation steps to be performed.

## References

1. [Unscented Kalman filter for long-distance vessel tracking in geodetic coordinates][1]
1. [Unscented Filtering and Nonlinear Estimation][2]

[1]: https://doi.org/10.1016/j.apor.2022.103205
[2]: https://ieeexplore.ieee.org/document/1271397
