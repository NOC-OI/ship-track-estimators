# Estimators of ship tracks

This Python library implements an Unscented Kalman Filter with geodetic dynamics, as detailed in references [1] and [2].
Additionally, the library includes the implementation of the Unscented Rauch-Tung-Striebel Smoother (URTSS) algorithm, which is discussed in reference [3].

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
track_estimator -i input.json -o trial -t data/historical_ships/historical_ship_data.csv -s 01203792 -ic "primary.id" -lat "lat" -lon "lon"
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
- `-rts` or `-rts-smoother`: Apply the Rauch-Tung-Striebel (RTS) smoother

#### Example of `input.json`

```json
{
  "dim": 4,
  "H": [1, 1, 0, 0],
  "R": [0.25, 0.25, 0, 0],
  "Q": [1e-4, 1e-4, 1e-6, 1e-6],
  "P": [1.0, 1.0, 1.0, 1.0],
  "dt": 1,
  "nsteps": 500
}
```

Here, `dim` represents the dimensions of matrices, `H` is the measurement matrix, `R` is the measurement noise covariance matrix, `Q` the process noise covariance, and `P` is the estimate error covariance matrix. Furthermore, `dt` is the time step (which can be either a constant time step or a list of numbers), and `nsteps` indicates the number of estimation steps to be performed. When `dt` is `-1`, `0` or `null`, the `nsteps` parameter indicates the number of sub-steps that should be performed within each main time step, as determined by the measurements. For example:

```json
{
  "dim": 4,
  "H": [1, 1, 0, 0],
  "R": [0.001, 0.001, 0, 0],
  "Q": [1e-2, 1e-2, 1e-4, 1e-4],
  "P": [1.0, 1.0, 1.0, 1.0],
  "dt": -1,
  "nsteps": 2
}
```

This configuration will result in performing 2 sub-steps within each main step, with the length of each main step determined by the measurement data.

## References

1. [Unscented Kalman filter for long-distance vessel tracking in geodetic coordinates][1]
1. [Unscented Filtering and Nonlinear Estimation][2]
1. [Unscented Rauch-Tung-Striebel Smoother][3]

[1]: https://doi.org/10.1016/j.apor.2022.103205
[2]: https://ieeexplore.ieee.org/document/1271397
[3]: http://ieeexplore.ieee.org/document/4484208/
