# Estimators of ship tracks

Python library that implements an Unscented Kalman Filter with geodetic dynamics (see refs. [1] and [2]).

## Conda environment

To utilize this package, it should be installed within a dedicated Conda environment. You can create this environment using the following command:

```
conda env create -n shiptrack-estimators -f environment.yml python=3.10
```

To activate the conda environment use:

```
conda activate shiptrack-estimators
```

## Install the package

To install this library, execute the following command within the source directory:

```
pip install -e .
```

## Examples

TODO

## References

1. [Unscented Kalman filter for long-distance vessel tracking in geodetic coordinates][1]
1. [Unscented Filtering and Nonlinear Estimation][2]

[1]: https://doi.org/10.1016/j.apor.2022.103205
[2]: https://ieeexplore.ieee.org/document/1271397
