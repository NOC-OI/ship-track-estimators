"""track_estimators command line interface."""
import logging
import os
import sys
from typing import Tuple

import numpy as np

from ..kalman_filters.non_linear_process import geodetic_dynamics
from ..kalman_filters.unscented import UnscentedKalmanFilter
from ..ship_track import ShipTrack
from ..utils import generate_dts, smooth
from .argument_parser import __version__, create_parser
from .json_loader import load_input_json

logger = logging.getLogger(__name__)


def banner():
    """Log the track estimator banner."""
    logger.info(
        r"""
             /|~~~

           ///|

         /////|

       ///////|

     /////////|

   \==========|===/
~~~~~~~~~~~~~~~~~~~~~
""",
        extra={"simple": True},
    )


def start_banner():
    """Log the track estimator start banner."""
    banner()
    logger.info(f"version: {__version__}", extra={"simple": True})


def exit_banner():
    """Log the track estimator exit banner."""
    # banner()
    logger.info(
        "Track estimator has terminated succesfully! :)", extra={"simple": True}
    )


def track_estimator():
    """Run the track estimator."""
    logging.basicConfig(
        format="Track estimator | %(levelname)s | %(asctime)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    start_banner()

    parser = create_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        logger.error(f"Input file '{args.input_file}' does not exist.")
        exit_banner()
        return

    if not os.path.isfile(args.track_file):
        logger.error(f"Track file '{args.track_file}' does not exist.")
        exit_banner()
        return

    logger.info(f"Reading input JSON from '{args.input_file}'...")
    settings = load_input_json(args.input_file)
    dim, dt, nsteps, H, Q, R, P, smooth_control = get_input_settings(settings)

    # -------------------------------------------------------------- #
    #                        Ship Track                              #
    # -------------------------------------------------------------- #
    if not args.reverse:
        reverse = False
    else:
        reverse = True

    ship_track = ShipTrack()
    ship_track.read_csv(
        args.track_file,
        ship_id=args.ship_id,
        id_col=args.id_col,
        lat_col=args.lat_id,
        lon_col=args.lon_id,
        reverse=reverse,
    )

    if smooth_control not in [-1, 0, 1, None]:
        logger.info(f"Smoothing SOG and COG by {smooth_control}.")
        ship_track.calculate_cog()
        ship_track.calculate_sog()
        ship_track.sog = smooth(ship_track.sog, smooth_control)
        ship_track.cog = smooth(ship_track.cog, smooth_control)

    z = ship_track.get_measurements(include_sog=True, include_cog=True)
    ship_track.calculate_cog_rate()
    ship_track.calculate_sog_rate()
    x0 = z[:, 0].reshape(-1, 1).copy()

    # -------------------------------------------------------------- #
    #                        Generate dts                            #
    # -------------------------------------------------------------- #
    if dt in [-1, 0, None]:
        dt_array = generate_dts(ship_track.dts, nsteps)
    else:
        dt_array = generate_dts(ship_track.dts, 1)

    # Number of time steps
    nsteps = len(dt_array)

    # -------------------------------------------------------------- #
    #                     Unscented Kalman Filter                    #
    # -------------------------------------------------------------- #
    logger.info("Running the Unscented Kalman Filter.")
    ukf = UnscentedKalmanFilter(
        H=H, Q=Q, R=R, P=P, x0=x0, non_linear_process=geodetic_dynamics
    )

    predictions, estimate_vars = ukf.run(nsteps, dt_array, ship_track)
    logger.info("Finished running the Unscented Kalman Filter.")

    # -------------------------------------------------------------- #
    #                          RTS Smoother                          #
    # -------------------------------------------------------------- #
    # Run the RTS smoother
    if args.apply_rts_smoother:
        predictions_smoothed, estimate_vars_smoothed = ukf.run_rts_smoother(
            ship_track=ship_track
        )

    # -------------------------------------------------------------- #
    #                          Write outputs                         #
    # -------------------------------------------------------------- #
    logger.info(f"Writing outputs with prefix '{args.output_prefix}'.")
    np.savetxt(
        f"{args.output_prefix}_{args.ship_id}_predictions.txt", np.asarray(predictions)
    )
    np.savetxt(
        f"{args.output_prefix}_{args.ship_id}_variances.txt",
        np.diagonal(np.asarray(estimate_vars), axis1=1, axis2=2),
    )
    np.savetxt(f"{args.output_prefix}_{args.ship_id}_dts.txt", np.asarray(dt_array))
    np.savetxt(
        f"original_{args.ship_id}_track.txt",
        np.array((ship_track.lon, ship_track.lat)).T,
    )

    if args.apply_rts_smoother:
        np.savetxt(
            f"{args.output_prefix}_{args.ship_id}_predictions_smoothed.txt",
            np.asarray(predictions_smoothed),
        )
        np.savetxt(
            f"{args.output_prefix}_{args.ship_id}_variances_smoothed.txt",
            np.diagonal(np.asarray(estimate_vars_smoothed), axis1=1, axis2=2),
        )

    exit_banner()


def get_input_settings(
    settings: dict,
) -> Tuple[int, float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the input settings.

    Parameters
    ----------
    settings
        The input settings.

    Returns
    -------
    dim, dt, nsteps, H, Q, R, P
        dim represents the dimensions of matrices,
        dt is the time step,
        nsteps is the number of estimation steps,
        H is the measurement matrix,
        R is the measurement noise covariance matrix,
        Q the process noise covariance,
        P is the estimate error covariance matrix.
    """
    if "dim" in settings:
        dim = int(settings["dim"])
    else:
        raise KeyError("dim not found in input settings")

    if "dt" in settings:
        dt = settings["dt"]
    else:
        raise KeyError("dt not found in input settings")

    if "nsteps" in settings:
        nsteps = int(settings["nsteps"])
    else:
        raise KeyError("nsteps not found in input settings")

    if "smooth" in settings:
        smooth_control = int(settings["smooth"])
    else:
        smooth_control = None

    H = _get_input_matrix(settings, "H", dim)
    Q = _get_input_matrix(settings, "Q", dim)
    R = _get_input_matrix(settings, "R", dim)
    P = _get_input_matrix(settings, "P", dim)

    return dim, dt, nsteps, H, Q, R, P, smooth_control


def _get_input_matrix(settings: dict, matrix_name: str, dim: int) -> np.ndarray:
    """
    Get the input matrix from the settings dictionary.

    Parameters
    ----------
    settings
        The input settings.
    matrix_name
        The name of the matrix.
    dim
        The dimension of the matrix.

    Returns
    -------
    matrix
        The matrix as a numpy array.
    """
    if matrix_name in settings:
        matrix = np.asarray(settings[matrix_name])

        assert (
            matrix.shape[0] == dim
        ), f"Dimension mismatch: {matrix.shape[0]} != {dim} for {matrix_name}"

        if matrix.ndim == 1:
            matrix = np.diag(matrix)
        elif matrix.ndim == 2:
            assert (
                matrix.shape[1] == dim
            ), f"Dimension mismatch: {matrix.shape[1]} != {dim} for {matrix_name}"
        else:
            raise ValueError("{matrix_name} must be 1 or 2 dimensional")
    else:
        raise KeyError(f"{matrix_name} not found in input settings")

    return matrix
