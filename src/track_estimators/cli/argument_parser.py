"""Argument parser module."""
import argparse

from ..__init__ import __version__


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Ship track estimator {__version__} command line interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        default="input.json",
        help="Filepath to the input JSON file",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        default="output",
        help="Output file prefix",
    )

    parser.add_argument(
        "-t",
        "--track-file",
        dest="track_file",
        required=True,
        help="Filepath to the ship track data",
    )

    parser.add_argument(
        "-s",
        "--ship-id",
        dest="ship_id",
        required=True,
        help="Ship ID",
    )

    parser.add_argument(
        "-lat",
        "--latitude-id",
        dest="lat_id",
        required=True,
        help="Name of the latitude column",
    )

    parser.add_argument(
        "-lon",
        "--longitude-id",
        dest="lon_id",
        required=True,
        help="Name of the longitude column",
    )

    parser.add_argument(
        "-ic",
        "--id-col",
        dest="id_col",
        required=True,
        help="Name of the ship ID column",
    )

    parser.add_argument(
        "-rts",
        "--rts-smoother",
        dest="apply_rts_smoother",
        action="store_true",
        help="Apply the Rauch-Tung-Striebel (RTS) smoother",
    )

    parser.add_argument(
        "-rev",
        "--reverse",
        dest="reverse",
        action="store_true",
        help="Reverse the trajectory",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    return parser
