import json


def load_input_json(input_file: str = "input.json") -> dict:
    """
    Load the input JSON file.

    Parameters
    ----------
    input_file
        The input JSON file.

    Returns
    -------
    Dictionary with the input data.
    """
    with open(input_file, "r") as f:
        return json.load(f)
