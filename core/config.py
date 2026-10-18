"""Portable desktop configuration, independent of the sensor and ML runtime."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def asset_path(variable, default=None):
    value = os.environ.get(variable) or default
    if not value:
        raise ValueError(f"Set {variable} to the required local asset path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    expected = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
    if sorted(config["SKELETON_USABLE_JOINTS_NUMBER"].values()) != expected:
        raise ValueError("The bundled model requires the documented 17 Kinect joints.")
    for edge in config["SKELETON_LINES"]:
        if len(edge) != 2 or any(not isinstance(i, int) or not 0 <= i < 17 for i in edge):
            raise ValueError("SKELETON_LINES must contain pairs of indices from 0 to 16.")
    return config
