import pandas as pd
from scipy.io import loadmat
import xarray as xr

from ._utils import IDENTIFIER


def create_assembly(mouse: str, date: str) -> xr.Dataset:
    raw = loadmat(f"natimg2800_{mouse}_{date}.mat", simplify_cells=True)
    return xr.Dataset(
        data_vars={
            "stimulus-related activity": (
                ("presentation", "neuroid"),
                raw["stim"]["resp"],
            ),
            "spontaneous activity": (("time", "neuroid"), raw["stim"]["spont"]),
        },
        coords={
            "stimulus_id": (
                "presentation",
                [
                    "blank" if i_image == 2800 else f"image{i_image:04}"
                    for i_image in raw["stim"]["istim"] - 1
                ],
            ),
            "x": ("neuroid", raw["med"][:, 0]),
            "y": ("neuroid", raw["med"][:, 1]),
            "z": ("neuroid", raw["med"][:, 2]),
            "noise_level": (
                "neuroid",
                pd.DataFrame(raw["stat"])["noiseLevel"].values,
            ),
        },
        attrs={
            "identifier": f"{IDENTIFIER}-{mouse}_{date}",
            "stimulus_set_identifier": IDENTIFIER,
        },
    )
