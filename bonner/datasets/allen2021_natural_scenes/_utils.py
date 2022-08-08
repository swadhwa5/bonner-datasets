import functools
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .._utils import groupby_reset

IDENTIFIER = "allen2021.natural-scenes"

BUCKET_NAME = "natural-scenes-dataset"
N_SUBJECTS = 8
N_SESSIONS = (40, 40, 32, 30, 40, 32, 40, 30)
N_SESSIONS_HELD_OUT = 3
N_MAX_SESSIONS = 40
N_TRIALS_PER_SESSION = 750
N_STIMULI = 73000
ROIS = {
    "surface": (
        "streams",
        "prf-visualrois",
        "prf-eccrois",
        "floc-places",
        "floc-faces",
        "floc-bodies",
        "floc-words",
        "HCP_MMP1",
        "Kastner2015",
        "nsdgeneral",
        "corticalsulc",
    ),
    "volume": ("MTL", "thalamus"),
}


def format_stimulus_id(idx: int) -> str:
    return f"image{idx:05}"


def load_stimulus_metadata() -> pd.DataFrame:
    """Load and format stimulus metadata.

    :return: stimulus metadata
    """
    metadata = pd.read_csv(
        Path.cwd() / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv",
        sep=",",
    ).rename(columns={"Unnamed: 0": "stimulus_id"})
    metadata["stimulus_id"] = metadata["stimulus_id"].apply(
        lambda idx: format_stimulus_id(idx)
    )
    return metadata


def get_shared_stimulus_ids(assemblies: Iterable[xr.DataArray]) -> list[str]:
    """Gets the IDs of the stimuli shared across all the participants in the experiment.

    :return: shared_stimulus_ids
    """
    return list(
        functools.reduce(
            lambda x, y: x & y,
            [set(assembly["stimulus_id"].values) for assembly in assemblies],
        )
    )


def average_across_reps(assembly: xr.DataArray) -> xr.DataArray:
    """Average NeuroidAssembly across repetitions of conditions.

    :param assembly: neural data
    :return: assembly with data averaged across repetitions along "stimulus_id" coordinate
    """
    groupby = assembly.groupby("stimulus_id")
    assembly = groupby.mean(skipna=True, keep_attrs=True)
    assembly = groupby_reset(assembly, "stimulus_id", "presentation")
    return assembly


def compute_nc(assembly: xr.DataArray) -> np.ndarray:
    """Compute the noise ceiling for a subject's fMRI data using the method described in the NSD Data Manual (https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/6CusMRYfk0) under the "Conversion of ncsnr to noise ceiling percentages" section.

    :param assembly: neural data
    :return: noise ceilings for all voxels
    """
    ncsnr = assembly["ncsnr"].values
    groupby = assembly["stimulus_id"].groupby("stimulus_id")

    counts = np.array([len(reps) for reps in groupby.groups.values()])

    ncsnr_squared = ncsnr**2
    if counts is None:
        fraction = 1
    else:
        unique, counts = np.unique(counts, return_counts=True)
        reps = dict(zip(unique, counts))
        fraction = (reps[1] + reps[2] / 2 + reps[3] / 3) / (reps[1] + reps[2] + reps[3])
    nc = ncsnr_squared / (ncsnr_squared + fraction)
    return nc
