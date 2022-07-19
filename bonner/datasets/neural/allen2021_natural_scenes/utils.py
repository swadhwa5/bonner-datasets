from typing import Tuple, AbstractSet
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ...utils.brainio.assembly import load as load_assembly_
from ...utils.brainio.stimulus_set import load as load_stimulus_set_
from ...utils import groupby_reset

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
    :rtype: pd.DataFrame
    """
    metadata = pd.read_csv(
        Path.cwd() / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv",
        sep=",",
    ).rename(columns={"Unnamed: 0": "stimulus_id"})
    metadata["stimulus_id"] = metadata["stimulus_id"].apply(
        lambda idx: format_stimulus_id(idx)
    )
    return metadata


def load_assembly(
    subject: int, catalog_name: str = "bonner-brainio", check_integrity: bool = True
) -> xr.DataArray:
    return load_assembly_(
        catalog_name=catalog_name,
        identifier=f"{IDENTIFIER}-subject{subject}",
        check_integrity=check_integrity,
    )


def load_stimulus_set(
    catalog_name: str = "bonner-brainio", check_integrity: bool = True
) -> Tuple[pd.DataFrame, Path]:
    return load_stimulus_set_(
        catalog_name=catalog_name,
        identifier=IDENTIFIER,
        check_integrity=check_integrity,
    )


def get_shared_stimulus_ids() -> AbstractSet[str]:
    """Gets the IDs of the stimuli shared across all the participants in the experiment.

    :return: shared_stimulus_ids
    :rtype: Set
    """
    assemblies = {
        subject: load_assembly(subject=subject, check_integrity=False)
        for subject in range(N_SUBJECTS)
    }
    shared_stimulus_ids = functools.reduce(
        lambda x, y: x & y,
        [
            set(assemblies[subject]["stimulus_id"].values)
            for subject in range(N_SUBJECTS)
        ],
    )
    return shared_stimulus_ids


def average_across_reps(assembly: xr.DataArray) -> xr.DataArray:
    """Average NeuroidAssembly across repetitions of conditions.

    :param assembly: neural data
    :type assembly: xr.DataArray
    :return: assembly with data averaged across repetitions along "stimulus_id" coordinate
    :rtype: xr.DataArray
    """
    groupby = assembly.groupby("stimulus_id")
    assembly = groupby.mean(skipna=True, keep_attrs=True)
    assembly = groupby_reset(assembly, "stimulus_id", "presentation")
    return assembly


def compute_nc(assembly: xr.DataArray) -> np.ndarray:
    """Compute the noise ceiling for a subject's fMRI data using the method described in the NSD Data Manual (https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/6CusMRYfk0) under the "Conversion of ncsnr to noise ceiling percentages" section.

    :param assembly: neural data
    :type assembly: xr.DataArray
    :return: noise ceilings for all voxels
    :rtype: np.ndarray
    """
    ncsnr = assembly["ncsnr"].values
    groupby = assembly.groupby("stimulus_id")

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
