import functools
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .._utils import groupby_reset, download_from_s3

IDENTIFIER = "allen2021.natural_scenes"
RESOLUTION = "1pt8mm"
PREPROCESSING = "fithrf_GLMdenoise_RR"
BUCKET_NAME = "natural-scenes-dataset"
N_SUBJECTS = 8
N_STIMULI = 73000
N_SESSIONS = (40, 40, 32, 30, 40, 32, 40, 30)
N_SESSIONS_HELD_OUT = 3
N_TRIALS_PER_SESSION = 750
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
BIBTEX = """
@article{Allen2021,
    doi = {10.1038/s41593-021-00962-x},
    url = {https://doi.org/10.1038/s41593-021-00962-x},
    year = {2021},
    month = dec,
    publisher = {Springer Science and Business Media {LLC}},
    volume = {25},
    number = {1},
    pages = {116--126},
    author = {Emily J. Allen and Ghislain St-Yves and Yihan Wu and Jesse L. Breedlove and Jacob S. Prince and Logan T. Dowdle and Matthias Nau and Brad Caron and Franco Pestilli and Ian Charest and J. Benjamin Hutchinson and Thomas Naselaris and Kendrick Kay},
    title = {A massive 7T {fMRI} dataset to bridge cognitive neuroscience and artificial intelligence},
    journal = {Nature Neuroscience}
}
"""


def load_stimulus_metadata() -> pd.DataFrame:
    """Load and format stimulus metadata.

    :return: stimulus metadata
    """
    filepath = Path("nsddata") / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    download_from_s3(filepath, bucket=BUCKET_NAME)
    metadata = pd.read_csv(filepath, sep=",").rename(
        columns={"Unnamed: 0": "stimulus_id"}
    )
    metadata["stimulus_id"] = metadata["stimulus_id"].apply(
        lambda idx: f"image{idx:05}"
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
