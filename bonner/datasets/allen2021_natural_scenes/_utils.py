from pathlib import Path
from collections.abc import Iterable, Mapping

import numpy as np
import xarray as xr
from .._utils.xarray import groupby_reset

IDENTIFIER = "allen2021.natural_scenes"
BUCKET_NAME = "natural-scenes-dataset"
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


def open_assembly(filepath: Path, subject: int = None, **kwargs) -> xr.Dataset:
    """Opens a subject's assembly.

    :param filepath: path to the dataset
    :param subject: subject ID
    :param **kwargs: passed on to xr.open_dataset
    :return:
    """
    if subject is None:
        return xr.open_dataset(filepath, group=f"/", **kwargs)
    else:
        return xr.open_dataset(filepath, group=f"/subject-{subject}", **kwargs)


def compute_shared_stimulus_ids(assemblies: Iterable[xr.Dataset]) -> set[str]:
    """Gets the IDs of the stimuli shared across all the participants in the experiment.

    :return: shared_stimulus_ids
    """
    return set.intersection(
        *(set(assembly["stimulus_id"].values) for assembly in assemblies)
    )


def compute_noise_ceiling(assembly: xr.Dataset) -> xr.DataArray:
    """Compute the noise ceiling for a subject's fMRI data using the method described in the NSD Data Manual (https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/6CusMRYfk0) under the "Conversion of ncsnr to noise ceiling percentages" section.

    :param assembly: neural data
    :return: noise ceilings for all voxels
    """
    groupby = assembly["stimulus_id"].groupby("stimulus_id")

    counts = np.array([len(reps) for reps in groupby.groups.values()])

    if counts is None:
        fraction = 1
    else:
        unique, counts = np.unique(counts, return_counts=True)
        reps = dict(zip(unique, counts))
        fraction = (reps[1] + reps[2] / 2 + reps[3] / 3) / (reps[1] + reps[2] + reps[3])

    ncsnr_squared = assembly["ncsnr"].sel(split=np.nan, drop=True) ** 2
    return ncsnr_squared / (ncsnr_squared + fraction)


def average_betas_across_reps(betas: xr.DataArray) -> xr.DataArray:
    return groupby_reset(
        betas.load()
        .groupby("stimulus_id")
        .mean()
        .assign_attrs({"average_across_reps": True}),
        groupby_coord="stimulus_id",
        groupby_dim="presentation",
    ).transpose("neuroid", "presentation")


def filter_betas_by_roi(
    betas: xr.DataArray,
    *,
    masks: xr.DataArray,
    selectors: Iterable[Mapping[str, str]],
    identifier: str = None,
) -> xr.DataArray:
    masks = masks.load().set_index({"roi": ("source", "label", "hemisphere")})
    selections = []
    for selector in selectors:
        selection = masks.sel(selector).values
        if selection.ndim == 1:
            selection = np.expand_dims(selection, axis=0)
        selections.append(selection)
    mask = np.any(np.concatenate(selections, axis=0), axis=0)
    betas = betas.load().isel({"neuroid": mask})
    if identifier:
        betas = betas.assign_attrs({"voxels": identifier})
    return betas


def filter_betas_by_stimulus_id(
    betas: xr.DataArray, *, stimulus_ids: set[str], exclude: bool = False
) -> xr.DataArray:
    selection = np.isin(betas["stimulus_id"].values, list(stimulus_ids))
    if exclude:
        selection = ~selection
    return betas.isel({"presentation": selection})
