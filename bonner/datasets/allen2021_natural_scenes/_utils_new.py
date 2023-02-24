from pathlib import Path
from collections.abc import Collection, Mapping

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


def open_subject_assembly(subject: int, *, filepath: Path, **kwargs) -> xr.Dataset:
    """Opens a subject's assembly.

    Args:
        filepath: path to the dataset
        subject: subject ID
        kwargs: passed on to xr.open_dataset

    Returns:
        subject's assembly
    """
    return xr.open_dataset(filepath, group=f"/subject-{subject}", **kwargs)


def compute_shared_stimulus_ids(
    assemblies: Collection[xr.DataArray], *, n_repetitions: int = 1
) -> set[str]:
    """Gets the IDs of the stimuli shared across all the provided assemblies.

    Args:
        assemblies: assemblies for different subjects
        n_repetitions: minimum number of repetitions for the shared stimuli in each subject

    Returns:
        shared stimulus ids
    """
    try:
        return set.intersection(
            *[
                set(
                    assembly["stimulus_id"].values[
                        (assembly["rep_id"] == n_repetitions - 1).values
                    ]
                )
                for assembly in assemblies
            ]
        )
    except:
        return set.intersection(
            *[set(assembly["stimulus_id"].values) for assembly in assemblies]
        )


def compute_noise_ceiling(assembly: xr.Dataset) -> xr.DataArray:
    """Compute the noise ceiling for a subject's fMRI data using the method described in the NSD Data Manual (https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/6CusMRYfk0) under the "Conversion of ncsnr to noise ceiling percentages" section.

    Args:
        assembly: a subject's neural data

    Returns:
        noise ceilings for all voxels
    """
    groupby = assembly["stimulus_id"].groupby("stimulus_id")

    counts = np.array([len(reps) for reps in groupby.groups.values()])

    if counts is None:
        fraction = 1
    else:
        unique, counts = np.unique(counts, return_counts=True)
        reps = dict(zip(unique, counts))
        fraction = (reps[1] + reps[2] / 2 + reps[3] / 3) / (reps[1] + reps[2] + reps[3])

    ncsnr_squared = assembly["ncsnr"] ** 2
    return (ncsnr_squared / (ncsnr_squared + fraction)).rename("noise ceiling")


def z_score_betas_within_sessions(betas: xr.DataArray) -> xr.DataArray:
    def z_score(betas: xr.DataArray) -> xr.DataArray:
        mean = betas.mean("presentation")
        std = betas.std("presentation")
        return (betas - mean) / std

    return (
        betas.load()
        .groupby("session_id")
        .map(func=z_score, shortcut=True)
        .assign_attrs(betas.attrs)
        .rename(betas.name)
    )


def z_score_betas_within_runs(betas: xr.DataArray) -> xr.DataArray:
    # even-numbered trials (i.e. Python indices 1, 3, 5, ...) had 62 trials
    # odd-numbered trials (i.e. Python indices 0, 2, 4, ...) had 63 trials
    n_runs_per_session = 12
    n_sessions = len(np.unique(betas["session_id"]))
    run_id = []
    for i_session in range(n_sessions):
        for i_run in range(n_runs_per_session):
            n_trials = 63 if i_run % 2 == 0 else 62
            run_id.extend([i_run + i_session * n_runs_per_session] * n_trials)
    betas["run_id"] = ("presentation", run_id)

    def z_score(betas: xr.DataArray) -> xr.DataArray:
        mean = betas.mean("presentation")
        std = betas.std("presentation")
        return (betas - mean) / std

    return (
        betas.load()
        .groupby("run_id")
        .map(func=z_score, shortcut=True)
        .assign_attrs(betas.attrs)
        .rename(betas.name)
    )


def remove_invalid_voxels_from_betas(
    betas: xr.DataArray, validity: xr.DataArray
) -> xr.DataArray:
    neuroid_filter = np.all(
        validity.stack({"neuroid": ("x_", "y_", "z_")}, create_index=True)[:-1, :],
        axis=0,
    )
    neuroid_filter = neuroid_filter.isel({"neuroid": neuroid_filter})
    valid_voxels = set(neuroid_filter.indexes["neuroid"].values)
    index = betas.set_index({"neuroid": ("x", "y", "z")}).indexes["neuroid"].values
    betas = betas.isel({"neuroid": [voxel in valid_voxels for voxel in index]})
    return betas


def average_betas_across_reps(betas: xr.DataArray) -> xr.DataArray:
    """Average the provided betas across repetitions of stimuli.

    Args:
        betas: betas

    Returns:
        averaged betas
    """
    return groupby_reset(
        betas.load()
        .groupby("stimulus_id")
        .mean()
        .assign_attrs(betas.attrs)
        .rename(betas.name),
        groupby_coord="stimulus_id",
        groupby_dim="presentation",
    ).transpose("neuroid", "presentation")


def filter_betas_by_roi(
    betas: xr.DataArray,
    *,
    rois: xr.DataArray,
    selectors: Collection[Mapping[str, str]],
) -> xr.DataArray:
    rois = rois.load().set_index({"roi": ("source", "label", "hemisphere")})
    selections = []
    for selector in selectors:
        selection = rois.sel(selector).values
        if selection.ndim == 1:
            selection = np.expand_dims(selection, axis=0)
        selections.append(selection)
    selection = np.any(np.concatenate(selections, axis=0), axis=0)
    betas = betas.load().isel({"neuroid": selection})
    return betas


def filter_betas_by_stimulus_id(
    betas: xr.DataArray, *, stimulus_ids: set[str], exclude: bool = False
) -> xr.DataArray:
    selection = np.isin(betas["stimulus_id"].values, list(stimulus_ids))
    if exclude:
        selection = ~selection
    return betas.isel({"presentation": selection})
