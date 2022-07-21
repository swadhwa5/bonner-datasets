import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import nibabel as nib
from tqdm import tqdm

from ...utils import load_nii
from ...utils.brainio.assembly import package
from .utils import (
    IDENTIFIER,
    N_SUBJECTS,
    ROIS,
    N_SESSIONS,
    N_SESSIONS_HELD_OUT,
    N_MAX_SESSIONS,
    N_TRIALS_PER_SESSION,
    format_stimulus_id,
    load_stimulus_metadata,
)


def package_assemblies(
    catalog_name: str, location_type: str, location: str, **kwargs: str
) -> None:
    stimulus_ids = _extract_stimulus_ids()

    for subject in range(N_SUBJECTS):
        mask = _load_brain_mask(subject)
        neuroid_metadata = _format_roi_metadata(subject)[mask.values]

        assembly = (
            xr.concat(
                [
                    _load_activations(
                        subject=subject,
                        session=session,
                        stimulus_ids=stimulus_ids[subject, session, :],
                    ).sel({"neuroid": mask})
                    for session in tqdm(
                        # TODO remove N_SESSIONS_HELD_OUT all data are released
                        range(N_SESSIONS[subject] - N_SESSIONS_HELD_OUT),
                        desc="session",
                    )
                ],
                dim="presentation",
            )
            .rename(f"{IDENTIFIER}-subject{subject}")
            .assign_coords(
                {
                    "ncsnr": (
                        "neuroid",
                        _load_ncsnr(subject).sel({"neuroid": mask}).data,
                    ),
                    "ncsnr_split1": (
                        "neuroid",
                        _load_ncsnr(subject, split=1).sel({"neuroid": mask}).data,
                    ),
                    "ncsnr_split2": (
                        "neuroid",
                        _load_ncsnr(subject, split=2).sel({"neuroid": mask}).data,
                    ),
                }
            )
            .assign_coords(
                {
                    coord: ("neuroid", series.values)
                    for coord, series in neuroid_metadata.iteritems()
                }
            )
            .assign_attrs(
                {
                    "resolution": "1.8 mm",
                    "preprocessing": "GLMsingle",
                    "brain_dimensions": mask.attrs["brain_dimensions"],
                    "structural_scan": _load_structural_scan(subject)
                    .sel({"neuroid": mask})
                    .data,
                    "identifier": f"{IDENTIFIER}-subject{subject}",
                    "stimulus_set_identifier": IDENTIFIER,
                }
            )
        )

        package(
            assembly=assembly,
            catalog_name=catalog_name,
            location_type=location_type,
            location=location,
        )


def _extract_stimulus_ids() -> xr.DataArray:
    """Extract and format image IDs for all trials.

    :return: stimulus_ids seen at each trial with "subject", "session" and "trial" dimensions
    :rtype: xr.DataArray
    """
    metadata = load_stimulus_metadata()
    metadata = np.array(metadata.iloc[:, 17:])
    indices = np.nonzero(metadata)
    trials = metadata[indices[0], indices[1]] - 1  # fix 1-indexing

    _stimulus_ids = [format_stimulus_id(idx) for idx in indices[0]]
    subject_ids = indices[1] // 3  # each subject has 3 columns, 1 for each possible rep
    session_ids = trials // N_TRIALS_PER_SESSION
    intra_session_trial_ids = trials % N_TRIALS_PER_SESSION

    stimulus_ids = xr.DataArray(
        data=np.full(
            (N_SUBJECTS, N_MAX_SESSIONS, N_TRIALS_PER_SESSION), "", dtype="<U10"
        ),
        dims=("subject", "session", "trial"),
    )
    stimulus_ids.values[
        subject_ids, session_ids, intra_session_trial_ids
    ] = _stimulus_ids
    return stimulus_ids


def _load_roi_mapping(
    *,
    subject: int,
    roi_type: str,
    roi_group: str,
    hemisphere: str,
) -> tuple[np.ndarray, dict[int, str]]:
    """Load a brain volume containing ROI integer labels and a mapping to string labels.

    :param subject: subject ID
    :type subject: int
    :param roi_type: type of ROI, can be "surface" or "volume"
    :type roi_type: str
    :param roi_group: ROI group label, listed in `ROIS.values()`
    :type roi_group: str
    :param hemisphere: "lh" or "rh"
    :type hemisphere: str
    :return: integer-labelled brain volume with mapping to ROI names
    :rtype: tuple[np.ndarray, dict[int, str]]
    """
    volume = nib.load(
        Path.cwd()
        / "nsddata"
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "func1pt8mm"
        / "roi"
        / f"{hemisphere}.{roi_group}.nii.gz"
    ).get_fdata()

    if roi_type == "surface":
        filepath = (
            Path.cwd()
            / "nsddata"
            / "freesurfer"
            / f"subj{subject + 1:02}"
            / "label"
            / f"{roi_group}.mgz.ctab"
        )
    elif roi_type == "volume":
        filepath = Path.cwd() / "nsddata" / "templates" / f"{roi_group}.ctab"
    mapping = (
        pd.read_csv(
            filepath,
            delim_whitespace=True,
            names=("label", "roi"),
        )
        .set_index("roi")
        .to_dict()["label"]
    )

    return volume, mapping


def _format_roi_metadata(subject: int) -> pd.DataFrame:
    """Collate and format the metadata for all ROIs.

    :param subject: subject ID
    :type subject: int
    :return: metadata for all ROIs
    :rtype: pd.DataFrame
    """
    voxels = []
    roi_indices = {}
    for roi_type, roi_groups in ROIS.items():
        for roi_group, hemisphere in itertools.product(roi_groups, ("lh", "rh")):
            volume, mapping = _load_roi_mapping(
                subject=subject,
                roi_type=roi_type,
                roi_group=roi_group,
                hemisphere=hemisphere,
            )
            for roi, label in mapping.items():
                if label != 0:
                    x, y, z = np.nonzero(volume == label)
                    roi_indices[f"roi_{roi_group}_{roi}_{hemisphere}"] = np.arange(
                        volume.size
                    ).reshape(volume.shape)[x, y, z]

    metadata = (
        pd.DataFrame(np.arange(volume.size), columns=["voxel"])
        .set_index("voxel")
        .assign(
            **{coord[:-3]: np.full(volume.size, False) for coord in roi_indices.keys()}
        )
        .assign(**{"hemisphere": np.full(volume.size, np.nan)})
    )

    voxels = np.unique(np.concatenate(list(roi_indices.values())))
    for coord, voxels in roi_indices.items():
        metadata.loc[voxels, coord[:-3]] = True
        metadata.loc[voxels, "hemisphere"] = coord[-2:]

    return metadata


def _load_ncsnr(subject: int, *, split: int = None) -> xr.DataArray:
    """Load and format noise-ceiling signal-to-noise ratios (NCSNR).

    :param subject: subject ID
    :type subject: int
    :param split: the ncsnr split used, can be `1`, `2`, or `None` (all), defaults to None
    :type split: int, optional
    :return: linearized NCSNR data
    :rtype: xr.DataArray
    """
    if split is None:
        suffix = ""
    else:
        suffix = f"_split{split}"

    return load_nii(
        Path.cwd()
        / "nsddata_betas"
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "func1pt8mm"
        / "betas_fithrf_GLMdenoise_RR"
        / f"ncsnr{suffix}.nii.gz"
    )


def _load_structural_scan(subject: int) -> xr.DataArray:
    """Load and format the structural scan registered to the functional data.

    :param subject: subject ID
    :type subject: int
    :return: linearized structural scan
    :rtype: xr.DataArray
    """
    return load_nii(
        Path.cwd()
        / "nsddata"
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "func1pt8mm"
        / "T1_to_func1pt8mm.nii.gz"
    )


def _load_brain_mask(subject: int) -> xr.DataArray:
    """Load and format a Boolean brain mask for the functional data.

    :param subject: subject ID
    :type subject: int
    :return: linearized Boolean brain mask
    :rtype: xr.DataArray
    """
    return load_nii(
        Path.cwd()
        / "nsddata"
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "func1pt8mm"
        / "brainmask.nii.gz"
    ).astype(bool)


def _load_activations(
    *,
    subject: int,
    session: int,
    stimulus_ids: xr.DataArray,
) -> xr.DataArray:
    """Load functional activations.

    :param subject: subject ID
    :type subject: int
    :param session: session ID
    :type session: int
    :param stimulus_ids: image IDs presented during the session
    :type stimulus_ids: xr.DataArray
    :return: functional activations with "presentation" and "neuroid" dimensions
    :rtype: xr.DataArray
    """
    activations = h5py.File(
        Path.cwd()
        / "nsddata_betas"
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "func1pt8mm"
        / "betas_fithrf_GLMdenoise_RR"
        / f"betas_session{session + 1:02}.hdf5",
        "r",
    )["betas"]
    return (
        xr.DataArray(
            data=np.array(activations, dtype=np.int16),
            dims=("presentation", "z", "y", "x"),
            coords={
                "stimulus_id": ("presentation", stimulus_ids.data),
                "session": (
                    "presentation",
                    session * np.ones(activations.shape[0], dtype=np.int64),
                ),
                "trial": ("presentation", np.arange(activations.shape[0])),
            },
        )
        .transpose("presentation", "x", "y", "z")  # HDF5 file has shape (N, Z, Y, X)
        .stack({"neuroid": ("x", "y", "z")})
        .reset_index("neuroid")
    )
