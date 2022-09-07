from typing import Iterable
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

from bonner.datasets._utils import load_nii, download_from_s3
from bonner.datasets.allen2021_natural_scenes._utils import (
    IDENTIFIER,
    RESOLUTION,
    PREPROCESSING,
    BUCKET_NAME,
    N_SESSIONS,
    N_SESSIONS_HELD_OUT,
    N_TRIALS_PER_SESSION,
    ROIS,
    BIBTEX,
    load_stimulus_metadata,
)


def extract_stimulus_ids(subject: int) -> xr.DataArray:
    """Extract and format image IDs for all trials.

    :return: stimulus_ids seen at each trial with "subject", "session" and "trial" dimensions
    """
    metadata = load_stimulus_metadata()
    metadata = np.array(
        metadata.loc[
            :, [f"subject{subject + 1}_" in column for column in metadata.columns]
        ]
    )
    assert metadata.shape[-1] == 3
    indices = np.nonzero(metadata)
    trials = metadata[indices] - 1  # fix 1-indexing

    stimulus_ids_ = [f"image{idx:05}" for idx in indices[0]]
    session_ids = trials // N_TRIALS_PER_SESSION
    intra_session_trial_ids = trials % N_TRIALS_PER_SESSION

    stimulus_ids = xr.DataArray(
        data=np.full((N_SESSIONS[subject], N_TRIALS_PER_SESSION), "", dtype="<U10"),
        dims=("session", "trial"),
    )
    stimulus_ids.values[session_ids, intra_session_trial_ids] = stimulus_ids_
    return stimulus_ids.assign_coords(
        {dim: (dim, np.arange(stimulus_ids.sizes[dim])) for dim in stimulus_ids.dims}
    )


def load_brain_mask(*, subject: int, resolution: str) -> xr.DataArray:
    """Load and format a Boolean brain mask for the functional data.

    :param subject: subject ID
    :param resolution: "1pt8mm" or "1mm"
    :return: Boolean brain mask
    """
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / f"func{resolution}"
        / "brainmask.nii.gz"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME)
    return load_nii(Path(filepath)).astype(bool)


def load_betas(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
) -> xr.DataArray:
    """Load betas.

    :param subject: subject ID
    :param resolution: "1pt8mm" or "1mm"
    :param preprocessing: "fithrf_GLMdenoise_RR", "fithrf", or "assumehrf"
    :return: betas
    """
    betas = []
    stimulus_ids = extract_stimulus_ids(subject)
    # TODO remove N_SESSIONS_HELD_OUT
    sessions = np.arange(N_SESSIONS[subject] - N_SESSIONS_HELD_OUT)
    for session in tqdm(sessions, desc="session"):
        filepath = (
            Path("nsddata_betas")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"betas_{preprocessing}"
            / f"betas_session{session + 1:02}.hdf5"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME)
        betas_session = (
            xr.open_dataset(filepath)["betas"]
            .rename(
                {
                    "phony_dim_0": "presentation",
                    "phony_dim_1": "z",
                    "phony_dim_2": "y",
                    "phony_dim_3": "x",
                }
            )
            .transpose("presentation", "x", "y", "z")
            .load()
        )
        betas.append(
            betas_session.assign_coords(
                {
                    "stimulus_id": (
                        "presentation",
                        stimulus_ids.sel(session=session).data,
                    ),
                    "session": (
                        "presentation",
                        session
                        * np.ones(betas_session.sizes["presentation"], dtype=np.uint32),
                    ),
                    "trial": (
                        "presentation",
                        np.arange(betas_session.sizes["presentation"]),
                    ),
                }
            )
        )
    return xr.concat(betas, dim="presentation")


def load_ncsnr(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
) -> xr.DataArray:
    """Load and format noise-ceiling signal-to-noise ratios (NCSNR).

    :param subject: subject ID
    :param resolution: "1pt8mm" or "1mm"
    :param preprocessing: "fithrf_GLMdenoise_RR", "fithrf", or "assumehrf
    :return: noise-ceiling SNRs
    """
    ncsnr = []
    for split in (None, 1, 2):
        if split is None:
            suffix = ""
        else:
            suffix = f"_split{split}"
        filepath = (
            Path("nsddata_betas")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"betas_{preprocessing}"
            / f"ncsnr{suffix}.nii.gz"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME)
        ncsnr.append(load_nii(filepath).expand_dims(split=[split]))
    return xr.concat(ncsnr, dim="split")


def load_structural_scans(
    *,
    subject: int,
    resolution: str,
) -> xr.DataArray:
    """Load and format the structural scans registered to the functional data.

    :param subject: subject ID
    :param resolution: "1pt8mm" or "1mm"
    :return: structural scans
    """
    scans = []
    for scan in ("T1", "T2", "SWI", "TOF"):
        filepath = (
            Path("nsddata")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"{scan}_to_func{resolution}.nii.gz"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME)
        scans.append(load_nii(filepath).expand_dims(scan=[scan]))
    return xr.concat(scans, dim="scan")


def load_rois(
    *,
    subject: int,
    rois: dict[str, Iterable[str]],
    resolution: str,
) -> xr.DataArray:
    """Load the ROI masks for a subject.

    :param subject: subject ID
    :param rois: dict with keys "surface" and "volume", each mapping to an Iterable of ROIs
    :param resolution: "1pt8mm" or "1mm"
    :return: ROI masks
    """
    roi_masks = []
    for roi_type, roi_groups in rois.items():
        for roi_group in roi_groups:
            if roi_type == "surface":
                filepath = (
                    Path("nsddata")
                    / "freesurfer"
                    / f"subj{subject + 1:02}"
                    / "label"
                    / f"{roi_group}.mgz.ctab"
                )
            elif roi_type == "volume":
                filepath = Path("nsddata") / "templates" / f"{roi_group}.ctab"
            download_from_s3(filepath, bucket=BUCKET_NAME)

            mapping = (
                pd.read_csv(
                    filepath,
                    delim_whitespace=True,
                    names=("label", "roi"),
                )
                .set_index("roi")
                .to_dict()["label"]
            )

            volumes = {}
            for hemisphere in ("lh", "rh"):
                filepath = (
                    Path("nsddata")
                    / "ppdata"
                    / f"subj{subject + 1:02}"
                    / f"func{resolution}"
                    / "roi"
                    / f"{hemisphere}.{roi_group}.nii.gz"
                )
                download_from_s3(filepath, bucket=BUCKET_NAME)
                volumes[hemisphere] = load_nii(filepath)

            for roi, label in mapping.items():
                if label != 0:
                    roi_masks.append(
                        ((volumes["lh"] == label) & (volumes["rh"] == label))
                        .expand_dims(roi=[roi])
                        .assign_coords(
                            {
                                "group": ("roi", [roi_group]),
                                "type": ("roi", [roi_type]),
                            },
                        )
                    )
    return xr.concat(roi_masks, dim="roi")


def load_prf_data(subject: int, resolution: str) -> xr.DataArray:
    """Load population receptive field mapping data.

    :param subject: subject ID
    :param resolution: "1pt8mm" or "1mm"
    :return: pRF data
    """
    prf_data = []
    for variable in (
        "angle",
        "eccentricity",
        "exponent",
        "gain",
        "R2",
        "size",
    ):
        filepath = (
            Path("nsddata")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"prf_{variable}.nii.gz"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME)
        prf_data.append(load_nii(filepath).expand_dims(variable=[variable]))
    return xr.concat(prf_data, dim="variable")


def load_functional_contrasts(subject: int, resolution: str) -> xr.DataArray:
    """Load functional contrasts.

    :param subject: subject ID
    :param resolution: "1pt8mm" or "1mm"
    :return: functional contrasts
    """
    categories = []
    for filename in ("domains", "categories"):
        filepath = Path("nsddata") / "experiments" / "floc" / f"{filename}.tsv"
        download_from_s3(filepath, bucket=BUCKET_NAME)

        categories += list(pd.read_csv(filepath, sep="\t").iloc[:, 0].values)

    floc_data = {}
    for category in categories:
        floc_data[category] = []
        for metric in ("tval", "anglemetric"):
            filepath = (
                Path("nsddata")
                / "ppdata"
                / f"subj{subject + 1:02}"
                / f"func{resolution}"
                / f"floc_{category}{metric}.nii.gz"
            )
            download_from_s3(filepath, bucket=BUCKET_NAME)

            floc_data[category].append(
                load_nii(filepath).expand_dims(
                    {
                        "category": [category],
                        "metric": [metric],
                    }
                )
            )
        floc_data[category] = xr.concat(floc_data[category], dim="metric")

    return xr.concat(list(floc_data.values()), dim="category")


def create_data_assembly(
    subject: int,
    resolution: str = RESOLUTION,
    preprocessing: str = PREPROCESSING,
) -> xr.Dataset:
    """Create a data assembly.

    :param subject: subject ID
    :param resolution: "1pt8mm" or "1mm", defaults to "1pt8mm"
    :param preprocessing: "fithrf_GLMdenoise_RR", "fithrf", or "assumehrf, defaults to "fithrf_GLMdenoise_RR"
    :return: data assembly
    """
    assembly = xr.Dataset(
        data_vars={
            "betas": load_betas(
                subject=subject,
                resolution=resolution,
                preprocessing=preprocessing,
            ),
            "brain_mask": load_brain_mask(subject=subject, resolution=resolution),
            "rois": load_rois(subject=subject, rois=ROIS, resolution=resolution),
            "ncsnr": load_ncsnr(
                subject=subject,
                resolution=resolution,
                preprocessing=preprocessing,
            ),
            "structural_scans": load_structural_scans(
                subject=subject, resolution=resolution
            ),
            "prf": load_prf_data(subject=subject, resolution=resolution),
            "contrasts": load_functional_contrasts(
                subject=subject, resolution=resolution
            ),
        },
        attrs={
            "identifier": f"{IDENTIFIER}-subject{subject}",
            "stimulus_set_identifier": IDENTIFIER,
            "subject": subject,
            "preprocessing": preprocessing,
            "resolution": resolution,
            "reference": BIBTEX,
        },
    )
    return assembly
