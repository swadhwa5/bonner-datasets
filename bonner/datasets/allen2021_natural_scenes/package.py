# TODO add localizer T-values and other voxel metadata (https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/G5dUBGBxMo)

from argparse import Namespace
import itertools
import os
from pathlib import Path
import shutil
from typing import Dict, Tuple
from unittest.mock import patch
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
import h5py
from PIL import Image
import nibabel as nib
from tqdm import tqdm
import boto3

from bonner.brainio.assembly import upload as package_assembly
from bonner.brainio.stimulus_set import upload as package_stimulus_set
from bonner.datasets import parser, load_nii
from bonner.utils.environments import working_directory

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


def get_assembly_identifier(subject: int) -> str:
    return f"{IDENTIFIER}-subject{subject}"


def download_dataset(force: bool = False) -> None:
    """Download the (1.8 mm)-resolution, GLMsingle preparation of the Natural Scenes Dataset.

    :param force: whether to force downloads even if files exist, defaults to False
    :type force: bool, optional
    """
    files = [
        "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",  # stimulus images
        "nsddata/experiments/nsd/nsd_stim_info_merged.csv",  # stimulus metadata
    ]
    for subject in range(N_SUBJECTS):
        for roi_type in ("surface", "volume"):
            for roi_group in ROIS[roi_type]:
                for hemisphere in ("lh", "rh"):
                    files.append(
                        f"nsddata/ppdata/subj{subject + 1:02}/func1pt8mm/roi/{hemisphere}.{roi_group}.nii.gz"
                    )  # roi masks

                # roi labels
                if roi_type == "surface":
                    files.append(
                        f"nsddata/freesurfer/subj{subject + 1:02}/label/{roi_group}.mgz.ctab"
                    )
                elif roi_type == "volume":
                    files.append(f"nsddata/templates/{roi_group}.ctab")

        # TODO once the later sessions are released, remove the N_SESSIONS_HELD_OUT
        for session in range(N_SESSIONS[subject] - N_SESSIONS_HELD_OUT):
            files.append(
                f"nsddata_betas/ppdata/subj{subject + 1:02}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{session + 1:02}.hdf5"
            )  # betas
        for suffix in ("", "_split1", "_split2"):
            files.append(
                f"nsddata_betas/ppdata/subj{subject + 1:02}/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr{suffix}.nii.gz"
            )  # ncsnr
        files.extend(
            [
                f"nsddata/ppdata/subj{subject + 1:02}/func1pt8mm/T1_to_func1pt8mm.nii.gz",
                f"nsddata/ppdata/subj{subject + 1:02}/func1pt8mm/brainmask.nii.gz",
            ]
        )

    s3 = boto3.client("s3")
    for file in [Path.cwd() / file for file in files]:
        if force or not file.exists():
            file.parent.mkdir(exist_ok=True, parents=True)
            with open(file, "wb") as f:
                s3.download_fileobj(BUCKET_NAME, file, f)


def save_image(args: Tuple[Image.Image, Path]) -> None:
    """Save an image to a filepath.

    :param args: an image and the filepath it should be saved to
    :type args: Tuple[Image.Image, Path]
    """
    image, filepath = args
    if not filepath.exists():
        image.save(filepath)


def format_image_id(idx: int) -> str:
    return f"image{idx:05}"


def save_images() -> Dict[str, Path]:
    """Save HDF5-formatted image stimuli as PNG files.

    :return: dictionary of stimulus paths
    :rtype: Dict[str, Path]
    """
    stimuli = h5py.File(
        Path.cwd() / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5", "r"
    )["imgBrick"]

    images_dir = Path.cwd() / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {
        f"{format_image_id(image)}": images_dir / f"{format_image_id(image)}.png"
        for image in range(N_STIMULI)
    }
    images = (
        Image.fromarray(stimuli[stimulus, :, :, :])
        for stimulus in range(stimuli.shape[0])
    )
    with Pool() as pool:
        list(
            pool.imap(
                save_image,
                tqdm(
                    zip(images, image_paths.values()),
                    total=len(image_paths),
                    desc="images",
                ),
                chunksize=1000,
            ),
        )
    return image_paths


def load_stimulus_metadata() -> pd.DataFrame:
    """Load and format stimulus metadata.

    :return: stimulus metadata
    :rtype: pd.DataFrame
    """
    metadata = pd.read_csv(
        Path.cwd() / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv",
        sep=",",
    ).rename(columns={"Unnamed: 0": "image_id"})
    metadata["image_id"] = metadata["image_id"].apply(lambda idx: format_image_id(idx))
    return metadata


def package_stimuli() -> None:
    """Package image stimuli as a BrainIO StimulusSet."""
    stimulus_set = load_stimulus_metadata()
    #TODO finish packaging
    raise NotImplementedError()
    package_stimulus_set()


def extract_image_ids() -> xr.DataArray:
    """Extract and format image IDs for all trials.

    :return: image_ids seen at each trial with "subject", "session" and "trial" dimensions
    :rtype: xr.DataArray
    """
    metadata = load_stimulus_metadata()
    metadata = np.array(metadata.iloc[:, 17:])
    indices = np.nonzero(metadata)
    trials = metadata[indices[0], indices[1]] - 1  # fix 1-indexing

    _image_ids = [format_image_id(idx) for idx in indices[0]]
    subject_ids = indices[1] // 3  # each subject has 3 columns, 1 for each possible rep
    session_ids = trials // N_TRIALS_PER_SESSION
    intra_session_trial_ids = trials % N_TRIALS_PER_SESSION

    image_ids = xr.DataArray(
        data=np.full(
            (N_SUBJECTS, N_MAX_SESSIONS, N_TRIALS_PER_SESSION), "", dtype="<U10"
        ),
        dims=("subject", "session", "trial"),
    )
    image_ids.values[subject_ids, session_ids, intra_session_trial_ids] = _image_ids
    return image_ids


def load_roi_mapping(
    *,
    subject: int,
    roi_type: str,
    roi_group: str,
    hemisphere: str,
) -> Tuple[np.ndarray, Dict]:
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
    :rtype: Tuple[np.ndarray, Dict]
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


def format_roi_metadata(subject: int) -> pd.DataFrame:
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
            volume, mapping = load_roi_mapping(
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


def load_ncsnr(subject: int, *, split: int = None) -> xr.DataArray:
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


def load_structural_scan(subject: int) -> xr.DataArray:
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


def load_brain_mask(subject: int) -> xr.DataArray:
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


def load_activations(
    *,
    subject: int,
    session: int,
    image_ids: xr.DataArray,
) -> xr.DataArray:
    """Load functional activations.

    :param subject: subject ID
    :type subject: int
    :param session: session ID
    :type session: int
    :param image_ids: image IDs presented during the session
    :type image_ids: xr.DataArray
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
                "image_id": ("presentation", image_ids.astype(str)),
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


def package_assemblies() -> None:
    """Package DataAssemblies."""
    image_ids = extract_image_ids()

    for subject in range(N_SUBJECTS):
        mask = load_brain_mask(subject)
        neuroid_metadata = format_roi_metadata(subject)[mask.values]

        activations = (
            xr.concat(
                [
                    load_activations(
                        subject=subject,
                        session=session,
                        image_ids=image_ids[subject, session, :],
                    ).sel({"neuroid": mask})
                    for session in tqdm(
                        # TODO remove N_SESSIONS_HELD_OUT all data are released
                        range(N_SESSIONS[subject] - N_SESSIONS_HELD_OUT),
                        desc="session",
                    )
                ],
                dim="presentation",
            )
            .rename(get_assembly_identifier(subject))
            .assign_coords(
                {
                    "ncsnr": (
                        "neuroid",
                        load_ncsnr(subject).sel({"neuroid": mask}),
                    ),
                    "ncsnr_split1": (
                        "neuroid",
                        load_ncsnr(subject, split=1).sel({"neuroid": mask}),
                    ),
                    "ncsnr_split2": (
                        "neuroid",
                        load_ncsnr(subject, split=2).sel({"neuroid": mask}),
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
                    "structural_scan": load_structural_scan(subject)
                    .sel({"neuroid": mask})
                    .values,
                }
            )
        )

        activations.to_netcdf(create_assembly_path(get_assembly_identifier(subject)))

        #TODO finish packaging
        raise NotImplementedError()
        package_assembly()


def main(args: Namespace) -> None:
    """Package the Natural Scenes Dataset"""
    dataset_dir = Path(args.brainio_download_cache) / IDENTIFIER
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with working_directory(dataset_dir), patch.dict(
        os.environ,
        {key.upper(): value for key, value in vars(args).items() if type(value) is str},
    ):
        download_dataset(args.force_download)
        package_stimuli()
        package_assemblies()
    if args.clear_cache:
        shutil.rmtree(dataset_dir)


if __name__ == "__main__":
    parser.description = f"package the {IDENTIFIER} dataset"
    parser.add_argument(
        "-a",
        "--aws-shared-credentials-file",
        type=str,
        default="~/.aws/credentials",
        help="path to AWS credentials file",
    )
    args = parser.parse_args()
    main(args)
