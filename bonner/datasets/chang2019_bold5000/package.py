from argparse import Namespace
import os
from pathlib import Path
import shutil
from typing import List
import subprocess
import json
from unittest.mock import patch

from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
import xarray as xr

from bonner.brainio.stimulus_set import upload as package_stimulus_set
from bonner.brainio.assembly import upload as package_assembly
from bonner.utils.files import download_file, unzip_file
from bonner.utils.environments import working_directory
from bonner.datasets import parser, load_nii, BRAINIO_CATALOG_NAME, BRAINIO_LOCATION_TYPE, BRAINIO_LOCATION, BRAINIO_DOWNLOAD_CACHE

IDENTIFIER = "chang2019.bold5000"
FIGSHARE_API_BASE_URL = "https://api.figshare.com/v2"
FIGSHARE_BOLD5000_V2_ARTICLE_ID = 14456124
FIGSHARE_BOLD5000_V1_ARTICLE_ID = 6459449
URL_IMAGES = "https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1"
S3_ROI_MASKS = "s3://openneuro.org/ds001499/derivatives/spm"

N_SUBJECTS = 4
N_SESSIONS = (15, 15, 15, 9)
ROIS = (
    "EarlyVis",
    "LOC",
    "OPA",
    "PPA",
    "RSC",
)


def get_betas_filename(subject: int, session: int) -> str:
    return f"CSI{subject + 1}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-{session + 1:02}.nii.gz"


def get_brain_mask_filename(subject: int) -> str:
    return f"CSI{subject + 1}_brainmask.nii.gz"


def get_imagenames_filename(subject: int) -> str:
    return f"CSI{subject + 1}_imgnames.txt"


def load_image_filename_stems(subject: int) -> List:
    with open(Path.cwd() / get_imagenames_filename(subject), "r") as f:
        # strip newlines, extension
        return [Path(line[:-1]).stem for line in f.readlines()]


def load_brain_mask(subject: int) -> xr.DataArray:
    return load_nii(Path.cwd() / get_brain_mask_filename(subject)).astype(bool)


def load_activations(subject: int, session: int) -> xr.DataArray:
    return load_nii(Path.cwd() / get_betas_filename(subject, session)).astype(
        np.float32
    )


def load_roi_mask(subject: int, hemisphere: str, roi: str) -> xr.DataArray:
    return load_nii(
        Path.cwd()
        / f"sub-CSI{subject + 1}/sub-CSI{subject + 1}_mask-{hemisphere}{roi}.nii.gz"
    ).astype(bool)


def load_structural_scan(subject: int) -> xr.DataArray:
    return load_nii(
        Path.cwd()
        / f"BOLD5000_Structural/CSI{subject + 1}_Structural/T1w_MPRAGE_CSI{subject + 1}.nii"
    )


def download_dataset(force: bool = False) -> None:
    # get URLs for all files in BOLD5000 Release 2 using figshare API (https://docs.figshare.com/#article_files)
    files = json.loads(
        requests.get(
            f"{FIGSHARE_API_BASE_URL}/articles/{FIGSHARE_BOLD5000_V2_ARTICLE_ID}/files"
        ).content
    )
    urls = {file["name"]: file["download_url"] for file in files}

    for subject in tqdm(range(N_SUBJECTS), desc="subject"):
        filenames = [
            get_brain_mask_filename(subject),  # brain masks
            get_imagenames_filename(subject),  # image names
        ]
        for filename in filenames:
            download_file(urls[filename], Path.cwd() / filename, force=force)
        for session in tqdm(range(N_SESSIONS[subject]), desc="subject"):
            filename = get_betas_filename(subject, session)  # betas
            download_file(urls[filename], Path.cwd() / filename, force=force)

    files = json.loads(
        requests.get(
            f"{FIGSHARE_API_BASE_URL}/articles/{FIGSHARE_BOLD5000_V1_ARTICLE_ID}/files"
        ).content
    )
    urls = {file["name"]: file["download_url"] for file in files}
    urls = (
        urls["BOLD5000_Structural.zip"],  # anatomical scans
        URL_IMAGES,  # stimulus images
    )
    for url in urls:
        filepath = download_file(url, force=force)
        unzip_file(filepath)

    # ROI masks
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            f"{S3_ROI_MASKS}",
            f"{Path.cwd()}",
        ]
    )
    # rename some inconsistently named files
    remapping = {
        "sub-CSI2/sub-CSI2_mask-LHLO.nii.gz": "sub-CSI2/sub-CSI2_mask-LHLOC.nii.gz",
        "sub-CSI2/sub-CSI2_mask-RHLO.nii.gz": "sub-CSI2/sub-CSI2_mask-RHLOC.nii.gz",
        "sub-CSI2/sub-CSI2_mask-RHRRSC.nii.gz": "sub-CSI2/sub-CSI2_mask-RHRSC.nii.gz",
        "sub-CSI3/sub-CSI3_mask-LHLO.nii.gz": "sub-CSI3/sub-CSI3_mask-LHLOC.nii.gz",
        "sub-CSI3/sub-CSI3_mask-RHLO.nii.gz": "sub-CSI3/sub-CSI3_mask-RHLOC.nii.gz",
    }
    for filename_old, filename_new in remapping.items():
        (Path.cwd() / filename_old).replace(Path.cwd() / filename_new)


def package_stimuli() -> None:
    parent_dir = Path.cwd() / "BOLD5000_Stimuli" / "Scene_Stimuli" / "Presented_Stimuli"
    image_paths = list(parent_dir.rglob("*.*"))
    stimulus_set = pd.DataFrame.from_dict(
        {
            "image_id": [path.stem for path in image_paths],
            "dataset": [path.parent.name for path in image_paths],
            "filename": [
                str(path.relative_to(parent_dir).with_suffix(""))
                for path in image_paths
            ],
        }
    )

    raise NotImplementedError()
    # TODO finish packaging
    package_stimulus_set()


def load_neuroid_metadata(subject: int) -> xr.DataArray:
    brain_mask = load_brain_mask(subject)
    n_voxels = np.sum(brain_mask.values)
    metadata = xr.DataArray(np.full(n_voxels, np.nan), dims="neuroid").assign_coords(
        {"hemisphere": ("neuroid", [""] * n_voxels)}
    )

    for roi in ROIS:
        metadata = metadata.assign_coords(
            {f"roi_{roi}": ("neuroid", [False] * n_voxels)}
        )
        for hemisphere in ("LH", "RH"):
            roi_mask = load_roi_mask(subject, hemisphere, roi).sel(
                {"neuroid": brain_mask}
            )
            metadata[f"roi_{roi}"][roi_mask] = True
            metadata["hemisphere"][roi_mask] = hemisphere
    return metadata


def package_assemblies() -> None:
    for subject in tqdm(range(N_SUBJECTS), desc="subject"):
        mask = load_brain_mask(subject)
        structural_scan = load_structural_scan(subject)
        neuroid_metadata = load_neuroid_metadata(subject)

        assembly = (
            xr.concat(
                [
                    load_activations(subject, session).sel({"neuroid": mask})
                    for session in range(N_SESSIONS[subject])
                ],
                dim="presentation",
            )
            .astype(np.float32)
            .rename(f"{IDENTIFIER}-subject{subject}")
            .assign_coords(
                {
                    "image_id": (
                        "presentation",
                        load_image_filename_stems(subject),
                    ),
                }
            )
            .assign_coords(
                {coord: neuroid_metadata[coord] for coord in neuroid_metadata.coords}
            )
            .assign_attrs(
                {
                    "brain_dimensions": mask.attrs["brain_dimensions"],
                    "structural_scan": structural_scan.values,
                    "structural_scan_brain_dimensions": structural_scan.attrs[
                        "brain_dimensions"
                    ],
                }
            )
            .dropna(dim="neuroid", how="any")
        )

        # TODO finish packaging assembly
        raise NotImplementedError()
        package_assembly()


def main(args: Namespace) -> None:
    """Package the BOLD5000 Dataset"""
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
    args = parser.parse_args()
    main(args)
