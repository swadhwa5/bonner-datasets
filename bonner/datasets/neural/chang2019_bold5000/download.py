from typing import Mapping
from pathlib import Path
import subprocess

from tqdm import tqdm

from utils.figshare import get_url_dict
from ...utils import download_file, unzip_file
from .utils import (
    FIGSHARE_ARTICLE_ID_V1,
    FIGSHARE_ARTICLE_ID_V2,
    _URL_IMAGES,
    _S3_ROI_MASKS,
    N_SUBJECTS,
    N_SESSIONS,
    _get_brain_mask_filename,
    _get_imagenames_filename,
    _get_betas_filename,
)


def download_dataset(force_download: bool = False, **kwargs: Mapping[str, str]) -> None:
    urls = get_url_dict(FIGSHARE_ARTICLE_ID_V2)

    for subject in tqdm(range(N_SUBJECTS), desc="subject", leave=False):
        filenames = [
            _get_brain_mask_filename(subject),  # brain masks
            _get_imagenames_filename(subject),  # image names
        ]
        for filename in filenames:
            download_file(urls[filename], Path(filename), force=force_download)
        for session in tqdm(range(N_SESSIONS[subject]), desc="session", leave=False):
            filename = _get_betas_filename(subject, session)  # betas
            download_file(urls[filename], Path(filename), force=force_download)

    urls = get_url_dict(FIGSHARE_ARTICLE_ID_V1)
    urls = {
        "BOLD5000_Structural.zip": urls["BOLD5000_Structural.zip"],
        "stimuli.zip": _URL_IMAGES,
    }
    for filename, url in urls.items():
        filepath = download_file(url, filepath=Path(filename), force=force_download)
        unzip_file(filepath, extract_dir=Path.cwd(), remove_zip=False)

    # ROI masks
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            f"{_S3_ROI_MASKS}",
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
        Path(filename_old).replace(Path(filename_new))
