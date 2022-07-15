from pathlib import Path
import json
import requests
import subprocess

from tqdm import tqdm

from ..utils import download_file, unzip_file
from .utils import _FIGSHARE_API_BASE_URL, _FIGSHARE_BOLD5000_V1_ARTICLE_ID, _FIGSHARE_BOLD5000_V2_ARTICLE_ID, _URL_IMAGES, _S3_ROI_MASKS, N_SUBJECTS, N_SESSIONS, _get_brain_mask_filename, _get_imagenames_filename, _get_betas_filename


def download_dataset(force_download: bool = False, **kwargs) -> None:
    # get URLs for all files in BOLD5000 Release 2 using figshare API (https://docs.figshare.com/#article_files)
    files = json.loads(
        requests.get(
            f"{_FIGSHARE_API_BASE_URL}/articles/{_FIGSHARE_BOLD5000_V2_ARTICLE_ID}/files"
        ).content
    )
    urls = {file["name"]: file["download_url"] for file in files}

    for subject in tqdm(range(N_SUBJECTS), desc="subject"):
        filenames = [
            _get_brain_mask_filename(subject),  # brain masks
            _get_imagenames_filename(subject),  # image names
        ]
        for filename in filenames:
            download_file(urls[filename], Path(filename), force=force_download)
        for session in tqdm(range(N_SESSIONS[subject]), desc="session"):
            filename = _get_betas_filename(subject, session)  # betas
            download_file(urls[filename], Path(filename), force=force_download)

    files = json.loads(
        requests.get(
            f"{_FIGSHARE_API_BASE_URL}/articles/{_FIGSHARE_BOLD5000_V1_ARTICLE_ID}/files"
        ).content
    )
    urls = {file["name"]: file["download_url"] for file in files}
    urls = (
        urls["BOLD5000_Structural.zip"],  # anatomical scans
        _URL_IMAGES,  # stimulus images
    )
    for url in urls:
        filepath = download_file(url, force=force_download)
        unzip_file(filepath)

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
