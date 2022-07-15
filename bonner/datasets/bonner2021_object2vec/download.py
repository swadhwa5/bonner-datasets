from pathlib import Path

from ..utils import download_file, unzip_file
from .utils import N_SUBJECTS, _URLS, _FILENAMES


def download_dataset(force_download: bool, **kwargs) -> None:
    filepath = download_file(
        _URLS["stimuli"], filepath=Path(_FILENAMES["stimuli"]), force=force_download,
    )
    unzip_file(filepath, extract_dir=Path.cwd(), remove_zip=False)

    download_file(_URLS["conditions"], filepath=Path(_FILENAMES["conditions"]), force=force_download)

    for subject in range(N_SUBJECTS):
        for filetype in ("activations", "noise_ceilings", "rois", "cv_sets"):
            download_file(
                _URLS[filetype][subject],
                filepath=Path(_FILENAMES[filetype][subject]),
                force=force_download,
            )
        for urls, filenames in zip(
            _URLS["contrasts"].values(), _FILENAMES["contrasts"].values()
        ):
            download_file(urls[subject], filepath=Path(filenames[subject]), force=force_download)
