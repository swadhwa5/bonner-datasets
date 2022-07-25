from pathlib import Path

from .._utils import download_file, unzip_file
from ._utils import N_SUBJECTS, URLS, FILENAMES


def download_dataset(force_download: bool) -> None:
    filepath = download_file(
        URLS["stimuli"],
        filepath=Path(FILENAMES["stimuli"]),
        force=force_download,
    )
    unzip_file(filepath, extract_dir=Path.cwd(), remove_zip=False)

    download_file(
        URLS["conditions"],
        filepath=Path(FILENAMES["conditions"]),
        force=force_download,
    )

    for subject in range(N_SUBJECTS):
        for filetype in ("activations", "noise_ceilings", "rois", "cv_sets"):
            download_file(
                URLS[filetype][subject],
                filepath=Path(FILENAMES[filetype][subject]),
                force=force_download,
            )
        for urls, filenames in zip(
            URLS["contrasts"].values(), FILENAMES["contrasts"].values()
        ):
            download_file(
                urls[subject], filepath=Path(filenames[subject]), force=force_download
            )
