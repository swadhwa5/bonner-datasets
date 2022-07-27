from bonner.brainio import Catalog

from ._download import download_dataset, save_images
from .utils import IDENTIFIER


def package(
    catalog: Catalog,
    location_type: str,
    location: str,
    force_download: bool,
) -> None:
    download_dataset(force_download=force_download)
    save_images()
    
    