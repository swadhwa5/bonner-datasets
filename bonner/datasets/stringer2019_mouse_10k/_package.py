from bonner.brainio import Catalog

from .._brainio import package_data_assembly, package_stimulus_set

from ._download import download_dataset, save_images
from ._utils import IDENTIFIER, SESSIONS
from ._data_assemblies import create_assembly
from ._stimulus_set import create_stimulus_set


def package(
    catalog: Catalog,
    location_type: str,
    location: str,
    force_download: bool,
) -> None:
    download_dataset(force_download=force_download)
    paths = save_images()

    stimulus_set = create_stimulus_set(paths=paths)
    package_stimulus_set(
        catalog=catalog,
        identifier=IDENTIFIER,
        stimulus_set=stimulus_set,
        location_type=location_type,
        location=location,
        class_csv="",
        class_zip="",
    )

    for session in SESSIONS:
        assembly = create_assembly(mouse=session["mouse"], date=session["date"])
        package_data_assembly(
            catalog=catalog,
            assembly=assembly,
            location_type=location_type,
            location=location,
            class_="",
        )
