from bonner.brainio import Catalog

from .._utils import brainio

from ._download import download_dataset
from ._utils import IDENTIFIER
from ._data_assemblies import create_data_assembly
from ._stimulus_set import create_stimulus_set


def package(
    catalog: Catalog,
    location_type: str,
    location: str,
    force: bool,
) -> None:
    download_dataset(force=force)

    stimulus_set = create_stimulus_set()
    brainio.package_stimulus_set(
        catalog=catalog,
        identifier=IDENTIFIER,
        stimulus_set=stimulus_set,
        location_type=location_type,
        location=location,
        class_csv="",
        class_zip="",
    )

    path = create_data_assembly()
    brainio.package_data_assembly(
        catalog=catalog,
        path=path,
        location_type=location_type,
        location=location,
        class_="",
    )
