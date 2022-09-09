from bonner.brainio import Catalog

from ._utils import IDENTIFIER
from ._stimulus_set import save_images, create_stimulus_set
from ._data_assembly import create_data_assembly
from .._utils import brainio


def package_stimulus_set(
    catalog: Catalog,
    location_type: str,
    location: str,
) -> None:
    save_images()
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


def package_data_assembly(
    catalog: Catalog, location_type: str, location: str, **kwargs: str
) -> None:
    filepath = create_data_assembly(**kwargs)
    brainio.package_data_assembly(
        catalog=catalog,
        path=filepath,
        location_type=location_type,
        location=location,
        class_="",
    )
