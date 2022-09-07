from bonner.brainio import Catalog

from ._utils import IDENTIFIER, N_SUBJECTS
from ._stimulus_set import save_images
from ._stimulus_set import create_stimulus_set
from ._data_assemblies import create_data_assembly

from .._brainio import package_data_assembly, package_stimulus_set


def package(
    catalog: Catalog,
    location_type: str,
    location: str,
) -> None:
    # save_images()
    # stimulus_set = create_stimulus_set()

    # package_stimulus_set(
    #     catalog=catalog,
    #     identifier=IDENTIFIER,
    #     stimulus_set=stimulus_set,
    #     location_type=location_type,
    #     location=location,
    #     class_csv="",
    #     class_zip="",
    # )

    for subject in range(N_SUBJECTS):
        assembly = create_data_assembly(subject)
        package_data_assembly(
            catalog=catalog,
            assembly=assembly,
            location_type=location_type,
            location=location,
            class_="",
        )
