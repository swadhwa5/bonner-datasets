from bonner.brainio import Catalog

from data_assembly import create_data_assembly
from .._utils.brainio import package_data_assembly


def package(
    catalog: Catalog,
    location_type: str,
    location: str,
) -> None:
    assembly = create_data_assembly()
    package_data_assembly(
        catalog=catalog,
        assembly=assembly,
        location_type=location_type,
        location=location,
        class_="",
    )
