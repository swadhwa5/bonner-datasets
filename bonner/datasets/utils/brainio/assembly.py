import pandas as pd
import xarray as xr

from bonner.brainio import BONNER_BRAINIO_HOME, fetch, package_assembly


def load(
    *,
    catalog_name: str,
    identifier: str,
    check_integrity: bool = True,
) -> xr.DataArray:
    """Load a BrainIO assembly from a catalog as a DataArray.

    :param catalog_name: name of the BrainIO catalog
    :param identifier: identifier of the assembly, as defined in the BrainIO specification
    :param check_integrity: whether to check the SHA1 hash of the file, defaults to True
    :return: the BrainIO assembly
    """
    filepath = fetch(
        catalog_name=catalog_name,
        identifier=identifier,
        lookup_type="assembly",
        class_="netcdf4",
        check_integrity=check_integrity,
    )
    assembly = xr.open_dataarray(filepath)
    return assembly


def package(
    *,
    assembly: xr.DataArray,
    catalog_name: str,
    location_type: str,
    location: str,
) -> None:
    """Package a DataArray as a BrainIO assembly.

    :param assembly: the DataArray
    :param catalog_name: name of the BrainIO catalog
    :param location_type: location_type of the assembly, as defined in the BrainIO specification
    :param location: location of the assembly, as defined in the BrainIO specification
    """
    identifier = assembly.attrs["identifier"]
    filepath = BONNER_BRAINIO_HOME / catalog_name / f"{identifier}.nc"
    assembly = assembly.to_dataset(name=identifier, promote_attrs=True)
    assembly.to_netcdf(filepath)

    package_assembly(
        filepath=filepath,
        class_="netcdf4",
        catalog_name=catalog_name,
        location_type=location_type,
        location=f"{location}/{filepath.name}",
    )


def merge(assembly: xr.DataArray, stimulus_set: pd.DataFrame) -> xr.DataArray:
    """Merge the metadata columns from a stimulus set into an assembly.

    :param assembly: the BrainIO assembly
    :param stimulus_set: the BrainIO stimulus set
    :return: the updated BrainIO assembly
    """
    assembly = assembly.load()
    stimulus_set = stimulus_set.loc[
        stimulus_set["stimulus_id"].isin(assembly["stimulus_id"].values), :
    ]
    for column in stimulus_set.columns:
        if column == "stimulus_id" or column == "filename":
            continue
        assembly[column] = ("presentation", stimulus_set[column])
    return assembly
