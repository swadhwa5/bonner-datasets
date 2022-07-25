from pathlib import Path
import shutil
import zipfile

import pandas as pd
import xarray as xr

from bonner.brainio import Catalog


def load_data_assembly(
    catalog: Catalog,
    identifier: str,
    use_cached: bool = True,
    check_integrity: bool = True,
    validate: bool = True,
) -> xr.DataArray:
    path = catalog.load_data_assembly(
        identifier=identifier,
        use_cached=use_cached,
        check_integrity=check_integrity,
        validate=validate,
    )
    return xr.open_dataarray(path)


def package_data_assembly(
    catalog: Catalog,
    assembly: xr.DataArray,
    location_type: str,
    location: str,
    class_: str,
) -> None:
    identifier = assembly.attrs["identifier"]
    path = catalog.cache_directory / f"{identifier}.nc"

    assembly = assembly.to_dataset(name=identifier, promote_attrs=True)
    assembly.to_netcdf(path)

    catalog.package_data_assembly(
        path=path,
        location_type=location_type,
        location=f"{location}/{path.name}",
        class_=class_,
    )


def load_stimulus_set(
    catalog: Catalog,
    identifier: str,
    use_cached: bool = True,
    check_integrity: bool = True,
    validate: bool = True,
) -> tuple[pd.DataFrame, Path]:
    paths = catalog.load_stimulus_set(
        identifier=identifier,
        use_cached=use_cached,
        check_integrity=check_integrity,
        validate=validate,
    )

    csv = pd.read_csv(paths["csv"])

    path_cache = catalog.cache_directory / identifier

    if not all([(path_cache / subpath).exists() for subpath in csv["filename"]]):
        if path_cache.exists():
            shutil.rmtree(path_cache)
        path_cache.mkdir(parents=True)
        with zipfile.ZipFile(paths["zip"], "r") as f:
            f.extractall(path_cache)

    return csv, path_cache


def package_stimulus_set(
    catalog: Catalog,
    identifier: str,
    stimulus_set: pd.DataFrame,
    stimulus_dir: Path,
    location_type: str,
    location: str,
    class_csv: str,
    class_zip: str,
) -> None:
    path_csv = catalog.cache_directory / f"{identifier}.csv"
    stimulus_set.to_csv(path_csv, index=False)

    path_zip = catalog.cache_directory / f"{identifier}.zip"
    with zipfile.ZipFile(path_zip, "w") as zip:
        for filename in stimulus_set["filename"]:
            zip.write(filename, arcname=filename)

    catalog.package_stimulus_set(
        identifier=identifier,
        path_csv=path_csv,
        path_zip=path_zip,
        location_type=location_type,
        location_csv=f"{location}/{path_csv.name}",
        location_zip=f"{location}/{path_zip.name}",
        class_csv=class_csv,
        class_zip=class_zip,
    )
