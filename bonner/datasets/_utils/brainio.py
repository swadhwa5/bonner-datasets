from pathlib import Path
import shutil
import zipfile

from loguru import logger
import pandas as pd
import xarray as xr

from bonner.brainio import Catalog


def package_data_assembly(
    catalog: Catalog,
    path: Path,
    location_type: str,
    location: str,
    class_: str,
) -> None:
    identifier = xr.open_dataset(path, group="/").attrs["identifier"]
    path_in_catalog = catalog.cache_directory / f"{identifier}.nc"
    path.replace(path_in_catalog)

    catalog.package_data_assembly(
        path=path_in_catalog,
        location_type=location_type,
        location=f"{location}/{path.name}",
        class_=class_,
    )


def package_stimulus_set(
    catalog: Catalog,
    identifier: str,
    stimulus_set: pd.DataFrame,
    location_type: str,
    location: str,
    class_csv: str,
    class_zip: str,
) -> None:
    path_csv = catalog.cache_directory / f"{identifier}.csv"
    logger.debug(f"Writing stimulus set {identifier} CSV file to {path_csv}")
    stimulus_set.to_csv(path_csv, index=False)

    path_zip = catalog.cache_directory / f"{identifier}.zip"
    logger.debug(f"Zipping stimulus set {identifier} stimuli to {path_zip}")
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
        logger.debug(f"The stimulus set {identifier} at {path_cache} is incomplete")
        if path_cache.exists():
            logger.debug(
                f"Deleting the existing stimulus set {identifier} at {path_cache}"
            )
            shutil.rmtree(path_cache)

        path_cache.mkdir(parents=True)
        logger.debug(
            f"Extracting the stimulus set {identifier} from {paths['zip']} to"
            f" {path_cache}"
        )
        with zipfile.ZipFile(paths["zip"], "r") as f:
            f.extractall(path_cache)

    return csv, path_cache
