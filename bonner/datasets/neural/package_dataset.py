from importlib import import_module

import os
from pathlib import Path

import click
from bonner.brainio import Catalog


BONNER_DATASETS_CATALOG_IDENTIFIER = os.getenv(
    "BONNER_DATASETS_CATALOG_IDENTIFIER",
)
BONNER_DATASETS_CATALOG_CSV_PATH = Path(os.getenv("BONNER_DATASETS_CATALOG_CSV_PATH"))
BONNER_DATASETS_CATALOG_CACHE_DIRECTORY = Path(
    os.getenv(
        "BONNER_DATASETS_CATALOG_CACHE_DIRECTORY",
        str(Path.home() / ".cache" / "bonner-brainio"),
    )
)
BONNER_DATASETS_CACHE = Path(
    os.getenv("BONNER_DATASETS_CACHE", str(Path.home() / ".cache" / "bonner-datasets"))
)
BONNER_DATASETS_LOCATION_TYPE = os.getenv("BONNER_DATASETS_LOCATION_TYPE")
BONNER_DATASETS_LOCATION = os.getenv("BONNER_DATASETS_LOCATION")


@click.command()
@click.argument("identifier")
@click.option(
    "-n",
    "--catalog-identifier",
    envvar="BONNER_DATASETS_CATALOG_IDENTIFIER",
    show_envvar=True,
    help="identifier of the BrainIO Catalog",
)
@click.option(
    "-c",
    "--catalog-csv-path",
    envvar="BONNER_DATASETS_CATALOG_CSV_PATH",
    show_envvar=True,
    type=click.Path(path_type=Path),
    help="path to the Catalog CSV file",
)
@click.option(
    "-d",
    "--catalog-cache-directory",
    envvar="BONNER_DATASETS_CATALOG_CACHE_DIRECTORY",
    default=Path.home() / ".cache" / "bonner-brainio",
    show_envvar=True,
    type=click.Path(path_type=Path),
    help="path to the Catalog CSV file",
)
@click.option(
    "-D",
    "--cache-directory",
    envvar="BONNER_DATASETS_CACHE",
    default=Path.home() / ".cache" / "bonner-datasets",
    show_default=True,
    show_envvar=True,
    type=click.Path(path_type=Path),
    help="cache directory for downloaded files",
)
@click.option(
    "-t",
    "--location-type",
    envvar="BONNER_DATASETS_LOCATION_TYPE",
    show_envvar=True,
    help="BrainIO 'location_type' of the files to be packaged",
)
@click.option(
    "-l",
    "--location",
    envvar="BONNER_DATASETS_LOCATION",
    show_envvar=True,
    help="BrainIO 'location' of the files to be packaged",
)
@click.option(
    "-f",
    "--force-download",
    is_flag=True,
    default=False,
    show_default=True,
    help="whether to re-download previously cached files",
)
def package_neural_dataset(
    identifier: str,
    catalog_identifier: Path = BONNER_DATASETS_CATALOG_IDENTIFIER,
    catalog_csv_path: Path = BONNER_DATASETS_CATALOG_CSV_PATH,
    catalog_cache_directory: Path = BONNER_DATASETS_CATALOG_CACHE_DIRECTORY,
    cache_directory: Path = BONNER_DATASETS_CACHE,
    location_type: str = BONNER_DATASETS_LOCATION_TYPE,
    location: str = BONNER_DATASETS_LOCATION,
    force_download: bool = False,
) -> None:
    """Package a neural dataset to an existing BrainIO Catalog.
    \f

    :param identifier: identifier of the neural dataset (e.g. bonner2021_object2vec)
    :param catalog_identifier: identifier of the BrainIO Catalog
    :param cache_directory: cache directory for downloaded files
    :param location_type: BrainIO 'location_type' of the files to be packaged
    :param location: BrainIO 'location' of the files to be packaged
    :param force_download: whether to re-download previously cached files
    """
    catalog = Catalog(
        catalog_identifier,
        csv_file=catalog_csv_path,
        cache_directory=catalog_cache_directory,
    )

    module = import_module(f"bonner.datasets.neural.{identifier}")
    package_fn = getattr(module, "package")
    identifier = getattr(module, "IDENTIFIER")

    cache_directory = cache_directory / identifier
    cache_directory.mkdir(parents=True, exist_ok=True)

    os.chdir(cache_directory)
    package_fn(
        catalog=catalog,
        location=location,
        location_type=location_type,
        force_download=force_download,
    )


if __name__ == "__main__":
    package_neural_dataset.callback(
        identifier="allen2021_natural_scenes",
    )
