from importlib import import_module

from pathlib import Path

import click
from bonner.brainio import Catalog
from bonner.datasets.neural._utils import working_directory


@click.command()
@click.argument("identifier")
@click.option(
    "-c",
    "--catalog-identifier",
    envvar="BONNER_DATASETS_CATALOG_IDENTIFIER",
    show_envvar=True,
    help="identifier of the BrainIO Catalog",
)
@click.option(
    "-d",
    "--cache-directory",
    envvar="BONNER_DATASETS_CACHE",
    default=Path.home() / ".cache" / "bonner-datasets",
    show_default=True,
    show_envvar=True,
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
    catalog_identifier: str,
    cache_directory: Path,
    location_type: str,
    location: str,
    force_download: bool,
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
    catalog = Catalog(catalog_identifier)

    module = import_module(f"bonner.datasets.neural.{identifier}")
    package_fn = getattr(module, "package")
    identifier = getattr(module, "IDENTIFIER")

    cache_directory.mkdir(parents=True, exist_ok=True)

    with working_directory(cache_directory):
        package_fn(
            catalog=catalog,
            location=location,
            location_type=location_type,
            force_download=force_download,
        )
