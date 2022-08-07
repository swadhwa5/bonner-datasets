from importlib import import_module
from typing import Any
import tomli as tomllib

import os
from pathlib import Path

from click_extra import extra_command, argument, option, config_option
from click_extra import Path as ClickPath
from bonner.brainio import Catalog


@extra_command()
@argument("identifier")
@config_option(default=Path.home() / ".config" / "bonner-datasets" / "config.toml")
@option(
    "-n",
    "--catalog-identifier",
    default="bonner-datasets",
    show_default=True,
    help="identifier of the Catalog",
)
@option(
    "-c",
    "--catalog-csv",
    type=ClickPath(path_type=Path),
    default=None,
    show_default=True,
    help="path to the Catalog CSV file",
)
@option(
    "-d",
    "--catalog-cache",
    type=ClickPath(path_type=Path),
    default=None,
    show_default=True,
    help="path to the Catalog cache directory",
)
@option(
    "-t",
    "--upload-location-type",
    default="local",
    show_default=True,
    help="BrainIO 'location_type' of the file to be packaged",
)
@option(
    "-l",
    "--upload-location",
    type=ClickPath(path_type=Path),
    default=Path(
        os.getenv("BONNER_BRAINIO_HOME", str(Path.home() / ".cache" / "bonner-brainio"))
    ),
    show_default=True,
    help="BrainIO 'location' of the files to be packaged",
)
@option(
    "-D",
    "--download-cache",
    type=ClickPath(path_type=Path),
    default=Path.home() / ".cache" / "bonner-datasets",
    show_default=True,
    help="cache directory for downloaded files",
)
@option(
    "-f",
    "--download-force",
    is_flag=True,
    default=False,
    show_default=True,
    help="whether to re-download previously cached files",
)
def package_dataset(
    identifier: str,
    config: Path,
    catalog_identifier: str,
    catalog_csv: Path | None,
    catalog_cache: Path | None,
    upload_location_type: str,
    upload_location: str,
    download_cache: Path,
    download_force: bool,
) -> None:
    """Package a neural dataset to an existing BrainIO Catalog."""

    catalog = Catalog(
        catalog_identifier,
        csv_file=catalog_csv,
        cache_directory=catalog_cache,
    )

    module = import_module(f"bonner.datasets.{identifier}")
    package_fn = getattr(module, "package")
    identifier = getattr(module, "IDENTIFIER")

    download_cache = download_cache / identifier
    download_cache.mkdir(parents=True, exist_ok=True)

    os.chdir(download_cache)
    package_fn(
        catalog=catalog,
        location=upload_location,
        location_type=upload_location_type,
        force_download=download_force,
    )
