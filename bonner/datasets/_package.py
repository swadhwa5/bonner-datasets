from importlib import import_module
import os
from pathlib import Path

import click
import click_extra
from click_extra import Path as ClickPath
from bonner.brainio import Catalog


@click.command(no_args_is_help=True)
@click.argument("identifier")
@click_extra.config_option(
    type=ClickPath(path_type=Path),
    default=Path.home() / ".config" / "bonner-datasets" / "config.toml",
)
@click.option(
    "-n",
    "--catalog-identifier",
    default="bonner-datasets",
    show_default=True,
    help="identifier of the Catalog",
)
@click.option(
    "-c",
    "--catalog-csv",
    type=ClickPath(path_type=Path),
    default=None,
    show_default=True,
    help="path to the Catalog CSV file",
)
@click.option(
    "-d",
    "--catalog-cache",
    type=ClickPath(path_type=Path),
    default=None,
    show_default=True,
    help="path to the Catalog cache directory",
)
@click.option(
    "-t",
    "--upload-location-type",
    default="local",
    show_default=True,
    help="BrainIO 'location_type' of the file to be packaged",
)
@click.option(
    "-l",
    "--upload-location",
    type=ClickPath(path_type=Path),
    default=Path(
        os.getenv("BONNER_BRAINIO_HOME", str(Path.home() / ".cache" / "bonner-brainio"))
    ),
    show_default=True,
    help="BrainIO 'location' of the files to be packaged",
)
@click.option(
    "-D",
    "--download-cache",
    type=ClickPath(path_type=Path),
    default=Path.home() / ".cache" / "bonner-datasets",
    show_default=True,
    help="cache directory for downloaded files",
)
@click.option(
    "-f",
    "--download-force",
    is_flag=True,
    default=False,
    show_default=True,
    help="whether to re-download previously cached files",
)
def package_dataset(
    identifier: str,
    catalog_identifier: str,
    catalog_csv: Path | None,
    catalog_cache: Path | None,
    upload_location_type: str,
    upload_location: str,
    download_cache: Path,
    download_force: bool,
) -> None:
    """Package a dataset to a BrainIO Catalog.

    \b
    The following datasets are available:
        allen2021_natural_scenes
        bonner2021_object2vec
        chang2019_bold5000
        stringer2019_mouse_10k
    """
    click.echo(f"{catalog_csv}")

    # catalog = Catalog(
    #     catalog_identifier,
    #     csv_file=catalog_csv,
    #     cache_directory=catalog_cache,
    # )

    # module = import_module(f"bonner.datasets.{identifier}")
    # package_fn = getattr(module, "package")
    # identifier = getattr(module, "IDENTIFIER")

    # download_cache = download_cache / identifier
    # download_cache.mkdir(parents=True, exist_ok=True)

    # os.chdir(download_cache)
    # package_fn(
    #     catalog=catalog,
    #     location=upload_location,
    #     location_type=upload_location_type,
    #     force_download=download_force,
    # )
