from pathlib import Path

from ...utils.brainio.stimulus_set import package
from .utils import IDENTIFIER, load_stimulus_metadata


def package_stimulus_set(
    catalog_name: str, location_type: str, location: str, **kwargs: str
) -> None:
    stimulus_set = load_stimulus_metadata()
    stimulus_set["filename"] = stimulus_set["stimulus_id"] + ".png"
    stimulus_set = stimulus_set.rename(
        columns={column: column.lower() for column in stimulus_set.columns}
    )
    package(
        identifier=IDENTIFIER,
        stimulus_set=stimulus_set,
        stimulus_dir=Path("images"),
        catalog_name=catalog_name,
        location_type=location_type,
        location=location,
    )
