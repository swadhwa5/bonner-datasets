from typing import Mapping
from pathlib import Path

import pandas as pd

from ...utils.brainio.stimulus_set import package
from .utils import IDENTIFIER


def package_stimulus_set(
    catalog_name: str,
    location_type: str,
    location: str,
    **kwargs: Mapping[str, str],
) -> None:
    parent_dir = Path("BOLD5000_Stimuli") / "Scene_Stimuli" / "Presented_Stimuli"
    image_paths = list(parent_dir.rglob("*.*"))
    stimulus_set = pd.DataFrame.from_dict(
        {
            "stimulus_id": [path.stem for path in image_paths],
            "dataset": [path.parent.name for path in image_paths],
            "filename": [
                str(path.relative_to(parent_dir))
                for path in image_paths
            ],
        }
    )
    package(
        identifier=IDENTIFIER,
        stimulus_set=stimulus_set,
        stimulus_dir=parent_dir,
        catalog_name=catalog_name,
        location_type=location_type,
        location=location,
    )
