from pathlib import Path
import pandas as pd


def create_stimulus_set(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stimulus_id": [path.stem for path in paths],
            "filename": paths,
        }
    )
