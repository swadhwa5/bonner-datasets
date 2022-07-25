import pandas as pd

from ._utils import load_stimulus_metadata


def create_stimulus_set() -> pd.DataFrame:
    stimulus_set = load_stimulus_metadata()
    stimulus_set["filename"] = "images/" + stimulus_set["stimulus_id"] + ".png"
    stimulus_set = stimulus_set.rename(
        columns={column: column.lower() for column in stimulus_set.columns}
    )
    return stimulus_set
