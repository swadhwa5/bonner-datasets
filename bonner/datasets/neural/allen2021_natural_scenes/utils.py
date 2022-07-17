from pathlib import Path

import pandas as pd

IDENTIFIER = "allen2021.natural-scenes"

BUCKET_NAME = "natural-scenes-dataset"
N_SUBJECTS = 8
N_SESSIONS = (40, 40, 32, 30, 40, 32, 40, 30)
N_SESSIONS_HELD_OUT = 3
N_MAX_SESSIONS = 40
N_TRIALS_PER_SESSION = 750
N_STIMULI = 73000
ROIS = {
    "surface": (
        "streams",
        "prf-visualrois",
        "prf-eccrois",
        "floc-places",
        "floc-faces",
        "floc-bodies",
        "floc-words",
        "HCP_MMP1",
        "Kastner2015",
        "nsdgeneral",
        "corticalsulc",
    ),
    "volume": ("MTL", "thalamus"),
}


def format_stimulus_id(idx: int) -> str:
    return f"image{idx:05}"


def load_stimulus_metadata() -> pd.DataFrame:
    """Load and format stimulus metadata.

    :return: stimulus metadata
    :rtype: pd.DataFrame
    """
    metadata = pd.read_csv(
        Path.cwd() / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv",
        sep=",",
    ).rename(columns={"Unnamed: 0": "stimulus_id"})
    metadata["stimulus_id"] = metadata["stimulus_id"].apply(
        lambda idx: format_stimulus_id(idx)
    )
    return metadata
