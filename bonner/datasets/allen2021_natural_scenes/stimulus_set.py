from pathlib import Path

from tqdm import tqdm
from PIL import Image
import pandas as pd
import xarray as xr

from .utils import (
    load_stimulus_metadata,
    BUCKET_NAME,
    N_STIMULI,
)
from .._utils import s3


def create_stimulus_set() -> pd.DataFrame:
    stimulus_set = load_stimulus_metadata()
    stimulus_set["filename"] = "images/" + stimulus_set["stimulus_id"] + ".png"
    stimulus_set = stimulus_set.rename(
        columns={column: column.lower() for column in stimulus_set.columns}
    )
    return stimulus_set


def save_images() -> None:
    filepath = Path("nsddata_stimuli") / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    s3.download(filepath, bucket=BUCKET_NAME)

    stimuli = xr.open_dataset(filepath)["imgBrick"]

    images_dir = Path("images")
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [images_dir / f"image{idx:05}.png" for idx in range(N_STIMULI)]
    images = (
        Image.fromarray(stimuli[stimulus, :, :, :].values)
        for stimulus in range(stimuli.shape[0])
    )
    for image, image_path in tqdm(zip(images, image_paths), desc="image", leave=False):
        if not image_path.exists():
            image.save(image_path)
