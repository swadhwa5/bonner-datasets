from logging.config import IDENTIFIER
from pathlib import Path

import xarray as xr

from bonner.datasets._utils.s3 import download

IDENTIFIER = "majaj2015_monkey"
BIBTEX = """
@article{Majaj2015,
  doi = {10.1523/jneurosci.5181-14.2015},
  url = {https://doi.org/10.1523/jneurosci.5181-14.2015},
  year = {2015},
  month = sep,
  publisher = {Society for Neuroscience},
  volume = {35},
  number = {39},
  pages = {13402--13418},
  author = {N. J. Majaj and H. Hong and E. A. Solomon and J. J. DiCarlo},
  title = {Simple Learned Weighted Sums of Inferior Temporal Neuronal Firing Rates Accurately Predict Human Core Object Recognition Performance},
  journal = {Journal of Neuroscience}
}
"""


def create_data_assembly() -> xr.Dataset:
    filepath = Path("majaj2015_monkey.nc")
    download(
        "assy_dicarlo_MajajHong2015_public.nc",
        bucket="brainio.dicarlo",
        filepath_local=filepath,
    )
    assembly = xr.open_dataarray(filepath)
    return (
        assembly.rename(
            {
                "image_id": "stimulus_id",
                "arr": "array",
                "col": "column",
            }
        )
        .assign_attrs(
            {
                "identifier": IDENTIFIER,
                "stimulus_set_identifier": IDENTIFIER,
                "time_bin": (
                    assembly["time_bin_start"].values[0],
                    assembly["time_bin_end"].values[0],
                ),
                "reference": BIBTEX,
            }
        )
        .drop_vars(["time_bin_start", "time_bin_end", "neuroid_id"])
        .isel(time_bin=0, drop=True)
        .transpose("presentation", "neuroid")
    )
