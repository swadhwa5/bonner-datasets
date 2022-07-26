from contextlib import contextmanager
import os
from pathlib import Path
import zipfile
import tarfile
import requests
import uuid

import nibabel as nib
import xarray as xr


@contextmanager
def working_directory(directory: Path):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)


def download_file(
    url: str,
    filepath: Path = None,
    stream: bool = True,
    allow_redirects: bool = True,
    chunk_size: int = 1024**2,
    force: bool = True,
) -> Path:
    if filepath is None:
        filepath = Path("/tmp") / f"{uuid.uuid4()}"
    elif filepath.exists():
        if not force:
            return filepath
        else:
            filepath.unlink()
    r = requests.Session().get(url, stream=stream, allow_redirects=allow_redirects)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    return filepath


def untar_file(
    filepath: Path, extract_dir: Path = None, remove_tar: bool = True
) -> Path:
    if extract_dir is None:
        extract_dir = Path("/tmp")
    with tarfile.open(filepath) as tar:
        tar.extractall(path=extract_dir)
    if remove_tar:
        filepath.unlink()
    return extract_dir


def unzip_file(
    filepath: Path, extract_dir: Path = None, remove_zip: bool = True
) -> Path:
    if extract_dir is None:
        extract_dir = Path("/tmp")
    with zipfile.ZipFile(filepath, "r") as f:
        f.extractall(extract_dir)
    if remove_zip:
        filepath.unlink()
    return extract_dir


def load_nii(filepath: Path, non_spatial_dim: int = -1) -> xr.DataArray:
    """Format an NII file as a DataArray.

    :param filepath: path to NII file [must be a 3D array (x, y, z) or 4D array (presentation, x, y, z)]
    :param non_spatial_dim: index of presentation dimension: if -1, assumed to be last dimension, else first
    :return: linearized brain volume with a "neuroid" dimension
    """
    nii = nib.load(filepath).get_fdata()

    if nii.ndim == 3:
        dims = ["x", "y", "z"]
        brain_dimensions = nii.shape
    elif nii.ndim == 4:
        if non_spatial_dim == -1:
            dims = ["x", "y", "z", "presentation"]
            brain_dimensions = nii.shape[:-1]
        else:
            dims = ["presentation", "x", "y", "z"]
            brain_dimensions = nii.shape[1:]

    return (
        xr.DataArray(
            data=nii,
            dims=dims,
        )
        .stack({"neuroid": ("x", "y", "z")})
        .reset_index("neuroid")
        .assign_attrs({"brain_dimensions": brain_dimensions})
    )


def groupby_reset(
    x: xr.DataArray, dim_groupby: str, dim_original_name: str
) -> xr.DataArray:
    return (
        x.reset_index(list(x.indexes))
        .rename({dim_groupby: dim_original_name})
        .assign_coords({dim_groupby: (dim_original_name, x[dim_groupby].values)})
        .drop(dim_original_name)
    )
