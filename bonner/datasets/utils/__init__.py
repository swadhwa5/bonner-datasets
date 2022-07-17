from typing import Generator, Iterable, Callable
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import zipfile
import tarfile
import requests
import uuid

import nibabel as nib
import xarray as xr

DATASETS_HOME = Path(os.getenv("DATASETS_HOME", str(Path.home() / "brainio")))


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
    :type filepath: Path
    :param non_spatial_dim: index of presentation dimension: if -1, assumed to be last dimension, else first
    :type non_spatial_dim: int
    :return: linearized brain volume with a "neuroid" dimension
    :rtype: xr.DataArray
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


def package(identifier: str, pipeline: Iterable[Callable]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--catalog-name",
        type=str,
        default="bonner-brainio",
        help="BrainIO catalog name",
    )
    parser.add_argument(
        "-t",
        "--location-type",
        type=str,
        default="rsync",
        help="rsync or S3",
    )
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default="cogsci-ml.win.ad.jhu.edu:/export/data2/shared/brainio/bonner-brainio",
        help="URL to remote directory",
    )
    parser.add_argument(
        "-c",
        "--clear-cache",
        type=bool,
        default=False,
        help="whether to delete cached files on exit",
    )
    parser.add_argument(
        "-f",
        "--force-download",
        type=bool,
        default=False,
        help="whether to re-download existing cached files",
    )
    parser.description = f"package the {identifier} dataset"
    args = parser.parse_args()

    dir_cache = DATASETS_HOME / identifier
    dir_cache.mkdir(parents=True, exist_ok=True)

    with working_directory(dir_cache):
        for pipe in pipeline:
            pipe(**vars(args))

    if args.clear_cache:
        shutil.rmtree(dir_cache)
