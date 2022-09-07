from pathlib import Path
import zipfile
import tarfile
import requests
import uuid

import nibabel as nib
import boto3
import xarray as xr


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


def download_from_s3(filepath: Path, *, bucket: str, use_cached: bool = True) -> None:
    """Download file(s) from S3.

    :param filepath: path of file in S3
    :param bucket: S3 bucket name
    :param use_cached: use existing file or re-download, defaults to True
    """
    s3 = boto3.client("s3")
    if (not use_cached) or (not filepath.exists()):
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "wb") as f:
            s3.download_fileobj(bucket, str(filepath), f)


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


def load_nii(
    filepath: Path,
    non_spatial_dim: int = -1,
    non_spatial_dim_label: str = "presentation",
) -> xr.DataArray:
    """Format an NII file as a DataArray.

    :param filepath: path to NII file [must be a 3D array (x, y, z) or 4D array e.g. (presentation, x, y, z)]
    :param non_spatial_dim: index of non-spatial dimension: if -1, assumed to be last dimension, else first
    :param non_spatial_dim_label: label to use for non-spatial dimension
    :return: linearized brain volume with a "neuroid" dimension
    """
    nii = nib.load(filepath).get_fdata()

    if nii.ndim == 3:
        dims = ["x", "y", "z"]
    elif nii.ndim == 4:
        if non_spatial_dim == -1:
            dims = ["x", "y", "z", non_spatial_dim_label]
        else:
            dims = [non_spatial_dim_label, "x", "y", "z"]

    return xr.DataArray(
        data=nii,
        dims=dims,
    )


def groupby_reset(
    x: xr.DataArray, dim_groupby: str, dim_original_name: str
) -> xr.DataArray:
    return (
        x.reset_index(list(x.indexes))
        .rename({dim_groupby: dim_original_name})
        .assign_coords({dim_groupby: (dim_original_name, x[dim_groupby].values)})
        .drop_vars(dim_original_name)
    )
