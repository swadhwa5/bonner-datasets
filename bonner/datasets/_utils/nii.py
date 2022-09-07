from pathlib import Path

import xarray as xr
import nibabel as nib


def to_dataarray(
    filepath: Path,
    *,
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
