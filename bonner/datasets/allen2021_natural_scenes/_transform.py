"""
Adapted from https://github.com/cvnlab/nsdcode
"""

from collections.abc import Collection
from pathlib import Path

from tqdm import tqdm
import numpy as np
from scipy.ndimage import map_coordinates
import nibabel as nib

from .._utils import s3
from ._utils import BUCKET_NAME

MNI_ORIGIN = np.asarray([183 - 91, 127, 73]) - 1
MNI_RESOLUTION = 1


def load_transformation(
    subject: int, *, source_space: str, target_space: str, suffix: str
) -> np.ndarray:
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "transforms"
        / f"{source_space}-to-{target_space}{suffix}"
    )

    s3.download(filepath, bucket=BUCKET_NAME)
    transformation = nib.load(filepath).get_fdata()
    return transformation


def load_native_surface(
    subject: int, *, hemisphere: str, surface_type: str = "w-g.pct.mgh"
) -> tuple[np.ndarray, np.ndarray]:
    filepath = (
        Path("nsddata")
        / "freesurfer"
        / f"subj{subject + 1:02}"
        / f"surf"
        / f"{hemisphere}.{surface_type}"
    )
    s3.download(filepath, bucket=BUCKET_NAME)
    return filepath
    # image = nib.freesurfer.mghformat.load(filepath)
    # return image.header, image.affine


def _interpolate(
    volume: np.ndarray, *, coordinates: np.ndarray, interpolation_type: str = "cubic"
) -> np.ndarray:
    """
    Wrapper for ba_interp3. Normal calls to ba_interp3 assign values to interpolation points that lie outside the original data range. We ensure that coordinates outside the original field-of-view (i.e. if the value along a dimension is less than 1 or greater than the number of voxels in the original volume along that dimension) are returned as NaN and coordinates that have any NaNs are returned as NaN.

    Args:
        volume: 3D matrix (can be complex-valued)
        coordinates: (3, N) matrix coordinates to interpolate at
        interpolation_type: "nearest", "linear", or "cubic"
    """
    # input
    if interpolation_type == "cubic":
        order = 3
    elif interpolation_type == "linear":
        order = 1
    elif interpolation_type == "nearest":
        order = 0
    else:
        raise ValueError("interpolation method not implemented.")

    # bad locations must get set to NaN
    bad = np.any(np.isinf(coordinates), axis=0)
    coordinates[:, bad] = 1

    # out of range must become NaN, too
    bad = np.any(
        np.c_[
            bad,
            coordinates[0, :] < 1,
            coordinates[0, :] > volume.shape[0],
            coordinates[1, :] < 1,
            coordinates[1, :] > volume.shape[1],
            coordinates[2, :] < 1,
            coordinates[2, :] > volume.shape[2],
        ],
        axis=1,
    ).astype(bool)

    # resample the volume
    if not np.any(np.isreal(volume)):
        # we interpolate the real and imaginary parts independently
        transformed_data = map_coordinates(
            np.nan_to_num(np.real(volume)).astype(np.float64),
            coordinates,
            order=order,
            mode="nearest",
        ) + 1j * map_coordinates(
            np.nan_to_num(np.imag(volume)).astype(np.float64),
            coordinates,
            order=order,
            mode="nearest",
        )
    else:
        # consider using mode constant with a cval.
        transformed_data = map_coordinates(
            np.nan_to_num(volume).astype(np.float64),
            coordinates,
            order=order,
            mode="nearest",
        )
        transformed_data[bad] = np.nan

    return transformed_data


def _transform(
    data: np.ndarray,
    *,
    transformation: np.ndarray,
    interpolation_type: str,
    target_type: str,
) -> np.ndarray:
    """_summary_

    Args:
        data: data to be transformed from one space to another
        transformation: transformation matrix
        interpolation_type: passed to _interpolate
        target_type: "volume" or "surface"

    Returns:
        Transformed data
    """
    target_shape = transformation.shape[:3]

    coordinates = np.c_[
        transformation[:, :, :, 0].ravel(order="F"),
        transformation[:, :, :, 1].ravel(order="F"),
        transformation[:, :, :, 2].ravel(order="F"),
    ].T

    coordinates[coordinates == 9999] = np.nan
    coordinates -= 1  # Kendrick's 1-based indexing.

    transformed_data: list[np.ndarray] = []

    if data.ndim == 3:
        data = np.expand_dims(data, axis=0)

    transformed_data = []
    for data_ in tqdm(data, desc="volume", leave=False):
        data_ = _interpolate(
            data_, coordinates=coordinates, interpolation_type=interpolation_type
        )
        data_ = np.nan_to_num(data_)
        if target_type == "volume":
            data_ = data_.reshape(target_shape, order="F")

        transformed_data.append(data_)

    if len(transformed_data) == 1:
        return transformed_data[0]
    else:
        return np.vstack(transformed_data)


def _ndarray_to_nifti1image(
    data: np.ndarray, *, resolution: float, origin: np.ndarray = None
) -> nib.Nifti1Image:
    header = nib.Nifti1Header()
    header.set_data_dtype(data.dtype)

    affine = np.diag([resolution] * 3 + [1])
    if origin is None:
        origin = (([1, 1, 1] + np.asarray(data.shape)) / 2) - 1
    affine[0, -1] = -origin[0] * resolution
    affine[1, -1] = -origin[1] * resolution
    affine[2, -1] = -origin[2] * resolution

    return nib.Nifti1Image(data, affine, header)


def transform_volume_to_mni(
    data: np.ndarray, *, subject: int, source_space: str, interpolation_type: str
) -> nib.Nifti1Image:
    transformation = load_transformation(
        subject=subject, source_space=source_space, target_space="MNI", suffix=".nii.gz"
    )
    transformed_data = _transform(
        data=data,
        transformation=transformation,
        target_type="volume",
        interpolation_type=interpolation_type,
    )
    return _ndarray_to_nifti1image(
        transformed_data, resolution=MNI_RESOLUTION, origin=MNI_ORIGIN
    )


def transform_volume_to_native_surface(
    data: np.ndarray,
    *,
    subject: int,
    source_space: str = "func1pt8",
    interpolation_type: str = "cubic",
    layers: Collection[str] = (
        "layerB1",
        "layerB2",
        "layerB3",
    ),
    average_across_layers: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    native_surface: dict[str, dict[str, np.ndarray]] = {}
    for hemisphere in ("lh", "rh"):
        native_surface[hemisphere] = {}
        for layer in layers:
            transformation = load_transformation(
                subject=subject,
                source_space=f"{hemisphere}.{source_space}",
                target_space=layer,
                suffix=".mgz",
            )

            native_surface[hemisphere][layer] = _transform(
                data,
                transformation=transformation,
                target_type="surface",
                interpolation_type=interpolation_type,
            )

        if average_across_layers:
            native_surface[hemisphere] = {
                "average": np.vstack(list(native_surface[hemisphere].values())).mean(
                    axis=0
                )
            }
    return native_surface
    # header, affine = load_native_surface_header(
    #     subject=subject, hemisphere=hemisphere
    # )
    # for layer, data in native_surface.items():
    #     vol_h = data[:, np.newaxis].astype(np.float64)
    #     v_img = nib.freesurfer.mghformat.MGHImage(
    #         vol_h, affine, header=header, extra={}
    #     )
    #     v_img.to_filename(f"temp")
