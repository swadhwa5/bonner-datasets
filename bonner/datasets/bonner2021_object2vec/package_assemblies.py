import itertools

import numpy as np
import xarray as xr
from scipy.io import loadmat
import nibabel as nib
from brainio.assembly import package

from ..utils import package
from .utils import load_conditions
from .constants import IDENTIFIER, N_SUBJECTS, URLS, FILENAMES, ROIS, BRAIN_DIMENSIONS


def create_assembly(subject: int) -> xr.DataArray:
    """Load and format functional activations.

    :param subject: subject ID
    :type subject: int
    :return: functional activations with "presentation" and "neuroid" dimensions
    :rtype: xr.DataArray
    """
    activations = loadmat(FILENAMES["activations"][subject], simplify_cells=True)[
        "betas"
    ]
    x, y, z = np.unravel_index(
        np.arange(np.product(BRAIN_DIMENSIONS)), BRAIN_DIMENSIONS
    )
    n_voxels = activations.shape[1]

    roi_indices = loadmat(FILENAMES["rois"][subject], simplify_cells=True)["indices"]
    hemisphere = np.full(n_voxels, fill_value="", dtype="<U1")
    for roi, _hemisphere in itertools.product(ROIS.keys(), ("L", "R")):
        hemisphere[roi_indices[roi][_hemisphere]] = _hemisphere

    noise_ceilings = loadmat(
        FILENAMES["noise_ceilings"][subject], simplify_cells=True
    )

    # TODO check whether MATLAB's ordering differs from Python (FORTRAN vs C)
    assembly = (
        xr.DataArray(
            data=activations,
            dims=("condition", "neuroid", "repetition"),
            coords={
                "stimulus_id": ("condition", load_conditions()),
                "x": ("neuroid", x),
                "y": ("neuroid", y),
                "z": ("neuroid", z),
                "hemisphere": ("neuroid", hemisphere),
                "repetition": ("repetition", np.arange(activations.shape[2])),
            },
        )
        .stack({"presentation": ("condition", "repetition")})
        .reset_index("presentation")
        .transpose("presentation", "neuroid")
        .assign_coords(
            {
                f"roi_{roi}": (
                    "neuroid",
                    [
                        True if idx in roi_indices[roi][hemisphere] else False
                        for idx in range(n_voxels)
                    ],
                )
                for roi, hemisphere in itertools.product(ROIS.keys(), ("L", "R"))
            }
        )
        .assign_coords(
            {
                f"contrast_{contrast}": (
                    "neuroid",
                    nib.load(FILENAMES["contrasts"][contrast][subject])
                    .get_fdata()
                    .reshape(-1),
                )
                for contrast in URLS["contrasts"].keys()
            }
        )
        .assign_coords(
            {
                noise_ceiling: ("neuroid", noise_ceilings[noise_ceiling])
                for noise_ceiling in ("upperR", "lowerR", "splitR")
            }
        )
        .dropna(dim="neuroid", how="all")
        .assign_attrs(
            {
                "identifier": f"{IDENTIFIER}-subject{subject}",
                "stimulus_set_identifier": IDENTIFIER,
                "brain_dimensions": BRAIN_DIMENSIONS,
            }
        )
        .drop_vars("condition")
    )
    return assembly


def package_assemblies(catalog_name: str, location_type: str, location: str, **kwargs) -> None:
    for subject in range(N_SUBJECTS):
        package(
            identifier=f"{IDENTIFIER}-subject{subject}",
            assembly=create_assembly(subject),
            catalog_name=catalog_name,
            location_type=location_type,
            location=location,
        )
