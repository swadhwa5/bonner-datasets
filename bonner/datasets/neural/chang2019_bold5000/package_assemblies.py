from pathlib import Path
from tqdm import tqdm
import numpy as np
import xarray as xr

from ...utils import load_nii
from ...utils.brainio.assembly import package
from .utils import (
    IDENTIFIER,
    N_SUBJECTS,
    N_SESSIONS,
    ROIS,
    _get_betas_filename,
    _get_brain_mask_filename,
    _get_imagenames_filename,
)


def package_assemblies(
    catalog_name: str, location_type: str, location: str, **kwargs: str
) -> None:
    for subject in tqdm(range(N_SUBJECTS), desc="subject"):
        mask = _load_brain_mask(subject)
        structural_scan = _load_structural_scan(subject)
        neuroid_metadata = _load_neuroid_metadata(subject)

        assembly = (
            xr.concat(
                [
                    _load_activations(subject, session).sel({"neuroid": mask})
                    for session in range(N_SESSIONS[subject])
                ],
                dim="presentation",
            )
            .astype(np.float32)
            .rename(f"{IDENTIFIER}-subject{subject}")
            .assign_coords(
                {
                    "stimulus_id": (
                        "presentation",
                        _load_image_filename_stems(subject),
                    ),
                }
            )
            .assign_coords(
                {coord: neuroid_metadata[coord] for coord in neuroid_metadata.coords}
            )
            .assign_attrs(
                {
                    "brain_dimensions": mask.attrs["brain_dimensions"],
                    "structural_scan": structural_scan.values,
                    "structural_scan_brain_dimensions": structural_scan.attrs[
                        "brain_dimensions"
                    ],
                    "identifier": f"{IDENTIFIER}-subject{subject}",
                    "stimulus_set_identifier": IDENTIFIER,
                }
            )
            .dropna(dim="neuroid", how="any")
        )
        package(
            assembly=assembly,
            catalog_name=catalog_name,
            location_type=location_type,
            location=location,
        )


def _load_neuroid_metadata(subject: int) -> xr.DataArray:
    brain_mask = _load_brain_mask(subject)
    n_voxels = np.sum(brain_mask.values)
    metadata = xr.DataArray(np.full(n_voxels, np.nan), dims="neuroid").assign_coords(
        {"hemisphere": ("neuroid", [""] * n_voxels)}
    )

    for roi in ROIS:
        metadata = metadata.assign_coords(
            {f"roi_{roi}": ("neuroid", [False] * n_voxels)}
        )
        for hemisphere in ("LH", "RH"):
            roi_mask = _load_roi_mask(subject, hemisphere, roi).sel(
                {"neuroid": brain_mask}
            )
            metadata[f"roi_{roi}"][roi_mask] = True
            metadata["hemisphere"][roi_mask] = hemisphere
    return metadata


def _load_brain_mask(subject: int) -> xr.DataArray:
    return load_nii(Path.cwd() / _get_brain_mask_filename(subject)).astype(bool)


def _load_activations(subject: int, session: int) -> xr.DataArray:
    return load_nii(Path.cwd() / _get_betas_filename(subject, session)).astype(
        np.float32
    )


def _load_roi_mask(subject: int, hemisphere: str, roi: str) -> xr.DataArray:
    return load_nii(
        Path.cwd()
        / f"sub-CSI{subject + 1}/sub-CSI{subject + 1}_mask-{hemisphere}{roi}.nii.gz"
    ).astype(bool)


def _load_structural_scan(subject: int) -> xr.DataArray:
    return load_nii(
        Path.cwd()
        / f"BOLD5000_Structural/CSI{subject + 1}_Structural/T1w_MPRAGE_CSI{subject + 1}.nii"
    )


def _load_image_filename_stems(subject: int) -> list[Path]:
    with open(_get_imagenames_filename(subject), "r") as f:
        # strip newlines, extension
        return [Path(line[:-1]).stem for line in f.readlines()]
