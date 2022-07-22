from pathlib import Path
import shutil
import zipfile

import pandas as pd

from bonner.brainio import BONNER_BRAINIO_HOME, fetch, package_stimulus_set


def load(
    catalog_name: str, identifier: str, check_integrity: bool = True
) -> tuple[pd.DataFrame, Path]:
    """Load a stimulus set from a catalog.

    :param catalog_name: name of the BrainIO catalog
    :param identifier: identifier of the stimulus set, as defined in the BrainIO specification
    :param check_integrity: whether to check the SHA1 hash of the file, defaults to True
    :return: the stimulus set metadata and the path to the stimuli
    """
    filepaths = {
        filetype: fetch(
            catalog_name=catalog_name,
            identifier=identifier,
            lookup_type="stimulus_set",
            class_=filetype,
            check_integrity=check_integrity,
        )
        for filetype in ("csv", "zip")
    }

    csv = pd.read_csv(filepaths["csv"])

    stimuli_dir = BONNER_BRAINIO_HOME / catalog_name / f"{identifier}"

    if not all([(stimuli_dir / subpath).exists() for subpath in csv["filename"]]):
        if stimuli_dir.exists():
            shutil.rmtree(stimuli_dir)
        stimuli_dir.mkdir(parents=True)
        with zipfile.ZipFile(filepaths["zip"], "r") as f:
            f.extractall(stimuli_dir)

    return csv, stimuli_dir


def package(
    *,
    identifier: str,
    stimulus_set: pd.DataFrame,
    stimulus_dir: Path,
    catalog_name: str,
    location_type: str,
    location: str,
) -> None:
    """Package a stimulus set.

    :param identifier: identifier of the stimulus set, as defined in the BrainIO specification
    :param stimulus_set: stimulus set metadata
    :param stimulus_dir: directory containing the stimuli
    :param catalog_name: name of the BrainIO catalog
    :param location_type: location_type of the stimulus set, as defined in the BrainIO specification
    :param location: location of the stimulus set, as defined in the BrainIO specification
    """

    filepaths = {
        "csv": _create_csv(
            identifier=identifier,
            stimulus_set=stimulus_set,
            catalog_name=catalog_name,
        ),
        "zip": _create_zip(
            identifier=identifier,
            stimulus_set=stimulus_set,
            stimulus_dir=stimulus_dir,
            catalog_name=catalog_name,
        ),
    }

    package_stimulus_set(
        identifier=identifier,
        filepath_csv=filepaths["csv"],
        filepath_zip=filepaths["zip"],
        class_csv="csv",
        class_zip="zip",
        location_csv=f"{location}/{filepaths['csv'].name}",
        location_zip=f"{location}/{filepaths['zip'].name}",
        catalog_name=catalog_name,
        location_type=location_type,
    )


def _create_csv(
    *, identifier: str, stimulus_set: pd.DataFrame, catalog_name: str
) -> Path:
    """Creates a CSV file of the stimulus set metadata.

    :param identifier: identifier of the stimulus set, as defined in the BrainIO specification
    :param stimulus_set: the stimulus set metadata
    :param catalog_name: name of the BrainIO catalog
    :return: path to the CSV file
    """
    filepath = BONNER_BRAINIO_HOME / catalog_name / f"{identifier}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    stimulus_set.to_csv(filepath, index=False)
    return filepath


def _create_zip(
    *,
    identifier: str,
    stimulus_set: pd.DataFrame,
    stimulus_dir: Path,
    catalog_name: str,
) -> Path:
    """Creates a ZIP archive of the stimulus set stimuli.

    :param identifier: identifier of the stimulus set, as defined in the BrainIO specification
    :param stimulus_set: the stimulus set metadata
    :param stimulus_dir: directory containing the stimuli
    :param catalog_name: name of the BrainIO catalog
    :return: path to the ZIP archive
    """
    filepath = BONNER_BRAINIO_HOME / catalog_name / f"{identifier}.zip"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(filepath, "w") as zip:
        for filename in stimulus_set["filename"]:
            zip.write(stimulus_dir / filename, arcname=filename)
    return filepath
