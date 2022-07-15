from ..utils import package
from .constants import IDENTIFIER
from .download import download_dataset
from .package_stimulus_set import package_stimulus_set
from .package_assemblies import package_assemblies


if __name__ == "__main__":
    package(
        IDENTIFIER,
        pipeline=[
            download_dataset,
            package_stimulus_set,
            package_assemblies,
        ]
    )
