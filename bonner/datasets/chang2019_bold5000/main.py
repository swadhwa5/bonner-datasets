from bonner.datasets.utils import package
from bonner.datasets.chang2019_bold5000.utils import IDENTIFIER
from bonner.datasets.chang2019_bold5000.download import download_dataset
from bonner.datasets.chang2019_bold5000.package_stimulus_set import package_stimulus_set
from bonner.datasets.chang2019_bold5000.package_assemblies import package_assemblies


if __name__ == "__main__":
    package(
        IDENTIFIER,
        pipeline=[
            download_dataset,
            package_stimulus_set,
            package_assemblies,
        ],
    )
