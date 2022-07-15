from bonner.datasets.utils import package
from bonner.datasets.bonner2021_object2vec.utils import IDENTIFIER
from bonner.datasets.bonner2021_object2vec.download import download_dataset
from bonner.datasets.bonner2021_object2vec.package_stimulus_set import package_stimulus_set
from bonner.datasets.bonner2021_object2vec.package_assemblies import package_assemblies


if __name__ == "__main__":
    package(
        IDENTIFIER,
        pipeline=[
            download_dataset,
            package_stimulus_set,
            package_assemblies,
        ]
    )
