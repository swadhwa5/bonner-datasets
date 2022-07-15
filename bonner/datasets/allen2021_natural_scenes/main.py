from bonner.datasets.utils import package
from bonner.datasets.allen2021_natural_scenes.utils import IDENTIFIER
from bonner.datasets.allen2021_natural_scenes.download import download_dataset, save_images
from bonner.datasets.allen2021_natural_scenes.package_stimulus_set import package_stimulus_set
from bonner.datasets.allen2021_natural_scenes.package_assemblies import package_assemblies


if __name__ == "__main__":
    package(
        IDENTIFIER,
        pipeline=[
            download_dataset,
            save_images,
            package_stimulus_set,
            package_assemblies,
        ]
    )
