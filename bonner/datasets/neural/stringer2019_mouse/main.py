from bonner.datasets.utils import package
from bonner.datasets.neural.stringer2019_mouse.download import download_dataset
from bonner.datasets.neural.stringer2019_mouse.utils import IDENTIFIER


if __name__ == "__main__":
    package(
        IDENTIFIER,
        pipeline=[
            download_dataset,
            # save_images,
            # package_stimulus_set,
            # package_assemblies,
        ],
    )
