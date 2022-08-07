__all__ = [
    "load_stimulus_set",
    "package_dataset",
]

from ._brainio import load_stimulus_set
from ._package import package_dataset

if __name__ == "__main__":
    package_dataset(auto_envvar_prefix="BONNER_DATASETS")
