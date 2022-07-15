import numpy as np
from scipy.io import loadmat

from .constants import FILENAMES


def load_conditions() -> np.ndarray:
    return loadmat(FILENAMES["conditions"], simplify_cells=True)["stacked"]["conds"]
