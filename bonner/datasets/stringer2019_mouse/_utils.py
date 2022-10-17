import xarray as xr
from scipy.sparse.linalg import eigsh


BIBTEX = """
@article{Stringer2019,
  doi = {10.1038/s41586-019-1346-5},
  url = {https://doi.org/10.1038/s41586-019-1346-5},
  year = {2019},
  month = jun,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {571},
  number = {7765},
  pages = {361--365},
  author = {Carsen Stringer and Marius Pachitariu and Nicholas Steinmetz and Matteo Carandini and Kenneth D. Harris},
  title = {High-dimensional geometry of population responses in visual cortex},
  journal = {Nature}
}
"""
IDENTIFIER = "stringer2019.mouse"
SESSIONS = (
    {
        "mouse": "M160825_MP027",
        "date": "2016-12-14",
    },
    {
        "mouse": "M161025_MP030",
        "date": "2017-05-29",
    },
    {
        "mouse": "M170604_MP031",
        "date": "2017-06-28",
    },
    {
        "mouse": "M170714_MP032",
        "date": "2017-08-07",
    },
    {
        "mouse": "M170714_MP032",
        "date": "2017-09-14",
    },
    {
        "mouse": "M170717_MP033",
        "date": "2017-08-20",
    },
    {
        "mouse": "M170717_MP034",
        "date": "2017-09-11",
    },
)


def preprocess_assembly(assembly: xr.Dataset) -> xr.DataArray:
    spontaneous = assembly["spontaneous activity"]
    mean = spontaneous.mean("time")
    std = spontaneous.std("time") + 1e-6

    spontaneous = (spontaneous - mean) / std

    stimulus_related = (assembly["stimulus-related activity"] - mean) / std
    stimulus_related = stimulus_related.isel(
        {"presentation": stimulus_related["stimulus_id"].values != "blank"}
    )

    _, eigenvectors = eigsh(spontaneous.values.T @ spontaneous.values, k=32)
    stimulus_related -= (stimulus_related.values @ eigenvectors) @ eigenvectors.T
    stimulus_related -= stimulus_related.mean("presentation")
    return stimulus_related
