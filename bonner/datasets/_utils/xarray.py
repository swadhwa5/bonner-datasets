from collections.abc import Hashable

import xarray as xr


def groupby_reset(
    x: xr.DataArray, *, groupby_coord: str, groupby_dim: Hashable
) -> xr.DataArray:
    return (
        x.reset_index(groupby_coord)
        .rename({groupby_coord: groupby_dim})
        .assign_coords({groupby_coord: (groupby_dim, x[groupby_coord].values)})
        .drop_vars(groupby_dim)
    )
