from collections.abc import Mapping, Hashable

import dask.array as da
import numpy as np
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


def groupby_mean_chunked(
    x: xr.DataArray,
    *,
    groupby_coord: str,
    chunks: Mapping[str, int] = {},
) -> xr.DataArray:
    original_dims = x.dims
    groupby_dim = x[groupby_coord].dims[0]
    groupby = x[groupby_coord].groupby(groupby_coord).groups
    stimulus_ids = np.array(list(groupby.keys()))
    non_groupby_dims = [dim for dim in x.dims if dim != groupby_dim]

    template = xr.DataArray(
        name=x.name,
        data=da.empty(
            [
                x.sizes[dim] if dim in non_groupby_dims else len(groupby)
                for dim in original_dims
            ],
            chunks={original_dims.index(dim): size for dim, size in chunks.items()},
            dtype=np.float32,
        ),
        dims=original_dims,
        coords={dim: (dim, x[dim].values) for dim in non_groupby_dims}
        | {groupby_coord: (groupby_dim, stimulus_ids)},
    )

    def _helper_ufunc(
        betas: xr.DataArray,
    ) -> xr.DataArray:
        return groupby_reset(
            betas.load().groupby(groupby_coord).mean(),
            groupby_coord=groupby_coord,
            groupby_dim=groupby_dim,
        ).transpose(*original_dims)

    return x.chunk(chunks=chunks).map_blocks(_helper_ufunc, template=template)
