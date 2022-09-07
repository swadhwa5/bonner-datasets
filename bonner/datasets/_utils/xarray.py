import xarray as xr


def groupby_reset(
    x: xr.DataArray, dim_groupby: str, dim_original_name: str
) -> xr.DataArray:
    return (
        x.reset_index(list(x.indexes))
        .rename({dim_groupby: dim_original_name})
        .assign_coords({dim_groupby: (dim_original_name, x[dim_groupby].values)})
        .drop_vars(dim_original_name)
    )
