"""Resolution-matching transforms for gridded canopy datasets."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import xarray as xr

_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def match_vertical_resolution(
    dataset: xr.Dataset,
    *,
    target_resolution: float,
    source_resolution: float = 0.0,
    dimension: str = "z",
    variables: Iterable[str] | None = None,
    truncate: float = 4.0,
) -> xr.Dataset:
    """Blur vertical profiles to a target Gaussian FWHM resolution.

    Coordinate sampling is not resolution. This transform preserves the
    original vertical coordinates while convolving variables along ``dimension``
    with the additional Gaussian blur needed to reach ``target_resolution``.
    Edge samples are normalized by the available kernel weight.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if dimension not in dataset.coords:
        raise ValueError(f"Dataset has no {dimension!r} coordinate")
    coordinate = dataset[dimension]
    if coordinate.dims != (dimension,) or coordinate.size < 2:
        raise ValueError(f"Coordinate {dimension!r} must contain at least two samples")
    target = _positive(target_resolution, "target_resolution")
    source = _nonnegative(source_resolution, "source_resolution")
    if target < source and not np.isclose(target, source):
        raise ValueError("target_resolution cannot be finer than source_resolution")
    if not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("truncate must be a positive finite value")

    values = np.asarray(coordinate.data, dtype=float)
    differences = np.diff(values)
    if not np.isfinite(values).all() or np.any(differences <= 0):
        raise ValueError(f"Coordinate {dimension!r} must be finite and increasing")
    sampling = float(np.median(differences))
    if not np.allclose(differences, sampling, rtol=1e-5, atol=1e-10):
        raise ValueError("Vertical resolution matching requires regular sampling")

    selected = _selected_variables(dataset, variables, dimension)
    additional_fwhm = float(np.sqrt(max(0.0, target**2 - source**2)))
    if np.isclose(additional_fwhm, 0.0):
        result = dataset.copy(deep=False)
    else:
        sigma_samples = additional_fwhm / (_FWHM_TO_SIGMA * sampling)
        radius = max(1, int(np.ceil(truncate * sigma_samples)))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        weights = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
        weights /= weights.sum()
        kernel_dimension = f"_{dimension}_resolution_kernel"
        weight_array = xr.DataArray(weights, dims=kernel_dimension)

        result = dataset.copy(deep=False)
        for name in selected:
            source_array = dataset[name]
            window = source_array.rolling(
                {dimension: weights.size}, center=True, min_periods=1
            ).construct(kernel_dimension)
            valid_weights = weight_array.where(window.notnull())
            smoothed = (window * valid_weights).sum(
                kernel_dimension, skipna=True
            ) / valid_weights.sum(kernel_dimension, skipna=True)
            smoothed.attrs = dict(source_array.attrs)
            smoothed.attrs.update(
                vertical_resolution=float(target),
                vertical_resolution_units=coordinate.attrs.get("units", "m"),
                resolution_kernel="gaussian_fwhm",
            )
            result[name] = smoothed

    result.attrs = dict(dataset.attrs)
    result.attrs.update(
        vertical_sampling=float(sampling),
        vertical_sampling_units=coordinate.attrs.get("units", "m"),
        vertical_resolution=float(target),
        vertical_resolution_units=coordinate.attrs.get("units", "m"),
        source_vertical_resolution=float(source),
        resolution_kernel="gaussian_fwhm",
    )
    return result


def _selected_variables(
    dataset: xr.Dataset,
    variables: Iterable[str] | None,
    dimension: str,
) -> tuple[str, ...]:
    if variables is None:
        selected = tuple(
            name
            for name, variable in dataset.data_vars.items()
            if dimension in variable.dims
        )
    else:
        selected = tuple(variables)
        unknown = sorted(set(selected).difference(dataset.data_vars))
        if unknown:
            raise ValueError(f"Unknown variables: {', '.join(unknown)}")
        without_dimension = [
            name for name in selected if dimension not in dataset[name].dims
        ]
        if without_dimension:
            raise ValueError(
                f"Variables do not use dimension {dimension!r}: "
                f"{', '.join(without_dimension)}"
            )
    if not selected:
        raise ValueError(f"No variables use dimension {dimension!r}")
    return tuple(dict.fromkeys(selected))


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return number


def _nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative finite value")
    return number
