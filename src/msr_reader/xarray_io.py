'''
xarray integration: read stacks and stack groups as labelled xarray.DataArray.

xarray is an optional dependency (pip install msr-reader[xarray]); this
module is only imported when the read_stack_xr/read_stack_group_xr methods
of OBFFile are called, so msr_reader itself keeps working without it.

Conventions:
- dims are normalized from the Imspector dimension labels ('ExpControl X'
  -> 'x', 'TimeLapse T'/'ExpControl T' -> 't'); unlabelled axes fall back to
  the positional OBF convention (file axes 0-3 are x/y/z/t, later axes
  'dim_<n>'). the original label is kept in the coord attrs.
- coordinates are physical positions in the axis SI unit (meters, seconds),
  computed from the axis offset/length; column positions from the footer, if
  present, are used instead (non-uniform axes). the unit string is in the
  coord attrs as 'units' (CF-style), the axis step as 'pixel_size'.
- size-1 axes are dropped, consistent with read_stack()/shapes.
- a group is returned as one DataArray with a 'channel' dim labelled with
  the channel names, mirroring read_stack_group().
'''
import re

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

from .obffile import _parse_dtype


def _require_xarray():
    if xr is None:
        raise ImportError(
            'xarray is required for the read_stack_xr/read_stack_group_xr methods; '
            'install it with: pip install msr-reader[xarray]'
        )


# positional fallback names for unlabelled axes: OBF stores the axes with X
# first (fastest-varying), so file axis 0/1/2/3 conventionally is x/y/z/t
_POSITION_NAMES = ('x', 'y', 'z', 't')


def _dim_name(label, file_axis, used):
    'normalized xarray dim name from an Imspector dimension label'
    name = None
    if label and label != 'None':
        token = re.sub(r'[^a-z0-9_]', '_', label.split()[-1].lower())
        if token and not token[0].isdigit():
            name = token
    if name is None:
        name = _POSITION_NAMES[file_axis] if file_axis < len(_POSITION_NAMES) else f'dim_{file_axis}'
    if name in used:
        base, candidate, n = name, name, 1
        while candidate in used:
            candidate = f'{base}_{n}'
            n += 1
        name = candidate
    used.add(name)
    return name


def _stack_dims_coords(obf, idx):
    '''
    (dims, coords, has_samples_axis) for stack idx, in numpy axis order.
    coords maps dim name -> (dims, values, attrs) tuples. size-1 axes are
    dropped, consistent with read_stack()/shapes.
    '''
    header = obf.stack_headers[idx]
    footer = obf.stack_footers[idx]
    rank = len(header.size)
    # kept axes (size > 1), in numpy order (file order is reversed)
    kept = [i for i in range(rank) if header.size[i] > 1][::-1]

    dims = []
    coords = {}
    used = set()
    for fi in kept:
        label = footer.dimension_labels[fi]
        name = _dim_name(label, fi, used)
        dims.append(name)
        n = header.size[fi]

        attrs = {}
        if label:
            attrs['imspector_label'] = label
        if footer.si_dimensions is not None:
            attrs['units'] = str(footer.si_dimensions[fi])

        if footer.col_positions is not None and footer.col_positions[fi] is not None:
            # explicit per-sample positions (non-uniform axis), in axis units
            values = np.asarray(footer.col_positions[fi], dtype=np.float64)
        else:
            step = header.length[fi] / n
            # physical center of sample k along the axis, per the OBF spec:
            # off[i] + (0.5 + k) * len[i] / res[i]
            values = header.offset[fi] + (0.5 + np.arange(n, dtype=np.float64)) * step
            attrs['pixel_size'] = step
        coords[name] = ((name,), values, attrs)

        if footer.col_labels is not None and footer.col_labels[fi] is not None:
            coords[f'{name}_labels'] = ((name,), list(footer.col_labels[fi]), {})

    # RGB-style stacks carry an extra trailing samples axis
    try:
        samples_per_pixel = _parse_dtype(header.dtype)[1]
    except ValueError:
        samples_per_pixel = 1

    return dims, coords, samples_per_pixel > 1


def _stack_attrs(obf, idx):
    header = obf.stack_headers[idx]
    # same placeholder as for group channel names of unnamed stacks
    attrs = {'stack_name': header.name if header.name else f'stack {idx}'}
    series, channel = obf._parse_stack_name(idx)
    if channel is not None:
        attrs['series'] = series
        attrs['channel_name'] = channel
    return attrs


def stack_to_xarray(obf, idx):
    """
    read stack idx into an xarray.DataArray with labelled dims and physical
    coordinates (SI units of the file: meters for spatial axes, seconds for
    time). requires the optional xarray dependency.
    """
    _require_xarray()
    data = obf.read_stack(idx)
    dims, coords, has_samples_axis = _stack_dims_coords(obf, idx)
    if has_samples_axis:
        dims = dims + ['samples']
        coords['samples'] = (('samples',), np.arange(data.shape[-1]), {})
    attrs = _stack_attrs(obf, idx)
    name = attrs.get('channel_name', attrs['stack_name'])
    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs, name=name)


def stack_group_to_xarray(obf, group_idx, axis=0):
    """
    read all stacks of a stack group into one xarray.DataArray, stacked along
    a new 'channel' axis (at position axis, default first) labelled with the
    channel names. requires the optional xarray dependency.
    """
    _require_xarray()
    group = obf.stack_groups[group_idx]
    data = obf.read_stack_group(group_idx, axis=axis)
    dims, coords, has_samples_axis = _stack_dims_coords(obf, group.indices[0])
    if has_samples_axis:
        dims = dims + ['samples']
        coords['samples'] = (('samples',), np.arange(data.shape[-1]), {})
    # normalize axis the same way np.stack does for the stacked result
    axis = axis % (len(dims) + 1)
    channel_dim = 'channel'
    n = 1
    while channel_dim in dims:
        channel_dim = f'channel_{n}'
        n += 1
    dims.insert(axis, channel_dim)
    coords[channel_dim] = ((channel_dim,), np.array(group.channel_names), {})
    attrs = {'series': group.series,
             'stack_names': [obf.stack_names[i] for i in group.indices]}
    name = str(group.series) if group.series is not None else f'group_{group_idx}'
    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs, name=name)
