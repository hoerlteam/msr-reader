'''
Tests for the optional xarray integration (OBFFile.read_stack_xr /
read_stack_group_xr). The whole module is skipped when xarray is not
installed; msr_reader itself must work without it.
'''
import os
import struct
import unittest
from unittest import mock

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    xr = None
    HAS_XARRAY = False

from msr_reader import OBFFile
from .synthetic_obf import StackSpec, build_obf, obf_file

INT16 = 0x8
UINT8 = 0x1

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _imaging_file():
    'two channels of series 0 (ZYX stacks), one stack of series 1, a rescue mask'
    zyx = dict(lengths=(4e-7, 3e-7, 2e-6), si_units=['m', 'm', 'm'],
               dimension_labels=['ExpControl X', 'ExpControl Y', 'ExpControl Z'])
    specs = [
        StackSpec('Chan A {0}', (4, 3, 2), INT16, struct.pack('<24h', *range(24)), **zyx),
        StackSpec('Chan B {0}', (4, 3, 2), INT16, struct.pack('<24h', *range(24, 48)), **zyx),
        StackSpec('Chan A {1}', (4, 3, 2), INT16, struct.pack('<24h', *range(24)), **zyx),
        StackSpec('Chan A {0}_rescue_info', (4, 3), UINT8, bytes(12),
                  dimension_labels=['ExpControl X', 'ExpControl Y']),
    ]
    return build_obf(specs)


@unittest.skipUnless(HAS_XARRAY, 'xarray not installed')
class TestStackXarray(unittest.TestCase):

    def test_dims_coords_units(self):
        with obf_file(_imaging_file()) as obf:
            da = obf.read_stack_xr(0)
            ref = obf.read_stack(0)
        self.assertIsInstance(da, xr.DataArray)
        # file axis order is X, Y, Z; the numpy/xarray axes are reversed
        self.assertEqual(da.dims, ('z', 'y', 'x'))
        self.assertEqual(da.shape, (2, 3, 4))
        # sample centers per the OBF spec: off + (0.5 + k) * len/res (off is 0 here)
        np.testing.assert_allclose(da['z'].values, [0.5e-6, 1.5e-6])
        np.testing.assert_allclose(da['x'].values, [0.5e-7, 1.5e-7, 2.5e-7, 3.5e-7])
        self.assertEqual(da['x'].attrs['units'], 'm')
        self.assertEqual(da['z'].attrs['units'], 'm')
        self.assertAlmostEqual(da['z'].attrs['pixel_size'], 1e-6)
        self.assertEqual(da['x'].attrs['imspector_label'], 'ExpControl X')
        self.assertEqual(da.attrs['stack_name'], 'Chan A {0}')
        self.assertEqual(da.attrs['series'], 0)
        self.assertEqual(da.attrs['channel_name'], 'Chan A')
        self.assertEqual(da.name, 'Chan A')
        self.assertEqual(da.dtype, np.int16)
        np.testing.assert_array_equal(da.values, ref)

    def test_size_one_axes_dropped(self):
        'consistent with read_stack()/shapes: size-1 axes do not appear'
        specs = [StackSpec('img {0}', (5, 1, 4), INT16, struct.pack('<20h', *range(20)),
                           lengths=(5e-7, 1.0, 4e-7), si_units=['m', 'dimensionless', 'm'],
                           dimension_labels=['ExpControl X', 'ExpControl Y', 'ExpControl Z'])]
        with obf_file(build_obf(specs)) as obf:
            da = obf.read_stack_xr(0)
            self.assertEqual(list(da.shape), obf.shape(0))
        self.assertEqual(da.dims, ('z', 'x'))
        self.assertEqual(da.shape, (4, 5))

    def test_time_axis_in_seconds(self):
        specs = [StackSpec('tl {0}', (2, 5), INT16, struct.pack('<10h', *range(10)),
                           lengths=(2e-7, 50.0), si_units=['m', 's'],
                           dimension_labels=['ExpControl X', 'TimeLapse T'])]
        with obf_file(build_obf(specs)) as obf:
            da = obf.read_stack_xr(0)
        self.assertEqual(da.dims, ('t', 'x'))
        np.testing.assert_allclose(da['t'].values, [5.0, 15.0, 25.0, 35.0, 45.0])
        self.assertEqual(da['t'].attrs['units'], 's')
        self.assertEqual(da['t'].attrs['imspector_label'], 'TimeLapse T')

    def test_col_positions_override_regular_grid(self):
        specs = [StackSpec('line {0}', (4,), INT16, struct.pack('<4h', *range(4)),
                           lengths=(1.0,), si_units=['m'],
                           dimension_labels=['ExpControl X'],
                           col_positions=[[1e-6, 2e-6, 5e-6, 9e-6]])]
        with obf_file(build_obf(specs)) as obf:
            da = obf.read_stack_xr(0)
        self.assertEqual(da.dims, ('x',))
        np.testing.assert_allclose(da['x'].values, [1e-6, 2e-6, 5e-6, 9e-6])
        self.assertEqual(da['x'].attrs['units'], 'm')
        self.assertNotIn('pixel_size', da['x'].attrs)

    def test_unlabelled_axes_get_positional_fallback_names(self):
        'unlabelled axes fall back to the positional OBF convention x/y/z/t'
        specs = [StackSpec('', (3, 2), INT16, struct.pack('<6h', *range(6)))]
        with obf_file(build_obf(specs)) as obf:
            da = obf.read_stack_xr(0)
        self.assertEqual(da.dims, ('y', 'x'))
        self.assertEqual(da.attrs['stack_name'], 'stack 0')
        self.assertNotIn('series', da.attrs)

    def test_unlabelled_axis_beyond_t(self):
        'file axes beyond t get dim_<n> names'
        specs = [StackSpec('', (2, 3, 4, 5, 6), INT16, struct.pack('<720h', *range(720)))]
        with obf_file(build_obf(specs)) as obf:
            da = obf.read_stack_xr(0)
        self.assertEqual(da.dims, ('dim_4', 't', 'z', 'y', 'x'))

    def test_missing_xarray_raises_informative_error(self):
        from msr_reader import xarray_io
        with mock.patch.object(xarray_io, 'xr', None):
            with obf_file(_imaging_file()) as obf:
                with self.assertRaises(ImportError) as cm:
                    obf.read_stack_xr(0)
        self.assertIn('msr-reader[xarray]', str(cm.exception))


@unittest.skipUnless(HAS_XARRAY, 'xarray not installed')
class TestStackGroupXarray(unittest.TestCase):

    def test_group_channel_dim(self):
        with obf_file(_imaging_file()) as obf:
            da = obf.read_stack_group_xr(0)
            ref = obf.read_stack_group(0)
        self.assertEqual(da.dims, ('channel', 'z', 'y', 'x'))
        self.assertEqual(da.shape, (2, 2, 3, 4))
        self.assertEqual(list(da['channel'].values), ['Chan A', 'Chan B'])
        self.assertEqual(da.attrs['series'], 0)
        self.assertEqual(da.attrs['stack_names'], ['Chan A {0}', 'Chan B {0}'])
        np.testing.assert_array_equal(da.values, ref)
        # spatial coords are shared along the channel axis
        np.testing.assert_allclose(da['x'].values, [0.5e-7, 1.5e-7, 2.5e-7, 3.5e-7])

    def test_single_member_group_gets_length_one_channel_axis(self):
        with obf_file(_imaging_file()) as obf:
            da = obf.read_stack_group_xr(1)  # series 1
        self.assertEqual(da.dims, ('channel', 'z', 'y', 'x'))
        self.assertEqual(da.shape, (1, 2, 3, 4))
        self.assertEqual(list(da['channel'].values), ['Chan A'])
        self.assertEqual(da.attrs['series'], 1)

    def test_group_axis_param(self):
        with obf_file(_imaging_file()) as obf:
            da = obf.read_stack_group_xr(0, axis=-1)
        self.assertEqual(da.dims, ('z', 'y', 'x', 'channel'))
        self.assertEqual(da.shape, (2, 3, 4, 2))

    def test_hierarchical_obf_group(self):
        'Lightbox-style names: series is the acquisition path, channels the last component'
        specs = [
            StackSpec('Overview 1/14:01:41/STAR RED', (2, 3), INT16, struct.pack('<6h', *range(6)),
                      lengths=(2e-7, 3e-7), si_units=['m', 'm'],
                      dimension_labels=['ExpControl X', 'ExpControl Y']),
            StackSpec('Overview 1/14:01:41/STAR ORANGE', (2, 3), INT16, struct.pack('<6h', *range(6, 12)),
                      lengths=(2e-7, 3e-7), si_units=['m', 'm'],
                      dimension_labels=['ExpControl X', 'ExpControl Y']),
        ]
        with obf_file(build_obf(specs), filename='synthetic.obf') as obf:
            da = obf.read_stack_group_xr(0)
        self.assertEqual(da.dims, ('channel', 'y', 'x'))
        self.assertEqual(list(da['channel'].values), ['STAR RED', 'STAR ORANGE'])
        self.assertEqual(da.attrs['series'], 'Overview 1/14:01:41')
        self.assertEqual(da.name, 'Overview 1/14:01:41')


@unittest.skipUnless(HAS_XARRAY, 'xarray not installed')
class TestRealFilesXarray(unittest.TestCase):

    def test_4d_nonstandard_axis_order(self):
        'TZYX file axis order: the footer labels define the xarray dims, T in seconds'
        with OBFFile(os.path.join(DATA_DIR, '20261019tests_4d.msr')) as obf:
            da = obf.read_stack_xr(0)
            grp = obf.read_stack_group_xr(0)
        self.assertEqual(da.dims, ('t', 'y', 'z', 'x'))
        self.assertEqual(da.shape, (5, 20, 4, 20))
        # sample centers: off + (0.5 + k) * len/res
        np.testing.assert_allclose(da['t'].values, (0.5 + np.arange(5)) * 10.0, rtol=1e-6)
        self.assertEqual(da['t'].attrs['units'], 's')
        np.testing.assert_allclose(da['z'].values, -1e-6 + (0.5 + np.arange(4)) * 5e-7, rtol=1e-6)
        self.assertEqual(da['z'].attrs['units'], 'm')
        self.assertEqual(da['x'].attrs['imspector_label'], 'ExpControl X')
        self.assertEqual(grp.dims, ('channel', 't', 'y', 'z', 'x'))
        self.assertEqual(list(grp['channel'].values), ['STAR RED_CONF'])

    def test_continuous_point_measurement_axis_is_t(self):
        '1D time series (conf01): the footer label ExpControl T wins over the positional x fallback'
        with OBFFile(os.path.join(DATA_DIR, 'conf01_continuous_point_measurement.msr')) as obf:
            da = obf.read_stack_xr(0)
        self.assertEqual(da.dims, ('t',))
        self.assertEqual(da.shape, (1024,))
        self.assertEqual(da['t'].attrs['imspector_label'], 'ExpControl T')
        self.assertEqual(da['t'].attrs['units'], 'dimensionless')

    def test_lightbox_hierarchical_groups(self):
        with OBFFile(os.path.join(DATA_DIR, '20261019tests_lightbox.obf')) as obf:
            self.assertEqual(len(obf.stack_groups), 16)
            da = obf.read_stack_group_xr(0)
        self.assertEqual(list(da['channel'].values), ['STAR RED', 'STAR ORANGE'])
        self.assertEqual(da.attrs['series'], 'Overview 1/14:01:41')
        # all footer labels are empty in this file: positional fallback for
        # x/y, dim_<n> for the unlabelled size-4 axis (file axis 4)
        self.assertEqual(da.dims, ('channel', 'dim_4', 'y', 'x'))
        self.assertEqual(da.shape, (2, 4, 400, 400))
        self.assertEqual(da['x'].attrs['units'], 'm')
        self.assertEqual(da['dim_4'].attrs['units'], 'dimensionless')


if __name__ == '__main__':
    unittest.main()
