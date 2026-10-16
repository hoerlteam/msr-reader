'''
Tests against real Imspector files (tests/data/).

The data files are a small fixed corpus of actual acquisitions; expectations
here were validated by hand during development and pin current reader
behavior against real writer output. The corpus is expected to grow with
purpose-acquired samples (chunked storage, column positions, truncated
measurements) as they become available.
'''
import os
import unittest

import numpy as np

from msr_reader import OBFFile, OBFSIUnit, StackGroup, StackSizes

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _path(name):
    return os.path.join(DATA_DIR, name)


class TestOldFiles(unittest.TestCase):
    'format version 1 files with version 5 stack footers (2015 era)'

    def test_conf01(self):
        with OBFFile(_path('conf01.msr')) as obf:
            self.assertEqual(obf.main_header.format_version, 1)
            self.assertEqual(obf.get_ome_xml_metadata(), '')

            self.assertEqual(obf.num_stacks, 1)
            header, footer = obf.stack_headers[0], obf.stack_footers[0]
            self.assertEqual(header.stack_version, 5)
            self.assertTrue(header.compressed)
            self.assertEqual(header.name, '')
            self.assertEqual(obf.shape(0), [1024])

            # unnamed 1D stack: single-member group with a placeholder name
            groups = obf.stack_groups
            self.assertEqual(len(groups), 1)
            self.assertIsNone(groups[0].series)
            self.assertEqual(groups[0].channel_names, ['stack 0'])
            self.assertIsInstance(obf.get_imspector_xml_metadata(0), str)
            self.assertNotEqual(obf.get_imspector_xml_metadata(0), '')

            data = obf.read_stack(0)
            self.assertEqual(data.dtype, np.int16)
            self.assertEqual(data.shape, (1024,))

    def test_groupa_cell01(self):
        with OBFFile(_path('groupa_cell01.msr')) as obf:
            self.assertEqual(obf.main_header.format_version, 1)
            self.assertEqual(obf.num_stacks, 7)
            self.assertTrue(all(h.stack_version == 5 and h.compressed for h in obf.stack_headers))

            # four identical images, two short line profiles, one long one
            shapes = [obf.shape(i) for i in range(7)]
            self.assertEqual(shapes, [[490, 450]] * 4 + [[18], [18], [665]])
            self.assertEqual([h.numpy_dtype for h in obf.stack_headers],
                             [np.int16] * 4 + [np.float64] * 3)

            # unnamed stacks stay singletons by default
            self.assertEqual(len(obf.stack_groups), 7)

            for i in range(7):
                obf.read_stack(i)

    def test_groupa_cell01_unnamed_grouping(self):
        with OBFFile(_path('groupa_cell01.msr'), group_unnamed_stacks=True) as obf:
            groups = [(g.series, g.channel_names, g.indices) for g in obf.stack_groups]
            self.assertEqual(groups, [
                (None, ['stack 0', 'stack 1', 'stack 2', 'stack 3'], [0, 1, 2, 3]),
                (None, ['stack 4', 'stack 5'], [4, 5]),
                (None, ['stack 6'], [6]),
            ])
            data = obf.read_stack_group(0)
            self.assertEqual(data.shape, (4, 490, 450))


class TestModernFiles(unittest.TestCase):
    'format version 2 files with version 6/7 stack footers'

    def test_overview_1(self):
        with OBFFile(_path('2889b245416cd9783c627f413caa45c9_overview_1.msr')) as obf:
            self.assertEqual(obf.main_header.format_version, 2)
            self.assertNotEqual(obf.get_ome_xml_metadata(), '')

            header, footer = obf.stack_headers[0], obf.stack_footers[0]
            self.assertEqual(header.name, 'STAR RED_CONF {0}')
            self.assertEqual(header.stack_version, 6)
            self.assertTrue(header.compressed)
            self.assertEqual(footer.samples_written, 40 * 333 * 333)
            self.assertEqual(obf.shape(0), [40, 333, 333])

            # SI axis units of the modern file: meters along the spatial axes
            self.assertIsInstance(footer.si_value, OBFSIUnit)
            for dim in ('ExpControl X', 'ExpControl Y', 'ExpControl Z'):
                axis_unit = footer.si_dimensions[footer.dimension_labels.index(dim)]
                self.assertEqual(axis_unit.meters.numerator, 1)
            self.assertEqual(footer.si_dimensions[0].scale_factor, 1.0)
            self.assertEqual(str(footer.si_dimensions[0]), 'm')

            # pixel sizes are SI meters: 250 nm z-step, 150 nm lateral
            np.testing.assert_allclose(obf.pixel_size(0), [2.5e-7, 1.5e-7, 1.5e-7], rtol=1e-6)

            groups = obf.stack_groups
            self.assertEqual([(g.series, g.channel_names) for g in groups], [(0, ['STAR RED_CONF'])])
            self.assertEqual(obf.read_stack_group(0).shape, (1, 40, 333, 333))

    def test_mesc_yoyo2(self):
        with OBFFile(_path('mesc_yoyo2_img_001.msr')) as obf:
            self.assertEqual(obf.num_stacks, 2)
            self.assertEqual([h.name for h in obf.stack_headers], ['gfp {0}', 'gfp {1}'])
            self.assertEqual([obf.shape(i) for i in range(2)], [[250, 250], [50, 235, 270]])

            # same channel name, but different series: never merged
            groups = [(g.series, g.channel_names) for g in obf.stack_groups]
            self.assertEqual(groups, [(0, ['gfp']), (1, ['gfp'])])

            np.testing.assert_allclose(obf.pixel_size(0), [2.0e-7, 2.0e-7], rtol=1e-6)
            np.testing.assert_allclose(obf.pixel_size(1), [3.0e-7, 1.0e-7, 1.0e-7], rtol=1e-6)

    def test_bad44489_field_0(self):
        'two channels in one series, version 7 stack footers (beyond the spec)'
        with OBFFile(_path('bad44489_field_0.msr')) as obf:
            self.assertEqual([h.name for h in obf.stack_headers], ['594 {0}', '640 {0}'])

            for footer in obf.stack_footers:
                self.assertEqual(footer.size, 1528)          # v7 layout: v6 fields + 60 unknown bytes
                self.assertEqual(footer.samples_written, 28 * 333 * 333)
                # real files have min_format_version 0, not the 1 the spec claims
                self.assertEqual(footer.min_format_version, 0)

            groups = obf.stack_groups
            self.assertEqual([(g.series, g.channel_names, g.indices) for g in groups],
                             [(0, ['594', '640'], [0, 1])])
            data = obf.read_stack_group(0)
            self.assertEqual(data.shape, (2, 28, 333, 333))
            self.assertEqual(data.dtype, np.int16)

    def test_bad44489_sted(self):
        with OBFFile(_path('bad44489_field_0_sted_0.msr')) as obf:
            groups = obf.stack_groups
            self.assertEqual([(g.series, g.channel_names) for g in groups],
                             [(0, ['ATTO 594_STED', 'STAR 635P_STED'])])
            self.assertEqual(obf.read_stack_group(0).shape, (2, 17, 110, 120))


if __name__ == '__main__':
    unittest.main()
