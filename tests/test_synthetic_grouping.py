'''
Synthetic tests for the automatic grouping of stacks into channels.

Grouping relies on Imspector naming behavior (the series number in curly
braces, the rescue_info convention), which is not part of the file format
specification: the synthetic files here pin our assumptions about it. The
grouping rules themselves (same series + identical geometry) were validated
against real multi-channel acquisitions.
'''
import struct
import unittest

import numpy as np

from .synthetic_obf import StackSpec, build_obf, obf_file

INT16 = 0x8
UINT8 = 0x1


def _grouping_file():
    'stacks: two channels of series 0, one of series 1, a rescue mask, two unnamed, one odd shape'
    specs = [
        StackSpec('Chan A {0}', (2, 3), INT16, struct.pack('<6h', *range(6)), lengths=(2.0, 3.0)),
        StackSpec('Chan B {0}', (2, 3), INT16, struct.pack('<6h', *range(6, 12)), lengths=(2.0, 3.0)),
        StackSpec('Chan A {1}', (2, 3), INT16, struct.pack('<6h', *range(6)), lengths=(2.0, 3.0)),
        StackSpec('Chan A {0}_rescue_info', (2, 3), UINT8, bytes(6)),
        StackSpec('', (2, 3), INT16, struct.pack('<6h', *range(6)), lengths=(2.0, 3.0)),
        StackSpec('', (2, 3), INT16, struct.pack('<6h', *range(6)), lengths=(2.0, 3.0)),
        StackSpec('Chan D {0}', (2, 4), INT16, struct.pack('<8h', *range(8)), lengths=(2.0, 4.0)),
    ]
    return build_obf(specs)


def _hierarchical_file():
    'Lightbox-style names: path/channel, same channel and geometry at different paths, a rescue mask'
    specs = [
        StackSpec('Overview 1/14:01:41/STAR RED', (2, 3), INT16, struct.pack('<6h', *range(6)), lengths=(2.0, 3.0)),
        StackSpec('Overview 1/14:01:41/STAR ORANGE', (2, 3), INT16, struct.pack('<6h', *range(6, 12)), lengths=(2.0, 3.0)),
        StackSpec('Overview 1/14:02:03/STAR RED', (2, 3), INT16, struct.pack('<6h', *range(6)), lengths=(2.0, 3.0)),
        StackSpec('Overview 1/Image 1/14:03:01/STAR RED', (2, 3), INT16, struct.pack('<6h', *range(6)), lengths=(2.0, 3.0)),
        StackSpec('Overview 1/Image 1/14:03:01/STAR ORANGE', (2, 3), INT16, struct.pack('<6h', *range(6, 12)), lengths=(2.0, 3.0)),
        StackSpec('Overview 1/14:01:41/STAR RED_rescue_info', (2, 3), UINT8, bytes(6)),
    ]
    return build_obf(specs)


def _hierarchical_groups():
    return [
        # path up to the last component identifies the acquisition
        ('Overview 1/14:01:41', ['STAR RED', 'STAR ORANGE'], [0, 1]),
        # same channel and geometry, different path: separate acquisition
        ('Overview 1/14:02:03', ['STAR RED'], [2]),
        ('Overview 1/Image 1/14:03:01', ['STAR RED', 'STAR ORANGE'], [3, 4]),
        # auxiliary stacks never group, even with hierarchical names
        (None, ['Overview 1/14:01:41/STAR RED_rescue_info'], [5]),
    ]


class TestStackGroups(unittest.TestCase):

    def test_grouping_by_series_and_geometry(self):
        with obf_file(_grouping_file()) as obf:
            groups = [(g.series, g.channel_names, g.indices) for g in obf.stack_groups]
        expected = [
            (0, ['Chan A', 'Chan B'], [0, 1]),   # same series, same geometry
            (1, ['Chan A'], [2]),                # same geometry, different series
            (None, ['Chan A {0}_rescue_info'], [3]),  # auxiliary stacks never group
            (None, ['stack 4'], [4]),            # unnamed stacks stay singletons
            (None, ['stack 5'], [5]),
            (0, ['Chan D'], [6]),                # same series, different geometry
        ]
        self.assertEqual(groups, expected)

    def test_group_unnamed_stacks_option(self):
        with obf_file(_grouping_file(), group_unnamed_stacks=True) as obf:
            groups = [(g.series, g.channel_names, g.indices) for g in obf.stack_groups]
        expected = [
            (0, ['Chan A', 'Chan B'], [0, 1]),
            (1, ['Chan A'], [2]),
            (None, ['Chan A {0}_rescue_info'], [3]),
            (None, ['stack 4', 'stack 5'], [4, 5]),  # grouped by geometry alone
            (0, ['Chan D'], [6]),
        ]
        self.assertEqual(groups, expected)

    def test_groups_partition_all_stacks(self):
        for kwargs in ({}, {'group_unnamed_stacks': True}):
            with obf_file(_grouping_file(), **kwargs) as obf:
                seen = [i for g in obf.stack_groups for i in g.indices]
                self.assertEqual(sorted(seen), list(range(obf.num_stacks)))

    def test_hierarchical_names_stay_singletons_in_msr_by_default(self):
        '.msr files (Imspector): hierarchical names are not grouped by default'
        with obf_file(_hierarchical_file(), filename='synthetic.msr') as obf:
            groups = [(g.series, g.channel_names, g.indices) for g in obf.stack_groups]
        expected = [
            (None, ['Overview 1/14:01:41/STAR RED'], [0]),
            (None, ['Overview 1/14:01:41/STAR ORANGE'], [1]),
            (None, ['Overview 1/14:02:03/STAR RED'], [2]),
            (None, ['Overview 1/Image 1/14:03:01/STAR RED'], [3]),
            (None, ['Overview 1/Image 1/14:03:01/STAR ORANGE'], [4]),
            (None, ['Overview 1/14:01:41/STAR RED_rescue_info'], [5]),
        ]
        self.assertEqual(groups, expected)

    def test_hierarchical_names_grouped_in_obf_by_default(self):
        '.obf files (Lightbox): hierarchical names are grouped by default'
        with obf_file(_hierarchical_file(), filename='synthetic.obf') as obf:
            groups = [(g.series, g.channel_names, g.indices) for g in obf.stack_groups]
        self.assertEqual(groups, _hierarchical_groups())

    def test_group_hierarchical_stacks_option(self):
        with obf_file(_hierarchical_file(), filename='synthetic.msr',
                       group_hierarchical_stacks=True) as obf:
            groups = [(g.series, g.channel_names, g.indices) for g in obf.stack_groups]
        self.assertEqual(groups, _hierarchical_groups())

        with obf_file(_hierarchical_file(), filename='synthetic.msr',
                       group_hierarchical_stacks=True) as obf:
            seen = [i for g in obf.stack_groups for i in g.indices]
            self.assertEqual(sorted(seen), list(range(obf.num_stacks)))
            data = obf.read_stack_group(0)
            self.assertEqual(data.shape, (2, 3, 2))  # (C, Y, X)
            self.assertEqual(data[0].flatten().tolist(), [0, 1, 2, 3, 4, 5])
            self.assertEqual(data[1].flatten().tolist(), [6, 7, 8, 9, 10, 11])

    def test_group_hierarchical_stacks_overrides_extension(self):
        'explicit True/False wins over the extension-based default'
        with obf_file(_hierarchical_file(), filename='synthetic.obf',
                       group_hierarchical_stacks=False) as obf:
            self.assertEqual(len(obf.stack_groups), 6)
        with obf_file(_hierarchical_file(), filename='synthetic.msr',
                       group_hierarchical_stacks=True) as obf:
            self.assertEqual(len(obf.stack_groups), 4)

    def test_read_stack_group_stacks_along_first_axis(self):
        with obf_file(_grouping_file()) as obf:
            data = obf.read_stack_group(0)
            self.assertEqual(data.shape, (2, 3, 2))  # (C, Y, X)
            self.assertEqual(data[0].flatten().tolist(), [0, 1, 2, 3, 4, 5])
            self.assertEqual(data[1].flatten().tolist(), [6, 7, 8, 9, 10, 11])
            # single-member groups get an axis of length one
            self.assertEqual(obf.read_stack_group(1).shape, (1, 3, 2))

    def test_group_metadata_matches_members(self):
        with obf_file(_grouping_file()) as obf:
            group = obf.stack_groups[0]
            self.assertEqual(group.shape, (3, 2))
            # dtype is a np.dtype instance (like arr.dtype), not the scalar class
            self.assertEqual(group.dtype, np.dtype('int16'))
            self.assertEqual(group.dtype.name, 'int16')
            self.assertEqual(group.series, 0)


if __name__ == '__main__':
    unittest.main()
