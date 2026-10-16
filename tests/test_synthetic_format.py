'''
Synthetic tests for the OBF file format parsing.

The files here are built by tests/synthetic_obf.py from our reading of the
file format specification, not from actual Imspector output: they pin that
interpretation and protect against regressions, but can not validate it.
Paths exercised only by these tests (version-0 stacks, truncation, column
positions/labels) have no real-world test data yet.
'''
import struct
import unittest
import warnings

import numpy as np

from msr_reader import OBFSIFraction, OBFSIUnit
from msr_reader.obffile import _parse_dtype
from .synthetic_obf import StackSpec, build_obf, obf_file

INT16 = 0x8  # OMAS_DT_SINT16


class TestSIUnitFormatting(unittest.TestCase):

    def _unit(self, exponents=(), scale_factor=1.0):
        names = ('meters', 'kilograms', 'seconds', 'amperes', 'kelvin',
                 'moles', 'candela', 'radian', 'steradian')
        fractions = [OBFSIFraction(0, 1)] * 9
        for name, numerator, denominator in exponents:
            fractions[names.index(name)] = OBFSIFraction(numerator, denominator)
        return OBFSIUnit(*fractions, scale_factor)

    def test_common_units(self):
        self.assertEqual(str(self._unit()), 'dimensionless')
        self.assertEqual(str(self._unit([('meters', 1, 1)])), 'm')
        self.assertEqual(str(self._unit([('meters', 1, 1), ('seconds', -1, 1)])), 'm * s^-1')
        self.assertEqual(str(self._unit([('meters', 2, 1), ('kilograms', 1, 1),
                                        ('seconds', -3, 1), ('amperes', -1, 1)])),
                         'm^2 * kg * s^-3 * A^-1')  # volt

    def test_fractional_exponents_and_scale(self):
        # sqrt(Hz) style noise spectral densities are why exponents are fractions
        self.assertEqual(str(self._unit([('seconds', -1, 2)])), 's^(-1/2)')
        # scale factor is only printed when it is not 1
        self.assertEqual(str(self._unit([('meters', 1, 1)], scale_factor=1e-9)), 'm * 1e-09')
        self.assertEqual(str(self._unit(scale_factor=1e-9)), 'dimensionless * 1e-09')

    def test_repr_matches_str(self):
        # notebooks display repr, both should show the pretty form
        unit = self._unit([('meters', 1, 1)])
        self.assertEqual(repr(unit), 'm')


class TestDtypeParsing(unittest.TestCase):

    def test_scalar_dtypes(self):
        expected = {
            0x1: (np.uint8, 1), 0x2: (np.int8, 1), 0x4: (np.uint16, 1),
            0x8: (np.int16, 1), 0x10: (np.uint32, 1), 0x20: (np.int32, 1),
            0x40: (np.float32, 1), 0x80: (np.float64, 1),
            0x1000: (np.uint64, 1), 0x2000: (np.int64, 1),
            0x10000: (np.bool_, 1),
        }
        for flags, (dtype, samples_per_pixel) in expected.items():
            self.assertEqual(_parse_dtype(flags), (dtype, samples_per_pixel))

    def test_rgb_dtypes(self):
        self.assertEqual(_parse_dtype(0x400), (np.uint8, 3))
        self.assertEqual(_parse_dtype(0x800), (np.uint8, 4))

    def test_complex_dtypes(self):
        self.assertEqual(_parse_dtype(0x40 | 0x40000000), (np.complex64, 1))
        self.assertEqual(_parse_dtype(0x80 | 0x40000000), (np.complex128, 1))

    def test_unsupported_dtypes(self):
        for flags in (0x0, 0x200, 0x8 | 0x40000000):
            with self.assertRaises(ValueError):
                _parse_dtype(flags)


class TestFileHeader(unittest.TestCase):

    def test_format_version_1_has_no_metadata(self):
        blob = build_obf([StackSpec('a {0}', (4,), INT16, struct.pack('<4h', 0, 1, 2, 3))],
                         file_format_version=1)
        with obf_file(blob) as obf:
            self.assertEqual(obf.main_header.format_version, 1)
            self.assertIsNone(obf.main_header.metadata_position)
            self.assertIsNone(obf.main_header.metadata)
            # used to raise TypeError when metadata was None
            self.assertEqual(obf.get_ome_xml_metadata(), '')

    def test_format_version_2_file_metadata(self):
        blob = build_obf([StackSpec('a {0}', (4,), INT16, struct.pack('<4h', 0, 1, 2, 3))],
                         file_metadata={'ome_xml': '<ome/>'})
        with obf_file(blob) as obf:
            self.assertEqual(obf.main_header.format_version, 2)
            self.assertEqual(obf.get_ome_xml_metadata(), '<ome/>')


class TestStackFooterVersions(unittest.TestCase):

    def test_version_0_stack_has_no_footer(self):
        # version-0 stacks are written without footer and variable section
        blob = build_obf([StackSpec('v0', (4,), INT16, struct.pack('<4h', 0, 1, 2, 3),
                                    stack_version=0)],
                         file_format_version=1)
        with obf_file(blob) as obf:
            self.assertEqual(obf.num_stacks, 1)
            footer = obf.stack_footers[0]
            self.assertIsNone(footer.size)
            self.assertEqual(footer.col_positions, [None])
            self.assertEqual(footer.col_labels, [None])
            self.assertEqual(footer.tag_dictionary, {})
            self.assertEqual(footer.dimension_labels, [''])
            self.assertEqual(obf.read_stack(0).tolist(), [0, 1, 2, 3])
            self.assertEqual(obf.shapes[0].sizes, [4])

    def test_higher_footer_version_is_skipped_via_size(self):
        # a footer version newer than the reader implements must be skipped
        # via its size member (footer V6 fields + 60 unknown bytes = V7 layout)
        blob = build_obf([StackSpec('v7 {0}', (3, 4), INT16, struct.pack('<12h', *range(12)),
                                    stack_version=7, extra_footer_bytes=60,
                                    tag_dict={'imspector': '<x/>'})])
        with obf_file(blob) as obf:
            footer = obf.stack_footers[0]
            self.assertEqual(footer.size, 1528)
            self.assertEqual(footer.samples_written, 12)
            self.assertEqual(footer.num_chunk_positions, 0)
            self.assertEqual(footer.tag_dictionary, {'imspector': '<x/>'})
            self.assertEqual(obf.read_stack(0).flatten().tolist(), list(range(12)))


class TestColumnPositionsAndLabels(unittest.TestCase):
    'no real-world test data for these exists yet, ordering follows the spec text'

    def test_positions_and_labels_are_parsed(self):
        blob = build_obf([StackSpec(
            'cols {0}', (3, 4), INT16, struct.pack('<12h', *range(12)),
            lengths=(3.0, 8.0), dimension_labels=['x', 'y'],
            col_positions=[[0.0, 0.5, 1.0], [10.0, 20.0, 30.0, 40.0]],
            col_labels=[['a', 'bb', 'ccc'], None],
            metadata='<meta/>', tag_dict={'imspector': '<x/>'})])
        with obf_file(blob) as obf:
            footer = obf.stack_footers[0]
            # all position arrays come before the label arrays (spec text!)
            self.assertEqual(footer.col_positions[0].tolist(), [0.0, 0.5, 1.0])
            self.assertEqual(footer.col_positions[1].tolist(), [10.0, 20.0, 30.0, 40.0])
            self.assertEqual(footer.col_labels[0], ['a', 'bb', 'ccc'])
            self.assertIsNone(footer.col_labels[1])
            # the variable section is still aligned after skipping col data
            self.assertEqual(footer.metadata, '<meta/>')
            self.assertEqual(footer.tag_dictionary, {'imspector': '<x/>'})
            self.assertEqual(footer.dimension_labels, ['x', 'y'])
            self.assertEqual(obf.read_stack(0).shape, (4, 3))


class TestTruncatedStacks(unittest.TestCase):
    'no real-world truncated test file exists yet, semantics from the spec'

    SIZE = (3, 4)  # 12 int16 samples, 24 bytes

    def _truncated_spec(self, data, samples_written, compression=False):
        return StackSpec('trunc {0}', self.SIZE, INT16, data,
                         compression=compression, samples_written=samples_written)

    def test_truncated_stack_is_zero_padded(self):
        blob = build_obf([self._truncated_spec(struct.pack('<6h', *range(6)), 6)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with obf_file(blob) as obf:
                data = obf.read_stack(0)
        self.assertEqual(data.flatten().tolist(), [0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0])
        self.assertTrue(any('truncated' in str(w.message) for w in caught))

    def test_truncated_compressed_stack_is_zero_padded(self):
        # a zlib stream can only contain what was actually compressed, so a
        # truncated compressed stack decompresses to fewer samples. StackSpec
        # takes uncompressed data and compresses it on write
        blob = build_obf([self._truncated_spec(struct.pack('<6h', *range(6)), 6,
                                               compression=True)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with obf_file(blob) as obf:
                data = obf.read_stack(0)
        self.assertEqual(data.flatten().tolist(), [0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0])
        self.assertTrue(any('truncated' in str(w.message) for w in caught))

    def test_samples_written_zero_means_full(self):
        blob = build_obf([self._truncated_spec(struct.pack('<12h', *range(12)), 0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with obf_file(blob) as obf:
                data = obf.read_stack(0)
        self.assertEqual(data.flatten().tolist(), list(range(12)))
        self.assertEqual(caught, [])


class TestSamplesPerPixel(unittest.TestCase):

    def test_rgb_stack_shape(self):
        # OMAS_DT_RGB: uint8, 3 samples per pixel, appended as a trailing axis
        blob = build_obf([StackSpec('rgb {0}', (2, 3), 0x400, bytes(range(18)),
                                    samples_written=18)])
        with obf_file(blob) as obf:
            data = obf.read_stack(0)
            self.assertEqual(data.shape, (3, 2, 3))
            self.assertEqual(data.dtype, np.uint8)


class TestDataLessStack(unittest.TestCase):
    'reserved == 0 marks a stack without data on disk, no real-world sample exists'

    def test_data_less_stack_reads_as_zeros_with_warning(self):
        blob = build_obf([StackSpec('empty {0}', (3, 4), INT16, b'', reserved=0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with obf_file(blob) as obf:
                data = obf.read_stack(0)
        self.assertEqual(data.shape, (4, 3))
        self.assertEqual(data.tolist(), [[0] * 3] * 4)
        self.assertTrue(any('data-less' in str(w.message) for w in caught))

    def test_data_carrying_stack_does_not_warn(self):
        # real files carry reserved = 1, reading them must stay warning-free
        blob = build_obf([StackSpec('full {0}', (3, 4), INT16, struct.pack('<12h', *range(12)))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with obf_file(blob) as obf:
                data = obf.read_stack(0)
        self.assertEqual(data.flatten().tolist(), list(range(12)))
        self.assertEqual(caught, [])


if __name__ == '__main__':
    unittest.main()
