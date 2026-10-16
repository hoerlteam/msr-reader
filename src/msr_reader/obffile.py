import re
import struct
import warnings
from collections import namedtuple
from io import BytesIO
from zlib import decompress

import numpy as np

MAXIMAL_NUMBER_OF_DIMENSIONS = 15
MAGIC_FILE_HEADER = b'OMAS_BF\n\xff\xff'
MAGIC_STACK_HEADER = b'OMAS_BF_STACK\n\xff\xff'

OBFHeader = namedtuple('OBFHeader', ['format_version', 'description', 'first_stack_position', 'metadata_position', 'metadata'])

class OBFStackHeader(namedtuple('OBFStackHeader',
    ['stack_version', 'size', 'length', 'offset', 'dtype', 'compressed', 'name',
     'description', 'data_position', 'data_length', 'next_stack_position'])):

    @property
    def numpy_dtype(self):
        '''
        numpy dtype (np.dtype instance, like arr.dtype) of the stack data. note
        that dtype holds the raw OMAS_DT flag from the file (including the
        complex bit and, for RGB types, the number of samples per pixel), which
        is not fully captured by a numpy dtype. raises ValueError for
        unsupported flags.
        '''
        return np.dtype(_parse_dtype(self.dtype)[0])

OBFSIFraction = namedtuple('OBFSIFraction', ['numerator', 'denominator'])
class OBFSIUnit(namedtuple('OBFSIUnit', ['meters', 'kilograms', 'seconds', 'amperes', 'kelvin', 'moles',
                            'candela', 'radian', 'steradian', 'scale_factor'])):
    '''
    an SI unit: the 9 base/supplementary units with rational exponents
    (OBFSIFraction each, numerator 0 = unused), multiplied by scale_factor,
    i.e. unit = scale_factor * m^p1 * kg^p2 * s^p3 * ...
    '''

    # symbols matching the exponent field order above
    _SYMBOLS = ('m', 'kg', 's', 'A', 'K', 'mol', 'cd', 'rad', 'sr')

    def __str__(self):
        parts = []
        for symbol, fraction in zip(self._SYMBOLS, self[:9]):
            if fraction.numerator == 0:
                continue
            if fraction.denominator == 1:
                parts.append(symbol if fraction.numerator == 1 else f'{symbol}^{fraction.numerator}')
            else:
                parts.append(f'{symbol}^({fraction.numerator}/{fraction.denominator})')
        unit = ' * '.join(parts) if parts else 'dimensionless'
        if self.scale_factor != 1:
            unit += f' * {self.scale_factor:g}'
        return unit

    # make notebooks show the pretty form, not the namedtuple field wall
    __repr__ = __str__

OBFStackFooter = namedtuple('OBFStackFooter', ['size', 'has_col_positions', 'has_col_labels', 'col_positions', 'col_labels', 'metadata', 'si_value', 'si_dimensions',
                                               'flush_positions', 'flush_block_size', 'tag_dictionary', 'stack_end_disk',
                                               'min_format_version', 'stack_end_used_disk', 'samples_written', 'num_chunk_positions',
                                               'dimension_labels'])

StackSizes = namedtuple('StackSizes', ['name', 'sizes', 'dimension_names'])

# pattern of default Imspector stack names: a user-settable channel name
# followed by the series number of the acquisition, e.g. 'STAR RED_CONF {0}'
STACK_NAME_PATTERN = re.compile(r'^(?P<base>.+) \{(?P<series>\d+)\}$')

StackGroup = namedtuple('StackGroup', ['series', 'channel_names', 'indices',
                                       'shape', 'dtype', 'pixel_sizes', 'dimension_names'])

def _read_file_header(fd):
    header_fmt = '<10sLQL'
    magic_header, format_version, first_stack_pos, descr_len = struct.unpack(header_fmt, fd.read(struct.calcsize(header_fmt)))

    if not magic_header == MAGIC_FILE_HEADER:
        raise ValueError('Can not parse file, no magic header found')

    descr = fd.read(descr_len).decode('utf-8')
    if format_version > 1:
        (meta_data_position, ) = struct.unpack('<Q', fd.read(struct.calcsize('<Q')))
        fd.seek(meta_data_position)
        metadata = _read_tag_dict(fd)
    else:
        meta_data_position, metadata = None, None
    
    return OBFHeader(format_version, descr, first_stack_pos, meta_data_position, metadata)
    

def _read_stack_header(fd, position):
    
    fd.seek(position)
    
    stack_header_fmt1 = '<16sLL'
    magic_header, version, rank = struct.unpack(stack_header_fmt1, fd.read(struct.calcsize(stack_header_fmt1)))

    if not magic_header == MAGIC_STACK_HEADER:
        raise ValueError('Can not parse stack, no magic header found')

    size = struct.unpack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}L', fd.read(struct.calcsize(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}L')))[:rank]
    length = struct.unpack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}d', fd.read(struct.calcsize(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}d')))[:rank]
    offset = struct.unpack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}d', fd.read(struct.calcsize(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}d')))[:rank]

    stack_header_fmt2 = '<LLLLLQQQ'
    data_type, compression_type, compression_level, name_len, descr_len, _, data_len, next_stack_pos = struct.unpack(stack_header_fmt2, fd.read(struct.calcsize(stack_header_fmt2)))
    
    name = fd.read(name_len).decode('utf-8')
    descr = fd.read(descr_len).decode('utf-8')

    data_pos = fd.tell()

    return OBFStackHeader(version, size, length, offset, data_type, compression_type>0, name, descr, data_pos, data_len, next_stack_pos)

def _parse_dtype(dtype_flags: int):

    # parse complex flag
    is_complex = (dtype_flags & 0x40000000) > 0
    if is_complex:
        dtype_flags ^= 0x40000000

    if dtype_flags == 0:
        raise ValueError('automatic data type determination is not supported yet')
    elif dtype_flags == 1:
        dtype, samples_per_pixel = np.uint8, 1
    elif dtype_flags == 2:
        dtype, samples_per_pixel = np.int8, 1
    elif dtype_flags == 4:
        dtype, samples_per_pixel = np.uint16, 1
    elif dtype_flags == 8:
        dtype, samples_per_pixel = np.int16, 1
    elif dtype_flags == 16:
        dtype, samples_per_pixel = np.uint32, 1
    elif dtype_flags == 32:
        dtype, samples_per_pixel = np.int32, 1
    elif dtype_flags == 64:
        dtype, samples_per_pixel = np.float32, 1
    elif dtype_flags == 128:
        dtype, samples_per_pixel = np.float64, 1
    elif dtype_flags == 0x00000400:
        dtype, samples_per_pixel = np.uint8, 3
    elif dtype_flags == 0x00000800:
        dtype, samples_per_pixel = np.uint8, 4
    elif dtype_flags == 0x00001000:
        dtype, samples_per_pixel = np.uint64, 1
    elif dtype_flags == 0x00002000:
        dtype, samples_per_pixel = np.int64, 1
    elif dtype_flags == 0x00010000:
        dtype, samples_per_pixel = np.bool_, 1
    else:
        raise ValueError(f'unknown data type with flag {dtype_flags}')

    # promote float/double to complex versions, error on other dtypes
    if is_complex and dtype == np.float32:
        dtype = np.complex64
    elif is_complex and dtype == np.float64:
        dtype = np.complex128
    elif is_complex:
        raise ValueError(f'complex data is only supported for float and double.')

    return dtype, samples_per_pixel

def _read_stack(fd, stack_header: OBFStackHeader, samples_written=None):
    fd.seek(stack_header.data_position)
    buffer = fd.read(stack_header.data_length)

    # zlib-decompress if necessary
    if stack_header.compressed:
        buffer = decompress(buffer)

    dtype, samples_per_pixel = _parse_dtype(stack_header.dtype)

    # a measurement ended early only writes samples_written samples (footer V6+),
    # by convention the missing data is zero (see file format spec), so pad short
    # buffers up to the full stack size (and drop any excess padding)
    num_samples = int(np.prod(stack_header.size)) * samples_per_pixel
    expected_bytes = num_samples * np.dtype(dtype).itemsize
    if len(buffer) < expected_bytes:
        buffer += b'\x00' * (expected_bytes - len(buffer))
    elif len(buffer) > expected_bytes:
        buffer = buffer[:expected_bytes]

    # samples_written == 0 means the full stack is on disk (see spec)
    if samples_written and samples_written < num_samples:
        warnings.warn(f'stack {stack_header.name!r} was truncated, only {samples_written} of '
                      f'{num_samples} samples were written, the remainder is zero-padded')

    return np.frombuffer(buffer, dtype=dtype).reshape(stack_header.size[::-1] + (samples_per_pixel, )).squeeze()

def _read_tag_dict(buffer):
    tag_dict = {}
    # Tag dictionary is of form key_length(uint32) key value_length(uint32) value
    # key, value are utf8-encoded strings
    # the end of the tag dictionary is indicated by a 0-valued uint32
    # TODO: check length (stored in footer) to make sure we do not run into infinite loops?
    while True:
        # read key length, if it is zero, we are at the end
        len_key, = struct.unpack('<L', buffer.read(4))
        if len_key == 0:
            break
        key = buffer.read(len_key)
        len_val, = struct.unpack('<L', buffer.read(4))
        val = buffer.read(len_val)
        tag_dict[key.decode('utf8')] = val.decode('utf8')
    return tag_dict

def _to_si_unit(unpack_output):
    # unpack output should be 9x2 int32 (numerator, denominator) pairs and one double scale factor
    unit_fractions = [OBFSIFraction(n, d) for n,d in zip(unpack_output[::2], unpack_output[1::2])]
    return OBFSIUnit(*unit_fractions, unpack_output[-1])
    

def _empty_stack_footer(stack_header: OBFStackHeader):
    # stack footers only exist from stack version 1 on. version-0 stacks carry no
    # footer, labels or tag dictionary on disk, so return empty/default values.
    # dimension_labels are padded to the stack rank so they stay aligned with
    # the stack axes (they are zipped together in OBFFile.shapes/pixel_sizes).
    return OBFStackFooter(
        size=None, has_col_positions=None, has_col_labels=None,
        col_positions=[None] * len(stack_header.size), col_labels=[None] * len(stack_header.size),
        metadata='',
        si_value=None, si_dimensions=None, flush_positions=[], flush_block_size=None,
        tag_dictionary={}, stack_end_disk=None, min_format_version=None,
        stack_end_used_disk=None, samples_written=None, num_chunk_positions=None,
        dimension_labels=[''] * len(stack_header.size))

def _read_stack_footer(fd, stack_header: OBFStackHeader):

    # version-0 stacks are written without any footer
    if stack_header.stack_version < 1:
        return _empty_stack_footer(stack_header)

    # seek to footer start (after binary data)
    fd.seek(stack_header.data_position + stack_header.data_length)

    # V1
    if stack_header.stack_version >= 1:
        (size,) = struct.unpack(f'<L', fd.read(struct.calcsize(f'<L')))
        has_col_info_fmt = f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}L'
        has_col_positions = struct.unpack(has_col_info_fmt, fd.read(struct.calcsize(has_col_info_fmt)))[:len(stack_header.size)]
        has_col_labels = struct.unpack(has_col_info_fmt, fd.read(struct.calcsize(has_col_info_fmt)))[:len(stack_header.size)]
        (metadata_length,) = struct.unpack(f'<L', fd.read(struct.calcsize(f'<L')))
    else:
        size, has_col_positions, has_col_labels, metadata_length = None, None, None, None

    # V2
    if stack_header.stack_version >= 2:
        si_fmt = '<18ld'
        si_value = _to_si_unit(struct.unpack(si_fmt, fd.read(struct.calcsize(si_fmt))))
        si_dimensions = []
        for _ in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
            si_dimensions.append(_to_si_unit(struct.unpack(si_fmt, fd.read(struct.calcsize(si_fmt)))))
    else:
        si_value, si_dimensions = None, None

    # V3
    if stack_header.stack_version >= 3:
        flush_fmt = '<QQ'
        num_flush_points, flush_block_size = struct.unpack(flush_fmt, fd.read(struct.calcsize(flush_fmt)))
    else:
        num_flush_points, flush_block_size = 0, None

    # V4
    if stack_header.stack_version >= 4:
        (tag_dictionary_length,) = struct.unpack('<Q', fd.read(struct.calcsize('<Q')))
    else:
        tag_dictionary_length = None

    # V5
    if stack_header.stack_version >= 5:
        stack_end_disk, min_format_version, stack_end_used_disk = struct.unpack('<QLQ', fd.read(struct.calcsize('<QLQ')))
    else:
        stack_end_disk, min_format_version, stack_end_used_disk = None, None, None

    # V6
    if stack_header.stack_version >= 6:
        samples_written, num_chunk_positions = struct.unpack('<QQ', fd.read(struct.calcsize('<QQ')))
    else:
        samples_written, num_chunk_positions = None, None

    # in any case: seek to end of footer
    fd.seek(stack_header.data_position + stack_header.data_length + size)

    # extra info after footer
    labels = []
    for _ in range(len(stack_header.size)):
        (n,) = struct.unpack(f'<L', fd.read(struct.calcsize(f'<L')))
        (s,) = struct.unpack(f'<{n}s', fd.read(struct.calcsize(f'<{n}s')))
        labels.append(s.decode('utf8'))

    if stack_header.stack_version >= 1:
        # TODO: handle chunked storage
        if num_chunk_positions is not None and num_chunk_positions > 0:
            raise ValueError('chunked storage not implemented yet')

        # column positions: for each axis with has_col_positions[i] != 0 an array
        # of res[i] float64 values follows.
        # entries are None for axes without positions.
        # NOTE: we assume all position arrays come before the column label arrays
        # TODO: needs testing on actual data!
        col_positions = []
        for i, has_pos in enumerate(has_col_positions):
            if has_pos:
                col_positions.append(np.frombuffer(fd.read(8 * stack_header.size[i]), dtype='<f8'))
            else:
                col_positions.append(None)

        # column labels: for each axis with has_col_labels[i] != 0 an array of
        # res[i] strings follows (same encoding as the dimension labels above).
        # entries are None for axes without column labels.
        col_labels = []
        for i, has_lab in enumerate(has_col_labels):
            if has_lab:
                labels_i = []
                for _ in range(stack_header.size[i]):
                    (n,) = struct.unpack(f'<L', fd.read(struct.calcsize(f'<L')))
                    labels_i.append(fd.read(n).decode('utf8'))
                col_labels.append(labels_i)
            else:
                col_labels.append(None)

        metadata = fd.read(metadata_length).decode('utf-8')
    else:
        metadata = ""

    # FLUSH POSITIONS (only >=V3, but since num_flush_points defaults to 0 this will do nothing)
    flush_positions = []
    for _ in range(num_flush_points):
        (fp,) = struct.unpack(f'<Q', fd.read(struct.calcsize(f'<Q')))
        flush_positions.append(fp)

    if stack_header.stack_version >= 4:
        tag_dict_buf = BytesIO(fd.read(tag_dictionary_length))
        tag_dict = _read_tag_dict(tag_dict_buf)
    else:
        tag_dict = {}

    return OBFStackFooter(size, has_col_positions, has_col_labels, col_positions, col_labels, metadata, si_value, si_dimensions, flush_positions,
                   flush_block_size, tag_dict, stack_end_disk, min_format_version, stack_end_used_disk,
                    samples_written, num_chunk_positions, labels)


class OBFFile:

    """
    Pure Python reader for OBF files (basis of Imspector MSR) as described at:
    https://imspectordocs.readthedocs.io/en/latest/fileformat.html
    """

    def __init__(self, path, group_unnamed_stacks=False):
        self.fd = open(path, 'rb')
        self._group_unnamed_stacks = group_unnamed_stacks
        self._parse_headers()
        self.stack_names = [s.name for s in self.stack_headers]
        self._stack_groups = None

    def _parse_headers(self):
        self.main_header = _read_file_header(self.fd)
        self.stack_headers = []
        self.stack_footers = []
        next_position = self.main_header.first_stack_position
        while next_position > 0:
            stack_header = _read_stack_header(self.fd, next_position)
            self.stack_headers.append(stack_header)
            next_position = stack_header.next_stack_position

            stack_footer = _read_stack_footer(self.fd, stack_header)
            self.stack_footers.append(stack_footer)

    def read_stack(self, idx):
        return _read_stack(self.fd, self.stack_headers[idx], self.stack_footers[idx].samples_written)
    
    @property
    def num_stacks(self):
        return len(self.stack_headers)

    @property
    def shapes(self):
        shapes = []
        for i in range(self.num_stacks):
            header_i : OBFStackHeader = self.stack_headers[i]
            footer_i : OBFStackFooter = self.stack_footers[i]

            sizes_stack = []
            dimension_labels = []
            for siz, label in zip(reversed(header_i.size), reversed(footer_i.dimension_labels)):
                if siz > 1:
                    sizes_stack.append(siz)
                    dimension_labels.append(label)
            shapes.append(StackSizes(header_i.name, sizes_stack, dimension_labels))
        return shapes
    

    @property
    def pixel_sizes(self):
        pixel_sizes = []
        for i in range(self.num_stacks):
            header_i : OBFStackHeader = self.stack_headers[i]
            footer_i : OBFStackFooter = self.stack_footers[i]

            sizes_stack = []
            dimension_labels = []
            for siz, length, label in zip(reversed(header_i.size), reversed(header_i.length), reversed(footer_i.dimension_labels)):
                if siz > 1:
                    sizes_stack.append(length / siz)
                    dimension_labels.append(label)
            pixel_sizes.append(StackSizes(header_i.name, sizes_stack, dimension_labels))
        return pixel_sizes

    def shape(self, stack_idx):
        """
        get size in pixels (array shape) of stack with index stack_idx
        """
        return self.shapes[stack_idx].sizes

    def pixel_size(self, stack_idx):
        """
        get pixel size (in meters) of stack with index stack_idx
        """
        return self.pixel_sizes[stack_idx].sizes

    def _group_key(self, stack_idx):
        # geometry part of the stack grouping key: shape, data type, pixel
        # sizes and dimension labels.
        header_i = self.stack_headers[stack_idx]
        shapes_i = self.shapes[stack_idx]
        pixel_sizes_i = self.pixel_sizes[stack_idx]
        # round pixel sizes to a fixed number of significant digits to make the
        # comparison robust against float noise: scientific notation rounds with
        # relative precision, unlike round()/np.round() which round to decimal
        # places (absolute precision) and would merge distinct pixel sizes
        # around the nanometer scale (or collapse tiny values onto each other)
        pixel_sizes_rounded = tuple(float(f'{p:.9e}') for p in pixel_sizes_i.sizes)
        return (tuple(shapes_i.sizes), header_i.dtype, pixel_sizes_rounded, tuple(shapes_i.dimension_names))

    @property
    def stack_groups(self):
        """
        stacks grouped into channels: stacks acquired in the same series
        (the number in curly braces in the stack name) with identical geometry
        (shape, data type, pixel sizes, dimension labels) are considered
        channels of the same acquisition. stacks that fit nowhere (e.g.
        auxiliary stacks like rescue_info maps, or stacks without a parseable
        name) form single-member groups. every stack is part of exactly one
        group, so the groups are always a partition of all stacks.
        """
        if self._stack_groups is None:
            self._stack_groups = self._compute_stack_groups()
        return self._stack_groups

    def _compute_stack_groups(self):
        groups = []
        key_to_group = {}
        for idx in range(self.num_stacks):
            name = self.stack_names[idx]

            # auxiliary data like ResCue switch-off maps is never treated as a channel
            if 'rescue_info' in name:
                key = None
            else:
                match = STACK_NAME_PATTERN.match(name)
                if match is not None:
                    key = (int(match.group('series')),) + self._group_key(idx)
                elif name == '' and self._group_unnamed_stacks:
                    # opt-in: group unnamed stacks by geometry alone (series unknown)
                    key = (None,) + self._group_key(idx)
                else:
                    key = None

            if key is None:
                groups.append([idx])
            elif key in key_to_group:
                groups[key_to_group[key]].append(idx)
            else:
                key_to_group[key] = len(groups)
                groups.append([idx])

        return [self._make_stack_group(indices) for indices in groups]

    def _make_stack_group(self, indices):
        header_i = self.stack_headers[indices[0]]
        shapes_i = self.shapes[indices[0]]
        pixel_sizes_i = self.pixel_sizes[indices[0]]

        series = None
        channel_names = []
        for idx in indices:
            name = self.stack_names[idx]
            match = STACK_NAME_PATTERN.match(name)
            if match is not None:
                series = int(match.group('series'))
                channel_names.append(match.group('base'))
            elif name:
                channel_names.append(name)
            else:
                channel_names.append(f'stack {idx}')

        try:
            dtype = np.dtype(_parse_dtype(header_i.dtype)[0])
        except ValueError:
            dtype = None

        return StackGroup(series, channel_names, indices, tuple(shapes_i.sizes), dtype,
                          list(pixel_sizes_i.sizes), list(shapes_i.dimension_names))

    def read_stack_group(self, group_idx, axis=0):
        """
        read all stacks of a group, stacked along a new axis (default: first,
        i.e. (C, Z, Y, X)). single-member groups get an axis of length one.
        """
        group = self.stack_groups[group_idx]
        return np.stack([self.read_stack(i) for i in group.indices], axis=axis)

    def get_imspector_xml_metadata(self, stack_idx):
        """
        get XML metadata string of stack with index stack_idx
        """

        # get footer of stack_idx
        stack_footer = self.stack_footers[stack_idx] 

        # check if we actually have metadata for our stack (some like rescue_info don't seem to have it)
        # return empty string if no metadata is present
        if 'imspector' in stack_footer.tag_dictionary:
            xml_imspector_metadata = stack_footer.tag_dictionary['imspector']
        else:
            xml_imspector_metadata = ''

        return xml_imspector_metadata

    def get_ome_xml_metadata(self):
        """
        get OME-XML metadata string
        """

        # check that we have metadata (will be available through main header,
        # but only from file format version 2 on)
        if self.main_header.metadata and 'ome_xml' in self.main_header.metadata:
            ome_xml_metadata = self.main_header.metadata['ome_xml']
        else:
            ome_xml_metadata = ''

        return ome_xml_metadata

    def close(self):
        self.fd.close()

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()