import struct
from collections import namedtuple
from zlib import decompress

import numpy as np

MAXIMAL_NUMBER_OF_DIMENSIONS = 15
MAGIC_FILE_HEADER = b'OMAS_BF\n\xff\xff'
MAGIC_STACK_HEADER = b'OMAS_BF_STACK\n\xff\xff'

OBFHeader = namedtuple('OBFHeader', ['format_version', 'description', 'first_stack_position', 'metadata_position', 'metadata'])

OBFStackHeader = namedtuple('OBFStackHeader',
    ['stack_version', 'size', 'length', 'offset', 'dtype', 'compressed', 'name',
     'description', 'data_position', 'data_length', 'next_stack_position'])

OBFSIFraction = namedtuple('OBFSIFraction', ['numerator', 'denominator'])
OBFSIUnit = namedtuple('OBFSIUnit', ['meters', 'kilograms', 'seconds', 'amperes', 'kelvin', 'moles',
                            'candela', 'radian', 'steradian', 'scale_factor'])

OBFStackFooter = namedtuple('OBFStackFooter', ['size', 'has_col_positions', 'has_col_labels', 'metadata', 'si_value', 'si_dimensions',
                                               'flush_positions', 'flush_block_size', 'tag_dictionary', 'stack_end_disk',
                                               'min_format_version', 'stack_end_used_disk', 'samples_written', 'num_chunk_positions',
                                               'dimension_labels'])

StackSizes = namedtuple('StackSizes', ['name', 'sizes', 'dimension_names'])

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
    
    if version < 5:
        raise ValueError('stack versions <5 not implemented yet')
    
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

def _read_stack(fd, stack_header: OBFStackHeader):
    fd.seek(stack_header.data_position)
    buffer = fd.read(stack_header.data_length)

    # zlib-decompress if necessary
    if stack_header.compressed:
        buffer = decompress(buffer)

    dtype, samples_per_pixel = _parse_dtype(stack_header.dtype)

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
    

def _read_stack_footer(fd, stack_header: OBFStackHeader):

    fd.seek(stack_header.data_position + stack_header.data_length)

    (size,) = struct.unpack(f'<L', fd.read(struct.calcsize(f'<L')))

    has_col_info_fmt = f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}L'
    has_col_positions = struct.unpack(has_col_info_fmt, fd.read(struct.calcsize(has_col_info_fmt)))[:len(stack_header.size)]
    has_col_labels = struct.unpack(has_col_info_fmt, fd.read(struct.calcsize(has_col_info_fmt)))[:len(stack_header.size)]

    (metadata_length,) = struct.unpack(f'<L', fd.read(struct.calcsize(f'<L')))

    si_fmt = '<18ld'
    si_value = _to_si_unit(struct.unpack(si_fmt, fd.read(struct.calcsize(si_fmt))))
    si_dimensions = []
    for _ in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
        si_dimensions.append(_to_si_unit(struct.unpack(si_fmt, fd.read(struct.calcsize(si_fmt)))))

    if stack_header.stack_version > 5:
        stack_footer_fmt = '<QQQQLQQQ'
        unpack_out = struct.unpack(stack_footer_fmt, fd.read(struct.calcsize(stack_footer_fmt)))
        (num_flush_points, flush_block_size, tag_dictionary_length,
          stack_end_disk, min_format_version, stack_end_used_disk, samples_written, num_chunk_positions) = unpack_out
    else:
        stack_footer_fmt = '<QQQQL'
        unpack_out = struct.unpack(stack_footer_fmt, fd.read(struct.calcsize(stack_footer_fmt)))
        (num_flush_points, flush_block_size, tag_dictionary_length,
          stack_end_disk, min_format_version, stack_end_used_disk, samples_written, num_chunk_positions) = unpack_out + (None, None, None)

    labels = []
    for _ in range(len(stack_header.size)):
        (n,) = struct.unpack(f'<L', fd.read(struct.calcsize(f'<L')))
        (s,) = struct.unpack(f'<{n}s', fd.read(struct.calcsize(f'<{n}s')))
        labels.append(s.decode('utf8'))

    # TODO: handle col_positions, col_labels, chunked storage
    if any(has_col_labels) or any(has_col_positions):
        raise ValueError('col_label and col_position parsing not implemented yet')
    if num_chunk_positions is not None and num_chunk_positions > 0:
        raise ValueError('chunked storage not implemented yet')
    
    metadata = fd.read(metadata_length).decode('utf-8')

    # FLUSH POSITIONS
    flush_positions = []
    for _ in range(num_flush_points):
        (fp,) = struct.unpack(f'<Q', fd.read(struct.calcsize(f'<Q')))
        flush_positions.append(fp)

    # tag_dict_buf = BytesIO(fd.read(tag_dictionary_length))
    tag_dict = _read_tag_dict(fd)    
    

    return OBFStackFooter(size, has_col_positions, has_col_labels, metadata, si_value, si_dimensions, flush_positions,
                   flush_block_size, tag_dict, stack_end_disk, min_format_version, stack_end_used_disk,
                    samples_written, num_chunk_positions, labels)


class OBFFile:

    """
    Pure Python reader for OBF files (basis of Imspector MSR) as described at:
    https://imspectordocs.readthedocs.io/en/latest/fileformat.html
    """

    def __init__(self, path):
        self.fd = open(path, 'rb')
        self._parse_headers()
        self.stack_names = [s.name for s in self.stack_headers]

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
        return _read_stack(self.fd, self.stack_headers[idx])
    
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

    def get_xml_metadata(self, stack_idx):
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

    def close(self):
        self.fd.close()

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()