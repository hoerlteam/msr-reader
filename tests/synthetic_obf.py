'''
Builders for synthetic OBF/MSR files used by the test suite.

NOTE: these builders encode our reading of the OBF file format specification
(https://imspectordocs.readthedocs.io/en/latest/fileformat.html), not verified
behavior of the Imspector writer. They pin that interpretation and protect
against regressions, but a passing test suite built on them is not a
validation against actual files.
'''
import os
import struct
import tempfile
from contextlib import contextmanager
from zlib import compress

MAXIMAL_NUMBER_OF_DIMENSIONS = 15
MAGIC_FILE_HEADER = b'OMAS_BF\n\xff\xff'
MAGIC_STACK_HEADER = b'OMAS_BF_STACK\n\xff\xff'


def _tag_dict_bytes(tag_dict):
    # key_length(uint32) key value_length(uint32) value, terminated by a
    # 0-valued uint32
    out = b''
    for key, val in tag_dict.items():
        key_b, val_b = key.encode('utf8'), val.encode('utf8')
        out += struct.pack('<L', len(key_b)) + key_b
        out += struct.pack('<L', len(val_b)) + val_b
    return out + struct.pack('<L', 0)


def _length_prefixed(string):
    bytes_ = string.encode('utf8')
    return struct.pack('<L', len(bytes_)) + bytes_


def _si_unit_bytes(scale_factor=1.0):
    # 9 (numerator, denominator) int32 fractions + one double scale factor
    return struct.pack('<18ld', *([0] * 18), scale_factor)


class StackSpec:
    '''
    specification of a single stack in a synthetic OBF file

    :param name: stack name (empty string for unnamed stacks)
    :param size: number of pixels along each axis (determines the rank)
    :param dtype_flags: OMAS_DT data type flag(s)
    :param data: uncompressed (!) binary data, is compressed on write if
        compression is set
    :param stack_version: stack footer version, 0 omits the footer entirely
    :param lengths: physical axis lengths in SI units (default: 1.0 per axis)
    :param metadata: free metadata string written into the footer (V1A)
    :param tag_dict: tag dictionary written after the footer (V4+)
    :param samples_written: V6 samples_written field, defaults to the full
        stack size (0 also means full, as per spec)
    :param col_positions: per-axis list of float positions or None
    :param col_labels: per-axis list of label strings or None
    :param dimension_labels: per-axis dimension label (default: empty strings)
    :param reserved: OBF stack header reserved field, 0 marks a data-less
        stack (real files carry 1)
    :param extra_footer_bytes: bytes appended to the footer without updating
        the reader, emulates a higher footer version than we implement (V7+)
    '''

    def __init__(self, name, size, dtype_flags, data, stack_version=6,
                 lengths=None, compression=False, description='',
                 metadata='', tag_dict=None, samples_written=None,
                 col_positions=None, col_labels=None,
                 dimension_labels=None, reserved=1, extra_footer_bytes=0):
        self.name = name
        self.size = tuple(size)
        self.dtype_flags = dtype_flags
        self.data = data
        self.stack_version = stack_version
        self.lengths = lengths
        self.compression = compression
        self.description = description
        self.metadata = metadata
        self.tag_dict = tag_dict if tag_dict is not None else {}
        self.samples_written = samples_written
        self.col_positions = col_positions
        self.col_labels = col_labels
        self.dimension_labels = dimension_labels
        self.reserved = reserved
        self.extra_footer_bytes = extra_footer_bytes


def _stack_blob(spec, next_stack_pos):
    rank = len(spec.size)
    pad = MAXIMAL_NUMBER_OF_DIMENSIONS - rank

    data = compress(spec.data) if spec.compression else spec.data
    name_b = spec.name.encode('utf8')
    descr_b = spec.description.encode('utf8')
    lengths = spec.lengths if spec.lengths is not None else (1.0,) * rank

    out = bytearray()
    out += MAGIC_STACK_HEADER
    out += struct.pack('<LL', spec.stack_version, rank)
    out += struct.pack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}L', *spec.size, *([0] * pad))
    out += struct.pack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}d', *lengths, *([0.0] * pad))
    out += struct.pack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}d', *([0.0] * MAXIMAL_NUMBER_OF_DIMENSIONS))
    out += struct.pack('<LLLLLQQQ', spec.dtype_flags, 1 if spec.compression else 0, 0,
                       len(name_b), len(descr_b), spec.reserved, len(data), next_stack_pos)
    out += name_b + descr_b + data

    if spec.stack_version < 1:
        # version-0 stacks carry no footer and no variable section
        return bytes(out)

    col_positions = list(spec.col_positions) if spec.col_positions is not None else [None] * rank
    col_labels = list(spec.col_labels) if spec.col_labels is not None else [None] * rank
    metadata_b = spec.metadata.encode('utf8')
    tag_dict_b = _tag_dict_bytes(spec.tag_dict)

    num_samples = 1
    for n in spec.size:
        num_samples *= n
    samples_written = spec.samples_written if spec.samples_written is not None else num_samples

    # fixed-size footer fields per version, size member must cover all of them
    # (plus any unknown bytes of higher versions)
    fields = b''
    fields += struct.pack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}L',
                          *[1 if p is not None else 0 for p in col_positions], *([0] * pad))
    fields += struct.pack(f'<{MAXIMAL_NUMBER_OF_DIMENSIONS}L',
                          *[1 if l is not None else 0 for l in col_labels], *([0] * pad))
    fields += struct.pack('<L', len(metadata_b))
    if spec.stack_version >= 2:
        fields += _si_unit_bytes() * (1 + MAXIMAL_NUMBER_OF_DIMENSIONS)
    if spec.stack_version >= 3:
        fields += struct.pack('<QQ', 0, 0)
    if spec.stack_version >= 4:
        fields += struct.pack('<Q', len(tag_dict_b))
    if spec.stack_version >= 5:
        fields += struct.pack('<QLQ', 0, 0, 0)
    if spec.stack_version >= 6:
        fields += struct.pack('<QQ', samples_written, 0)
    footer_size = 4 + len(fields) + spec.extra_footer_bytes
    out += struct.pack('<L', footer_size)
    out += fields
    out += b'\x00' * spec.extra_footer_bytes

    # variable section: dimension labels, column positions, column labels,
    # metadata string, (flush positions), tag dictionary
    dimension_labels = spec.dimension_labels if spec.dimension_labels is not None else [''] * rank
    for label in dimension_labels:
        out += _length_prefixed(label)
    for positions in col_positions:
        if positions is not None:
            out += struct.pack(f'<{len(positions)}d', *positions)
    for labels in col_labels:
        if labels is not None:
            for label in labels:
                out += _length_prefixed(label)
    out += metadata_b
    if spec.stack_version >= 4:
        out += tag_dict_b

    return bytes(out)


def build_obf(stacks, file_format_version=2, file_description='', file_metadata=None):
    '''
    assemble a synthetic OBF file from stack specifications. stacks are
    written in order and chained via next_stack_pos, the file-level tag
    dictionary (format version 2+) is written at the end of the file.
    '''
    if file_format_version > 1 and file_metadata is None:
        file_metadata = {}
    description_b = file_description.encode('utf8')

    file_header_length = 26 + len(description_b) + (8 if file_format_version > 1 else 0)
    positions = []
    pos = file_header_length
    for spec in stacks:
        positions.append(pos)
        pos += len(_stack_blob(spec, 0))

    out = bytearray()
    out += MAGIC_FILE_HEADER
    out += struct.pack('<LQL', file_format_version, positions[0] if positions else 0, len(description_b))
    out += description_b
    if file_format_version > 1:
        out += struct.pack('<Q', pos)
    for i, spec in enumerate(stacks):
        next_stack_pos = positions[i + 1] if i + 1 < len(stacks) else 0
        out += _stack_blob(spec, next_stack_pos)
    if file_format_version > 1:
        out += _tag_dict_bytes(file_metadata)
    return bytes(out)


@contextmanager
def obf_file(blob, **kwargs):
    'write a synthetic file to a temporary directory and open it with OBFFile'
    from msr_reader import OBFFile
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, 'synthetic.obf')
        with open(path, 'wb') as fd:
            fd.write(blob)
        with OBFFile(path, **kwargs) as obf:
            yield obf
