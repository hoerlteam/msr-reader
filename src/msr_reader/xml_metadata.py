from xml.etree import ElementTree

LEAF_TYPE_INT_PREFIXES = ('int', 'uint')
LEAF_TYPE_FLOAT_PREFIXES = ('single', 'double')
LEAF_TYPE_PREFIXES = ('char', 'logical') + LEAF_TYPE_INT_PREFIXES + LEAF_TYPE_FLOAT_PREFIXES


def startswith_any_in_collection(s, prefixes):
    for prefix in prefixes:
        if s.startswith(prefix):
            return True
    return False


def reshape_list(l, shape):
    rows, cols = shape
    res = []
    row_list = []
    for i, elem in enumerate(l):
        row_list.append(elem)
        if i > 0 and (i + 1) % cols == 0:
            res.append(row_list)
            row_list = []
    return res


def handle_xml_element(element: ElementTree.Element):
    typ = element.attrib['type']
    if startswith_any_in_collection(typ, LEAF_TYPE_PREFIXES):
        return handle_xml_leaf(element)
    elif typ == 'struct':
        return handle_xml_struct(element)
    elif typ == 'cell':
        return handle_xml_cell(element)
    elif typ == 'empty':
        return None
    else:
        raise ValueError(f'unknown XML element of type: {typ}')


def handle_xml_cell(element: ElementTree.Element):
    result_list = []
    for child in element:
        if not child.tag == 'item':
            raise ValueError(f'expected only item elements in cell, but got {child.tag}')
        result_list.append(handle_xml_element(child))
    return result_list


def handle_xml_struct(element: ElementTree.Element):
    result_dict = {}
    for child in element:
        child_result = handle_xml_element(child)
        if child_result is not None:
            result_dict[child.tag] = child_result

    # set empty dict to None, so it will get dropped by outer call
    if len(result_dict) == 0:
        result_dict = None

    return result_dict


def handle_xml_leaf(element: ElementTree.Element):
    typ = element.attrib['type']
    if not startswith_any_in_collection(typ, LEAF_TYPE_PREFIXES):
        raise ValueError(f'leaf node in XML of type {typ} not supported.')

    # if parameters is char/str, we just return it and do not care about size
    if typ == 'char':
        # empty string will be None in XML, return an actual string
        if element.text is None:
            return ''
        return element.text

    # get size of leaf data,
    size = element.attrib['size']
    size = tuple(map(int, size.split(' ')))

    # empty parameter: return None
    if any((s == 0 for s in size)):
        return None

    # check if we need to return more than one value as list
    is_scalar = all((s == 1 for s in size))

    # check if we need to return nested lists (two dimensional parameter array)
    is_matrix = all((s > 1 for s in size))

    # get function to convert str to dtype
    if startswith_any_in_collection(typ, LEAF_TYPE_INT_PREFIXES):
        parse_fun = int
    elif startswith_any_in_collection(typ, LEAF_TYPE_FLOAT_PREFIXES):
        parse_fun = float
    else:
        # NOTE: to parse bool, we first convert it to int (0/1), then to bool
        # because bool('0') would return True
        parse_fun = lambda text: bool(int(text))

    # scalar: just parse
    if is_scalar:
        return parse_fun(element.text)
    # matrix/2d array -> make nested list
    elif is_matrix:
        return reshape_list(list(map(parse_fun, element.text.split(' '))), size)
    # vector: split on whitespace, parse individual elements, to list
    else:
        return list(map(parse_fun, element.text.split(' ')))


def imspector_xml_to_dict(xml_string):
    '''
    Convert the Imspector XML metadata available via OBFFile.get_imspector_xml_metadata
    into a nested dictionary as used by the specpy interface

    NOTE: Newer versions of Imspector may no longer save this information to .msr files
    '''
    return handle_xml_element(ElementTree.fromstring(xml_string))
