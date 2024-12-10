# msr

Pure Python and cross-platform reader for MSR/OBF image files as created by Imspector.

Implementation follows the description at: https://imspectordocs.readthedocs.io/en/latest/fileformat.html

**WARNING:** Only tested on data from our own Abberior Expert Line STED microscope with a narrow range of Imspector versions.

## Installation

```
pip install msr-reader
```

## Usage

```python
from msr_reader import OBFFile
from xml.etree import ElementTree
import ome_types

with OBFFile('file_path.msr') as f:

    # reading image data
    idx = 0
    img = f.read_stack(idx) # read stack with index idx into numpy array

    # metadata
    stack_shapes = f.shapes # list of stack shapes, including stack and dimension names
    pixel_sizes = f.pixel_sizes # like shapes, but with pixel sizes (unit: meters)

    # imspector metadata as xml string
    xml_imspector_metadata = f.get_imspector_xml_metadata(idx)
    
    # can be parsed with ElementTree, for example    
    et = ElementTree.fromstring(xml_imspector_metadata)
    # find, e.g. x pixel size in XML
    et.find('doc/ExpControl/scan/range/x/psz').text

    # OME-XML metadata as xml string
    xml_ome_metadata = f.get_ome_xml_metadata()
    # can be parsed with ome-types, for example
    ome_meta = ome_types.from_xml(xml_ome_metadata)
```