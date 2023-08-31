# msr

Pure Python and cross-platform reader for MSR/OBF image files as created by Imspector.

Implementation follows the description at: https://imspectordocs.readthedocs.io/en/latest/fileformat.html

**WARNING:** Only tested on data from our own Abberior Expert Line STED microscope with a narrow range of Imspector versions.

## Installation

```
pip install msr
```

## Usage

```python
from msr import OBFFile

with OBFFile('file_path.msr') as f:

    # reading image data
    img = f.read_stack(idx) # read stack with index idx into numpy array

    # metadata
    stack_sizes = f.sizes # list of stack sizes/shapes, including stack and dimension names
    pixel_sizes = f.pixel_sizes # like sizes, but with pixel sizes (unit: meters)

    # XML metadata is usually part of the tag dictionary of stack footer
    stack_footer = f.stack_footers[idx] # get footer of stack idx

    # imspector metadata as xml string
    xml_imspector_metadata = stack_footer.tag_dictionary['imspector']
    
    # can be parsed with ElementTree, for example
    from xml.etree import ElementTree    
    et = ElementTree.fromstring(xml_imspector_metadata)
    # find, e.g. x pixel size in XML
    et.find('doc/ExpControl/scan/range/x/psz').text

```