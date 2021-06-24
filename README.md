# napari-ndtiffs

[![License](https://img.shields.io/pypi/l/napari-ndtiffs.svg?color=green)](https://raw.githubusercontent.com/tlambert03/napari-ndtiffs/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-ndtiffs.svg?color=green)](https://pypi.org/project/napari-ndtiffs)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-ndtiffs.svg?color=green)](https://python.org)
[![tests](https://github.com/tlambert03/napari-ndtiffs/workflows/tests/badge.svg)](https://github.com/tlambert03/napari-ndtiffs/actions)
[![codecov](https://codecov.io/gh/tlambert03/napari-ndtiffs/branch/master/graph/badge.svg)](https://codecov.io/gh/tlambert03/napari-ndtiffs)

napari plugin for nd tiff folders with optional OpenCl-based deskewing.

Built-in support for folders of (skewed) lattice light sheet tiffs.

![napari-ndtiffs demo](https://github.com/tlambert03/napari-ndtiffs/raw/master/demo.gif)

----------------------------------

*This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.*

## Features

- Drag and drop a folder of tiffs onto napari window to view easily 
  - (currently designed to detect  lattice light sheet tiffs, but easily
    adjustable)
- If lattice `Settings.txt` file is found, will deskew automatically (only if
  necessary)
- Lazily loads dataset on demand.  quickly load preview your data.
- Handles `.zip` archives as well!  Just directly compress your tiff folder,
  then drop it into napari.
- All-openCL deskewing, works on GPU as well as CPU, falls back to scipy if
  pyopencl is unavailable.

It would not be hard to support arbitrary filenaming patterns!  If you have a
folder of tiffs with a consistent naming scheme and would like to take advantage
of this plugin, feel free to open an issue!

## Installation

You can install `napari-ndtiffs` via [pip]:

```shell
pip install napari-ndtiffs
```

To also install PyOpenCL (for faster deskewing):

```shell
pip install napari-ndtiffs[opencl]
```

## Usage

In most cases, just drop your folder onto napari, or use `viewer.open("path")`

### Overriding parameters

You can control things like voxel size and deskewing angle as follows:

```python
from napari_ndtiffs import parameter_override
import napari

viewer = napari.Viewer()
with parameter_override(angle=45, name="my image"):
    viewer.open("path/to/folder", plugin="ndtiffs")
```

Valid keys for `parameter_override` include:

- **dx**: (`float`) the pixel size, in microns
- **dz**: (`float`)the z step size, in microns
- **deskew**: (`bool`) whether or not to deskew, (by default, will deskew if angle > 0, or if a lattice metadata file is detected that requires deskewing) 
- **angle**: (`float`) the angle of the light sheet relative to the coverslip
- **padval**: (`float`) the value with which to pad the image edges when deskewing (default is 0)
- **contrast_limits**: (`2-tuple of int`) (min, max) contrast_limits to use when viewing the image
- **name**: (`str`) an optional name for the image

### Sample data

Try it out with test data: [download sample data](https://www.dropbox.com/s/up4ywrn2sckjunc/lls_mitosis.zip?dl=1)

You can unzip if you like, or just drag the zip file onto the napari window.

Or, from command line, use:

```bash
napari path/to/lls_mitosis.zip
```

## Debugging

To monitor file io and deskew activity, enter the following in the napari console:

```python
import logging
logging.getLogger('napari_llsfolder').setLevel('DEBUG')
```


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-ndtiffs" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/tlambert03/napari-ndtiffs/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
