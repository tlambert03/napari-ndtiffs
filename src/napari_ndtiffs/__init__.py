from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari-ndtiffs")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
__all__ = [
    "napari_get_reader",
    "parse_settings",
    "parameter_override",
    "reader_function",
]


from typing import Optional

from napari_plugin_engine import napari_hook_implementation

from .reader import (
    PathLike,
    ReaderFunction,
    has_lls_data,
    parameter_override,
    reader_function,
)
from .settingstxt import parse_settings


@napari_hook_implementation
def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    if isinstance(path, str) and has_lls_data(path):
        return reader_function  # type: ignore
    return None
