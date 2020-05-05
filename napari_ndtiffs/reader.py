"""Plugin to read lattice light sheet folders into napari."""
import glob
import logging
import os
import re
import zipfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from dask import array as da
from dask import delayed
from tifffile import TiffFile, imread

from .affine import get_deskew_func
from .settingstxt import parse_settings

logger = logging.getLogger(__name__)
logging.getLogger("tifffile").setLevel(logging.CRITICAL)


LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]

# this dict holds any overrides parameter overrides that the user wants
OVERRIDES: Dict[str, Any] = {}


@contextmanager
def parameter_override(**kwargs):
    global OVERRIDES
    old = OVERRIDES.copy()
    OVERRIDES.update(kwargs)
    yield
    OVERRIDES = old


lls_pattern = re.compile(
    r"""
    ^(?![_.])  # don't start with _ or .
    .*
    _ch(?P<channel>\d{1})
    _stack(?P<stack>\d{4})
    _(?P<wave>[^_]+)
    _(?P<reltime>\d{7})msec
    _.*\.tiff?$""",  # ends with tif or tiff
    re.VERBOSE,
)


read_counter = 0


def noisy_imread(path, in_zip=None):
    # to see, set:  logging.getLogger("napari_ndtiffs").setLevel(logging.DEBUG)
    global read_counter
    read_counter += 1
    logger.debug(f"reading {path}, (read count: {read_counter})")
    if in_zip:
        with zipfile.ZipFile(in_zip) as zf:
            with zf.open(path, "r") as f:
                return imread(f)
    return imread(path)


lazy_imread = delayed(noisy_imread)  # lazy reader


def alphanumeric_key(s):
    k = [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", s)]
    return k


def has_lls_data(path):
    path = os.path.abspath(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            filelist = zf.namelist()
    elif os.path.isdir(path):
        filelist = os.listdir(path)
    else:
        return False
    for fname in filelist:
        if fname.endswith((".tif", ".tiff")):
            match = lls_pattern.match(fname)
            if match:
                gdict = match.groupdict()
                if gdict.get("channel") and gdict.get("stack"):
                    return True
    return False


def get_tiff_meta(
    path: str, in_zip: str = None
) -> Tuple[Tuple[int, int], np.dtype, float, float, Tuple[int, int]]:
    dx, dz = 1.0, 1.0
    if in_zip:
        with zipfile.ZipFile(in_zip) as zf:
            with zf.open(path, "r") as f:
                return get_tiff_meta(f)

    with TiffFile(path) as tfile:
        nz = len(tfile.pages)
        if not nz:
            raise ValueError(f"tiff file {path} has no pages!")
        first_page = tfile.pages[0]
        shape = (nz,) + first_page.shape
        dtype = first_page.dtype
        _dx = first_page.tags.get("XResolution")
        if hasattr(_dx, "value"):
            dx = 1 / np.divide(*_dx.value)

        desc = first_page.tags.get("ImageDescription")
        if hasattr(desc, "value"):
            match = re.search(r"spacing=([\d\.]+)", desc.value)
            if match:
                dz = float(match.groups()[0])

        sample = tfile.asarray(key=(nz // 4, nz // 2, 3 * nz // 4))
        clims = sample.min(), sample.max()
    return shape, dtype, dx, dz, clims


def reader_function(path: PathLike) -> List[LayerData]:
    """Take a path or list of paths and return a list of LayerData tuples."""

    try:
        settings = parse_settings(path)
    except FileNotFoundError:
        settings = {}
    in_zip = str(path) if zipfile.is_zipfile(path) else None
    channels = dict()
    if in_zip:
        with zipfile.ZipFile(path) as zf:
            filelist = zf.namelist()
    else:
        filelist = glob.glob(os.path.join(path, "*.tif"))
    for fname in filelist:
        match = lls_pattern.match(fname)
        if match:
            gdict = match.groupdict()
            if gdict.get("channel") not in channels:
                channels[gdict.get("channel")] = (gdict.get("wave"), [])
            channels[gdict.get("channel")][1].append(fname)

    data = []
    names = []
    clims = []
    for i in sorted(channels.keys()):
        wave, filenames = channels[i]
        names.append(wave)
        shape, dtype, dx, dz, clims_ = get_tiff_meta(filenames[0], in_zip=in_zip)
        clims.append(clims_)
        lazy_arrays = [lazy_imread(fn, in_zip=in_zip) for fn in sorted(filenames)]
        dask_arrays = [
            da.from_delayed(delayed_reader, shape=shape, dtype=dtype)
            for delayed_reader in lazy_arrays
        ]
        stack = da.stack(dask_arrays, axis=0)
        data.append(stack)
    data = da.stack(data)

    dx = OVERRIDES.get("dx") or dx
    dz = OVERRIDES.get("dz") or dz

    dzdx_ratio = dz / dx
    if (
        settings.get("params", {}).get("samplescan", False) or OVERRIDES.get("angle")
    ) and OVERRIDES.get("deskew", True):
        # if the image is the same size or smaller than the Settings.txt file, we deskew
        angle = OVERRIDES.get("angle")
        if angle is None:
            angle = settings["params"]["angle"]

        if shape[-1] <= settings["params"]["ny"] and angle > 0:
            deskew_func, new_shape, dzdx_ratio = get_deskew_func(
                data.shape,
                dx=OVERRIDES.get("dx") or settings["params"]["dx"],
                dz=OVERRIDES.get("dz") or settings["params"]["dz"],
                angle=angle,
                padval=OVERRIDES.get("padval") or 0,
            )
            data = data.map_blocks(deskew_func, dtype="float32", chunks=new_shape)
    meta = {
        "channel_axis": 0,
        "scale": (1, dzdx_ratio, 1, 1),
        "multiscale": False,
        "contrast_limits": OVERRIDES.get("contrast_limits") or clims,
        "name": OVERRIDES.get("name") or names,
    }
    if settings:
        meta["metadata"] = settings

    return [(data, meta)]
