"""Deskew utilities"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def deskew_block(block, mat=None, out_shape=None, padval=0):
    try:
        from ._ocl_affine import affine_transform
    except ImportError:
        from scipy.ndimage.interpolation import affine_transform

        logger.warning(
            "Could not import pyopencl. "
            "Falling back to scipy for CPU affine transforms"
        )

    extradims = block.ndim - 3
    last3dims = (0,) * extradims + (slice(None),) * 3
    array = block[last3dims]

    deskewed = affine_transform(array, mat, output_shape=tuple(out_shape[-3:]), order=0)
    return deskewed[(None,) * extradims + (...,)]


deskew_counter = 0


def get_deskew_func(shape, dz=0.5, dx=0.1, angle=31.5, padval=0):
    # calculate affine matrix from globals
    deskewFactor = np.cos(np.deg2rad(angle)) * dz / dx
    mat = np.eye(4)
    mat[2, 0] = -deskewFactor

    # calculate shape of output array
    (nz, ny, nx) = shape[-3:]
    out_shape = [1] * (len(shape) - 3) + list(shape[-3:])
    # new nx
    out_shape[-1] = np.int(np.floor((nz - 1) * -mat[2, 0]) + nx)
    new_dzdx_ratio = np.sin(np.deg2rad(angle)) * dz / dx

    def noisy_deskew(arr):
        # to see, set:  logging.getLogger("napari_ndtiffs").setLevel(logging.DEBUG)
        global deskew_counter
        deskew_counter += 1
        logger.debug(f"deskew #{deskew_counter}")
        return deskew_block(arr, mat=mat, out_shape=out_shape, padval=padval)

    return noisy_deskew, out_shape, new_dzdx_ratio
