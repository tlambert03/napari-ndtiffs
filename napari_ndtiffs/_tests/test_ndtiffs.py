# -*- coding: utf-8 -*-

from napari_ndtiffs import napari_get_reader, affine
from scipy.ndimage import affine_transform
import numpy as np


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None


def test_affine_nearest():
    assert affine.cl

    arr = np.ones((64, 256, 256))
    tmat = np.eye(4)
    tmat[2, 0] = -2
    scipy_ = affine_transform(arr, tmat, order=0, output_shape=(64, 256, 396))
    ours = affine.affine_transform(arr, tmat, order=0, output_shape=(64, 256, 396))
    assert np.allclose(scipy_, ours)


def test_affine_linear():
    assert affine.cl

    arr = np.ones((64, 256, 256))
    tmat = np.eye(4)
    tmat[2, 0] = -2
    scipy_ = affine_transform(arr, tmat, order=1, output_shape=(81, 256, 396))
    ours = affine.affine_transform(arr, tmat, order=1, output_shape=(81, 256, 396))
    assert np.allclose(scipy_, ours)
