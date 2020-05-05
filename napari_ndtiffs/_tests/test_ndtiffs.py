# -*- coding: utf-8 -*-

from napari_ndtiffs import napari_get_reader, affine
from scipy.ndimage import affine_transform
import numpy as np
import pytest

cl = pytest.importorskip("pyopencl")


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None


devices = [d for p in cl.get_platforms() for d in p.get_devices()]
names = [d.name[:30] for d in devices]


@pytest.fixture(params=devices, ids=names)
def all_gpus(monkeypatch, request):
    def patched_func():
        class holder:
            pass

        GPU = holder()

        GPU.device = request.param
        GPU.ctx = cl.Context(devices=[GPU.device])
        GPU.queue = cl.CommandQueue(GPU.ctx)
        return GPU

    monkeypatch.setattr(affine, "get_gpu", patched_func)


def test_affine_nearest(all_gpus):
    assert affine.cl

    arr = np.ones((64, 256, 256))
    tmat = np.eye(4)
    tmat[2, 0] = -2
    scipy_ = affine_transform(arr, tmat, order=0, output_shape=(64, 256, 396))
    ours = affine.affine_transform(arr, tmat, order=0, output_shape=(64, 256, 396))
    assert np.allclose(scipy_, ours)


def test_affine_linear(all_gpus):
    assert affine.cl

    arr = np.ones((64, 256, 256))
    tmat = np.eye(4)
    tmat[2, 0] = -2
    scipy_ = affine_transform(arr, tmat, order=1, output_shape=(81, 256, 396))
    ours = affine.affine_transform(arr, tmat, order=1, output_shape=(81, 256, 396))
    assert np.allclose(scipy_, ours)
