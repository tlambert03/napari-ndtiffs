#!/usr/bin/env python
from setuptools import setup

# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "napari_ndtiffs/_version.py"}
setup(use_scm_version=use_scm)
