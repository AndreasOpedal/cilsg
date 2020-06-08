'''
This script is responsible for cythonizing the factorization.pyx file.
To run execute: python3 setup.py build_ext --inplace
'''

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize('factorization.pyx'),
    include_dirs=[numpy.get_include()]
)
