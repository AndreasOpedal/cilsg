from setuptools import setup
from Cython.Build import cythonize
import numpy

'''
This script is responsible for cythonizing the prediction.pyx file.
To run execute: python3 setup.py build ext --inplace
'''

setup(
    ext_modules=cythonize('prediction.pyx'),
    include_dirs=[numpy.get_include()]
)
