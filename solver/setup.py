from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(["moacp_noreinj.pyx", "moacp_mut.pyx"]),
      include_dirs=[numpy.get_include()])
