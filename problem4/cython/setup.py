from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='mc simulation cython version',
    ext_modules=cythonize("mc_sim.pyx"),
    include_dirs=[np.get_include()],
)
