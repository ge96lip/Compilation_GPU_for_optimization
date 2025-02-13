from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cythonfn2.pyx", annotate=True, language_level=3, compiler_directives={"optimize.use_switch": False}),
    include_dirs=[numpy.get_include()]
)