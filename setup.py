from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("word2vec.word2vec_c", ["./word2vec/word2vec_c.pyx"]),
    Extension("word2vec.model_c", ["./word2vec/model_c.pyx"]),
    Extension("word2vec.data.dataset_c", ["./word2vec/data/dataset_c.pyx"]),
]

setup(
    ext_modules=cythonize(ext_modules, language_level=3),
    include_dirs=np.get_include(),
    requires=['Cython'],
)
