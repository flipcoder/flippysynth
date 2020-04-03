from distutils.core import setup
from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

setup(
    name="flippysynth",
    ext_modules=cythonize("flippysynth.py"),
    entry_points="""
        flippysynth=flippysynth.__main__:main
    """,
)
