
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "qdr.ranker",
        sources=['qdr/ranker.pyx'],
        extra_compile_args=['-std=c++0x'],
        language="c++"),
]


setup(name='qdr',
    version='0.0',
    description='Query-Document Relevance',
    author='Moz Data Science',
    packages=['qdr'],
    package_dir={'qdr': 'qdr'},
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

