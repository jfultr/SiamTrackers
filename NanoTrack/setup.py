from distutils.core import setup
from distutils.extension import Extension


without_cython = False
try:
    from Cython.Build import cythonize
except ImportError:
    without_cython = True
    print('WARNING: Package installed WITHOUT region fucntion! please install cython=0.29 to use it')


if not without_cython:
    ext_modules = [
        Extension(
            name='toolkit.utils.region',
            sources=[
                'toolkit/utils/region.pyx',
                'toolkit/utils/src/region.c',
            ],
            include_dirs=[
                'toolkit/utils/src'
            ]
        )
    ]

if not without_cython:
    setup(
        name='toolkit',
        packages=['toolkit'],
        ext_modules=cythonize(ext_modules)
    )
else:
    setup(
        name='toolkit',
        packages=['toolkit'],
    ) 
