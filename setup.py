
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import numpy as np
from distutils.extension import Extension


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='p_utils',
      version='0.1',
      description='General utilities for data analysis',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5'
      ],
      keywords='',
      #url='http://github.com/',
      author='Pietro Marchesi',
      author_email='pietromarchesi92@gmail.com',
      license='new BSD',
      packages=['p_utils'],
      install_requires=[
          'numpy',
          'scipy'
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite = 'nose.collector',
      tests_require = ['nose'],
      #ext_modules=[Extension('pidpy.utilsc', ['pidpy/utilsc.c'])],
      include_dirs=[np.get_include()]
      )