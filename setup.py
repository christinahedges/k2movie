#!/usr/bin/env python
import os
import sys
from setuptools import setup

if "testpublish" in sys.argv[-1]:
    os.system("python setup.py sdist upload -r pypitest")
    sys.exit()
elif "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload -r pypi")
    sys.exit()

# Load the __version__ variable without importing the package
exec(open('k2movie/version.py').read())

entry_points = {'console_scripts':
                ['k2movie = k2movie.ui:k2movie']}

setup(name='k2movie',
      version=__version__,
      description="Creates movies if Kepler data based on target names. "
                  "Uses a custom data format.",
      long_description=open('README.rst').read(),
      author='Christina Hedges',
      author_email='christina.l.hedges@nasa.gov',
      license='MIT',
      packages=['k2movie'],
      package_data={'k2movie': ['data/*.csv']},
      install_requires=['astropy>=0.4',
                        'numpy',
                        'pandas',
                        'click',
                        'requests',
                        'imageio>=1',
                        'K2ephem'],
      entry_points=entry_points,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
