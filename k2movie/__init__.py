import os

from .version import __version__

PACKAGEDIR = os.path.dirname(os.path.abspath(__file__))

from .k2movie import *
from .build import build
