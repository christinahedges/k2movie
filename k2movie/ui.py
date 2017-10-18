"""Implements the k2movie command-line tools.

crunch....make a database
inspect...build an images
movie...make a movie
"""
from astropy.io import fits
from astropy.utils.data import download_file,clear_download_cache
from astropy.config.paths import get_cache_dir
import click
import numpy as np
import os
import dask.dataframe as dd
import pandas as pd
from . import PACKAGEDIR, __version__
from .build import build

WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def k2movie(**kwargs):
    pass


@k2movie.command()
@click.argument('dir', type=str,default='database/')
@click.option('-v', '--verbose', type=bool,
              default=True, metavar='<True/False>',
              help="Change verbosity")
@click.option('-o', '--overwrite', type=bool,
              default=False, metavar='<True/False>',
              help="Overwrite previously crunched files")
@click.option('-in', '--input', type=str, default=None,
              help='Directory containing TPFs. If blank, will download from mast')
@click.option('-lim', '--cachelim', type=int, default=30,
              help='Size limit for file caching in gb.')
@click.option('-c', '--campaign', type=int, default=-1,
              help='Campaign to crunch. Default: crunch all')
@click.option('-ch', '--channel', type=int, default=-1,
              help='Channel to crunch. Default: crunch all')


def database(dir,input,verbose,cachelim,overwrite,campaign,channel):
    '''Creates a database of HDF5 binaries filled with TPFs'''
    if campaign == -1:
        campaign=None
    else:
        campaign=[campaign]

    if channel == -1:
        channel=None
    else:
        channel=[channel]

    build(dir,input,verbose,cachelim,overwrite,campaign,channel)
    return



if __name__ == '__main__':
    k2movie()
