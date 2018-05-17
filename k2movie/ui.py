"""Implements the k2movie command-line tools.

crunch....make a database
inspect...build an images
movie...make a movie
"""
from astropy.io import fits
from astropy.utils.data import download_file, clear_download_cache
from astropy.config.paths import get_cache_dir
import click
import numpy as np
import os
import dask.dataframe as dd
import pandas as pd
from . import PACKAGEDIR, __version__
from .build import bld
from .build import log
from glob import glob

WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def k2movie(**kwargs):
    pass


@k2movie.command()
@click.option('-d', '--input', type=str, default=None,
              help='Directory containing TPFs. If blank, will download from mast')
@click.option('-c', '--campaign', type=int, default=-1,
              help='Campaign to crunch. Default: crunch all')
@click.option('-ch', '--channel', type=int, default=-1,
              help='Channel to crunch. Default: crunch all')
@click.option('-l', '--level', type=str,
              default='INFO', metavar='[DEBUG, INFO, WARNING, ERROR, CRITICAL]',
              help="Set logger level")
@click.option('-o', '--overwrite', type=bool,
              default=False, metavar='<True/False>',
              help="Overwrite previously crunched files")
@click.option('-lim', '--cachelim', type=int, default=30,
              help='Size limit for file caching in gb.')
@click.option('-out', '--output', type=str, default=None,
              help='Directory to output to.')
def build(input, campaign, channel, level, cachelim, overwrite, output):
    '''Creates a database of HDF5 binaries filled with TPFs'''
    if campaign == -1:
        campaign = None
    else:
        campaign = [campaign]

    if channel == -1:
        channel = None
    else:
        channel = [channel]
    log.setLevel(level.upper())
    bld(dir=output, indir=input, cachelim=cachelim, overwrite=overwrite,
        campaigns=campaign, channels=channel)
    return


@k2movie.command()
@click.argument('dir', type=str, default='database/')
def list(dir):
    '''Prints the number of campaigns that are stored'''
    print('----------------------')
    print('Database Completeness:')
    print('----------------------')
    print()
    cnames = ['c00', 'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08',
              'c09', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', ]

    # cnames = [c[len(dir):] for c in cnames]
    cnames = np.sort(cnames)
    for c in cnames:
        print('|{}|'.format(c), end='\t')
        mnames = glob('{}{}/*'.format(dir, c))
        ex = np.asarray([os.path.isfile('{}/0.h5'.format(m)) for m in mnames])
        for i in range((len(np.where(ex == True)[0]))//2):
            print('|', end='')
        for i in range((84-len(np.where(ex == True)[0]))//2):
            print('.', end='')
        print()
    return


@k2movie.command()
@click.option('-d', '--input', type=str, default=os.path.join(PACKAGEDIR, 'database/'),
              help='Directory built database')
@click.option('-c', '--campaign', type=int, default=None,
              help='Campaign to use')
@click.option('-ch', '--channel', type=int, default=None,
              help='Channel to use')
@click.option('-l', '--level', type=str,
              default='INFO', metavar='[DEBUG, INFO, WARNING, ERROR, CRITICAL]',
              help="Set logger level")
@click.option('-a', '--aperture', type=int,
              default=10, help="Pixel aperture size")
def target(targetname, input, campaign, channel, level):
    '''Creates a movie of a target'''
    if campaign == -1:
        campaign = None
    else:
        campaign = [campaign]

    if channel == -1:
        channel = None
    else:
        channel = [channel]
    log.setLevel(level.upper())
    bld(dir=output, indir=input, cachelim=cachelim, overwrite=overwrite,
        campaigns=campaign, channels=channel)
    return


if __name__ == '__main__':
    k2movie()
