from . import __version__, PACKAGEDIR
import glob
import pickle
import re
from tqdm import tqdm
from astropy.utils.data import download_file, clear_download_cache
from astropy.config.paths import get_cache_dir
from astropy.io import fits
from astropy.wcs import WCS
import fitsio
import numpy as np
import pandas as pd
import os
from k2mosaic import mast
from contextlib import contextmanager
import warnings
import sys

import logging
log = logging.getLogger('\tk2movie ')


@contextmanager
def silence():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')
DATA_DIR = os.path.join(PACKAGEDIR, 'data', 'database/')


def pickle_wcs(output_fn=WCS_DIR, ffi_store=None):
    '''Writes the WCS solution function of each campaign and module to a pickled
    file.
    This will enable each X/Y coordinate to have an RA and Dec
    '''
    if (os.path.isdir(WCS_DIR) == False):
        os.makedirs(WCS_DIR)

    if ffi_store == None:
        ffi_filenames = []
        while len(ffi_filenames) == 0:
            ffi_store = input('Enter Directory with FFIs: ')
            ffi_filenames = glob.glob(os.path.join(ffi_store, '*cal.fits'))
            if len(ffi_filenames) == 0:
                log.warning('Cannot find FFIs')
    else:
        ffi_filenames = glob.glob(os.path.join(ffi_store, '*cal.fits'))
    with tqdm(total=len(ffi_filenames)) as pbar:
        for filename in ffi_filenames:
            basename = os.path.basename(filename)
            campaign = int(re.match(".*c([0-9]+)_.*", basename).group(1))
            if campaign <= 2:
                if ('wcs' in filename) == False:
                    continue
            fts = fits.open(filename)
            for ext in range(1, 85):
                pickle.dump(WCS(fts[ext].header), open('{}'.format(output_fn) +
                                                       'c{0:02}_'.format(campaign)+'{0:02}.p'.format(ext), 'wb'))
            pbar.update()
    log.debug("Built WCS")


def get_wcs(campaign=0, channel=1, xs=[500], ys=[500], dir=WCS_DIR):
    r = pickle.load(open(dir+'c{0:02}_'.format(campaign)+'{0:02}.p'.format(channel), 'rb'))
    RA, Dec = r.wcs_pix2world(xs, ys, 1)
    return RA, Dec


def hdf5_mosaic(tpf_filenames, campaign, channel,
                output_prefix='', memory_lim=4, dtype=np.float16):
    '''Mosaic a set of TPFS into a dataframe. NOTE: by default this will be a float16
    dataframe which saves a little on storage space, these can become very large. Use
    these at your own risk.

    Paramters
    ---------


    Returns
    -------
    '''
    if isinstance(tpf_filenames, str):
        log.debug('Parsing file...')
        tpf_filenames = np.loadtxt(tpf_filenames, dtype=str)
    wcs_file = WCS_DIR+'c{0:02}_'.format(campaign)+'{0:02}.p'.format(channel)
    if os.path.isfile(wcs_file) == False:
        log.error('No WCS found')
    log.debug('Loading WCS')
    with silence():
        r = pickle.load(open(wcs_file, 'rb'))
    if tpf_filenames[0].startswith("http"):
        with silence():
            tpf_filename = download_file(tpf_filenames[0], cache=True)
    else:
        tpf_filename = tpf_filenames[0]
    tpf = fitsio.FITS(tpf_filename)
    cadencelist = tpf[1]['CADENCENO'].read()
    tpf.close()
    cols = np.asarray(cadencelist, dtype=str)
    cols = np.append(['RA', 'Dec', 'Row', 'Column', 'APERFLAG'], cols)
    df = pd.DataFrame(columns=cols, dtype=dtype)
    edf = pd.DataFrame(columns=cols, dtype=dtype)
    fname = 'k2movie_c{0:02}_ch{1:02}'.format(campaign, channel)
    FILEPATH = '{}{}.h5'.format(output_prefix, fname)
    ERROR_FILEPATH = '{}{}_ERR.h5'.format(output_prefix, fname)
    log.debug('File path: {}'.format(FILEPATH))
    log.debug('File path: {}'.format(ERROR_FILEPATH))
    if os.path.isfile(FILEPATH):
        log.debug('Clearing old file')
        os.remove(FILEPATH)
    if os.path.isfile(ERROR_FILEPATH):
        log.debug('Clearing old error file')
        os.remove(ERROR_FILEPATH)

    totalpixels = 0
    with tqdm(total=len(tpf_filenames)) as bar:
        for i, tpf_filename in enumerate(tpf_filenames):
            if tpf_filename.startswith("http"):
                with silence():
                    tpf_filename = download_file(tpf_filename, cache=True)
            tpf = fitsio.FITS(tpf_filename)
            try:
                aperture = tpf[2].read()
            except:
                continue
            aperture_shape = aperture.shape
            # Get the pixel coordinates of the corner of the aperture
            hdr_list = tpf[1].read_header_list()
            hdr = {elem['name']: elem['value'] for elem in hdr_list}
            col, row = int(hdr['1CRV5P']), int(hdr['2CRV5P'])
            height, width = aperture_shape[0], aperture_shape[1]
            # Fill the data
            mask = aperture > 0
            y, x = np.meshgrid(np.arange(col, col+width), np.arange(row, row+height))
            x, y = x[mask], y[mask]
            totalpixels += len(x.ravel())
            flux = (tpf[1].read()['FLUX'])
            error = (tpf[1].read()['FLUX_ERR'])
            tpf.close()
            f = np.asarray([f[mask].ravel() for f in flux])
            ap = aperture[mask].ravel()
            e = np.asarray([e[mask].ravel() for e in error])
            RA, Dec = r.wcs_pix2world(y.ravel(), x.ravel(), 1)
            f = np.asarray(np.reshape(np.append([np.asarray(RA, dtype=float),
                                                 np.asarray(Dec, dtype=float),
                                                 np.asarray(x.ravel(), dtype=float),
                                                 np.asarray(y.ravel(), dtype=float),
                                                 ap],
                                                f),
                                      (np.shape(f)[0]+5, np.shape(f)[1])), dtype=float)
            e = np.asarray(np.reshape(np.append([np.asarray(RA, dtype=float),
                                                 np.asarray(Dec, dtype=float),
                                                 np.asarray(x.ravel(), dtype=float),
                                                 np.asarray(y.ravel(), dtype=float),
                                                 ap],
                                                e),
                                      (np.shape(e)[0]+5, np.shape(e)[1])), dtype=float)
            with silence():
                df = df.append(pd.DataFrame(f.T, columns=cols, dtype=dtype))
                edf = edf.append(pd.DataFrame(e.T, columns=cols, dtype=dtype))
            mem = np.nansum(df.memory_usage())/1E9
            if mem >= memory_lim/2:
                log.debug(
                    '\n{} gb memory limit reached after {} TPFs. Writing to file.'.format(memory_lim, i))
                df.to_hdf(FILEPATH, 'table', append=True)
                df = pd.DataFrame(columns=cols, dtype=dtype)
                edf.to_hdf(ERROR_FILEPATH, 'table', append=True)
                edf = pd.DataFrame(columns=cols, dtype=dtype)
            bar.update()
        df.to_hdf(FILEPATH, 'table', append=True)
        edf.to_hdf(ERROR_FILEPATH, 'table', append=True)
        log.debug('\n{} Pixels Written'.format(totalpixels))
    return


def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def bld(dir=None, indir=None, cachelim=30, overwrite=False,
        campaigns=None, channels=None, memory_lim=4):
    '''Creates a database of HDF5 files'''

    if dir is not None:
        if not os.path.isdir(dir):
            log.debug('Creating Directory')
            os.makedirs(dir)
    else:
        dir = ''

    log.debug('-------------------------------')
    log.debug('Building K2 TPF HDF5 database.')
    if (os.path.isdir(WCS_DIR) == False):
        log.error('No WCS Files Found')

    if indir is None:
        log.error('No input directory. Build URLS using k2mosaic.')
    else:
        log.debug('Input directory: {}'.format(indir))
        log.debug('Assuming MAST-like structure.')

    if overwrite:
        log.debug('Overwrite enabled.')

    if campaigns is None:
        campaigns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 91, 92, 101, 102, 111, 112, 12, 13, 14, 15]
    if channels is None:
        channels = range(1, 85)

    for campaign in campaigns:
        cdir = '{}'.format(dir)+'c{0:02}/'.format(campaign)
        if not os.path.isdir(cdir):
            os.makedirs(cdir)
        for ext in channels:
            edir = '{}'.format(cdir)+'{0:02}/'.format(ext)
            if not os.path.isdir(edir):
                os.makedirs(edir)
            if (os.path.isfile('{}'.format(edir)+'k2movie_c{0:02}_ch{1:02}.h5'.format(campaign, ext))):
                if overwrite == False:
                    log.info('File Exists. Skipping. Set overwrite to True to overwrite.')
                    continue
            try:
                urls = mast.get_tpf_urls('c{}'.format(campaign), ext)
            except mast.NoDataFoundException:
                log.info('Campaign {} Channel {} : No URLS found'.format(campaign, ext))
                continue
            cache_size = get_dir_size(get_cache_dir())/1E9

            log.debug('-------------------------------')
            log.debug('Campaign:\t {}'.format(campaign))
            log.debug('Channel:\t {}'.format(ext))
            log.debug('-------------------------------')
            log.debug('{} Files'.format(len(urls)))
            log.debug('{0:.2g} gb in astropy cache'.format(cache_size))

            if cache_size >= cachelim:
                log.debug('Cache hit limit of {} gb. Clearing.'.format(cachelim))
                clear_download_cache()

            if (indir is None) == False:
                log.debug('Building from input')
                tpf_filenames = np.asarray(['{}{}'.format(indir, u.split(
                    'https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1]) for u in urls])
                if os.path.isfile(tpf_filenames[0]) is False:
                    tpf_filenames = np.asarray(['{}{}'.format(indir, (u.split(
                        'https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1])).split('.gz')[0] for u in urls])
                if os.path.isfile(tpf_filenames[0]) is False:
                    log.debug('No MAST structure...trying again.')
                    tpf_filenames = np.asarray(['{}{}'.format(indir, (u.split(
                        'https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1]).split('/')[-1]) for u in urls])
                if os.path.isfile(tpf_filenames[0]) is False:
                    tpf_filenames = np.asarray(['{}{}'.format(indir, ((u.split(
                        'https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1]).split('/')[-1])).split('.gz')[0] for u in urls])
            else:
                log.debug('Downloading/Caching')
                tpf_filenames = [None]*len(urls)
                with click.progressbar(length=len(urls)) as bar:
                    for i, u in enumerate(urls):
                        with silence():
                            tpf_filenames[i] = download_file(u, cache=True)
                        bar.update(1)
                tpf_filenames = np.asarray(tpf_filenames)
            [log.debug(t) for t in tpf_filenames[0:10]]
            log.debug('...')
            log.debug('Building Campaign {} Channel {}'.format(campaign, ext))
            hdf5_mosaic(tpf_filenames, campaign, ext,
                        output_prefix='{}'.format(edir),
                        memory_lim=memory_lim)
            log.info('Campaign {} Complete'.format(campaign))
    log.info('ALL DONE')
    log.debug('-------------------------------')
