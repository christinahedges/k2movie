from . import __version__, PACKAGEDIR
import glob
import pickle
import re
from tqdm import tqdm
from astropy.utils.data import download_file,clear_download_cache
from astropy.config.paths import get_cache_dir
from astropy.io import fits
import fitsio
import numpy as np
import pandas as pd
import os
from k2mosaic import mast
from contextlib import contextmanager
import warnings
import sys
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


def pickle_wcs(output_fn=WCS_DIR, ffi_store=None):
    '''Writes the WCS solution function of each campaign and module to a pickled
    file.
    This will enable each X/Y coordinate to have an RA and Dec
    '''
    if (os.path.isdir(WCS_DIR) == False):
        os.makedirs(WCS_DIR)


    if ffi_store==None:
        ffi_filenames=[]
        while len(ffi_filenames)==0:
            ffi_store=input('Enter Directory with FFIs: ')
            ffi_filenames = glob.glob(os.path.join(ffi_store, '*cal.fits'))
            if len(ffi_filenames)==0:
                print('Cannot find FFIs')
    else:
        ffi_filenames = glob.glob(os.path.join(ffi_store, '*cal.fits'))
    with tqdm(total=len(ffi_filenames)) as pbar:
       for filename in ffi_filenames:
            basename = os.path.basename(filename)
            campaign = int(re.match(".*c([0-9]+)_.*", basename).group(1))
            if campaign<=2:
                if ('wcs' in filename) ==False:
                    continue
            fts = fits.open(filename)
            for ext in range(1, 85):
                pickle.dump(WCS(fts[ext].header),open('{}'.format(output_fn)+'c{0:02}_'.format(campaign)+'{0:02}.p'.format(ext),'wb'))
            pbar.update()
    print("Built WCS")

def get_wcs(campaign=0,channel=1,xs=[500],ys=[500],dir=WCS_DIR):
    r = pickle.load(open(dir+'c{0:02}_'.format(campaign)+'{0:02}.p'.format(channel),'rb'))
    RA,Dec = r.wcs_pix2world(xs,ys,1)
    return RA,Dec


def hdf5_mosaic(tpf_filenames, campaign, channel,
    output_prefix='', verbose=True,memory_lim=4,dtype=np.float16):
    '''Mosaic a set of TPFS into a dataframe'''
    if isinstance(tpf_filenames,str):
        print ('Parsing file...')
        tpf_filenames=np.loadtxt(tpf_filenames,dtype=str)
    wcs_file=WCS_DIR+'c{0:02}_'.format(campaign)+'{0:02}.p'.format(channel)
    if os.path.isfile(wcs_file)==False:
        print('No WCS found')
        pickle_wcs()
    if verbose:
        print('Loading WCS')
    with silence():
        r = pickle.load(open(wcs_file,'rb'))
    if tpf_filenames[0].startswith("http"):
        if verbose:
            tpf_filename = download_file(tpf_filenames[0],cache=True)
        else:
            with silence():
                tpf_filename = download_file(tpf_filenames[0],cache=True)
    else:
        tpf_filename=tpf_filenames[0]
    tpf = fitsio.FITS(tpf_filename)
    cadencelist=tpf[1]['CADENCENO'].read()
    tpf.close()
    cols=np.asarray(cadencelist,dtype=str)
    cols=np.append(['RA','Dec','X','Y'],cols)
    df=pd.DataFrame(columns=cols,dtype=dtype)
    fname=0
    if os.path.isfile('{}{}.h5'.format(output_prefix,fname)):
        if verbose:
            print('Clearing old file')
        os.remove('{}{}.h5'.format(output_prefix,fname))
    totalpixels=0
    with tqdm(total=len(tpf_filenames)) as bar:
        for i,tpf_filename in enumerate(tpf_filenames):
            if tpf_filename.startswith("http"):
                if verbose:
                    tpf_filename = download_file(tpf_filename,cache=True)
                else:
                    with silence():
                        tpf_filename = download_file(tpf_filename,cache=True)
            tpf = fitsio.FITS(tpf_filename)
            aperture_shape = tpf[1].read()['FLUX'][0].shape
            # Get the pixel coordinates of the corner of the aperture
            hdr_list = tpf[1].read_header_list()
            hdr = {elem['name']:elem['value'] for elem in hdr_list}
            col, row = int(hdr['1CRV5P']), int(hdr['2CRV5P'])
            height, width = aperture_shape[0], aperture_shape[1]
            # Fill the data
            mask = tpf[2].read() > 0
            y,x=np.meshgrid(np.arange(col,col+width),np.arange(row,row+height))
            x,y=x[mask],y[mask]
            totalpixels+=len(x.ravel())
            flux=(tpf[1].read()['FLUX'])
            tpf.close()

            f=np.asarray([f[mask].ravel() for f in flux])
            RA,Dec = r.wcs_pix2world(x.ravel(),y.ravel(),1)
            f=np.asarray(np.reshape(np.append([np.asarray(RA,dtype=float),
                                                np.asarray(Dec,dtype=float),
                                                np.asarray(x.ravel(),dtype=float),
                                                np.asarray(y.ravel(),dtype=float)],
                                                f),
                                                (np.shape(f)[0]+4,np.shape(f)[1])),dtype=float)
            with silence():
                df=df.append(pd.DataFrame(f.T,columns=cols,dtype=dtype))
            mem=np.nansum(df.memory_usage())/1E9
            #print ('\n',mem,np.nansum(df.count()),np.nansum(df.memory_usage())/(np.nansum(df.count())))
            if mem>=memory_lim:
                if verbose:
                    print('\n{} gb memory limit reached after {} TPFs. Writing to file.'.format(memory_lim,i))
                df.to_hdf('{}{}.h5'.format(output_prefix,fname),'table',append=True)
                df=pd.DataFrame(columns=cols,dtype=dtype)
            bar.update()
        df.to_hdf('{}{}.h5'.format(output_prefix,fname),'table',append=True)
        if verbose:
            print('\n{} Pixels Written'.format(totalpixels))
    return

def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def build(dir,indir=None,verbose=True,cachelim=30,overwrite=False):
    '''Creates a database of HDF5 files'''

    print ('-------------------------------')
    print ('Crunching K2 TPFS')
    if (os.path.isdir(WCS_DIR) == False):
        print ('No WCS Files Found')
        os.makedirs(WCS_DIR)
        pickle_wcs()

    if not os.path.isdir(dir):
        if verbose:
            print('Creating Directory')
        os.makedirs(dir)
    if indir is None:
        print('No input directory. Building URLS using k2mosaic.')

    if verbose:
        if overwrite:
            print('Overwrite enabled.')
    for campaign in range(14):
        cdir='{}'.format(dir)+'c{0:02}/'.format(campaign)
        if not os.path.isdir(cdir):
            os.makedirs(cdir)
            print (cdir)
        for ext in range(1,85):

            edir='{}'.format(cdir)+'{0:02}/'.format(ext)
            if not os.path.isdir(edir):
                os.makedirs(edir)
            if overwrite==False:
                if (os.path.isfile('{}'.format(edir)+'0.h5')):
                    if verbose:
                        print('File Exists. Skipping. Set overwrite to True to overwrite.')
                        continue
            try:
                urls = mast.get_tpf_urls('c{}'.format(campaign), ext)
            except:
                print('Channel {} : No URLS found?'.format(ext))
                continue
            cache_size=get_dir_size(get_cache_dir())/1E9

            if verbose:
                print ('-------------------------------')
                print ('Campaign:\t {}'.format(campaign))
                print ('Channel:\t {}'.format(ext))
                print ('-------------------------------')
                print ('{} URLs'.format(len(urls)))
                print ('{0:.2g} gb in astropy cache'.format(cache_size))

            if cache_size>=cachelim:
                print ('Cache hit limit of {} gb. Clearing.'.format(cachelim))
                clear_download_cache()

            print ('Downloading/Caching')
            return
            tpf_filenames=[None]*len(urls)
            if verbose:
                for i,u in enumerate(urls):
                    tpf_filenames[i] = download_file(u,cache=True)
            else:

                with click.progressbar(length=len(urls)) as bar:
                    for i,u in enumerate(urls):
                        with silence():
                            tpf_filenames[i] = download_file(u,cache=True)
                        bar.update(1)
            tpf_filenames=np.asarray(tpf_filenames)
            hdf5_mosaic(tpf_filenames, campaign, ext,
                        output_prefix='{}'.format(edir),verbose=verbose)
            if verbose:
                print ('Complete')
                print ('-------------------------------')
        break
    print ('ALL DONE')
    print ('-------------------------------')
