'''
Creates movies from HDF5 database of TPFs

Example usage coming soon.
'''

from astropy.utils.data import download_file
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
import dask.dataframe as dd
import astropy.units as u
import time
from tqdm import tqdm
import matplotlib.patheffects as path_effects
from matplotlib import animation
import K2ephem
from K2fov.K2onSilicon import onSiliconCheck, fields
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from astropy.wcs import FITSFixedWarning
from photutils import IRAFStarFinder
from photutils import Background2D, MedianBackground

from . import PACKAGEDIR
from .mast import *
from .build import *

DATA_DIR = os.path.join(PACKAGEDIR, 'data', 'database/')
TIME_FILE = os.path.join(PACKAGEDIR, 'data', 'campaign_times.txt')
WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')


class K2MovieInputError(Exception):
    pass


class movie(object):
    '''Class to create a movie of K2 data. Calling compute will run
    the movie creator.

    Parameters
    ----------
    name : str
        Name of object to resolve. Alternatively, pass an RA/Dec string or skycoord
    campaign : int
        Campaign number to search
    channel : int
        Channel to search
    aperture : int or np.ndarray of ones and zeros.
        aperture inside which to return the movie. If an int, will return a square of
        shape n x n. If an np.ndarray, will use that array.
    database_dir : str (default 'database/')
        location of the munged database.
    '''

    def __init__(self, name=None, campaign=None, channel=None, aperture=100, database_dir=DATA_DIR, output='out.mp4'):
        self.name = name
        if isinstance(aperture, int):
            self.aperture = np.ones((aperture, aperture))
        elif isinstance(aperture, np.ndarray):
            self.aperture = aperture
        else:
            raise K2MovieInputError('Input aperture is not an int or a numpy array.')
        self.tol_x = np.shape(self.aperture)[0]//2
        self.tol_y = np.shape(self.aperture)[1]//2

        self.campaign = campaign
        if (campaign not in np.arange(20, dtype=int)) and (campaign not in [91, 92, 101, 102, 111, 112]):
            raise K2MovieInputError('Campaign {} is not available.'.format(campaign))
        self.channel = channel
        if (channel not in np.arange(85, dtype=int)):
            raise K2MovieInputError('Channel {} is not available.'.format(channel))
        self.fname = '{0}c{1}/{2}/k2movie_c{1}_ch{2}.h5'.format(database_dir,
                                                                '{0:02}'.format(self.campaign), '{0:02}'.format(self.channel))
        self.err_fname = '{0}c{1}/{2}/k2movie_c{1}_ch{2}_ERR.h5'.format(database_dir,
                                                                        '{0:02}'.format(self.campaign), '{0:02}'.format(self.channel))
        if not os.path.isfile(self.fname):
            raise K2MovieInputError('No database file. Try running '
                                    '`k2movie build -c {} -ch {}``'
                                    ' in the terminal.'.format(campaign, channel))
        self.output = output
        self.times = pd.read_csv(TIME_FILE)
        log.debug('Campaign: {}'.format(self.campaign))
        log.debug('Channel: {}'.format(self.channel))
        log.debug('Output file: {}'.format(output))
        # Create times from the campaign number
        self.LC = 29.4295*u.min
        campaign_str = 'c{}'.format(self.campaign)
        self.start_time = np.asarray(
            self.times.StartTime[self.times.Campaign == campaign_str])[0]+2454833.
        self.end_time = np.asarray(
            self.times.EndTime[self.times.Campaign == campaign_str])[0]+2454833.
        self.start_cad = np.asarray(self.times.StartCad[self.times.Campaign == campaign_str])[0]
        self.end_cad = np.asarray(self.times.EndCad[self.times.Campaign == campaign_str])[0]
        self.ncad = self.end_cad - self.start_cad
        self.times = (np.arange(self.ncad)*self.LC).to(u.day).value+self.start_time
        try:
            self.ra, self.dec = findStatic(self.name)
            log.debug('Found static object {}'.format(self.name))
        except K2MovieNoObject:
            log.debug('No static object found for {}'.format(self.name))
        self.df = dd.read_hdf((self.fname), key='table')
        self.y, self.x = np.asarray(self.df[['Row', 'Column']].compute()).T
        self.cols = np.asarray(self.df.columns[5:], dtype=float)
        self.cadence_names = np.asarray(self.df.columns[5:], dtype=int)
        log.debug('Read in database')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            self.wcs_file = '{}{}'.format(WCS_DIR, 'c{0:02}_'.format(
                self.campaign)+'{0:02}.p'.format(self.channel))
            self.wcs = pickle.load(open(self.wcs_file, 'rb'))
        log.debug('Read in WCS')
        log.debug('Finding location on channel')
        self.x1, self.y1 = np.asarray([(np.asarray(self.wcs.wcs_world2pix(r, d, 1), dtype=float))
                                       for r, d in zip(self.ra, self.dec)]).T
        log.debug('Trimming data')

        self.df = self.df[(self.df['Column'] > self.x1.min()-self.tol_x) & (self.df['Column'] < self.x1.max()+self.tol_x) & (
            self.df['Row'] > self.y1.min()-self.tol_y) & (self.df['Row'] < self.y1.max()+self.tol_y)]
        self.populate()

    def populate(self):
        '''Fill an array with useful values'''
        log.debug('Creating data array')
        start = time.time()
        # Compute all cadences
        cols = list(self.df.columns[5:])
        a = np.asarray(self.df[cols].compute()).T
        if np.shape(a)[0] == 0:
            raise K2K2MovieInputError('No data found on this channel with this aperture.')
        x, y = np.asarray(self.df[['Column', 'Row']].compute()).T
        self.ar = np.zeros((self.aperture.shape[0], self.aperture.shape[1], np.shape(a)[0]))
        cols = list(self.df.columns[5:])
        # If there's only one location..
        if len(self.x1) == 1:
            xloc, yloc = x - self.x1[0] + self.tol_x, y - self.y1[0] + self.tol_x
            pos = np.where((xloc >= 0) & (xloc < self.tol_x * 2) &
                           (yloc >= 0) & (yloc < self.tol_y * 2))[0]

            for i, f in enumerate(a):
                self.ar[xloc[pos].astype(int), yloc[pos].astype(int), i] = f[pos]
        log.debug('Finished ({0:0.2g}s)'.format(time.time()-start))
        self.ar[~np.isfinite(self.ar)] = np.nan
        self.ar[self.ar == 0] = np.nan

    def movie(self, scale='linear', title='', text=True, **kwargs):
        '''Create a movie of a populated array'''
        log.debug('Creating movie')
        start = time.time()
        # If there's not a lot of movement the movie should be fixed
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        dat = (self.ar)
        if scale == 'log':
            dat = np.log10(self.ar)
        if 'vmin' not in kwargs:
            # Calculate color stretch...
            y = dat.ravel()
            y = y[np.isfinite(y)]
            kwargs['vmin'] = np.percentile(y, 10)
            kwargs['vmax'] = np.percentile(y, 90)

        im = ax.imshow(dat[:, :, 0].T, **kwargs)
        ax.axis('off')
        if text:
            text1 = ax.text(0.1, 0.9, title, fontsize=10,
                            color='white', transform=ax.transAxes)
            text1.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                    path_effects.Normal()])
            text2 = ax.text(0.1, 0.83, 'Campaign {}'.format(self.campaign),
                            fontsize=8, color='white', transform=ax.transAxes)
            text2.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                    path_effects.Normal()])
            text3 = ax.text(0.1, 0.78, 'Cadence: {}'.format(
                int(self.cadence_names[0])), fontsize=8, color='white', transform=ax.transAxes)
            text3.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                    path_effects.Normal()])

        def animate(i):
            im.set_array(dat[:, :, i].T)
            if text:
                text3.set_text('Cadence: {}'.format(int(self.cadence_names[i])))
                return im, text3,
            return im,
        anim = animation.FuncAnimation(fig, animate, frames=len(
            self.cadence_names), interval=30, blit=True)
        anim.save(self.output, dpi=150)
        log.debug('Saved. ({0:0.2g}s)'.format(time.time()-start))
        plt.close()
