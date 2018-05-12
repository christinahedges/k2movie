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

    def __init__(self, name=None, campaign=None, channel=None, aperture=8, database_dir=DATA_DIR, output='out.mp4'):
        self.name = name
        if isinstance(aperture, int):
            self.aperture = np.ones((aperture, aperture))
        elif isinstance(aperture, np.ndarray):
            self.aperture = aperture
        else:
            raise K2MovieInputError('Input aperture is not an int or a numpy array.')
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
        try:
            self.ra, self.dec = findStatic(self.name)
            log.debug('Found static object {}'.format(self.name))
        except K2MovieNoObject:
            try:
                self.ra, self.dec = findMoving(name, campaign, channel, time)
                log.debug('Found moving object {}'.format(self.name))
            except K2MovieNoObject:
                raise K2MovieInputError('No object {}'.format(self.name))

    def find(self):
        '''Find the ras and decs of the target'''
        if (self.loc is None) == False and (self.name is None) == False:
            print('Location and Target Name specified. Using Location')
        else:
            if self.verbose:
                print('Finding {}'.format(self.loc))
            # Query simbad?
            if self.objType == 'static':
                self.findStatic()
            # If it's still not right...Query JPL?
            if (self.loc is None) or (self.objType == 'moving'):
                self.findMoving()
            if self.loc is None:
                print('No target found?')
                print(self.channel)
            if self.channel is None:
                if self.verbose:
                    print('Finding channels...')
                ra, dec = self.loc
                k = fields.getKeplerFov(self.campaign)
                onsil = np.asarray(
                    list(map(onSiliconCheck, [ra.value], [dec.value], np.repeat(k, 1))))[0]
                if (onsil is False):
                    print('Not on silicon?')
                else:
                    self.channel = [(k.getChannelColRow(ra.value, dec.value)[0]).astype(int)]
                    if self.verbose:
                        print('Channel: {}'.format(self.channel))
        self.find_loc()

    def find_loc(self):
        '''Find the location of the target on the detector'''
        if (self.loc is None):
            print('No location specified')
            return None
        if self.verbose:
            print('Finding location on focal plane')

        xpos, ypos = self.loc[0], self.loc[1]
        radec = False
        if (isinstance(xpos, u.quantity.Quantity)):
            radec = True
            xpos, ypos = xpos.value, ypos.value
        if (isinstance(xpos, np.ndarray)):
            if (isinstance(xpos[0], u.quantity.Quantity)):
                radec = True
                xpos, ypos = xpos.value, ypos.value
        if radec == True:
            if len(self.channel) >= 1:
                wcs_file = '{}{}'.format(WCS_DIR, 'c{0:02}_'.format(
                    self.campaign)+'{0:02}.p'.format(self.channel[0]))
                r = pickle.load(open(wcs_file, 'rb'))
                x1, y1 = (np.asarray(r.wcs_world2pix(xpos, ypos, 1), dtype=float))
            else:
                ch0 = -1
                x1, y1 = np.zeros(len(xpos)), np.zeros(len(xpos))
                for i, ch, x, y in zip(range(len(xpos)), self.channel, xpos, ypos):
                    if ch != ch0:
                        wcs_file = '{}{}'.format(WCS_DIR, 'c{0:02}_'.format(
                            self.campaign)+'{0:02}.p'.format(ch.astype(int)))
                        r = pickle.load(open(wcs_file, 'rb'))
                        ch0 = ch
                    x1[i], y1[i] = (np.asarray(r.wcs_world2pix(x, y, 1), dtype=float))

        else:
            y1, x1 = ypos, xpos
        if isinstance(x1, int) or isinstance(x1, float) or isinstance(x1, np.int64):
            x1 = [x1]
            y1 = [y1]
        self.x1 = x1
        self.y1 = y1

    def trim(self):
        '''Cut down dataframe to relevant points'''
        if self.verbose:
            print('Trimming data')
        else:
            self.df = self.df[(self.df['Row'] > self.x1.min()-self.tol) & (self.df['Row'] < self.x1.max()+self.tol) & (
                self.df['Column'] > self.y1.min()-self.tol) & (self.df['Column'] < self.y1.max()+self.tol)]

    def populate(self):
        '''Fill an array with useful values'''
        if self.verbose:
            print('Creating data array')
            start = time.time()
        # Compute all cadences
        self.ar = np.reshape(np.zeros(self.tol*2*self.tol*2*len(self.cadence)),
                             (self.tol*2, self.tol*2, len(self.cadence)))
        cols = list(self.df.columns[5:][self.cadence])

        for ch in np.unique(self.channel):
            if self.verbose:
                print('Switching to channel {}'.format(ch))
            self.fname = self.dir +\
                'c{}/{}/0.h5'.format('{0:02}'.format(self.campaign), '{0:02}'.format(ch))
            if os.path.isfile(self.fname) == False:
                if self.verbose:
                    print('No file.')
                continue
            self.df = dd.read_hdf((self.fname), key='table')
            a = np.asarray(self.df[cols].compute()).T
            y, x = np.asarray(self.df[['Row', 'Column']].compute()).T
            self.y = y
            self.x = x
            ch0 = ch
            # If there's only one location...
            if isinstance(self.x1, np.ndarray) == False:
                xloc, yloc = x-self.x1+self.tol, y-self.y1+self.tol
                pos = np.where((xloc >= 0) & (xloc < self.tol*2) &
                               (yloc >= 0) & (yloc < self.tol*2))[0]

                for i, f in enumerate(a):
                    if len(self.channel) > 1:
                        if self.channel[i] != ch:
                            continue
                    if np.nansum(f[pos]) != 0:
                        self.ar[xloc[pos].astype(int), yloc[pos].astype(int), i] = f[pos]
            # If there are multiple locations...
            else:
                if self.verbose:
                    print('\tTracking...')
                for i, x2, y2, f in zip(range(len(self.x1)), self.x1, self.y1, a):
                    if len(self.channel) > 1:
                        if self.channel[i] != ch:
                            continue
                    xloc, yloc = np.copy(x)-x2+self.tol, np.copy(y)-y2+self.tol
                    pos = np.where((xloc >= 0) & (xloc < self.tol*2) &
                                   (yloc >= 0) & (yloc < self.tol*2))[0]
                    if np.nansum(f[pos]) != 0:
                        self.ar[xloc[pos].astype(int), yloc[pos].astype(int), i] = f[pos]
        self.ar[self.ar == 0] = np.nan
        if self.verbose:
            print('Finished ({0:0.2g}s)'.format(time.time()-start))

    def axes(self):
        '''Output the right axes, either in RA and Dec or X and Y'''
        if (self.return_radec != True) and (self.return_xy != True):
            return
        else:
            if self.verbose:
                print('Creating data axes')
            if len(self.x1) == 1:
                x = np.arange(-self.tol, self.tol)+self.x1[0]
                y = np.arange(-self.tol, self.tol)+self.y1[0]
                X, Y = np.meshgrid(y, x)
            else:
                x = np.arange(-self.tol, self.tol)
                y = np.arange(-self.tol, self.tol)
                X, Y = np.meshgrid(y, x)
                X3, Y3 = [], []
                for i in range(len(self.x1)):
                    X3.append(X+self.x1[i])
                    Y3.append(Y+self.y1[i])
                X, Y = np.asarray(X3), np.asarray(Y3)

            if self.return_xy == True:
                self.ax1 = X
                self.ax2 = Y
                if self.verbose:
                    print('Creating Pixel Axes')
            self.r = pickle.load(open(self.wcs_file, 'rb'))
            ra, dec = self.r.wcs_pix2world(X.ravel(), Y.ravel(), 1)
            ra, dec = np.reshape(ra, np.shape(X)), np.reshape(dec, np.shape(X))
            if self.return_radec == True:
                self.ax1 = ra
                self.ax2 = dec
                if self.verbose:
                    print('Creating RA/Dec Axes')

    def stabilize(self):
        '''Use photutils to find the source in the images and stabilise the movie'''
        if self.stabilize_corr == True:
            # If no stabilising...just make some limits based on tolerance
            if self.verbose:
                print('Stabilizing source motion...')
            self.xlim = np.zeros(self.ncad)+self.tol
            self.ylim = np.zeros(self.ncad)+self.tol
            for i, data in enumerate(self.ar.T):
                data[np.isfinite(data) == False] = 0
                mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
                daofind = IRAFStarFinder(fwhm=3, threshold=5.*std)
                sources = daofind(data - median)
                try:
                    self.xlim[i] = sources[np.argmax(sources['flux'])]['xcentroid']
                    self.ylim[i] = sources[np.argmax(sources['flux'])]['ycentroid']
                except:
                    continue
            # If background_corr then remove the background from all the images...
        else:
            self.xlim = np.zeros(self.ncad)+self.tol
            self.ylim = np.zeros(self.ncad)+self.tol

    def movie(self):
        '''Create a movie of a populated array'''
        plt.style.use(self.stylesheet)
        if self.verbose:
            print('Writing to movie')
            print('\tOutput file: {}'.format(self.outfile))
        cmap = plt.get_cmap(self.cmap)
        cmap.set_bad(self.badcol, 1.)
        # If there's not a lot of movement the movie should be fixed
        fig = plt.figure(figsize=(4, 4))
        if self.colorbar == True:
            fig = plt.figure(figsize=(5, 4))
        if self.colorbar is False:
            ax = fig.add_axes([0., 0., 1.01, 1.01, ])
        else:
            ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        if self.scale == 'log':
            dat = np.log10(self.ar)
        if self.scale == 'linear':
            dat = (self.ar)
        if self.vmin is None:
            # Calculate color stretch...
            if self.verbose:
                print('Calculating color stretch')
            y = dat.ravel()
            y = y[np.isfinite(y)]
            self.vmin = np.percentile(y, 10)
            self.vmax = np.percentile(y, 90)
            if self.verbose:
                print('Color stretch: {} - {}'.format(self.vmin, self.vmax))

        im = ax.imshow(dat[:, :, 0].T, cmap=cmap, vmin=self.vmin,
                       vmax=self.vmax, origin='bottom', interpolation='none')
        if self.colorbar == True:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=10)
            if self.scale == 'linear':
                cbar.set_label('e$^-$s$^-1$', fontsize=10)
            if self.scale == 'log':
                cbar.set_label('log10(e$^-$s$^-1$)', fontsize=10)

        ax.set_xlim(self.xlim[0]-self.tol, self.xlim[0]+self.tol)
        ax.set_ylim(self.ylim[0]-self.tol, self.ylim[0]+self.tol)
        if self.inset:
            ax.set_xlim(self.xlim[0]-self.tol, self.xlim[0]+self.tol)
            ax.set_ylim(self.ylim[0]-self.tol, self.ylim[0]+self.tol)
        # ax.patch.set_facecolor('black')
        ax.axis('off')
        if self.text:

            if (self.title is None) == False:
                text1 = ax.text(0.1, 0.9, self.title, fontsize=10,
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

        if self.inset:
            newax = fig.add_axes([0.62, 0.62, 0.25, 0.25], zorder=2)
            inset = newax.imshow(dat[:, :, 0].T, cmap=cmap, vmin=self.vmin,
                                 vmax=self.vmax, origin='bottom')
            newax.set_xlim(self.xlim[0]-self.inset_size, self.xlim[0]+self.inset_size)
            newax.set_ylim(self.ylim[0]-self.inset_size, self.ylim[0]+self.inset_size)
            newax.spines['bottom'].set_color('red')
            newax.spines['top'].set_color('red')
            newax.spines['right'].set_color('red')
            newax.spines['left'].set_color('red')
            newax.set_xticks([])
            newax.set_yticks([])

        def animate(i):
            im.set_array(dat[:, :, i].T)
            ax.set_xlim(self.xlim[i]-self.tol, self.xlim[i]+self.tol)
            ax.set_ylim(self.ylim[i]-self.tol, self.ylim[i]+self.tol)

            if self.inset:
                ax.set_xlim(self.xlim[i]-self.tol, self.xlim[i]+self.tol)
                ax.set_ylim(self.ylim[i]-self.tol, self.ylim[i]+self.tol)

                inset.set_array(dat[:, :, i].T)
                newax.set_xlim(self.xlim[i]-self.inset_size, self.xlim[i]+self.inset_size)
                newax.set_ylim(self.ylim[i]-self.inset_size, self.ylim[i]+self.inset_size)
            if self.text:
                text3.set_text('Cadence: {}'.format(int(self.cadence_names[i])))

            if self.inset:
                if self.text:
                    return im, inset, text3,
                else:
                    return im, inset,
            else:
                if self.text:
                    return im, text3,
                else:
                    return im,
        anim = animation.FuncAnimation(fig, animate, frames=len(
            self.cadence_names), interval=self.frameinterval, blit=True)
        if self.verbose:
            print('Saving...')
            start = time.time()
        anim.save(self.outfile, dpi=150)
        if self.verbose:
            print('Saved. ({0:0.2g}s)'.format(time.time()-start))
        plt.close()

    def compute(self, return_ar=False):
        self.produce()
        if len(np.where(np.isfinite(np.asarray(self.x1) * np.asarray(self.y1)))[0]) == 0:
            print('No data?')
            return
#        self.trim()
        self.populate()
        self.axes()
        self.stabilize()
        self.movie()
        if return_ar:
            if (self.return_xy == False) and (self.return_radec == False):
                return self.ar
            else:
                return self.ar, self.ax1, self.ax2

    def inspect(self, ax=None, radec=True, source=True, return_ar=False):
        plt.style.use(self.stylesheet)
        self.produce()
        if len(np.where(np.isfinite(np.asarray(self.x1) * np.asarray(self.y1)))[0]) == 0:
            print('No data?')
            return
        tol = self.tol
        if ax == None:
            fig, ax = plt.subplots(1, figsize=(8, 8))

        cmap = plt.get_cmap(self.cmap)
        cmap.set_bad(self.badcol, 1.)
        ax.set_facecolor(self.badcol)
        if self.verbose:
            print('Inspecting')
        if self.cadence is None:
            cadence = 0
        else:
            cadence = self.cadence[0]

        iterat = np.unique(self.channel)
        for ch in iterat:
            fname = self.dir + \
                'c{}/{}/0.h5'.format('{0:02}'.format(self.campaign), '{0:02}'.format(ch))
            if os.path.isfile(fname) is False:
                continue
            df = dd.read_hdf(fname, 'table')
            wcs_file = '{}{}'.format(WCS_DIR, 'c{0:02}_'.format(
                self.campaign)+'{0:02}.p'.format(ch))
            r = pickle.load(open(wcs_file, 'rb'))
            cols = df.columns[5:][cadence]
            x, y, f = np.asarray(df[['Row', 'Column', cols]].compute()).T
            ar = np.zeros((1150, 1150))*np.nan
            X, Y = np.meshgrid(np.arange(1150), np.arange(1150))
            ar[x.astype(int), y.astype(int)] = f
            if self.scale == 'log':
                ar = np.log10(ar)
            y = ar.ravel()
            y = y[np.isfinite(y)]
            vmin = np.percentile(y, 1)
            vmax = np.percentile(y, 99)
            if radec == True:
                ra, dec = r.wcs_pix2world(X.ravel(), Y.ravel(), 1)
                ra, dec = np.reshape(ra, np.shape(X)), np.reshape(dec, np.shape(X))
                ax.contourf(ra, dec, ar, cmap=cmap, vmin=vmin, vmax=vmax)
                # plt.text(np.mean(ra),np.mean(dec),'{}'.format(ch),fontsize=30)
            else:
                ax.contourf(X, Y, ar, cmap=cmap, vmin=vmin, vmax=vmax)

        if (self.loc is None) == False:
            if radec == True:
                for ch in iterat:
                    wcs_file = '{}{}'.format(WCS_DIR, 'c{0:02}_'.format(
                        self.campaign)+'{0:02}.p'.format(ch))
                    r = pickle.load(open(wcs_file, 'rb'))
                    ra, dec = r.wcs_pix2world(self.x1, self.y1, 1)
                    if source:
                        ax.scatter(ra, dec, edgecolor='C3', s=300, facecolor='None', label='Source')

                ax.set_xlim(np.asarray(ra).min()-(tol*4*u.arcsecond).to(u.deg).value,
                            np.asarray(ra).max()+(tol*4*u.arcsecond).to(u.deg).value)
                ax.set_ylim(np.asarray(dec).min()-(tol*4*u.arcsecond).to(u.deg).value,
                            np.asarray(dec).max()+(tol*4*u.arcsecond).to(u.deg).value)
                ax.set_ylabel('Dec ($^{\circ}$)', fontsize=15)
                ax.set_xlabel('RA ($^{\circ}$)', fontsize=15)
            else:
                if source:
                    ax.scatter(self.x1, self.y1, edgecolor='C3',
                               s=300, facecolor='None', label='Source')
                ax.set_xlim(np.asarray(self.x1).min()-tol, np.asarray(self.x1).max()+tol)
                ax.set_ylim(np.asarray(self.y1).min()-tol, np.asarray(self.y1).max()+tol)
                ax.set_ylabel('Y Pixel', fontsize=15)
                ax.set_xlabel('X Pixel', fontsize=15)
            if self.name is None:
                ax.set_title('Campaign: {}'.format(self.campaign), fontsize=20)
            else:
                ax.set_title('Campaign: {} Source: {}'.format(
                    self.campaign, self.name), fontsize=20)
            if source:
                ax.legend()
        if return_ar:
            if radec:
                return ar, ra, dec
            else:
                return ar, X, Y
