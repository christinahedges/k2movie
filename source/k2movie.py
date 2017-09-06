from astropy.utils.data import download_file
import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import pickle
import dask.dataframe as dd
import astropy.units as u
import time
from tqdm import tqdm
import matplotlib.patheffects as path_effects
from matplotlib import animation


class k2movie(object):
    '''Object to hold all the data'''
    def __init__(self,
                name=None,
                campaign=0,
                channel=1,
                cadence=0,
                loc=None,
                tol=50,
                return_radec=False,
                return_xy=False,
                data_dir='/Users/ch/K2/projects/chiron/database/',
                verbose=False,
                outfile='out.mp4',
                vmin=2,
                vmax=5,
                title=None,
                inset=True,
                inset_size=10):
        self.verbose=verbose
        self.dir=data_dir
        self.campaign = campaign
        self.channel = channel
        if isinstance(cadence,int):
            self.cadence = [cadence]
        else:
            self.cadence = cadence
        self.loc=loc
        self.tol=tol
        self.return_radec=return_radec
        self.return_xy=return_xy
        self.wcs_file='/Users/ch/K2/repos/k2mosaic/k2mosaic/data/wcs/c{0:02}_'.format(campaign)+'{0:02}.p'.format(channel)
        self.r = pickle.load(open(self.wcs_file,'rb'))
        self.outfile=outfile
        self.inset=inset
        self.vmin=vmin
        self.vmax=vmax
        self.title=title
        self.inset_size=inset_size
        if self.verbose:
            print('Initialised:')
            print('\tCampaign: {}'.format(self.campaign))
            print('\tChannel: {}'.format(self.channel))
            print('\tCadence: {}'.format(self.cadence))
            print('\tPixel Tolerance: {}'.format(self.tol))
        self.start_time=(2384.45356226+2454833.) #Need start times for all campaigns in JD
        self.LC=29.4295*u.min
        self.name=name
        if (self.name is None):
            find()

        if (self.channel is None):
            self.fname=(self.dir+'c{}/{}/0.h5'.format('{0:02}'.format(self.campaign),'*'))
        else:
            self.fname=(self.dir+'c{}/{}/0.h5'.format('{0:02}'.format(self.campaign),'{0:02}'.format(self.channel)))
        self.df=dd.read_hdf(self.fname, key='table')
        self.cols=np.asarray(self.df.columns[4:],dtype=float)
        self.cadence_names=np.asarray(self.df.columns[4:],dtype=int)
        if (self.cadence is None) == False:
                self.cadence_names=self.cadence_names[self.cadence]

    def find(self):
        if (loc is None) == False:
            print ('Location and Target Name specified. Using Target Name')
        else:
            if self.verbose:
                print('Finding {}'.format(self.name))
        #Query simbad?

        #Query JPL?
        ra,dec,channel=findMoving(self.name,self.campaign)
        self.loc=[ra,dec]
        self.channel=channel[0]



    def findMoving(self,name,campaign):
        cadtime=((self.cols-self.cols[0])*self.LC).to(u.day).value+self.start_time

        #Sparsely look to save time
        cad=np.arange(0,len(self.cols),100)

        #Get the ephemeris from JPL
        df=K2ephem.get_ephemeris_dataframe(name,campaign,campaign,step_size=1./(4))
        times=[t[0:23] for t in np.asarray(df.index,dtype=str)]
        df_jd=Time(times,format='isot').jd
        ra,dec=np.interp(cadtime[cad],df_jd,df.ra)*u.deg,np.interp(cadtime[cad],df_jd,df.dec)*u.deg

        #Find those on silicon
        k = fields.getKeplerFov(campaign)
        mask=list(map(onSiliconCheck,ra.value,dec.value,np.repeat(k,len(ra))))
        ra,dec=[ra[mask],dec[mask]]
        channel=np.asarray(np.unique(k.getChannelColRowList(ra.value,dec.value)),dtype=int)
        if self.verbose():
            print('Channels: '.format(channel))
        #Use only those that are actually on Silicon
        cad=cad[mask]
        cad=np.arange(cad[0],cad[-1])
        #Interpolate each cadence
        ra,dec=np.interp(cadtime[cad],df_jd,df.ra)*u.deg,np.interp(cadtime[cad],df_jd,df.dec)*u.deg
        return ra,dec,channel




    def find_loc(self):
        '''Find the location, should be it's own function. Should output the right channel number.'''
        if (self.loc is None):
            print ('No location specified')
            return None
        xpos,ypos=self.loc[0],self.loc[1]
        radec=False
        if (isinstance(xpos,u.quantity.Quantity)):
            radec=True
            xpos,ypos=xpos.value,ypos.value
        if (isinstance(xpos,np.ndarray)):
            if (isinstance(xpos[0],u.quantity.Quantity)):
                radec=True
                xpos,ypos=xpos.value,ypos.value
        if radec == True:
            x1,y1=(np.asarray(self.r.wcs_world2pix(xpos,ypos,1),dtype=int))
        else:
            y1,x1=ypos,xpos


        if isinstance(x1,int) or isinstance(x1,float) or isinstance(x1,np.int64):
            x1=[x1]
            y1=[y1]
        self.x1=x1
        self.y1=y1
        if self.verbose:
            print('Finding location on focal plane')
            print('\tPixel Location: {} {}'.format(self.x1,self.y1))


    def trim(self):
        '''Cut down dataframe to relevant points'''
        if self.verbose:
            print('Trimming data')
        else:
            self.df=self.df[(self.df['X']>self.x1.min()-self.tol)&(self.df['X']<self.x1.max()+self.tol)&(self.df['Y']>self.y1.min()-self.tol)&(self.df['Y']<self.y1.max()+self.tol)]



    def populate(self):
        '''Fill an array with useful values'''
        if self.verbose:
            print('Creating data array')
            start=time.time()
        #Compute all cadences
        if self.cadence is None:
            a=np.asarray(self.df.compute())
            y,x=a[:,2:4].T
            a=a[:,4:].T
            self.ar=np.reshape(np.zeros(self.tol*2*self.tol*2*len(a)),(self.tol*2,self.tol*2,len(a)))
            xloc,yloc=x.astype(int)-self.x1+self.tol,y.astype(int)-self.y1+self.tol
            pos=np.where((xloc>=0) & (xloc<self.tol*2) & (yloc>=0) & (yloc<self.tol*2))[0]
            self.ar[xloc[pos],yloc[pos]]=a.T[pos]
        else:
            y,x=np.asarray(self.df[['X','Y']].compute()).T
            cols=list(self.df.columns[4:][self.cadence])
            a=np.asarray(self.df[cols].compute()).T
            self.ar=np.reshape(np.zeros(self.tol*2*self.tol*2*len(a)),(self.tol*2,self.tol*2,len(a)))
            #If there's only one location...
            if isinstance(self.x1,np.ndarray)==False:
                xloc,yloc=x.astype(int)-self.x1+self.tol,y.astype(int)-self.y1+self.tol
                pos=np.where((xloc>=0) & (xloc<self.tol*2) & (yloc>=0) & (yloc<self.tol*2))
                for i,f in enumerate(a):
                    self.ar[xloc[pos],yloc[pos],i]=f[pos]
            #If there are multiple locations...
            else:
                if self.verbose:
                    print('\tTracking...')
                for i,x2,y2,f in zip(range(len(self.x1)),self.x1,self.y1,a):
                    xloc,yloc=np.copy(x.astype(int))-x2+self.tol,np.copy(y.astype(int))-y2+self.tol
                    pos=np.where((xloc>=0) & (xloc<self.tol*2) & (yloc>=0) & (yloc<self.tol*2))[0]
                    self.ar[xloc[pos],yloc[pos],i]=f[pos]
        self.ar[self.ar==0]=np.nan
        if self.verbose:
            print('Finished ({0:0.2g}s)'.format(time.time()-start))


    def axes(self):
        '''Output the right axes, either in RA and Dec or X and Y'''
        if (self.return_radec != True) and (self.return_xy != True):
            return
        else:
            if self.verbose:
                print('Creating data axes')
            if len(self.x1)==1:
                x=np.arange(-self.tol,self.tol)+self.x1[0]
                y=np.arange(-self.tol,self.tol)+self.y1[0]
                X,Y=np.meshgrid(y,x)
            else:
                x=np.arange(-self.tol,self.tol)
                y=np.arange(-self.tol,self.tol)
                X,Y=np.meshgrid(y,x)
                X3,Y3=[],[]
                for i in range(len(self.x1)):
                    X3.append(X+self.x1[i])
                    Y3.append(Y+self.y1[i])
                X,Y=np.asarray(X3),np.asarray(Y3)

            if self.return_xy == True:
                self.ax1=X
                self.ax2=Y
                if self.verbose:
                    print('Creating Pixel Axes')
            self.r = pickle.load(open(self.wcs_file,'rb'))
            ra,dec=self.r.wcs_pix2world(X.ravel(),Y.ravel(),1)
            ra,dec=np.reshape(ra,np.shape(X)),np.reshape(dec,np.shape(X))
            if self.return_radec == True:
                self.ax1=ra
                self.ax2=dec
                if self.verbose:
                    print('Creating RA/Dec Axes')


    def stabilize(self):
        '''Use photutils to find the source in the images and stabilise the movie'''
        #If no stabilising...just make some limits based on tolerance
        if (self.cadence is None):
            length=np.shape(self.ar)[2]
        else:
            length=len(self.cadence)

        self.xlim=np.zeros(length)+self.tol
        self.ylim=np.zeros(length)+self.tol



    def movie(self):
        '''Create a movie of a populated array'''

        if self.verbose:
            print('Writing to movie')
            print ('\tOutput file: {}'.format(self.outfile))
        cmap = plt.get_cmap('gray')
        #cmap.set_bad('black',1.)
        #If there's not a lot of movement the movie should be fixed
        fig=plt.figure(figsize=(4,4))
        ax=fig.add_subplot(111)
        im=ax.imshow(np.log10(self.ar[:,:,0]),cmap=cmap,vmin=self.vmin,vmax=self.vmax,origin='bottom',interpolation='none')
        ax.set_xlim(self.xlim[0]-self.tol,self.xlim[0]+self.tol)
        ax.set_ylim(self.ylim[0]-self.tol,self.ylim[0]+self.tol)
        if self.inset:
            ax.set_xlim(self.xlim[0]-self.tol,self.xlim[0]+self.tol)
            ax.set_ylim(self.ylim[0]-self.tol,self.ylim[0]+self.tol)
        #ax.patch.set_facecolor('black')
        ax.axis('off')
        if (self.title is None)==False:
            text1=ax.text(0.1,0.9,self.title,fontsize=10,color='white',transform=ax.transAxes)
            text1.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])

        text2=ax.text(0.1,0.83,'Campaign {}'.format(self.campaign),fontsize=8,color='white',transform=ax.transAxes)
        text2.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                               path_effects.Normal()])
        text3=ax.text(0.1,0.78,'Cadence: {}'.format(int(self.cadence_names[0])),fontsize=8,color='white',transform=ax.transAxes)
        text3.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                               path_effects.Normal()])


        if self.inset:
            newax = fig.add_axes([0.62, 0.62 , 0.25, 0.25], zorder=2)
            inset=newax.imshow(np.log10(self.ar[:,:,0]),cmap=cmap,vmin=self.vmin,vmax=self.vmax,origin='bottom')
            newax.set_xlim(self.xlim[0]-self.inset_size,self.xlim[0]+self.inset_size)
            newax.set_ylim(self.ylim[0]-self.inset_size,self.ylim[0]+self.inset_size)
            newax.spines['bottom'].set_color('red')
            newax.spines['top'].set_color('red')
            newax.spines['right'].set_color('red')
            newax.spines['left'].set_color('red')
            newax.set_xticks([])
            newax.set_yticks([])



        def animate(i):
            im.set_array(np.log10(self.ar[:,:,i]))
            ax.set_xlim(self.xlim[i]-self.tol,self.xlim[i]+self.tol)
            ax.set_ylim(self.ylim[i]-self.tol,self.ylim[i]+self.tol)

            if self.inset:
                ax.set_xlim(self.xlim[i]-self.tol,self.xlim[i]+self.tol)
                ax.set_ylim(self.ylim[i]-self.tol,self.ylim[i]+self.tol)

                inset.set_array(np.log10(self.ar[:,:,i]))
                newax.set_xlim(self.xlim[i]-self.inset_size,self.xlim[i]+self.inset_size)
                newax.set_ylim(self.ylim[i]-self.inset_size,self.ylim[i]+self.inset_size)

            text3.set_text('Cadence: {}'.format(int(self.cadence_names[i])))
            if self.inset:
                return im,inset,text3,
            else:
                return im,text3,

        anim = animation.FuncAnimation(fig,animate,frames=len(self.cadence_names), interval=15, blit=True)
        if self.verbose:
            print ('Saving...')
            start=time.time()
        anim.save(self.outfile,dpi=150)
        if self.verbose:
            print ('Saved. ({0:0.2g}s)'.format(time.time()-start))
        plt.close()

    def compute(self,return_ar=False):
        self.find_loc()
        self.trim()
        self.populate()
        self.axes()
        self.stabilize()
        self.movie()
        if return_ar:
            if (self.return_xy==False) and (self.return_radec==False):
                return self.ar
            else:
                return self.ar,self.ax1,self.ax2

    def inspect(self,ax,radec=False):
        if self.cadence is None:
            cadence=0
        else:
            cadence=self.cadence[0]
        cols=self.df.columns[4:][cadence]
        x,y,f=np.asarray(self.df[['X','Y',cols]].compute()).T
        ar=np.zeros((1200,1200))*np.nan
        X,Y=np.meshgrid(np.arange(1200),np.arange(1200))
        ar[x.astype(int),y.astype(int)]=f
        if radec==True:
            ra,dec=self.r.wcs_pix2world(X.ravel(),Y.ravel(),1)
            ra,dec=np.reshape(ra,np.shape(X)),np.reshape(dec,np.shape(X))
            ax.contourf(ra,dec,ar)
        else:
            ax.contourf(X,Y,ar)

        if (self.loc is None) == False:
            self.find_loc()
            if radec==True:
                ra,dec=self.r.wcs_pix2world(self.x1,self.y1,1)
                ax.scatter(ra,dec,c='C3')
            else:
                ax.scatter(self.x1,self.y1)
