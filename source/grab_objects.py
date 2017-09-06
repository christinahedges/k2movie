from astropy.utils.data import download_file
import pandas as pd
import numpy as np
from tqdm import tqdm
from astroquery.simbad import Simbad
import logging
import warnings
from contextlib import contextmanager
import sys, os
import K2ephem
import matplotlib.pyplot as plt
import pickle
import json


@contextmanager
def silence():
    logger = logging.getLogger()
    logger.disabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


def reduce_list(infile,objfile,objname,overwrite=True):
    obj=pickle.load(open(objfile,'rb'))
    pos=np.where(obj.Name==objname)[0]
    if len(pos)!=1:
        print('Cannot find object')
        return None
    i=np.asarray(obj.loc[pos,['InvestigationID']])[0][0]
    campaign=np.asarray(obj.loc[pos,['Campaign']])[0][0]
    targlisturl='https://keplerscience.arc.nasa.gov/data/campaigns/c{}/K2Campaign{}targets.csv'.format(campaign,campaign)
    with silence():
        targlistfname=download_file(targlisturl,cache=True)
    df=pd.read_csv(targlistfname)
    ok=[(i in d[' Investigation IDs']) for j,d in df.iterrows()]
    df=df[ok].reset_index(drop=True)
    urls=np.loadtxt(infile,dtype=str)
    urls_i=np.asarray([(u.split('ktwo')[-1]).split('-')[0] for u in urls],dtype=int)
    df_i=np.asarray(df[df.columns[0]],dtype=int)
    ok=np.in1d(urls_i,df_i)
    if overwrite==True:
        np.savetxt(infile,urls[ok],fmt="%s")
        return
    else:
        return urls[ok]


def grab(outfile='out.p',plot=True):
    obj=pd.DataFrame(columns=['InvestigationID','Name','Campaign','dist','tpfs','minmag','maxmag'])
    k=0
    print('Finding TILES')
    for campaign in tqdm(range(15)):
        targlisturl='https://keplerscience.arc.nasa.gov/data/campaigns/c{}/K2Campaign{}targets.csv'.format(campaign,campaign)
        targlistfname=download_file(targlisturl,cache=True)
        df=pd.read_csv(targlistfname)

        mask=([('TILE' in d) for d in df[' Investigation IDs']])


        ids1=np.asarray(df[' Investigation IDs'][mask])

        ids=[]
        for i in ids1:
            for j in np.unique(i.split('|')):
                ids.append(j)
        ids=np.asarray(ids)
        tpfs=np.asarray([len(np.where(ids==i)[0]) for i in np.unique(ids)])

        ids=np.unique(ids)
        mask=[('TILE' in i) for i in ids]
        ids=ids[mask]
        tpfs=tpfs[mask]
        for i,t in zip(ids,tpfs):
            i=i.strip(' ')
            i2=np.asarray(i.split('_'))
            pos=np.where((i2!='LC')&(i2!='SSO')&(i2!='TILE')&(i2!='TROJAN')&(i2!='SC'))[0]
            i2=' '.join(i2[pos])
            i2=np.asarray(i2.split('-'))
            pos=np.where((i2!='LC')&(i2!='SSO')&(i2!='TILE')&(i2!='TROJAN')&(i2!='SC'))[0]
            i2=' '.join(i2[pos])
            obj.loc[k]=(np.transpose([i,i2,campaign,0,t,0,0]))
            k+=1

    obj=obj.reset_index(drop=True)
    print('Querying JPL')
    with tqdm(total=len(obj)) as pbar:
        for i,o in obj.iterrows():
            try:
                with silence():
                    df=K2ephem.get_ephemeris_dataframe(o.Name,o.Campaign,o.Campaign)
                obj.loc[i,['minmag']]=float(np.asarray(df.mag).min())
                obj.loc[i,['maxmag']]=float(np.asarray(df.mag).max())
                ra,dec=np.asarray(df.ra),np.asarray(df.dec)
                obj.loc[i,['dist']]=(np.nansum(((ra[1:]-ra[0:-1])**2+(dec[1:]-dec[0:-1])**2)**0.5))
                pbar.update()
            except:
                continue
                pbar.update()
    pickle.dump(obj,open(outfile,'wb'))
    print('Saved to {}'.format(outfile))
    if plot==True:
        minmag=np.asarray(obj.minmag,dtype=float)
        maxmag=np.asarray(obj.maxmag,dtype=float)
        dist=np.asarray(obj.dist,dtype=float)
        fig=plt.figure(figsize=(15,15))
        pos=dist!=0
        plt.scatter(dist[pos],maxmag[pos],c=np.asarray(obj.Campaign,dtype=float)[pos],s=100,cmap=plt.get_cmap('tab20b'),vmin=0,vmax=20)
        for n,m,d in zip(obj.Name,obj.maxmag,obj.dist):
            if float(m)!=0:
                plt.text(d+0.2,m+0.2,n,alpha=0.5)
        plt.xlabel('Distance Traveled in Campaign (deg)')
        plt.ylabel('Maximum Brightness (Mag)')
        plt.gca().invert_yaxis()
        cbar=plt.colorbar(ticks=np.arange(20))
        cbar.set_label('Campaign')
        plt.title('Moving Bodies in K2',fontsize=20)
        plt.savefig('bodies.png',dpi=200,bbox_inches='tight')
        print('Plot saved to bodies.png')


def plot_tracks(objfile='out.p'):
    obj=pickle.load(open(objfile,'rb'))
    fpurl='https://raw.githubusercontent.com/KeplerGO/K2FootprintFiles/master/json/k2-footprint.json'
    fpfname=download_file(fpurl,cache=True)
    fp = json.load(open(fpfname))


    with tqdm(total=14) as pbar:
        for campaign in range(14):
            o=obj[np.asarray(obj.Campaign,dtype=float)==campaign]
            srcs=len(o[np.asarray(o.dist,dtype=float)!=0])
            if srcs==0:
                pbar.update()
                continue
            fig,ax=plt.subplots(figsize=(10,10))
            for module in range(100):
                try:
                    ch = fp["c{}".format(campaign)]["channels"]["{}".format(module)]
                    ax.plot(ch["corners_ra"] + ch["corners_ra"][:1],
                            ch["corners_dec"] + ch["corners_dec"][:1],c='C0')
                    ax.text(np.mean(ch["corners_ra"] + ch["corners_ra"][:1]),np.mean(ch["corners_dec"] + ch["corners_dec"][:1]),'{}'.format(module),fontsize=10,color='C0',va='center',ha='center',alpha=0.3)
                except:
                    continue
            xlim,ylim=ax.get_xlim(),ax.get_ylim()

            for i,o in obj[np.asarray(obj.Campaign,dtype=float)==campaign].iterrows():
                if float(o.dist)==0:
                    continue
                try:
                    with silence():
                        df=K2ephem.get_ephemeris_dataframe(o.Name,campaign,campaign,step_size=2)
                except:
                    continue

                ra,dec=df.ra,df.dec
                if fp["c{}".format(campaign)]['ra']-np.mean(ra)>300:
                    ra+=360.
                ok=np.where((ra>xlim[0])&(ra<xlim[1])&(dec>ylim[0])&(dec<ylim[1]))[0]
                ra,dec=ra[ok],dec[ok]
                p=ax.plot(ra,dec,lw=(24.-float(o.maxmag))/2)
                c=p[0].get_color()
                ax.text(np.median(ra),np.median(dec)+0.2,o.Name,color=c,zorder=99,fontsize=10)


            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.title('Campaign {}'.format(campaign),fontsize=20)
            plt.xlabel('RA')
            plt.ylabel('Dec')

            fig.savefig('campaign{}.png'.format(campaign),bbox_inches='tight',dpi=200)
            pbar.update()
            plt.close()
