import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.stats import spearmanr
import fitsio
from glob import glob
import numpy as np
from tqdm import tqdm
from photutils import DAOStarFinder,IRAFStarFinder
from astropy.stats import sigma_clipped_stats
from matplotlib import animation
import pickle
import warnings
warnings.filterwarnings("ignore")


from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats

def movie(fnames,cad,xlim,ylim,outfile='out.mp4',n=35,n2=5,xoffset=0,title=None,vmin=0,vmax=3,campaign=None):
    with tqdm(total=(len(fnames))) as pbar:
        cmap = plt.get_cmap('gray')
        cmap.set_bad('black',1.)
        #If there's not a lot of movement the movie should be fixed
        if np.max(xlim)-np.min(xlim)<=n:
            print ('fixing axes')
            xlim=np.zeros(len(xlim))+np.mean(xlim)
            ylim=np.zeros(len(ylim))+np.mean(ylim)

        fig=plt.figure(figsize=(4,4))
        tpf = fitsio.FITS(fnames[0])
        ax=fig.add_subplot(111)
        im=ax.imshow(np.log10(tpf[1].read()),cmap=cmap,vmin=vmin,vmax=vmax,origin='bottom',interpolation='none')
        ax.set_xlim(xlim[0]-n,xlim[0]+n)
        ax.set_ylim(ylim[0]-n,ylim[0]+n)
        ax.axis('off')
        if (title is None)==False:
            text1=ax.text(0.1,0.9,'Title',fontsize=10,color='white',transform=ax.transAxes)
            text1.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])
        if (campaign is None)==False:
            text2=ax.text(0.1,0.83,'Campaign {}'.format(campaign),fontsize=8,color='white',transform=ax.transAxes)
            text2.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])
        text3=ax.text(0.1,0.78,'Cadence: {}'.format(int(cad[0])),fontsize=8,color='white',transform=ax.transAxes)
        text3.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                               path_effects.Normal()])


        newax = fig.add_axes([0.62, 0.62 , 0.25, 0.25], zorder=2)
        inset=newax.imshow(np.log10(tpf[1].read()),cmap=cmap,vmin=0,vmax=2.5,origin='bottom')
        newax.set_xlim(xlim[0]-n2+xoffset,xlim[0]+n2+xoffset)
        newax.set_ylim(ylim[0]-n2,ylim[0]+n2)
        newax.spines['bottom'].set_color('red')
        newax.spines['top'].set_color('red')
        newax.spines['right'].set_color('red')
        newax.spines['left'].set_color('red')
        newax.set_xticks([])
        newax.set_yticks([])



        def animate(i):
            tpf = fitsio.FITS(fnames[i])
            im.set_array(np.log10(tpf[1].read()))
            ax.set_xlim(xlim[i]-n+xoffset,xlim[i]+n+xoffset)
            ax.set_ylim(ylim[i]-n,ylim[i]+n)

            inset.set_array(np.log10(tpf[1].read()))
            newax.set_xlim(xlim[i]-n2+xoffset,xlim[i]+n2+xoffset)
            newax.set_ylim(ylim[i]-n2,ylim[i]+n2)

            text3.set_text('Cadence: {}'.format(int(cad[i])))
            pbar.update()
            return im,inset,text3,

        anim = animation.FuncAnimation(fig,animate,frames=len(fnames), interval=15, blit=True)
        print ('Writing to {} ...'.format(outfile))
        anim.save(outfile,dpi=150)

def add_family(s,cadence,dict=None):
    if dict is None:
        xloc,yloc,tol=s['xcentroid'],s['ycentroid'],15
        def nextpos(inx,iny,inc):
            dist=((inx-xloc)**2+(iny-yloc)**2)**0.5
            return dist,dist<tol
        newdict={'x':[s['xcentroid']],
                 'y':[s['ycentroid']],
                 'c':cadence,'nextpos':nextpos,'age':0}
    else:
        x=np.append(dict['x'],s['xcentroid'])
        y=np.append(dict['y'],s['ycentroid'])
        c=np.append(dict['c'],cadence)
        if (len(x)<=10):
            def nextpos(inx,iny,inc):
                xerr=np.max([np.mean(x)-np.min(x)+10.,np.max(x)-np.mean(x)+10.])
                yerr=np.max([np.mean(y)-np.min(y)+10.,np.max(y)-np.mean(y)+10.])
                xloc,yloc,tol=np.mean(x),np.mean(y),((xerr**2)+(yerr**2))**0.5
                dist=((inx-xloc)**2+(iny-yloc)**2)**0.5
                return dist,dist<tol
        else:
            def nextpos(inx,iny,inc):
                xloc=np.polyval(np.polyfit(c,x,4),inc)
                yloc=np.polyval(np.polyfit(c,y,4),inc)
                tol=30
                dist=((inx-xloc)**2+(iny-yloc)**2)**0.5
                return dist,dist<tol

        newdict={'x':x,'y':y,'c':c,'nextpos':nextpos,'age':0}
    return newdict


def track(fnames,cadences,tpf0,threshold=150,top=None):
    print('Finding source families')
    families=np.empty(0)
    with tqdm(total=len(fnames)) as pbar:
        for c,fname in zip(cadences,fnames):
            tpf = fitsio.FITS(fname)
            data=tpf[1].read()-tpf0
            tpf.close()

            mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
            data[np.isfinite(data)==False]=0
            if (threshold is None):
                threshold=10.*std
            daofind = DAOStarFinder(fwhm=3.0, threshold=threshold)
            sources = daofind(data - median)

            for s in sources:
                if len(families)==0:
                    families=np.append(families,add_family(s,c))
                    continue
                x,y=s['xcentroid'],s['ycentroid']
                d,ac=np.zeros(len(families)),np.zeros(len(families),dtype=bool)
                for i in range(len(families)):
                    d[i],ac[i]=(families[i]['nextpos'](x,y,c))

                if (np.any(ac)==True):
                    d[np.where(ac!=True)]=np.nan
                    families[np.nanargmin(d)]=add_family(s,c,families[np.nanargmin(d)])
                if (np.any(ac)==False):
                    families=np.append(families,add_family(s,c))

            #Prune old, small families
            for i in range(len(families)):
                families[i]['age']+=1
            ages=np.asarray([f['age'] for f in families])
            sizes=np.asarray([len(f['x']) for f in families])
            families=families[np.where((ages<=10)|(sizes>=3))[0]]
            pbar.update()
    return families


def run(fitsdir='../data/c12-ch30/',startingpoint=[920,610],reduce=10,calculate=False,stabilise=False,vmin=0,vmax=10):

    fnames=np.asarray(glob('{}*.fits'.format(fitsdir)))
    cad=np.asarray([float((f.split('-cad')[-1])[:-5]) for f in fnames])
    tpf0=fitsio.FITS(fnames[0])[1].read()
    lastx,lasty=startingpoint[0],startingpoint[1]
    xs,ys=np.zeros(len(fnames)),np.zeros(len(fnames))
    ts=np.linspace(0,len(fnames),len(fnames)/10,dtype=int)-1
    ts[0]=1

    print('Calculating X and Y positions')
    with tqdm(total=len(fnames[ts])) as pbar:
        for i,c,f in zip(ts,cad[ts],np.asarray(fnames)[ts]):
            tpf = fitsio.FITS(f)
            data=tpf[1].read()-tpf0
            tpf.close()
            mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
            data[np.isfinite(data)==False]=0
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
            sources = daofind(data - median)

            pos=np.where(sources['peak']>150)[0]
            if len(pos)==0:
                pbar.update()
                continue
            x,y=np.asarray(sources['xcentroid'][pos]),np.asarray(sources['ycentroid'][pos])
            dist=(((lastx-x)**2+(lasty-y)**2)**0.5)
            pos=np.where(dist<=30)[0]
            if len(pos)==0:
                pbar.update()
                continue
            src=pos[np.argmin(dist[pos])]

            xs[i],ys[i]=x[src],y[src]

            if len(np.where(xs!=0)[0])<5:
                lastx,lasty=x[src],y[src]
            else:
                try:
                    lastx=np.polyval(np.polyfit(cad[np.where(xs!=0)],xs[np.where(xs!=0)],2),cad[ts][i+1])
                    lasty=np.polyval(np.polyfit(cad[np.where(ys!=0)],ys[np.where(ys!=0)],2),cad[ts][i+1])
                except:
                    continue
            pbar.update()

    print ('Plotting X and Y Positions')
    plt.figure()
    plt.scatter(cad[np.where(xs!=0)],xs[np.where(xs!=0)])
    xlim=np.polyval(np.polyfit(cad[np.where(xs!=0)],xs[np.where(xs!=0)],4),cad)
    plt.plot(cad,xlim,c='C1')
    plt.xlabel('Cadence')
    plt.ylabel('X Detector Position')
    plt.title('X Axis')
    plt.savefig('xlim.png',dpi=150,bbox_inches='tight')
    plt.close()
    plt.figure()
    plt.scatter(cad[np.where(ys!=0)],ys[np.where(ys!=0)])
    ylim=np.polyval(np.polyfit(cad[np.where(ys!=0)],ys[np.where(ys!=0)],4),cad)
    plt.plot(cad,ylim,c='C1')
    plt.xlabel('Cadence')
    plt.ylabel('Y Detector Position')
    plt.title('Y Axis')
    plt.savefig('ylim.png',dpi=150,bbox_inches='tight')
    plt.close()
    print('Creating Movie')
    movie(fnames[0:100],cad[0:100],xlim,ylim,vmin=vmin,vmax=vmax)
    print('Complete')
