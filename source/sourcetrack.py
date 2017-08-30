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
        return im,inset,text3,

    anim = animation.FuncAnimation(fig,animate,frames=len(fnames), interval=15, blit=True)
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


def run(reduce=10,calculate=False,stabilise=False):

    fnames=np.asarray(glob('../data/*.fits'))
    cad=np.asarray([float((f.split('-cad')[-1])[:-5]) for f in fnames])

    if stabilise==True:
        blank = fitsio.FITS(fnames[0])[1].read()
        blank[np.isfinite(blank)==False]=0
        blank*=0.
        sfamilies=track(fnames,cad,blank,threshold=300)
        for f in sfamilies:
            f.pop('nextpos')
        bad=[]
        for i,f in enumerate(sfamilies):
            if len(f['x'])<10:
                bad.append(i)
        families=np.delete(sfamilies,bad)
        print ('{} Bright Sources Found'.format(len(sfamilies)))
        pickle.dump(sfamilies,open('stabilise.p','wb'))

    return
    ts=np.linspace(0,len(fnames),len(fnames)/reduce,dtype=int)-1
    ts[0]=1


    if calculate==True:
        print ('Calculating average frame')
        with tqdm(total=len(fnames[ts])) as pbar:
            for i,f in enumerate(fnames[ts]):
                tpf = fitsio.FITS(f)
                if i==0:
                    av=tpf[1].read()
                else:
                    av=np.nansum([av,tpf[1].read()],axis=0)
                tpf.close()
                pbar.update()
        av/=float(len(fnames[ts]))
        av[np.where(av==0)]=np.nan

        families=track(fnames[ts],cad[ts],av)
        for f in families:
            f.pop('nextpos')
        pickle.dump(families,open('families.p','wb'))
    else:
        families=pickle.load(open('families.p','rb'))

    bad=[]
    for i,f in enumerate(families):
        if len(f['x'])<10:
            bad.append(i)
            continue
        if  (((np.max(f['x'])-np.min(f['x']))**2+(np.max(f['y'])-np.min(f['y']))**2)**0.5)<10:
            bad.append(i)

    families=np.delete(families,bad)

    print('{} Families of Sources Found'.format(len(families)))

    '''print('Creating movies')
    with tqdm(total=len(families)) as pbar:
        for i,f in enumerate(families):
            xlim=np.polyval(np.polyfit(f['c'],f['x'],4),cad)
            ylim=np.polyval(np.polyfit(f['c'],f['y'],4),cad)
            pos=np.where((cad>=np.min(f['c']))&(cad<=np.max(f['c'])))[0]
            movie(fnames[pos],cad[pos],xlim[pos],ylim[pos],outfile='family{}.mp4'.format(i))
            pbar.update()
    print('Complete')
'''
