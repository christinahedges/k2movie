import fitsio
from time import sleep
import gc

fnames = ['/Users/ch/Downloads/ktwo200002562-c02_lpd-targ.fits.gz',
          '/Users/ch/Downloads/ktwo200069997-c92_lpd-targ.fits.gz',
          '/Users/ch/Downloads/ktwo200071159-c91_spd-targ.fits.gz',
          '/Users/ch/Downloads/ktwo200071208-c91_lpd-targ.fits.gz',
          '/Users/ch/Downloads/ktwo200083102-c102_lpd-targ.fits.gz',
          '/Users/ch/Downloads/ktwo200083127-c102_lpd-targ.fits.gz']
for f in fnames:
    tpf = fitsio.FITS(f)
    sleep(1)
    cadencelist = tpf[1]['CADENCENO'].read()
    sleep(1)
    tpf.close()
    sleep(1)
