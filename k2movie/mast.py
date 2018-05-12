import sys
import os
import time
import re
import json
from .build import log
import astropy.units as u
import K2ephem
import numpy as np

from .build import silence

try:  # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try:  # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib


class K2MovieNoObject(Exception):
    pass


def mastQuery(request):

    server = 'masttest.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content


def findMAST(epic):
    if (isinstance(epic, str) is False):
        print('Please pass a string to query.')
        return
    resolverRequest = {'service': 'Mast.Name.Lookup',
                       'params': {'input': epic,
                                  'format': 'json'},
                       }

    headers, resolvedObjectString = mastQuery(resolverRequest)
    resolvedObject = json.loads(resolvedObjectString)
    try:
        objRa = resolvedObject['resolvedCoordinate'][0]['ra']
        objDec = resolvedObject['resolvedCoordinate'][0]['decl']
        return objRa, objDec
    except:
        return None, None


def findStatic(name):
    '''Find a static object in K2'''
    ra, dec = findMAST(name)
    if (ra is not None) & (dec is not None):
        ra, dec = ra * u.deg, dec * u.deg
        return ra, dec
    else:
        raise K2MovieNoObject('No static target {}'.format(name))


def findMoving(name, campaign, channel, time):
    '''Find a moving object by querying JPL small bodies database'''
    # Get the ephemeris from JPL
    with silence():
        df = K2ephem.get_ephemeris_dataframe(
            name, campaign, campaign, step_size=1./(4))

    times = [t[0:23] for t in np.asarray(df.index, dtype=str)]
    df_jd = Time(times, format='isot').jd-lag
    # K2 Footprint...
    k = fields.getKeplerFov(campaign)

    # If no cadence specified...only use the on silicon cadences...
    mask = list(map(onSiliconCheck, ra.value, dec.value, np.repeat(k, len(ra))))
    if np.any(mask) == False:
        raise K2MovieNoObject('No moving target {}'.format(name))

    ra, dec = ra[mask], dec[mask]
    pos = np.where(mask)[0]
    cad = cad[pos.min()-1:pos.max()+1]
    cad = np.arange(cad[0], cad[-1])
    # Interpolate each cadence
    ra, dec = np.interp(time[cad], df_jd, df.ra) * \
        u.deg, np.interp(time[cad], df_jd, df.dec)*u.deg
    return ra, dec
