import sys
import os
import time
import re
import json

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
