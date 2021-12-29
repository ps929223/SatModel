# -*- coding: utf-8 -*-
'''
2일 전 VIIRS DNB 500m 격자자료를 읽어서 저장
초안: 2021.09.10
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

import datetime as dt
import os
import os.path
import shutil
import sys

''' Set date '''
yesterday=dt.datetime.today()-dt.timedelta(days=2)
yesterday_str=str(yesterday)
yesterday_str='2021-09-08'
year = yesterday_str[:4]
month = yesterday_str[5:7]
day = yesterday_str[8:10]

''' Set path '''
target_dir = 'D:/02_Satellite/VIIRS/DNB500mGrid/'+yesterday_str[:10]
os.makedirs(target_dir,exist_ok=True)

''' 날짜코드 계산 '''
d0 = dt.date(int(year), 1, 1)
d1 = dt.date(int(year), int(month), int(day))
delta = d1 - d0
cod = delta.days + 1

## URL 생성
ladsweb_url = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000/VNP46A1/{:04d}/{:03d}'.format(int(year),cod)

## token
token='bGFzb21zOmEyOXZjM056ZEVCcmFXOXpkQzVoWXk1cmNnPT06MTYxNzkzNDI2NTphNDJmMjIzMmU4YmFiZmZlYzZkZGM5N2NkNzQzYWE1N2NkYjUxN2U3'

''' 함수 시작 '''

## Python Version 확인해서 library 호출
try:
    from StringIO import StringIO   # python2
except ImportError:
    from io import StringIO         # python3


################################################################################


USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')


def geturl(url, token=None, out=None):
    headers = { 'user-agent' : USERAGENT }
    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        import ssl
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        if sys.version_info.major == 2:
            import urllib2
            try:
                fh = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read()
                else:
                    shutil.copyfileobj(fh, out)
            except urllib2.HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except urllib2.URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

        else:
            from urllib.request import urlopen, Request, URLError, HTTPError
            try:
                fh = urlopen(Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read().decode('utf-8')
                else:
                    shutil.copyfileobj(fh, out)
            except HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

    except AttributeError:
        # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
        import subprocess
        try:
            args = ['curl', '--fail', '-sS', '-L', '--get', url]
            for (k,v) in headers.items():
                args.extend(['-H', ': '.join([k, v])])
            if out is None:
                # python3's subprocess.check_output returns stdout as a byte string
                result = subprocess.check_output(args)
                return result.decode('utf-8') if isinstance(result, bytes) else result
            else:
                subprocess.call(args, stdout=out)
        except subprocess.CalledProcessError as e:
            print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)
        return None



################################################################################


DESC = "This script will recursively download all files if they don't exist from a LAADS URL and stores them to the specified path"


def sync(src, dest, tok):
    '''synchronize src url with dest directory'''
    try:
        import csv
        files = [ f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True) ]
    except ImportError:
        import json
        files = json.loads(geturl(src + '.json', tok))

    # use os.path since python 2/3 both support it while pathlib is 3.4+
    for f in files:
        # currently we use filesize of 0 to indicate directory
        filesize = int(f['size'])
        path = os.path.join(dest, f['name'])
        url = src + '/' + f['name']
        if filesize == 0:
            try:
                print('creating dir:', path)
                os.mkdir(path)
                sync(src + '/' + f['name'], path, tok)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path):
                    print('downloading: ' , path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping: ', path)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
    return 0


''' 함수 실행 '''
os.makedirs(target_dir, exist_ok=True)
sync(src=ladsweb_url,dest=target_dir,tok=token)