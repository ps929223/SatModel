'''
Sentinel 다운로드 프로그램
2021.12.23
Auth: Hokun Jeon
KIOST Marine Bigdata Center
'''
import geojson
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import timedelta, datetime
import os
import pandas as pd


def get_sector_footprint(area):
    import Lib.Map as Map
    coord = Map.sector()[area]

    # 대상영역
    footprint = 'MULTIPOLYGON((('+str(coord[0])+' '+str(coord[2])+', '+str(coord[1])+' '+str(coord[2])+', '+str(coord[1])\
                +' '+str(coord[3])+', '+str(coord[0])+' '+str(coord[3])+', '+str(coord[0])+' '+str(coord[2])+')))'

    return footprint

def get_geojson_footprint(path_geojson):
    footprint = geojson_to_wkt(read_geojson(path_geojson))
    return footprint


def sentinel_download_footprint(footprint, id, area, platform, date_from, date_to):
    pw = '#gunia4ever'
    api = SentinelAPI(id, pw, 'https://scihub.copernicus.eu/dhus')
    cols=['title']

    if platform == 's1':
        products = api.query(footprint, date=(date_from, date_to), platformname='Sentinel-1', filename='S1*',
                             producttype='GRD')
        print('S1: ' + str(len(products)))
        if len(products) == 0:
            print('No Data For S1' + area)
        else:
            # area = 'Dokdo'
            save_path = 'E:/02_Satellite/S1/' + area
            os.makedirs(save_path, exist_ok=True)
    elif platform == 's2':
        products = api.query(footprint, date=(date_from, date_to), platformname='Sentinel-2', producttype='S2MSI2A',
                             filename='S2*', cloudcoverpercentage=(0, 10))
        if len(products) == 0:
            print('No Data For S2' + area)
        else:
            save_path = 'E:/02_Satellite/S2/' + area
            os.makedirs(save_path, exist_ok=True)
    if len(products) > 0:
        DF = pd.DataFrame(products).T
        DF.to_csv(save_path + '/'+platform+'_%s-%s.csv' % (date_from, date_to))
        print(DF)
        api.download_all(products, save_path, max_attempts=2, n_concurrent_dl=2, lta_retry_delay=600)


'''
Test Code
- geojson을 읽어서 SKorea 해역 Boundary를 파악(제주,독도 포함 영역)
- Sentinel-1 다운로드
- 스케쥴은 매주 금요일 저녁 8시(KST)부터 14일전부터 7일전까지
  (관측 후 업로드까지 Delay는 약 5일로 보이나, 안정성을 위해 일주일 전 것까지 다운)
'''

path_geojson='E:/05_Bathymetry/Boundary/Sentinel_SKorea/Sentinel_SKorea.geojson'
footprint=get_geojson_footprint(path_geojson)

import datetime as dt

today=dt.datetime.today()
date_from=(today-dt.timedelta(days=14)).strftime('%Y%m%d')
date_to=(today-dt.timedelta(days=7)).strftime('%Y%m%d')

sentinel_download_footprint(footprint=footprint, id='ps929223', area='SKorea',
                            platform='s1', date_from=date_from, date_to=date_to)