'''
VBD 자료를 읽어서 NC에 저장
초기: 2021.09.10
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

'입력/출력, 소스코드 path 설정'

import sys, os

## 호군 로컬PC
path_VBD='D:/03_MarineTraffic/VBD'
path_NC='D:/20_Product/VBD/NC/ScatterDaily'
path_Lib='D:/programming/SatModel/Lib'
sys.path.append(path_Lib)

'라이브러리 호출'
import numpy as np
import os
import pandas as pd
import gzip
import datetime as dt

'경로 변경, 경로 생성'
os.makedirs(path_NC, exist_ok=True) # NC파일이 저장될 경로 생성

def oneday(path_month,date):
    '''
    VBD daily data(*.csv.gz)를 읽어 lonlat을 반환

    lonlat=oneday(path_month)
    path_month = 'D:/03_MarineTraffic/VBD/global-saa/daily/CSV/2020/M03'
    date='01'
    daily 데이터가 한달치씩 폴더 안에 있어야 함. 이 때 월 폴더명은 숫자앞에 'M'이 기입되어 있음
    daily 데이터 형식은 csv.gz 이어야 함
    '''

    day_list=np.array(os.listdir(path_month))
    day_list=day_list[np.array(['csv.gz' in name for name in day_list])] # gz만 추출
    day_list=day_list[np.array([day_list[kk].split('_')[2][7:9]==date for kk in range(len(day_list))])]  # 해당 날짜만 추출

    tp = pd.read_csv(gzip.open(os.path.join(path_month,list(day_list)[0]))) # gz 압축해제 후, DF 읽기
    lons = list(tp.Lon_DNB)
    lats = list(tp.Lat_DNB)
    lonlat=[lons, lats]
    Date_Mscan=list(tp.Date_Mscan)
    return lonlat, Date_Mscan

def make_oneday_NC(lonlat,Date_Mscan,year,month,date,area):
    '''
    oneday 함수에서 반환한 lonlat을 이용해 NC를 생성
    make_month_NC(lonlat,year,month,area)
    year='2020'
    month='03'
    date='14'
    area='EastAsia' # Donghae, Dokdo, Changwon

    daily 데이터가 한달치씩 폴더 안에 있어야 함. 이 때 월 폴더명은 숫자앞에 'M'이 기입되어 있음
    daily 데이터 형식은 csv.gz 이어야 함
    '''
    import Map
    sector=Map.sector()

    # area='Donghae'
    coord=sector[area]

    '''
    1. NC에 할당할 데이터 생성
    '''
    lats = np.array(lonlat[1])
    lons = np.array(lonlat[0])
    scans = np.array(Date_Mscan)

    # lons이 360보다 큰 경우 row삭제
    lats = np.delete(lats, lons>360)
    scans = np.delete(scans, lons > 360)
    lons = np.delete(lons, lons>360)


    if coord[1]>180: # int'nal dateline을 포함하는 경우
        lons[lons<0]=lons[lons<0]+360

    cond=((coord[0] <= lons) & (lons <= coord[1]))&((coord[2] <= lats) & (lats <= coord[3])) # 범위 내 자료 조건
    lons = lons[cond]
    lats = lats[cond]

    # import matplotlib.pyplot as plt
    # plt.scatter(lons,lats)

    scans = scans[cond]


    '''
    2. NC에 할당
    '''
    import netCDF4 as nc
    dir=path_NC+'/'+year
    os.makedirs(dir, exist_ok=True)
    fn = dir+'/ShipScatter'+'_'+area+'_'+year+'-'+month+'-'+date+'.nc'
    ds = nc.Dataset(fn, 'w', format='NETCDF4')

    lat=ds.createDimension('lat', len(lats))
    lon=ds.createDimension('lon', len(lons))
    scan=ds.createDimension('Date_Mscan', len(scans))

    ds.title='Ship Scatter '+area+'-'+year+'-'+month+'-'+date

    ds.area=area

    lat = ds.createVariable('lat', np.float32, ('lat'))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ds.createVariable('lon', np.float32, ('lon'))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    scan = ds.createVariable('Date_Mscan', np.str, ('Date_Mscan'))
    scan.units = 'Scan yyyy-mm-dd HH:MM:SS.S'
    scan.long_name = 'Scan Time'

    # 프로젝션 변수
    crs = ds.createVariable('VBD_map_projection', np.int32)
    crs.long_name = 'VBD Scatter Projection'
    crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs.EPSG_code = "EPSG:4326"
    # crs.latitude_of_projection_origin = min(lats)
    # crs.longitude_of_projection_origin = min(lons)
    crs.semi_major_axis = 6378137.0  # WGS84
    crs.semi_minor_axis = 6356752.5  # WGS84

    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    lat[:] = np.flipud(lats)
    lon[:] = lons
    scan[:]= np.flipud(scans)
    ds.close()


'처리코드'
areas=['Changwon']
years=['2017','2018','2019','2020']
for area in areas:
    # area = 'Donghae'
    # year = '2017'
    for year in years:
        month_list=os.listdir(os.path.join(path_VBD,year))
        # 해당 연도의 월별 자료 생산
        ii=0
        for ii in range(len(month_list)):
            path_month=os.path.join(path_VBD,year,month_list[ii])
            month = month_list[ii].split('M')[1]  # 앞에 M을 때어내고 월'을 나타내는 숫자만 남기기
            file_list=os.listdir(path_month)
            jj=0
            for jj in range(len(file_list)):
                date=file_list[jj].split('_')[2][7:9]
                lonlat,Date_Mscan=oneday(path_month,date)
                print(path_month)
                make_oneday_NC(lonlat, Date_Mscan, year, month, date, area)