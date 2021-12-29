'''
매일의 VBD 자료를 읽어서 DensityMap을 NC형태로 저장
초기 2021.09.01
Auth: Hokun Jeon
Marne Bigdata Center
KIOST
'''


'라이브러리 호출'
import numpy as np
import os, sys
import pandas as pd
import gzip
import datetime as dt

## 호군 로컬PC
path_VBD = 'E:/03_MarineTraffic/VBD'
path_out_NC='E:/20_Product/VBD/NC/DensityDaily'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def read_VBD_csvgz(path_file):
    '''
    VBD daily data(*.csv.gz)를 한달치를 읽어 전체 lonlat을 반환

    lonlat=read_VBD_csvgz(path_file)
    path_file = 'E:/03_MarineTraffic/VBD\\2020\\M01/VBD_npp_d20200101_global-saa_noaa_ops_v23.csv.gz'

    daily 데이터가 한달치씩 폴더 안에 있어야 함. 이 때 월 폴더명은 숫자앞에 'M'이 기입되어 있음
    daily 데이터 형식은 csv.gz 이어야 함

    '''

    DF = pd.read_csv(gzip.open(path_file)) # gz 압축해제 후, DF 읽기
    DF = DF[['Lon_DNB','Lat_DNB','Rad_DNB','Date_Mscan']]
    DF = DF.dropna()
    DF.columns=['lon','lat','rad','utc']
    return DF

def separate_time(DF):
    from Lib.lib_cluster import dbscan
    import pandas as pd
    import matplotlib.pyplot as plt
    # DF=n_DF
    DF = DF.sort_values(by='utc')
    DF['utc'] = pd.to_datetime(DF['utc'])

    DF['sec']=(DF['utc'] - pd.to_datetime('1970-01-01')).dt.total_seconds()
    DF['1']=1

    result=dbscan(data2col=DF[['sec','1']],eps=60*60*0.5,min_samples=2)
    DF=DF[['lon','lat','rad','utc','sec']]
    DF['label']=result['label']
    group_times=np.array(pd.to_datetime('1970-01-01')+pd.to_timedelta(DF.groupby('label').mean()['sec'], unit='sec'))
    labels=np.array(DF['label'].unique())
    DF['mean_utc']=np.nan
    for label in labels:
        DF['mean_utc'][DF['label']==label]=group_times[label==labels][0]
    DF=DF.reset_index(inplace=False).drop(columns='index')
    DF['mean_utc'] = pd.to_datetime(DF['mean_utc'])
    DF['mean_utc'] = DF['mean_utc'].dropna().apply(lambda x: x.strftime('%Y%m%d-%H%M%S'))
    return DF


def make_daily_NC(DF,area,resolution):
    '''
    days2month 함수에서 반환한 lonlat을 이용해 NC를 생성
    make_daily_NC(DF,area,resolution)
    area='Donghae' # Changwon
    resolution=0.25
    lonlat=read_lonlat(path_file)

    daily 데이터가 한달치씩 폴더 안에 있어야 함. 이 때 월 폴더명은 숫자앞에 'M'이 기입되어 있음
    daily 데이터 형식은 csv.gz 이어야 함
    '''

    import Lib.Map as Map
    sector=Map.sector()
    # area='Donghae'
    coord=sector[area]

    '''
    1. NC에 할당할 데이터 생성
    '''

    lons = np.array(DF['lon'])
    lats = np.array(DF['lat'])

    if coord[1]>180:
        lons[lons<0]=lons[lons<0]+360 # 표시하고자하는 값이 Int'nal Time Zone을 포함하는 경우
    cond = ((coord[0] <= lons) & (lons <= coord[1])) & ((coord[2] <= lats) & (lats <= coord[3]))  # 범위 내 자료 조건
    if sum(cond)!=0:
        n_DF=pd.DataFrame()
        n_DF['lon'] = lons[cond]
        n_DF['lat'] = lats[cond]
        n_DF['utc'] = np.array(DF['utc'][cond])
        n_DF['rad'] = np.array(DF['rad'][cond])
        n_DF=separate_time(n_DF)
        mean_utc=n_DF['mean_utc'].unique()

        for ii in range(len(mean_utc)):
            # ii=0
            cond=mean_utc[ii]==n_DF['mean_utc']
            lat_bins = np.linspace(coord[2], coord[3], int((coord[3] - coord[2])/resolution+1))
            lon_bins = np.linspace(coord[0], coord[1], int((coord[1] - coord[0])/resolution+1))
            density, _, _ = np.histogram2d(n_DF['lon'][cond], n_DF['lat'][cond], [len(lon_bins), len(lat_bins)])
            meshlon, meshlat=np.meshgrid(lon_bins, lat_bins)


            '''
            2. NC에 할당
            '''
            import netCDF4 as nc

            date=mean_utc[ii].split('-')[0]
            utc=mean_utc[ii].split('-')[1]

            year=date[:4]
            month=date[4:6]
            day=date[6:]

            dir = path_out_NC+'/'+year
            os.makedirs(dir,exist_ok=True)
            fn_path=dir+'/ShipDensity'+'_'+area+'_'+ year +'-'+month+'-'+day+'-'+utc+'_'+str(resolution).replace('.','p')+'.nc'
            ds = nc.Dataset(fn_path, 'w', format='NETCDF4')

            lat=ds.createDimension('lat', meshlon.shape[0])
            lon=ds.createDimension('lon', meshlon.shape[1])

            ds.title='Ship Density '+mean_utc[ii]+' Res:'+str(resolution)

            ds.area=area
            ds.resolution=resolution

            lat = ds.createVariable('meshlat', np.float32, ('lat','lon'))
            lat.units = 'degrees_north'
            lat.long_name = 'latitude'
            lon = ds.createVariable('meshlon', np.float32, ('lat','lon'))
            lon.units = 'degrees_east'
            lon.long_name = 'longitude'

            # 프로젝션 변수
            crs = ds.createVariable('VBD_map_projection', int)
            crs.long_name = 'VBD Density Grid Projection'
            crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
            crs.EPSG_code = "EPSG:4326"
            crs.latitude_of_projection_origin = np.min(meshlat)
            crs.longitude_of_projection_origin = np.min(meshlon)
            crs.semi_major_axis = 6378137.0  # WGS84
            crs.semi_minor_axis = 6356752.5  # WGS84
            crs.spatial_resolution = resolution

            # Define a 3D variable to hold the data
            Density = ds.createVariable('density',int,('lat','lon')) # note: unlimited dimension is leftmost
            Density.units = 'ship' # ship
            Density.standard_name = '# of ships' # this is a CF standard name

            # Write latitudes, longitudes.
            # Note: the ":" is necessary in these "write" statements
            lat[:] = meshlat
            lon[:] = meshlon
            Density[:]=density.T
            ds.close()




from Lib.lib_os import *
import pandas as pd


'Batch시에만 사용'
area='Donghae'
resolution=0.05


file_list=np.array(recursive_file(path_VBD,'*.csv.gz'))
date_list=pd.date_range('2017-01-01','2020-12-31').format(formatter=lambda x: x.strftime('%Y%m%d'))

count_date=[]
for ii in range(len(date_list)):
    # ii=0
    count_date.append(date_list[ii])
    print(date_list[ii])
    cond = [date_list[ii] in name for name in file_list]
    if sum(cond) == 0:
        print('Invalid for '+date_list[ii])
    else:
        path_file=file_list[np.array(cond)][0]
        DF=read_VBD_csvgz(path_file)
        make_daily_NC(DF, area, resolution)