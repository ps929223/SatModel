'''
GOCI1 Daily Mean(DM) NetCDF를 생성
해양빅데이터센터
전호군
초안: 2021.10.13
'''


import os, sys
import matplotlib.pyplot as plt
sys.path.append('D:/programming/SatModel/Lib')

def genNC(CHL,date,days,path_dir):
    '''
    path_dir='E:/CHL_DM'
    date='20180304'
    CHL = DM
    '''

    year = date[:4]
    month = date[4:6]
    day = date[6:8]

    import netCDF4 as nc

    path_dir_sub = path_dir + '/' + year + '/' + month
    os.makedirs(path_dir_sub, exist_ok=True)

    fname='GOCI1_DM'+str(days)+'_CHL_'+year+'-'+month+'-'+day
    # fname = 'GOCI1_DM' + '_CHL_' + year + '-' + month + '-' + day
    print(fname)
    path_file = path_dir_sub + '/' + fname + '.nc'
    ds = nc.Dataset(path_file, 'w', format='NETCDF4')

    r,c = mesh_lon.shape
    y = ds.createDimension('y', r)
    x = ds.createDimension('x', c)

    ds.title = fname
    ds.area = 'Korea'
    ds.time = fname.split('_')[-1]

    chl = ds.createVariable(varname='chl',datatype='f4',dimensions=('y','x')) # note: unlimited dimension is leftmost
    chl.units = 'mm m-3' # ship
    chl.standard_name = 'chl-a concentration' # this is a CF standard name

    # 프로젝션 변수
    crs = ds.createVariable('projection', np.int32)
    crs.long_name = 'Projection'
    crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs.EPSG_code = "EPSG:4326"
    # crs.latitude_of_projection_origin = min(lats)
    # crs.longitude_of_projection_origin = min(lons)
    crs.semi_major_axis = 6378137.0  # WGS84
    crs.semi_minor_axis = 6356752.5  # WGS84

    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    chl[:] = CHL
    ds.close()



from Lib.lib_GOCI1 import *
from Lib.lib_os import *
import pandas as pd

## 경로 설정
path_coord = 'D:/02_Satellite/GOCI1/Coordinates'
path_source_dir= 'E:/CHL'
path_output_dir= 'E:/CHL_DM'
os.makedirs(path_output_dir, exist_ok=True)

## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=np.array(recursive_file(path_source_dir, pattern="*.he5"))
date_list=list(pd.date_range('2019-05-08','2019-05-14').strftime('%Y%m%d').astype(str))

## 경위도 meshgrid를 불러옵니다
mesh_lon, mesh_lat = read_GOCI1_coordnates()
r,c=mesh_lon.shape

## Masking meshgrid를 불러옵니다
LM=read_GOCI1_LM()

## Daily Mean을 생산합니다
for ii in range(len(date_list)):
    # ii=0
    filesOfDate=file_list[np.array([date_list[ii] in file_path for file_path in file_list])]
    if len(filesOfDate) == 0:
        print('No data for '+date_list[ii])
        continue
    else:
        dataset=np.zeros((len(filesOfDate),r, c))
        for jj in range(len(filesOfDate)):
            # jj=0
            print(filesOfDate[jj])
            dataset[jj,:,:]=read_GOCI1_Chl(filesOfDate[jj])['CHL']
        ## 일주일 평균
        DM = np.nanmean(dataset, axis=0)
        DM[DM==0]=np.nan
        genNC(DM, date_list[ii], days=1, path_dir=path_output_dir)









## n-Days Mean을 생산합니다
import netCDF4 as nc
import datetime as dt
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('D:/programming/SatModel/Lib')
from Lib.lib_os import *
import pandas as pd
from Lib.lib_GOCI1 import *

## 경로 설정
path_coord = 'D:/02_Satellite/GOCI1/Coordinates'

## 경위도 meshgrid를 불러옵니다
mesh_lon, mesh_lat = read_GOCI1_coordnates()
r,c=mesh_lon.shape

## Masking meshgrid를 불러옵니다
LM=read_GOCI1_LM()


days=5
path_source_dir='E:/CHL_DM'
path_output_dir = 'E:/CHL_DM' + str(days)
os.makedirs(path_output_dir, exist_ok=True)


file_list = np.array(recursive_file(path_source_dir, pattern="*.nc"))
file_list = file_list[np.array(['_res' not in name for name in file_list])]
date_list=list(pd.date_range('2019-05-01','2021-05-30').astype(str))

for ii in range(int(np.floor(days / 2)), len(date_list) - int(np.floor(days / 2))):
# for ii in range(427, len(date_list) - int(np.floor(days / 2))):
    # ii=2

    ## 날짜에 해당하는 파일을 찾습니다

    print('----------------------------------')
    print(date_list[ii])
    print('----------------------------------')

    CHL = np.zeros((days, LM.shape[0], LM.shape[1]))

    ## 과거 현재 미래의 자료를 읽어서 쌓기
    daysbefore=str(dt.datetime.strptime(date_list[ii], '%Y-%m-%d')-dt.timedelta(int(np.floor(days / 2))))[:10]
    daysafter=str(dt.datetime.strptime(date_list[ii], '%Y-%m-%d')+dt.timedelta(int(np.floor(days / 2))))[:10]

    target_date_list=pd.date_range(daysbefore, daysafter).astype(str)

    kk = 0
    for jj in range(len(target_date_list)):
        try:
            fileOfDate = file_list[np.array([target_date_list[jj] in file_path for file_path in file_list])][0]
            print(fileOfDate)
            CHL[kk, :, :] = np.array(nc.Dataset(fileOfDate)['chl'])
        except:
            print('Inserted NaN because NoData ' + target_date_list[jj])
            CHL[kk, :, :] = np.array([np.nan] * LM.shape[0] * LM.shape[1]).reshape(LM.shape[0], LM.shape[1])
        kk += 1

    ## 일주일 평균
    CHL = np.nanmean(CHL, axis=0)
    CHL[CHL == 0] = np.nan
    genNC(CHL, date_list[ii][0:4] + date_list[ii][5:7] + date_list[ii][8:10], days, path_output_dir)




'''
가시화
'''
from Lib.lib_GOCI1 import *
from Lib.lib_os import *
import pandas as pd
import netCDF4 as nc
import os, sys
sys.path.append('D:/programming/SatModel/Lib')
import Lib.Map as Map
import matplotlib.pyplot as plt

## 경로 설정
path_coord = 'D:/02_Satellite/GOCI1/Coordinates'
path_source_dir = 'E:/CHL_DM9'
path_output_dir = 'D:/20_Product/GOCI1/CHL/Map'
os.makedirs(path_output_dir, exist_ok=True)

## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=np.array(recursive_file(path_source_dir, pattern="*.nc"))
file_list=file_list[np.array(['_res100' not in name for name in file_list])]
date_list=list(pd.date_range('2020-10-01','2020-10-10').strftime('%Y-%m-%d').astype(str))

## 경위도 meshgrid를 불러옵니다
mesh_lon, mesh_lat = read_GOCI1_coordnates()
r,c=mesh_lon.shape

## Masking meshgrid를 불러옵니다
LM=read_GOCI1_LM()


## 가시화합니다
fig1 = plt.figure(2, figsize=(13, 9))
for ii in range(len(date_list)):
    # ii=0
    filesOfDate=file_list[np.array([date_list[ii] in file_path for file_path in file_list])]
    if len(filesOfDate) == 0:
        print('No data for '+date_list[ii])
        continue
    else:
        dataset = np.array(nc.Dataset(filesOfDate[0])['chl'])
        dataset[dataset==0]=np.nan
        sector = Map.sector()
        area = 'Donghae'
        coord = sector[area]
        map_res1 = 'h'
        grid_res1 = 1
        m = Map.making_map(coord, map_res1, grid_res1)
        xx, yy = m(mesh_lon, mesh_lat)
        plt.pcolormesh(xx, yy, dataset, cmap='jet', alpha=.8, vmin=0, vmax=2)
        # plt.pcolormesh(xx, yy, density, cmap=cmap1, alpha=.8, vmin=np.log(ticks[0]), vmax=np.log(ticks[-1]))
        # Projection이 'cyl'인 경우 x,y 대신 mesh_lon, mesh_lat를 사용함
        # plt.clim(np.log(ticks[0]),np.log(ticks[0])]) # 범위 제한을 해서 Colorbar가 최대값을 표시하지 못하는 현상을 막음
        cb = plt.colorbar(shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%1.1f')
        cb.set_label('Chl-a Concentration [mg m-3]', size=14)

        dokdo = Map.dokdo_psn()
        x, y = m(dokdo[0], dokdo[1])
        m.scatter(x, y, facecolor='w', edgecolor='k', s=80, marker='o')

        title = 'GOCI-I DM9 '+date_list[ii]
        plt.title(title, fontsize=30)
        # plt.pause(2)
        plt.tight_layout()
        plt.savefig(path_output_dir+'/'+title.replace(' ','_')+'.png')
        plt.clf()

