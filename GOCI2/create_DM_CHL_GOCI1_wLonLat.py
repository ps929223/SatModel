'''
GOCI1 Daily Mean(DM) NetCDF를 생성
Subset과 경위도 meshgrid를 추가함
해양빅데이터센터
전호군
초안: 2021.10.18
'''


import os, sys
import matplotlib.pyplot as plt
sys.path.append('D:/programming/SatModel/Lib')


def genNC_subset_wMesh(mesh_lon, mesh_lat, CHL,date,path_dir,subset_idx, days):
    '''
    path_dir='E:/CHL_DM_Dokdo'
    date='2018-03-15'
    CHL = DM
    '''

    year = date[:4]
    month = date[5:7]
    day = date[8:10]

    import netCDF4 as nc

    path_dir_sub = path_dir + '/' + year + '/' + month
    os.makedirs(path_dir_sub, exist_ok=True)

    fname='GOCI1_DM'+str(days)+'_CHL_Dokdo_'+year+'-'+month+'-'+day
    path_file = path_dir_sub + '/' + fname + '.nc'
    ds = nc.Dataset(path_file, 'w', format='NETCDF4')

    sub_mesh_lon = mesh_lon[subset_idx[2]:subset_idx[3]:,subset_idx[0]:subset_idx[1]]
    sub_mesh_lat = mesh_lat[subset_idx[2]:subset_idx[3]:,subset_idx[0]:subset_idx[1]]
    sub_mesh_CHL = CHL[subset_idx[2]:subset_idx[3]:,subset_idx[0]:subset_idx[1]]

    r,c = sub_mesh_lon.shape
    y = ds.createDimension('y', r)
    x = ds.createDimension('x', c)

    ds.title = fname
    ds.area = 'Korea'
    ds.time = fname.split('_')[-1]

    lon = ds.createVariable(varname='longitude', datatype='f4', dimensions=('y', 'x'))
    lat = ds.createVariable(varname='latitude', datatype='f4', dimensions=('y', 'x'))
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
    lon[:] = sub_mesh_lon
    lat[:] = sub_mesh_lat
    chl[:] = sub_mesh_CHL
    ds.close()


from Lib.lib_GOCI1 import *
from Lib.lib_os import *
import pandas as pd

## 경로 설정
path_coord = 'D:/02_Satellite/GOCI1/Coordinates'
path_source_dir= 'E:/CHL_DM'
path_output_dir= 'E:/CHL_DM_Dokdo'
os.makedirs(path_output_dir, exist_ok=True)

## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=np.array(recursive_file(path_source_dir, pattern="*.nc"))
date_list=list(pd.date_range('2018-03-04','2021-03-26').strftime('%Y-%m-%d').astype(str))

## 경위도 meshgrid를 불러옵니다
mesh_lon, mesh_lat = read_GOCI1_coordnates()
r,c=mesh_lon.shape

## n-Days Mean을 생산합니다
import netCDF4 as nc

##
from Lib.lib_GOCI1 import *
import Lib.Map as Map

mesh_lon, mesh_lat=read_GOCI1_coordnates()
dokdo=Map.sector()['Dokdo']

minidx=find_nearst_idx(mesh_lon, mesh_lat, dokdo[0], dokdo[2])
maxidx=find_nearst_idx(mesh_lon, mesh_lat, dokdo[1], dokdo[3])
subset_idx=[int(minidx[1]),int(maxidx[1]),int(maxidx[0]),int(minidx[0])] # minlon, maxlon, minlat, maxlat
r, c= mesh_lon.shape
days=3

# for ii in range(len(date_list)):
for ii in range(1100, 1107):
    # ii=11
    filesOfDate=file_list[np.array([date_list[ii] in file_path for file_path in file_list])]
    CHL=np.zeros((days,r,c))

    ## 과거 현재 미래의 자료를 읽어서 쌓기
    kk=0
    for jj in range(int(ii-np.floor(days/2)),int(ii+np.floor(days/2))+1):
        file_name=file_list[jj].split('/')[-1]
        ## Chl
        CHL[kk,:,:]=np.array(nc.Dataset(file_list[jj])['chl'])
        kk+=1

    ## 일주일 평균
    DM = np.nanmean(CHL, axis=0)
    DM[DM==0]=np.nan
    genNC_subset_wMesh(mesh_lon, mesh_lat, DM,date_list[ii],path_output_dir,subset_idx,days=days)
