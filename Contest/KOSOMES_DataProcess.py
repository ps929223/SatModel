'''
해양환경안전학회(KOSMES)
VBD 및 모델자료를 읽어서 NC로 저장
'''
import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('D:/programming/SatModel/Lib')
import Lib.lib_CMEMS as cmm

def read_VBD(path_VBD):
    '''
    path_VBD='E:/20_Product/VBD/NC/DensityDaily/2017/ShipDensity_Donghae_2017-01-01_0p25.nc'
    '''
    import netCDF4 as nc
    import pandas as pd
    data=nc.Dataset(path_VBD)
    return np.array(data['meshlon']),np.array(data['meshlat']),np.array(data['density'])

def read_Bath(path_Bath, grid_y, grid_x):
    '''
    path_Bath='E:/05_Bathmetry/ETOPO/Donghae_ETOPO1(bedrock).tif'
    '''
    import PIL.Image as Image
    import numpy as np

    img=Image.open(path_Bath)
    img=np.flipud(np.array(img.resize((grid_x, grid_y), Image.BICUBIC)))*0.3048 # feet를 meter로 변환
    return img.astype(int)

path_VBD = 'E:/20_Product/VBD/NC/DensityDaily/2017/ShipDensity_Donghae_2017-01-01_0p05.nc'
path_BIO='E:/01_Model/CMEMS/GLOBAL_ANALYSIS_FORECAST_BIO_001_028/CHL/Donghae/GLO-ANAL-FCST-BIO-001-028_2020.nc'
path_SST='E:/01_Model/CMEMS/SST_GLO_SST_L4_REP_OBSERVATIONS_010_024/Donghae/C3S-GLO-SST-L4-REP-OBS-SST_2020.nc'
path_SLA='E:/01_Model/CMEMS/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/Donghae/dataset-duacs-nrt-global-merged-allsat-phy-l4_2020.nc'


from Lib.lib_os import *
import pandas as pd
file_list=recursive_file('E:/20_Product/VBD/NC/DensityDaily/2018','*0p05.nc')

mesh_lon, mesh_lat, Data_VBD=read_VBD(path_VBD)
grid_y, grid_x=mesh_lat.shape

## Bathymetry
Bath=read_Bath('E:/05_Bathymetry/ETOPO/Donghae_ETOPO1(bedrock).tif', grid_y=grid_y, grid_x=grid_x)
Bath[Bath>0]=999


### VBD의 날짜
date_list=np.array([file_list[ii].split('\\')[-1].split('_')[2] for ii in range(len(file_list))])

### 정상적인 날짜
date_list2=pd.date_range('2020-01-01','2020-12-31').astype(str)

### VBD에 없는 날짜 찾기
abscent_date=[]
abscent_idx=[]
for ii in range(len(date_list2)):
    if sum(date_list2[ii]==date_list) < 1 :
        abscent_date.append(date_list2[ii])
        abscent_idx.append(ii)
abscent_idx=np.array(abscent_idx)
abscent_mask=np.zeros(len(date_list2))
abscent_mask[abscent_idx]=True
abscent_mask=abscent_mask==1


r,c=mesh_lon.shape
N=len(file_list)
Data_VBD=np.zeros((N,r,c))
Data_VBD[:]=np.nan

## VBD 읽기
for ii in range(len(file_list)):
    mesh_lon, mesh_lat, Data_VBD[ii,:,:] = read_VBD(file_list[ii])


## 자료 읽기
Data_BIO=cmm.read_BIO_NC(path_BIO, grid_y=grid_y, grid_x=grid_x)
Data_SST=cmm.read_SST_NC(path_SST, grid_y=grid_y, grid_x=grid_x)
Data_SLA=cmm.read_SLA_NC(path_SLA, grid_y=grid_y, grid_x=grid_x)


## 동해좌표 읽기
import Lib.Map as Map
coord = Map.sector()['Donghae']

## 데이터 읽기
n_CHL=Data_BIO['chl'][~abscent_mask,:,:]
n_FE=Data_BIO['fe'][~abscent_mask,:,:]
n_NO3=Data_BIO['no3'][~abscent_mask,:,:]
n_NPPY=Data_BIO['nppy'][~abscent_mask,:,:]
n_O2=Data_BIO['o2'][~abscent_mask,:,:]
n_PH=Data_BIO['ph'][~abscent_mask,:,:]
n_PHYC=Data_BIO['phyc'][~abscent_mask,:,:]
n_PO4=Data_BIO['po4'][~abscent_mask,:,:]
n_SI=Data_BIO['si'][~abscent_mask,:,:]
n_SPCO2=Data_BIO['spco2'][~abscent_mask,:,:]

n_BIO_lon=np.array(Data_BIO['mesh_lon'])
n_BIO_lat=np.array(Data_BIO['mesh_lat'])

n_SST=Data_SST['SST']
n_SST=n_SST[~abscent_mask,:,:]
n_SST_lon=np.array(Data_SST['mesh_lon'])
n_SST_lat=np.array(Data_SST['mesh_lat'])

n_SLA=Data_SLA['SLA']
n_SLA=n_SLA[~abscent_mask,:,:]
n_SLA_lon=np.array(Data_SLA['mesh_lon'])
n_SLA_lat=np.array(Data_SLA['mesh_lat'])

def idxArray4match(mesh_lon,mesh_lat,target_meshlon, target_meshlat):
    '''
    mesh_lon/lat에 target_meshlon/lat에 위치한 데이터를 입력하는 방법으로
    최근접거리를 적용함

    target_meshlon=n_SLA_lon
    target_meshlat=n_SLA_lat
    '''

    import Lib.lib_GOCI1 as G1

    r,c=mesh_lon.shape
    id_x=np.zeros((r,c))
    id_x[:]=np.nan
    id_y=id_x.copy()

    for ii in range(r):
        for jj in range(c):
           y, x= G1.find_nearst_idx(mesh_lon[ii,jj],mesh_lat[ii,jj],target_meshlon, target_meshlat)
           if len(y) > 1 or len(x) >1: # 만약 최근접이 2개 이상 나오게 되면,
               y, x = min(y), min(x) # 최소값으로 선택함
           id_y[ii,jj], id_x[ii,jj]= int(y), int(x)

    return id_y, id_x


def matchedArray(mesh_lon,mesh_lat,target_meshlon, target_meshlat, target_z):
    '''
    VBD 경위도좌표계에 맞추어, CHL, SST, SLA의 값을 입력한다
    이때 방식은 Nearest Neighbor

    입력자료 예)
    mesh_lon = mesh_lon
    mesh_lat = mesh_lat
    target_meshlon = n_CHL_lon
    target_meshlat = n_CHL_lat
    target_z = n_CHL[ii,:,:]
    '''


    y,x=idxArray4match(mesh_lon,mesh_lat,target_meshlon, target_meshlat)
    y,x = y.astype(int),x.astype(int) # 실수르 정수로 변환해야 indexing이 가능함

    r,c=y.shape

    z=np.zeros((r,c)) # 빈공간만들고
    z[:]=np.nan # nan으로 채우기

    ## 2중 for문을 통해 cell 하나하나의 값을 찾음
    for ii in range(r):
        for jj in range(c):
            z[ii,jj]=target_z[y[ii,jj],x[ii,jj]]

    return z

## 격자크기는 VBD를 따름
r,c=mesh_lon.shape
## 날짜수는 BIO를 따름
N=len(n_CHL)

## 빈공간 만듬
n2_CHL=np.zeros((N,r,c))
n2_CHL[:]=np.nan
n2_FE=n2_CHL.copy()
n2_NO3=n2_CHL.copy()
n2_NPPY=n2_CHL.copy()
n2_O2=n2_CHL.copy()
n2_PH=n2_CHL.copy()
n2_PHYC=n2_CHL.copy()
n2_PO4=n2_CHL.copy()
n2_SI=n2_CHL.copy()
n2_SPCO2=n2_CHL.copy()
n2_SST=n2_CHL.copy()
n2_SLA=n2_CHL.copy()

from tqdm import tqdm

## CHL날짜수만큼 자료 입력
for ii in tqdm(range(N)):
    # ii=0
    ns_CHL = n_CHL[ii,:,:]
    ns_FE = n_FE[ii,:,:]
    ns_NO3 = n_NO3[ii,:,:]
    ns_NPPY = n_NPPY[ii,:,:]
    ns_O2 = n_O2[ii,:,:]
    ns_PH = n_PH[ii,:,:]
    ns_PHYC = n_PHYC[ii,:,:]
    ns_PO4 = n_PO4[ii,:,:]
    ns_SI = n_SI[ii,:,:]
    ns_SPCO2 = n_SPCO2[ii,:,:]
    ns_SST = n_SST[ii, :, :]
    ns_SLA = n_SLA[ii, :, :]

    n2_CHL[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_CHL))
    n2_FE[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_FE))
    n2_NO3[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_NO3))
    n2_NPPY[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_NPPY))
    n2_O2[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_O2))
    n2_PH[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_PH))
    n2_PHYC[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_PHYC))
    n2_PO4[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_PO4))
    n2_SI[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_BIO_lon, n_BIO_lat, np.array(ns_SI))
    n2_SPCO2[ii, :, :] = matchedArray(mesh_lon, mesh_lat, n_BIO_lon, n_BIO_lat, np.array(ns_SPCO2))
    n2_SST[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_SST_lon, n_SST_lat, np.array(ns_SST))
    n2_SLA[ii,:,:]=matchedArray(mesh_lon,mesh_lat,n_SLA_lon, n_SLA_lat, np.array(ns_SLA))


'''
################################
KOSOMES용 데이터를 하나의 NC로 만들기
################################
'''

import netCDF4 as nc

path_dir_sub = 'D:/30_Conference/2021-11_KOSOMES(PUS)/data'
os.makedirs(path_dir_sub, exist_ok=True)

fname='KOSMES'
path_file = path_dir_sub + '/' + fname + '_0p05.nc'
ds = nc.Dataset(path_file, 'w', format='NETCDF4')

lon = ds.createDimension('lat', r)
lat = ds.createDimension('lon', c)
tim = ds.createDimension('tim', len(date_list))

ds.title = fname
ds.area = 'Donghae'
ds.time = '2020'


lon = ds.createVariable(varname='longitude', datatype='f4', dimensions=('lat', 'lon'))
lat = ds.createVariable(varname='latitude', datatype='f4', dimensions=('lat', 'lon'))
time = ds.createVariable(varname='time', datatype=str, dimensions=('tim'))

bath = ds.createVariable(varname='bath',datatype=int,dimensions=('lat','lon')) # note: unlimited dimension is leftmost
bath.units = 'meter'
bath.standard_name = 'Bathymetry'

vbd = ds.createVariable(varname='vbd',datatype=int,dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
vbd.units = 'ships'
vbd.standard_name = 'VIIRS Boat Detection'

chl = ds.createVariable(varname='chl',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
chl.units = 'mg m-3'
chl.standard_name = 'chl-a concentration'

fe = ds.createVariable(varname='fe',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
fe.units = 'mmol m-3'
fe.standard_name = 'mole_concentration_of_dissolved_iron_in_sea_water'

no3 = ds.createVariable(varname='no3',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
no3.units = 'mmol m-3'
no3.standard_name = 'mole_concentration_of_nitrate_in_sea_water'

nppy = ds.createVariable(varname='nppy',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
nppy.units = 'mg m-3 day-1'
nppy.standard_name = 'Total Primary Production of Phyto'

o2 = ds.createVariable(varname='o2',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
o2.units = 'mmol m-3'
o2.standard_name = 'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water'

ph = ds.createVariable(varname='ph',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
ph.units = '1'
ph.standard_name = 'sea_water_ph_reported_on_total_scale'

phyc = ds.createVariable(varname='phyc',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
phyc.units = 'mg m-3'
phyc.standard_name = 'mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water'

po4 = ds.createVariable(varname='po4',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
po4.units = 'mg m-3'
po4.standard_name = 'mole_concentration_of_phosphate_in_sea_water'

si = ds.createVariable(varname='si',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
si.units = 'mg m-3'
si.standard_name = 'Dissolved Silicate'

spco2 = ds.createVariable(varname='spco2',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
spco2.units = 'Pa'
spco2.standard_name = 'surface_partial_pressure_of_carbon_dioxide_in_sea_water'

sst = ds.createVariable(varname='sst',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
sst.units = 'Celsius degree'
sst.standard_name = 'Sea Surface Temperature'

sla = ds.createVariable(varname='sla',datatype='f4',dimensions=('tim', 'lat','lon')) # note: unlimited dimension is leftmost
sla.units = 'm'
sla.standard_name = 'Sea Level Anomaly'


lon[:] = mesh_lon
lat[:] = mesh_lat
bath[:] = Bath
time[:] = date_list
vbd[:] = Data_VBD
chl[:] = n2_CHL
fe[:] = n2_FE
no3[:] = n2_NO3
nppy[:] = n2_NPPY
o2[:] = n2_O2
ph[:] = n2_PH
phyc[:] = n2_PHYC
si[:] = n2_SI
spco2[:] = n2_SPCO2
sst[:] = n2_SST
sla[:] = n2_SLA
ds.close()



