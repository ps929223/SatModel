'''

'''
import cv2
import netCDF4 as nc
import os, sys
sys.path.append('D:/programming/Dokdo')
from Lib.lib_os import *
from Lib.lib_GOCI1 import *
import numpy as np
import matplotlib.pyplot as plt

def interp(img,dsize=(300,300),method=cv2.INTER_CUBIC):
    resize_img=cv2.resize(img, dsize=dsize, interpolation=method)
    return resize_img

def genNC(img,lon, lat, lm, NonObsArea, path_output_nc):
    '''
    path_output_nc='E:/CHL_DM\\2021\\03\\GOCI1_DM_CHL_2021-03-25_res300.nc'
    '''

    # img=resize_img
    import netCDF4 as nc

    ds = nc.Dataset(path_output_nc, 'w', format='NETCDF4')

    r, c=img.shape
    y = ds.createDimension('y', r)
    x = ds.createDimension('x', c)

    ds.title = path_output_nc.split('\\')[-1][:-3]

    mesh_chl = ds.createVariable(varname='chl',datatype='f4',dimensions=('y','x')) # note: unlimited dimension is leftmost
    mesh_lon = ds.createVariable(varname='lon',datatype='f4',dimensions=('y','x'))
    mesh_lat = ds.createVariable(varname='lat',datatype='f4',dimensions=('y','x'))
    mesh_lm = ds.createVariable(varname='lm',datatype='f4',dimensions=('y','x'))
    mesh_NonObsArea = ds.createVariable(varname='NonObsArea', datatype='f4', dimensions=('y', 'x'))


    mesh_chl.units = 'mm m-3' # ship
    mesh_chl.standard_name = 'chl-a concentration' # this is a CF standard name
    mesh_lm.standard_name = 'land mask 0:sea, 1: land'  # this is a CF standard name
    mesh_NonObsArea.standard_name = 'Non-Observation Area 0: Obs, 1:NonObs'  # this is a CF standard name


    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    mesh_chl[:] = img
    mesh_lon[:] = lon
    mesh_lat[:] = lat
    mesh_lm[:] = lm
    mesh_NonObsArea[:] = NonObsArea

    ds.close()




mesh_lon, mesh_lat=read_GOCI1_coordnates()
LM = read_GOCI1_LM()
flag = read_GOCI1_flag()
NonObsArea=flag.copy()
NonObs_value=flag[5380,5390] # 가장자리 값
NonObsArea[NonObsArea>NonObs_value]=np.nan
NonObsArea=~np.isnan(NonObsArea)*1
resize_lon=interp(mesh_lon,dsize=(100,100), method=cv2.INTER_LINEAR)
resize_lat=interp(mesh_lat,dsize=(100,100), method=cv2.INTER_LINEAR)
resize_lm=interp(LM,dsize=(100,100), method=cv2.INTER_NEAREST)
resize_lm[resize_lm==130]=1 # 해안선도 육지로 고려
resize_NonObsArea=interp(NonObsArea,dsize=(100,100), method=cv2.INTER_NEAREST)

subset_lon=resize_lon[3:89,5:93]
subset_lat=resize_lat[3:89,5:93]
subset_lm=resize_lm[3:89,5:93]
subset_NonObsArea=resize_NonObsArea[3:89,5:93]


path_dir='E:/20_Product/CHL_DM/CHL_'
DMS=[path_dir+'DM1',path_dir+'DM3',path_dir+'DM5',path_dir+'DM7',path_dir+'DM9']

file_list=[]

for ii in range(len(DMS)):
    # ii=4
    file_list=file_list+recursive_file(DMS[ii], pattern='*.nc')

## 'res100' 자료는 목록에서 제외
file_list=np.array(file_list)[np.array(['_res100' not in name for name in file_list])]

error=[]
for ii in range(len(file_list)):
    print(file_list[ii])
    try:
        img=np.array(nc.Dataset(file_list[ii])['chl'])
        resize_img=interp(img,dsize=(100,100), method=cv2.INTER_CUBIC)
        resize_img[resize_img<0]=0 # 보간에 의해 0보다 작은 것은 0으로
        subset_img=resize_img[3:89,5:93]
        cond_nan=np.isnan(subset_img)

        genNC(subset_img, subset_lon, subset_lat, subset_lm, subset_NonObsArea,
              path_output_nc=file_list[ii][:-3]+'_y86x88.nc')
    except:
        error.append(file_list[ii])


