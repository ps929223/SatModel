'''
만들어진 일평균(DM)을 읽어서 독도영역만 추출
전호군, 해양빅데이터센터
업뎃 2021.12.04
hkjeon@kiost.ac.kr
'''

'''
격자데이터 생성
'''

import netCDF4 as nc
from Lib.lib_GOCI1 import *
import Lib.Map as Map
import os

### 데이터 경로 설정
path_source_dir = 'E:/20_Product/CHL_DM/CHL_DM1'
path_output_dir = 'E:/20_Product/CHL_Krig'
path_grid_dir=path_output_dir+'/grid'

os.makedirs(path_output_dir, exist_ok=True)
os.makedirs(path_grid_dir, exist_ok=True)

### GOCI1의 경위도
mesh_lon, mesh_lat = read_GOCI1_coordnates()

## 독도영역 최저최고 좌표
coord=Map.sector()['Dokdo']

## GOCI1으로부터 추출한 격자
cond_coord = (coord[0] < mesh_lon) & (mesh_lon < coord[1]) & (coord[2] < mesh_lat) & (mesh_lat < coord[3])
idy,idx=np.where(cond_coord)


np.savetxt(path_grid_dir+'/Dokdo_idy.csv',idy, fmt='%i', delimiter=',')
np.savetxt(path_grid_dir+'/Dokdo_idx.csv',idx, fmt='%i', delimiter=',')

bumper=20 ## 추후 모델과 비교시 테두리 빈공간이 생기지 않도록 범퍼를 설정
ext_lon=mesh_lon[min(idy)-bumper:max(idy)+bumper,min(idx)-bumper:max(idx)+bumper]
ext_lat=mesh_lat[min(idy)-bumper:max(idy)+bumper,min(idx)-bumper:max(idx)+bumper]

## GOCI1에 대한 격자 저장: 살짝 비뚤어져있음
np.savetxt(path_grid_dir+'/Dokdo_ext_lon.csv',ext_lon, fmt='%.4f', delimiter=',')
np.savetxt(path_grid_dir+'/Dokdo_ext_lat.csv',ext_lat, fmt='%.4f', delimiter=',')

## 독도영역 경위도 추출
import Lib.Map as Map
mesh_lon, mesh_lat = Map.gen_sector_mesh('Dokdo',n_row=100, n_col=100)
### 생성된 Grid 저장
np.savetxt(path_grid_dir+'/Dokdo_meshlon.csv',mesh_lon, fmt='%.4f', delimiter=',') ## 모델자료와 비교를 위한 격자
np.savetxt(path_grid_dir+'/Dokdo_meshlat.csv',mesh_lat, fmt='%.4f', delimiter=',')

### Cloud Mask 저장 2020-08-01
import Lib.lib_Geo as Geo
Data=nc.Dataset('E:/20_Product/CHL_DM/CHL_DM1/2020/08/GOCI1_DM_CHL_2020-08-01.nc')['chl']
ext_chl = Data[int(min(idy)) - bumper:int(max(idy)) + bumper, int(min(idx)) - bumper:int(max(idx)) + bumper]
cloud_mask=np.isnan(ext_chl)
cloud_mask=Geo.matchedArray(mesh_lon, mesh_lat, ext_lon, ext_lat, cloud_mask)
np.savetxt(path_grid_dir + '/Dokdo_cloud_2020-08-01.csv', cloud_mask, fmt='%f', delimiter=',')

'''
저장된 격자를 읽고, Cloud Mask 작업
'''


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import netCDF4 as nc
from tqdm import tqdm

import Lib.lib_draw as Kd
import Lib.lib_outlier as Ol
import Lib.lib_Geo as Geo
from Lib.lib_os import *
from Lib.lib_GOCI1 import *



### 데이터 경로 설정
path_source_dir = 'E:/20_Product/CHL_DM/CHL_DM1'
path_output_dir = 'E:/20_Product/CHL_Krig'

### 파일목록
file_list=np.array(recursive_file(path_source_dir,'*.nc'))
file_list=file_list[np.array(['res100' not in name for name in file_list])] ## 해상도 줄인거는 제외

### 격자파일 읽기
mesh_lon=np.genfromtxt(path_grid_dir+'/Dokdo_meshlon.csv', delimiter=',')
mesh_lat=np.genfromtxt(path_grid_dir+'/Dokdo_meshlat.csv', delimiter=',')
ext_lon=np.genfromtxt(path_grid_dir+'/Dokdo_ext_lon.csv', delimiter=',')
ext_lat=np.genfromtxt(path_grid_dir+'/Dokdo_ext_lat.csv', delimiter=',')
idx=np.genfromtxt(path_grid_dir+'/Dokdo_idx.csv', delimiter=',')
idy=np.genfromtxt(path_grid_dir+'/Dokdo_idy.csv', delimiter=',')
cloud_mask=np.genfromtxt(path_grid_dir+'/Dokdo_cloud_2020-08-01.csv', delimiter=',')
cloud_mask=cloud_mask==1

bumper=20 ## 추후 모델과 비교시 테두리 빈공간이 생기지 않도록 범퍼를 설정


### 기간설정
date_list=pd.date_range('2020-01-01','2020-12-31').astype(str)

### 결측률을 넣을 빈공간
Ratio_Orig=np.zeros(len(date_list))
Ratio_Orig[Ratio_Orig==0]=np.nan
Ratio_Mask=Ratio_Orig.copy()
Ratio_ORM=Ratio_Orig.copy()
os.makedirs(path_output_dir + '/Ratio', exist_ok=True)

### 날짜별 원본, 이상값제거본, 마스킹본 CSV, PNG 저장
### 위 각각에 대한 결측률 계산
for ii in tqdm(range(len(date_list))):
    # ii=61 # 2020-08-01
    # ii=0
    ## 해당 날짜만 찾겠음
    cond = np.array([date_list[ii] in name for name in file_list])
    path_file=file_list[cond]

    ## 데이터 읽고 추출
    data=np.array(nc.Dataset(path_file[0])['chl'])
    ext_chl = data[int(min(idy))-bumper:int(max(idy))+bumper,int(min(idx))-bumper:int(max(idx))+bumper] ## 추출
    n_chl=Geo.matchedArray(mesh_lon, mesh_lat, ext_lon, ext_lat, ext_chl) ## Downsampling

    MN=len(ext_chl.flatten()) # 전체 격자 수
    Ratio_Orig[ii]=np.sum(np.isnan(ext_chl))/MN # 결측비율

    if Ratio_Orig[ii]<0.3: ## 원본에서 결측률이 30% 이내인 경우에만 사용. 결측률이 너무 크면 Kriging자체가 어려움
        ## 원본
        Kd.pcolor(mesh_lon, mesh_lat, n_chl, cmap='turbo', vmin=0.01, vmax=0.4, ticks=np.arange(0,0.5,0.05))
        plt.savefig(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_Orig.png')
        plt.clf()
        DF_Dokdo = pd.DataFrame({'lon': mesh_lon.flatten(), 'lat': mesh_lat.flatten(), 'chl-a': n_chl.flatten()})
        DF_Dokdo.to_csv(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_Orig.csv', index=False)

        ## 이상값 제거방법1: IQR score
        n1_chl=Ol.ORM_speckle2D(n_chl)
        o_ext_chl=Ol.ORM_speckle2D(n1_chl)

        ## 이상값 제거방법2: 임의Threshold
        # o_ext_chl = n_chl.copy() # 복사본
        # o_ext_chl[n_chl>1]=np.nan # 복사본에서 Outelier 제거

        Kd.pcolor(mesh_lon, mesh_lat, o_ext_chl.reshape(n_chl.shape),
                  cmap='turbo', vmin=0.01, vmax=0.4, ticks=np.arange(0,0.5,0.05))
        plt.savefig(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_ORM.png')
        plt.clf()
        DF_Dokdo = pd.DataFrame({'lon': mesh_lon.flatten(), 'lat': mesh_lat.flatten(), 'chl-a': o_ext_chl.flatten()})
        DF_Dokdo.to_csv(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_ORM.csv', index=False)
        Ratio_ORM[ii] = np.sum(np.isnan(o_ext_chl)) / MN

        ## 구름마스킹
        m_ext_chl=o_ext_chl.copy()
        m_ext_chl[cloud_mask]=np.nan
        Kd.pcolor(mesh_lon, mesh_lat, m_ext_chl, cmap='turbo', vmin=0.01, vmax=0.4, ticks=np.arange(0,0.5,0.05))
        plt.savefig(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_Mask.png')
        plt.clf()
        DF_Dokdo = pd.DataFrame({'lon': mesh_lon.flatten(), 'lat': mesh_lat.flatten(), 'chl-a': m_ext_chl.flatten()})
        DF_Dokdo.to_csv(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_Mask.csv', index=False)
        Ratio_Mask[ii] = np.sum(np.isnan(m_ext_chl)) / MN
    else:
        continue

### 결측률 저장
path_ratio=path_output_dir+'/Ratio'
os.makedirs(path_ratio, exist_ok=True)
DF=pd.DataFrame({'Date':date_list,'Ratio_Orig':Ratio_Orig,'Ratio_ORM':Ratio_ORM,'Ratio_Mask':Ratio_Mask})\
    .to_csv(path_ratio+'/Dokdo_Ratio.csv', index=False)