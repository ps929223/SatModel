
'''
만들어진 일평균(DM)을 읽어서 독도영역만 추출
'''
import netCDF4 as nc
from Lib.lib_os import *
from Lib.lib_GOCI1 import *
import matplotlib.pyplot as plt
import Lib.Map as Map

### 데이터 경로 설정
path_source_dir = 'E:/20_Product/CHL_DM/CHL_DM1'
path_output_dir = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data'

### GOCI1의 경위도
mesh_lon, mesh_lat = read_GOCI1_coordnates()

## 독도영역 최저최고 좌표
coord=Map.sector()['Dokdo']

## GOCI1으로부터 추출한 격자
cond_coord = (coord[0] < mesh_lon) & (mesh_lon < coord[1]) & (coord[2] < mesh_lat) & (mesh_lat < coord[3])
idy,idx=np.where(cond_coord)

np.savetxt(path_output_dir+'/grid/Dokdo_idy.csv',idy, fmt='%i', delimiter=',')
np.savetxt(path_output_dir+'/grid/Dokdo_idx.csv',idx, fmt='%i', delimiter=',')

bumper=20 ## 추후 모델과 비교시 테두리 빈공간이 생기지 않도록 범퍼를 설정
ext_lon=mesh_lon[min(idy)-bumper:max(idy)+bumper,min(idx)-bumper:max(idx)+bumper]
ext_lat=mesh_lat[min(idy)-bumper:max(idy)+bumper,min(idx)-bumper:max(idx)+bumper]

## GOCI1에 대한 격자 저장: 살짝 비뚤어져있음
np.savetxt(path_output_dir+'/grid/Dokdo_ext_lon.csv',ext_lon, fmt='%.4f', delimiter=',')
np.savetxt(path_output_dir+'/grid/Dokdo_ext_lat.csv',ext_lat, fmt='%.4f', delimiter=',')

## 독도영역 경위도 추출

lons=np.linspace(coord[0],coord[1],100)
lats=np.linspace(coord[2],coord[3],100)
mesh_lon, mesh_lat=np.meshgrid(lons,lats)

### 생성된 Grid 저장
np.savetxt(path_output_dir+'/grid/Dokdo_meshlon.csv',mesh_lon, fmt='%.4f', delimiter=',') ## 모델자료와 비교를 위한 격자
np.savetxt(path_output_dir+'/grid/Dokdo_meshlat.csv',mesh_lat, fmt='%.4f', delimiter=',')

### Cloud Mask 저장 2020-08-01
import Lib.lib_Geo as Geo
Data=nc.Dataset('E:/20_Product/CHL_DM/CHL_DM1/2020/08/GOCI1_DM_CHL_2020-08-01.nc')['chl']
ext_chl = Data[int(min(idy)) - bumper:int(max(idy)) + bumper, int(min(idx)) - bumper:int(max(idx)) + bumper]
cloud_mask=np.isnan(ext_chl)
cloud_mask=Geo.matchedArray(mesh_lon, mesh_lat, ext_lon, ext_lat, cloud_mask)
np.savetxt(path_output_dir + '/grid/Dokdo_cloud_2020-08-01.csv', cloud_mask, fmt='%f', delimiter=',')

'''
### 저장된 격자를 읽고, Cloud Mask 작업
'''
import numpy as np
import pandas as pd
import netCDF4 as nc
from Lib.lib_os import *
from Lib.lib_GOCI1 import *
import matplotlib.pyplot as plt
import Lib.lib_Geo as Geo

### 데이터 경로 설정
path_source_dir = 'E:/20_Product/CHL_DM/CHL_DM1'
path_output_dir = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data'

### 기간설정
date_list=pd.date_range('2020-06-01','2020-08-31').astype(str)

### 파일목록
file_list=np.array(recursive_file(path_source_dir,'*.nc'))
file_list=file_list[np.array(['res100' not in name for name in file_list])] ## 해상도 줄인거는 제외

### 격자파일 읽기
mesh_lon=np.genfromtxt(path_output_dir+'/grid/Dokdo_meshlon.csv', delimiter=',')
mesh_lat=np.genfromtxt(path_output_dir+'/grid/Dokdo_meshlat.csv', delimiter=',')
ext_lon=np.genfromtxt(path_output_dir+'/grid/Dokdo_ext_lon.csv', delimiter=',')
ext_lat=np.genfromtxt(path_output_dir+'/grid/Dokdo_ext_lat.csv', delimiter=',')
idx=np.genfromtxt(path_output_dir+'/grid/Dokdo_idx.csv', delimiter=',')
idy=np.genfromtxt(path_output_dir+'/grid/Dokdo_idy.csv', delimiter=',')
cloud_mask=np.genfromtxt(path_output_dir+'/grid/Dokdo_cloud_2020-08-01.csv', delimiter=',')
cloud_mask=cloud_mask==1

bumper=20
from tqdm import tqdm
import Lib.lib_KAGIS_draw as Kd
import Lib.lib_outlier as Ol

Ratio_Orig=np.zeros(len(date_list))
Ratio_Orig[Ratio_Orig==0]=np.nan
Ratio_Mask=Ratio_Orig.copy()
Ratio_ORM=Ratio_Orig.copy()

import os
os.makedirs(path_output_dir + '/CHL_Orig', exist_ok=True)
os.makedirs(path_output_dir + '/QQ', exist_ok=True)
os.makedirs(path_output_dir + '/Ratio', exist_ok=True)

import Lib.lib_Geo as Geo

for ii in tqdm(range(len(date_list))):
# for ii in tqdm(range(4,6)):
    # ii=61 # 2020-08-01
    # ii=0
    ## 해당 날짜만 찾겠음
    cond = np.array([date_list[ii] in name for name in file_list])
    path_file=file_list[cond]

    ## 데이터 읽고 추출
    data=np.array(nc.Dataset(path_file[0])['chl'])
    ext_chl = data[int(min(idy))-bumper:int(max(idy))+bumper,int(min(idx))-bumper:int(max(idx))+bumper]
    n_chl=Geo.matchedArray(mesh_lon, mesh_lat, ext_lon, ext_lat, ext_chl)

    MN=len(ext_chl.flatten())
    Ratio_Orig[ii]=np.sum(np.isnan(ext_chl))/MN

    if Ratio_Orig[ii]<0.3: ## 원본에서 결측률이 30% 이내인 경우에만 사용
        ## 원본
        Kd.pcolor(mesh_lon, mesh_lat, n_chl)
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
        Kd.pcolor(mesh_lon, mesh_lat, o_ext_chl.reshape(n_chl.shape))
        plt.savefig(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_ORM.png')
        plt.clf()
        DF_Dokdo = pd.DataFrame({'lon': mesh_lon.flatten(), 'lat': mesh_lat.flatten(), 'chl-a': o_ext_chl.flatten()})
        DF_Dokdo.to_csv(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_ORM.csv', index=False)
        Ratio_ORM[ii] = np.sum(np.isnan(o_ext_chl)) / MN

        ## 구름마스킹
        m_ext_chl=o_ext_chl.copy()
        m_ext_chl[cloud_mask]=np.nan
        Kd.pcolor(mesh_lon, mesh_lat, m_ext_chl)
        plt.savefig(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_Mask.png')
        plt.clf()
        DF_Dokdo = pd.DataFrame({'lon': mesh_lon.flatten(), 'lat': mesh_lat.flatten(), 'chl-a': m_ext_chl.flatten()})
        DF_Dokdo.to_csv(path_output_dir + '/CHL_Orig/Dokdo_' + date_list[ii] + '_CHL_Mask.csv', index=False)
        Ratio_Mask[ii] = np.sum(np.isnan(m_ext_chl)) / MN
    else:
        continue

path_ratio=path_output_dir+'/Ratio'
os.makedirs(path_ratio, exist_ok=True)
DF=pd.DataFrame({'Date':date_list,'Ratio_Orig':Ratio_Orig,'Ratio_ORM':Ratio_ORM,'Ratio_Mask':Ratio_Mask})\
    .to_csv(path_ratio+'/Dokdo_Ratio.csv', index=False)



#
# ''' 데이터 입력 및 전처리 '''
# def read_DF(target_date_str):
#     path_input_csv='D:/30_학술대회/2021-11 한국지리정보학회(제주)/data/ORM_IQRscore/CHL_Orig/Dokdo_'+target_date_str+'_CHL.csv'
#     DF=pd.read_csv(path_input_csv)
#     DF=DF.dropna(subset=['chl-a'], axis=0)
#     return DF
#
#
# date_list=pd.date_range('2020-06-16','2020-08-31').astype(str)
# dates=[]
# nodata_date=[]
# for ii in range(len(date_list)):
#     # ii=0
#     print(date_list[ii])
#     target_date_str=date_list[ii]
#     DF=read_DF(target_date_str)
#     dates.append(date_list[ii])
#     if len(DF)<10:
#         print('No data date '+date_list[ii])
#         ranges.append(np.nan)
#         sills.append(np.nan)
#         nuggets.append(np.nan)
#         RMSEs.append(np.nan)
#         MADs.append(np.nan)
#     else:
#
#         plt.figure(4)
#         scatter(DF,target_date_str)
#         print('Scatter.. Done')
#
#         gen_chl_hist(DF,target_date_str)
#         print('Histogram.. Done')
#
#         krig_train, krig_test=split_chl_traintest(DF, target_date_str)
#         print('Train/Test Separation.. Done')
#
#         QQ(DF,target_date_str)
#         print('QQ.. Done')
#
#         DF_wCoord=add_xy_colum(ddm_lon, ddm_lat, DF)