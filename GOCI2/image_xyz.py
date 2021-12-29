import os, sys
import matplotlib.pyplot as plt
sys.path.append('D:/programming/SatModel/Lib')

########### 한달 평균 데이터 ###############

#### 1 특정 위도에 대한 데이터 추출
from Lib.lib_GOCI1 import *
from Lib.lib_os import *
import pandas as pd

## 경로 설정
path_coord = 'D:/02_Satellite/GOCI1/Coordinates'
path_source_dir= 'E:/CHL'
path_output_dir= 'D:/20_Product/GOCI1/CHL/Pixel_TimeSeries'
os.makedirs(path_output_dir, exist_ok=True)

## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=np.array(recursive_file(path_source_dir, pattern="*202006*.he5"))
date_list=list(pd.date_range('2020-06-01','2020-06-30').strftime('%Y%m%d').astype(str))

file_list_duration=[]
for ii in range(len(date_list)):
    tp=file_list[np.array([date_list[ii] in file_path for file_path in file_list])]
    file_list_duration=np.concatenate([file_list_duration, tp])


mesh_lon, mesh_lat = read_GOCI1_coordnates()

r,c=mesh_lon.shape
dataset=np.zeros((len(file_list_duration),r,c))

time=[]
for ii in range(len(file_list_duration)):
    ## 진행경과 표시
    # if np.remainder(ii/len(file_list),0.1) == 0:
    #     print(int(ii/len(file_list)*100))
    print(file_list_duration[ii].split('/')[-1])
    ## 데이터 읽기
    tp=read_GOCI1_Chl(file_list_duration[ii])
    dataset[ii,:,:]=tp['CHL']
    time.append(tp['time'])

## 일주일 평균
DSmean=np.nanmean(dataset, axis=0)

plt.pcolor(mesh_lon,mesh_lat,DSmean)
plt.grid()
# DSmean[np.isnan(DSmean)]='NaN'
DSmean=DSmean.astype(str)

r_idx=~np.isnan(np.nanmean(DSmean,axis=1))
c_idx=~np.isnan(np.nanmean(DSmean,axis=0))

ext_lon=mesh_lon[r_idx,:]
ext_lon=ext_lon[:,c_idx]
ext_lat=mesh_lat[r_idx,:]
ext_lat=ext_lat[:,c_idx]
ext_chl=DSmean[r_idx,:]
ext_chl=ext_chl[:,c_idx]


## Flatten
lons, lats, chls =ext_lon.flatten(), ext_lat.flatten(), ext_chl.flatten()

## DF
DF = pd.DataFrame({'lon':lons, 'lat':lats, 'chl-a': chls})
DF=DF.dropna['chl-a'](axis=0)

DF.info(memory_usage = "deep")

## downcasting loop
for column in DF:
    if DF[column].dtype == 'float32':
        DF[column]=pd.to_numeric(DF[column], downcast='float')
    if DF[column].dtype == 'int64':
        DF[column]=pd.to_numeric(DF[column], downcast='int')

DF.to_csv(path_output_dir+'/ALLPix_'+time[0][:10]+'_'+time[-1][:10]+'.csv', index=False)