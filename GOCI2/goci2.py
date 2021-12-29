import netCDF4 as nc
import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.path.append('D:/programming/SatModel/Lib')
import Map

########### Draw Map ###############
def draw_goci(fn, data_type):
    ds = nc.Dataset(fn)
    band=np.array(ds['geophysical_data'][data_type][:])
    flag=ds['geophysical_data']['flag']
    lat=np.array(ds['navigation_data']['latitude'][:])
    lon=np.array(ds['navigation_data']['longitude'][:])

    # 동해 34.0~43.7N, 128~138E
    # 독도 36.9~37.6N, 131.4~132.6E

    band[band == np.min(band)] = np.nan
    vmin, vmax = np.nanpercentile(band, [2.5, 97.5])
    cut_band = np.clip(band, vmin, vmax)

    sect = {'Donghae': [128, 138, 30, 40], 'Dokdo': [131.4, 132.6, 36.9, 37.6]}
    coord = sect['Donghae']

    m = Map.making_map(coord, 1)
    mesh_lon, mesh_lat = m(lon, lat)
    m.pcolor(mesh_lon, mesh_lat, cut_band, cmap='jet')
    m.colorbar()


def draw_GOCI2_CHL(path_file):
    '''
    path_file='D:/02_Satellite/GOCI2/CHL\\2021\\9\\9\\GK2B_GOCI2_L2_20210909_231530_LA_S004_Chl.nc'
    '''
    ds = nc.Dataset(path_file)
    band=np.array(ds['geophysical_data']['Chl'][:])
    mesh_lat=np.array(ds['navigation_data']['latitude'][:])
    mesh_lon=np.array(ds['navigation_data']['longitude'][:])
    # 동해 34.0~43.7N, 128~138E
    # 독도 36.9~37.6N, 131.4~132.6E

    band[band == np.min(band)] = np.nan
    vmin, vmax = np.nanpercentile(band, [2.5, 97.5])
    cut_band = np.clip(band, vmin, vmax)

    sect = {'Donghae': [128, 138, 30, 40], 'Dokdo': [131.4, 132.6, 36.9, 37.6]}
    coord = sect['Donghae']

    ## 독도위치
    target_lon, target_lat = Map.dokdo_psn()

    m = Map.making_map(coord, map_res='i', grid_res=1)
    xx,yy=m(mesh_lon,mesh_lat)
    m.pcolor(xx, yy, cut_band, cmap='jet')
    x,y=m(target_lon,target_lat)
    m.scatter(x, y, facecolor='w', edgecolor='k', s=80, marker='o')
    m.colorbar()


########### 특정위도선에 있는 모든 데이터를 추출 ###############

#### 1 특정 위도에 대한 데이터 추출
from Lib.lib_GOCI2 import *
from Lib.lib_os import *
import pandas as pd
import Lib.Map as Map
import matplotlib.pyplot as plt

## 경로 설정
path_source_dir= 'D:/02_Satellite/GOCI2/CHL'
path_output_dir= 'D:/20_Product/GOCI2/CHL/Pixel_TimeSeries'
os.makedirs(path_output_dir, exist_ok=True)

## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=recursive_file(path_source_dir, pattern="*S004*.nc")

## 독도위치
target_lon, target_lat=Map.dokdo_psn()

## 데이터 추출
tp=[];tp2=[]
ii=0
error=[]
for ii in range(510,len(file_list)):
    ## 진행경과 표시
    # if np.remainder(ii/len(file_list),0.1) == 0:
    #     print(int(ii/len(file_list)*100))
    ## 데이터 읽기
    try:
        dataset=read_GOCI2_Chl(file_list[ii])
        print(file_list[ii])
        ## 타겟위치로부터 가장 가까운 픽셀의 idx 읽기
        idx=find_nearst_idx(dataset['mesh_lon'], dataset['mesh_lat'], target_lon, target_lat)
        ## 추출된 idx를 통해 해당 row(위도)에 해당하는 모든 경도 데이터 추출
        ext_dataset=extract_for_specific_row(dataset['data'], dataset['mesh_lon'], target_row=idx[0])
        ## 경도데이터를 누적
        tp.append(ext_dataset['ext_data'])
        ##  Timestamp를 누적
        tp2.append(dataset['obs_time'])
    except:
        print('Error: '+file_list[ii])
        error.append(file_list[ii])


## DF변환 위해 자료형태 변환
tp=np.array(tp)
tp=tp.reshape(tp.shape[0],tp.shape[2])
## DF로 변환
lons=ext_dataset['ext_lon'][0]
str_lons=['%.3f'% lon for lon in lons]
DF=pd.DataFrame(tp, columns=str_lons, index=tp2)

## ext_DF의 index를 timestamp형식으로 변경
DF.index=pd.to_datetime(DF.index, format='%Y%m%d_%H%M%S')

## nan값만 있는 경도는 제거하기
cond=np.sum(~np.isnan(DF),0)>0 # 경도별 nan이 없는 것
ext_DF=DF[DF.keys()[cond]] # 위 조건에 유효한 경도값만 추출

## IQR에 해당하는 것만 뽑기 IQR=25~75%
vmin, vmax = np.nanpercentile(np.array(ext_DF).flatten(), (0, 99))

## DF 저장
path_output_csv = path_output_dir + '/' + 'CHL-Lat' + str('%.3f' % target_lat).replace('.','p')+'.csv'
ext_DF.to_csv(path_output_csv)

## 여기저기 사용할 경도명 추출
column_names=list(ext_DF.keys())
for ii in range(len(ext_DF.columns)):
# for ii in range(10):
    name=list(ext_DF.keys())[ii]
    print(name)
    name_str = 'CHL-Lat' + str('%.3f' % target_lat).replace('.', 'p') + '_Lon' + name.replace('.', 'p')
    ## 특정 경도데이터
    tp=ext_DF[name]
    ## 가시화
    plt.figure(1, figsize=(9,5))
    plt.plot(tp.index, tp, marker='.', linestyle='dashed', markerfacecolor='none', markersize=5, linewidth=0.5)
    plt.ylim(vmin*0.95,vmax*1.05)
    plt.title(name_str)
    plt.xlabel('Time [UTC]')
    plt.ylabel('Chl-a Concentration' +' ['+dataset['unit']+']')
    plt.grid()
    ## 저장
    path_output_png = path_output_dir + '/' + name_str + '.png'
    plt.savefig(path_output_png)
    ## figure 초기화
    plt.clf()



########### 가시화 배치처리 ###############

type_list=[]
for ii in range(len(file_list)):
    # ii=8
    type_name=file_list[ii].split('_')[-1][:-3]
    fn=path_dir+'/'+file_list[ii]
    plt.figure()
    draw_goci(fn,type_name)
    plt.title(file_list[ii])
    plt.savefig(file_list[ii][:-3]+'.png')
    plt.pause(10)
    plt.clf()

