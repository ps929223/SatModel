import os, sys
import matplotlib.pyplot as plt
sys.path.append('D:/programming/SatModel/Lib')

########### 특정위도선에 있는 모든 데이터를 추출 ###############

#### 1 특정 위도에 대한 데이터 추출
from Lib.lib_GOCI1 import *
from Lib.lib_os import *
import pandas as pd
import Lib.Map as Map
import matplotlib.pyplot as plt

## 경로 설정
path_coord = 'D:/02_Satellite/GOCI1/Coordinates'
path_source_dir= 'D:/02_Satellite/GOCI1/CHL'
path_output_dir= 'D:/20_Product/GOCI1/CHL/Pixel_TimeSeries'
os.makedirs(path_output_dir, exist_ok=True)

## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=recursive_file(path_source_dir, pattern="*.he5")

## 독도위치
target_lon, target_lat=Map.dokdo_psn()

## 데이터 추출
tp=[];tp2=[]
ii=0


for ii in range(len(file_list)):
    ## 진행경과 표시
    # if np.remainder(ii/len(file_list),0.1) == 0:
    #     print(int(ii/len(file_list)*100))
    print(file_list[ii].split('/')[-1])
    ## 데이터 읽기
    dataset=read_GOCI1_Chl(file_list[ii])
    ## 타겟위치로부터 가장 가까운 픽셀의 idx 읽기: 자르기 위함
    idx = find_nearst_idx(dataset['mesh_lon'], dataset['mesh_lat'], 129, target_lat)

    # 자르기
    dataset['mesh_lon']=dataset['mesh_lon'][:,idx[1][0]:]
    dataset['mesh_lat'] = dataset['mesh_lat'][:, idx[1][0]:]
    dataset['data']=dataset['data'][:, idx[1][0]:]

    ## 타겟위치로부터 가장 가까운 픽셀의 idx 읽기
    idx=find_nearst_idx(dataset['mesh_lon'], dataset['mesh_lat'], target_lon, target_lat)
    ## 추출된 idx를 통해 해당 row(위도)에 해당하는 모든 경도 데이터 추출
    ext_dataset=extract_for_specific_row(dataset['data'], dataset['mesh_lon'], target_row=idx[0])
    ## 경도데이터를 누적
    tp.append(ext_dataset['ext_data'])
    ##  Timestamp를 누적
    tp2.append(dataset['obs_time'])

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

## DF 저장
path_output_csv = path_output_dir + '/' + 'CHL-Lat' + str('%.3f' % target_lat).replace('.','p')+'.csv'
ext_DF.to_csv(path_output_csv)

## IQR에 해당하는 것만 뽑기 IQR=25~75%
vmin, vmax = np.nanpercentile(np.array(ext_DF).flatten(), (0, 100))


## 여기저기 사용할 경도별 추출
column_names=list(ext_DF.keys())
# ii=0
for ii in range(len(ext_DF.columns)):
    name=list(ext_DF.keys())[ii]
    name_str = 'CHL-Lat' + str('%.3f' % target_lat).replace('.', 'p') + '_Lon' + name.replace('.', 'p')
    ## 특정 경도데이터
    tp=ext_DF[name]
    ## 가시화
    plt.figure(1, figsize=(9,5))
    plt.plot(tp.index, tp, marker='.', linestyle='dashed', markerfacecolor='none', markersize=10, linewidth=0.5)
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
