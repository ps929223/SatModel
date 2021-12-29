'''
독도부이 구자료와 신자료 매칭
구자료(~2021.01)
신자료(2010.10~)

공통되는 것은 신자료 변수명에 맞추고,
각각 독립적인 것들은 독립적인 변수명을 이용함
'''

import pandas as pd
import os
import numpy as np


'''
옛날자료(2009~2021)기준 편집
'''

path_input_dir='D:/20_Product/Buoy/Dokdo'
file_name='Dokdo_2009-03-29-2021-01-07_추출.csv'
path_input_file=path_input_dir+'/'+file_name
DF1=pd.read_csv(path_input_file)


## 각 Key의 string 앞에 space가 있어서 제거하고 다시 입력해 줌
keys=list(DF1.keys())
for jj in range(len(keys)):
    for ii in range(30):  ## 넉넉하게 10번 반복
        keys[jj]=keys[jj].replace(' ','')
        keys[jj] = keys[jj].replace('.', '_')
## 새로운 Column명 입력
DF1.columns=keys

## 최근3년자료만 가져오기로 함
cond=(2018<=DF1['iy']) & (DF1['iy'] <= 2020)
DF1=DF1[cond]

## 시각 변수
DF1['yymmddHHMMSS']=DF1['iy'].astype(str)+DF1['im'].map('{0:02d}'.format)+DF1['id'].map('{0:02d}'.format)\
                   +DF1['jh'].map('{0:02d}'.format)+DF1['jm'].map('{0:02d}'.format)
DF1['yymmddHHMMSS']=pd.to_datetime(DF1['yymmddHHMMSS'], format='%Y%m%d%H%M')

## 병합에서 남기고 싶은 변수명
name_vars=['yymmddHHMMSS','bv', 'Wd', 'wsp', 'wmsp', 'CSTAR', 'Chloro', 'fhg', 'fpd', 'zhg', 'zpd', 'atemp', 'rh', 'bp', 'lat', 'lon']
## 변수명에 해당하는 DF만 추출
DF1=DF1[name_vars]
## 변수명을 신규자료에 맞추어 수정
DF1.columns=['yymmddHHMMSS','Volt', '1WD', '1WS', '1WG', 'ntu', 'chl', 'w_FH', 'w_FP', 'w_ZH', 'w_ZP', 'Temp_1',
             'HR', 'bP', 'Latitude', 'Longitude']

DF1=DF1.reset_index(drop=True)
DF1=DF1.loc[~DF1.index.duplicated(keep='first')]
DF1.to_csv(path_input_dir+'/Dokdo_2018-2020.csv', index=False)


'''
최근자료(21.10.19)기준 편집
'''
path_input_dir='D:/20_Product/Buoy/Dokdo'
file_name='Dokdo_2020-10-16-2021-10-15.csv'
path_input_file=path_input_dir+'/'+file_name
DF2=pd.read_csv(path_input_file)

## 각 Key의 string 앞에 space가 있어서 제거하고 다시 입력해 줌
keys=list(DF2.keys())
for jj in range(len(keys)):
    for ii in range(30):  ## 넉넉하게 10번 반복
        keys[jj] = keys[jj].replace(' ','')
        keys[jj] = keys[jj].replace('.', '_')
print(keys)
## 새로운 Column명 입력
DF2.columns=keys

## Datetime Column
DF2['yymmddHHMMSS']=DF2['yymmddHHMMSS'].astype(str)
DF20=pd.DataFrame({'Fore': ['20']*len(DF2)}) # 연도가 '20 이런식으로 되어 있어서 앞에 20을 더 붙힘
DF2['yymmddHHMMSS']=DF20['Fore']+DF2['yymmddHHMMSS']
del(DF20)
DF2['yymmddHHMMSS']=pd.to_datetime(DF2['yymmddHHMMSS'], format='%Y%m%d%H%M%S')
DF_tp=pd.DataFrame(columns=['deg','min','sec'])
DF_tp['deg']=np.floor(DF2['Latitude']/100).astype(int)
DF_tp['min']=np.floor(DF2['Latitude']-DF_tp['deg']*100).astype(int)/60
DF_tp['sec']=(DF2['Latitude']-DF_tp['deg']*100-DF_tp['min'])/60/60
DF2['Latitude']=DF_tp['deg']+DF_tp['min']+DF_tp['sec']

DF_tp['deg']=np.floor(DF2['Longitude']/100).astype(int)
DF_tp['min']=np.floor(DF2['Longitude']-DF_tp['deg']*100).astype(int)/60
DF_tp['sec']=(DF2['Longitude']-DF_tp['deg']*100-DF_tp['min'])/60/60
DF2['Longitude']=DF_tp['deg']+DF_tp['min']+DF_tp['sec']


## 특정기간 설정
cond=DF2['yymmddHHMMSS'].astype(str) >= '2021-01-01 00:00:00' # 2021년 이후자료
DF2=DF2[cond]
# DF2=DF2.reset_index(drop=True)
DF2=DF2.loc[~DF2.index.duplicated(keep='first')]
DF2.to_csv(path_input_dir+'/Dokdo_2021-01-01-2021-10-15.csv', index=False)


'''
이제 위의 오래된 것과 최신 것을 병합
'''

## 공통되는 변수명
name_comm_vars=list(set(DF1.keys()).intersection(DF2.keys()))
print(name_comm_vars)
name_comm_vars=['yymmddHHMMSS',  'Temp_1', 'bP',  'HR', 'chl', 'ntu', '1WS', '1WD', '1WG',
                'w_FH', 'w_FP', 'w_ZH', 'w_ZP', 'Volt', 'Longitude', 'Latitude']

DF1=DF1[name_comm_vars]
DF2=DF2[name_comm_vars]

DF_combine=pd.concat([DF1, DF2], axis=0)

DF_combine.to_csv(path_input_dir+'/Dokdo_2020-01-01_2021-10-15.csv', index=False)