import pandas as pd
import os
import numpy as np


'''
최근자료(21.10.19)기준 편집
'''
path_input_dir='D:/20_Product/Buoy/Dokdo'
file_name='Dokdo_2020-10-16-2021-10-15.csv'
path_input_file=path_input_dir+'/'+file_name
DF=pd.read_csv(path_input_file)
## 각 Key의 string 앞에 space가 있어서 제거하고 다시 입력해 줌
keys=list(DF.keys())
for ii in range(len(keys)): ## 몇번 반복해준다
    keys[ii].replace(' ','')

## Datetime Column 생성
DF.columns=keys
DF['yymmddHHMMSS']=DF['yymmddHHMMSS'].astype(str)
DF20=pd.DataFrame({'Fore': ['20']*len(DF)}) # 연도가 '20 이런식으로 되어 있어서 앞에 20을 더 붙힘
DF['yymmddHHMMSS']=DF20['Fore']+DF['yymmddHHMMSS']
DF['yymmddHHMMSS']=pd.to_datetime(DF['yymmddHHMMSS'], format='%Y%m%d%H%M%S')

DF.to_csv(path_input_file[:-4]+'_edit.csv')


'''
옛날자료(2009~2021)기준 편집
'''
path_input_dir='D:/20_Product/Buoy/Dokdo'
file_name='Dokdo_2009-03-29-2021-01-07_추출.csv'
path_input_file=path_input_dir+'/'+file_name
DF=pd.read_csv(path_input_file)

cond=(2018<=DF['iy']) & (DF['iy'] <= 2020)
DF=DF[cond]
DF['yymmddHHMMSS']=DF['iy'].astype(str)+DF['im'].map('{0:02d}'.format)+DF['id'].map('{0:02d}'.format)\
                   +DF['jh'].map('{0:02d}'.format)+DF['jm'].map('{0:02d}'.format)
DF['yymmddHHMMSS']=pd.to_datetime(DF['yymmddHHMMSS'], format='%Y%m%d%H%M')


import Lib.lib_TimeSeriesOutliers as TSO
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as GF
from scipy import interpolate

DF[' S37Tmp01'][DF[' S37Tmp01']<5]=np.nan
DF['A']=TSO.tsclean(DF[' S37Tmp01'], qmin=.25, qmax=.85)
xx=np.arange(0,len(DF))[~np.isnan(DF['A'])]
f=interpolate.interp1d(xx,DF['A'][~np.isnan(DF['A'])], kind='linear')
DF['B']=f(np.arange(len(DF)))
plt.plot(DF['Time'], DF['B'])
DF['TC']=GF(DF['B'], sigma=5, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
# nan=np.isnan(DF['A']-DF['TC2']>0.1)
plt.plot(DF['Time'], DF['TC'])
nan=np.isnan(TSO.tsclean(DF['B']-DF['TC'], qmin=.1, qmax=.9))
DF['C']=DF['B'].copy()
DF['C'][nan]=DF['TC'][nan]
plt.scatter(DF['Time'], DF['C'], s=.1)

plt.plot(DF['Time'], DF[' S37Tmp01'])
plt.plot(DF['Time'], DF['A1'])
plt.plot(DF['Time'], DF['TC'])
plt.plot(DF['Time'], DF['A']-DF['TC1'])
plt.plot(DF['Time'], DF['A']-DF['TC2'])