'''
GOCI 가용자료 정도를 파악하기 위함
DMn에 대해 해상도를 2,4,6,8배 낮추는 방식으로 진행
해양빅데이터센터
전호군
초안: 2021.11.29
'''


def draw_DM_kernel(path_file, MaskIn, ksize=(2,2)):
    '''
    ksize=(1,1)
    DM=1
    '''
    import netCDF4 as nc
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import Lib.lib_MovingWindow as MW

    MaskIn=MW.poolingNonOverlap(mat=MaskIn, ksize=ksize, method='max', pad=False)

    file_name = path_file.split('\\')[-1]
    print(file_name)

    ## DM
    DM = file_name.split('_')[1]
    if DM=='DM':
        DM='DM1'

    ## Date
    date_str=file_name[:-3].split('_')[-1]
    ## Chl
    CHL = np.array(nc.Dataset(path_file)['chl'])
    ## NaN
    CHL=MW.poolingNonOverlap(mat=CHL,ksize=ksize,method='mean',pad=False)
    CHL[~MaskIn] = 0  # MaskOut=0
    CHL[np.isnan(CHL)]=1 # 결측은 값으로 1을 입력
    CHL[~(CHL==0)&~(CHL==1)]=0.5 # 값이 있는 것은 0으로 입력

    nan = np.sum(CHL == 1)
    ttl = np.sum(MaskIn)
    mr = nan / ttl

    plt.figure(1, figsize=(4,4))
    plt.imshow(CHL, cmap='Greys_r')
    title = 'CHL_' + date_str + '_' + DM + '_K' + str(ksize[0])
    plt.title(title.replace('_',' ')+' MR: '+str(round(mr,4)))
    plt.tight_layout()

    path_png_dir='E:/20_Product/GOCI1/CHL/MissingRatio/DailyMap'
    os.makedirs(path_png_dir, exist_ok=True)
    plt.savefig(path_png_dir+'/'+title+'.png')
    plt.clf()
    return mr, nan, ttl


def visible_DM_kernel(path_file, MaskIn, ksize=(2,2)):
    '''
    path_file='E:/20_Product/CHL_DM\\CHL_DM1\\2018\\03\\GOCI1_DM_CHL_2018-03-04.nc'
    ksize=(1,1)
    '''
    import netCDF4 as nc
    import numpy as np
    import Lib.lib_MovingWindow as MW

    MaskIn = MW.poolingOverlap(mat=MaskIn, ksize=ksize, method='max', pad=False)

    file_name = path_file.split('\\')[-1]
    print(file_name)
    ## Date
    date_str=file_name.split('_')[-1][:-3]
    ## Chl
    CHL = np.array(nc.Dataset(path_file)['chl'])
    ## NaN
    CHL=MW.poolingNonOverlap(CHL,ksize,method='mean',pad=False)
    CHL[~MaskIn] = 0  # MaskOut 0 을 입력
    CHL[np.isnan(CHL)]=1 # 결측은 값으로 1을 입력
    CHL[~(CHL==0)&~(CHL==1)]=0.5 # 값이 있는 것은 0으로 입력

    nan = np.sum(CHL == 1)
    ttl = np.sum(MaskIn)
    mr = nan / ttl

    return mr, nan, ttl

'''
날짜별 Spatial, Temporal 해상도에 따른 결측률 지도 그림
'''

import numpy as np
from Lib.lib_os import *
import Lib.lib_MovingWindow as MW
import Lib.lib_GOCI1 as G1

## DM 경로 설정
path_source_dir = 'E:/20_Product/CHL_DM'

DMs=[1,3,5,7,9]
ksizes=[1,2,4,6,8]

file_list = np.array(recursive_file(path_source_dir, pattern="*DM*.nc"))
file_list = file_list[np.array(['res100' not in name for name in file_list])]
file_list = file_list[np.array(['Dokdo' not in name for name in file_list])]
date_list=[file_list[ii].split('\\')[-1].split('_')[3].split('.nc')[0] for ii in range(len(file_list))]
date_list=np.unique(date_list)
# path_file=file_list[0]


## Land Maksing
LM=G1.read_GOCI1_LM()
LM=LM>=1 # 해안선과 육지를 모두 Maksing
print('LandArea:'+str(round(np.sum(LM)/(LM.shape[0]*LM.shape[1]),4)))

## NonObsArea 관측외 영역
flag=G1.read_GOCI1_flag()
NonObsArea=flag.copy()
NonObsArea[NonObsArea>-2.6e+36]=np.nan
NonObsArea=~np.isnan(NonObsArea)
print('NonObsArea:'+str(round(np.sum(NonObsArea)/(NonObsArea.shape[0]*NonObsArea.shape[1]),4)))

MaskIn=~(LM|NonObsArea)


### Draw DM Kernel
for hh in range(len(date_list)):
    # hh=21
    date_files = np.array(recursive_file(path_source_dir, pattern="*DM*"+date_list[hh]+"*.nc"))
    date_files = date_files[np.array(['res100' not in name for name in date_files])]
    date_files = date_files[np.array(['Dokdo' not in name for name in date_files])]

    for ii in range(len(date_files)): # 파일이름에서 DM을 추출함
        # ii=0
        DM=date_files[ii].split('\\')[-1].split('_')[1].split('DM')[1]
        if DM=='':
            DM_file=np.array(recursive_file(path_source_dir, pattern="*DM_*"+date_list[hh]+".nc"))
            DM = '1'
        else:
            DM_file = np.array(recursive_file(path_source_dir, pattern="*DM"+DM+'*_*' + date_list[hh] + ".nc"))
            path_file = file_list[ii]
        for jj in range(len(DM_file)):
            path_file=DM_file[jj]
            for kk in range(len(ksizes)):
                print(path_file)
                draw_DM_kernel(path_file, MaskIn, ksize=(ksizes[kk],ksizes[kk]))


'''
전체기간 Spatial, Temporal 해상도에 따른 결측률 테이블
'''

from Lib.lib_os import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as GF

# 날짜 목록
file_list = np.array(recursive_file(path_source_dir, pattern="*DM*.nc"))
file_list = file_list[np.array(['res100' not in name for name in file_list])]
file_list = file_list[np.array(['Dokdo' not in name for name in file_list])]


## DM 경로 설정
path_source_dir = 'E:/20_Product/CHL_DM'

DMs=[1,3,5,7,9]
ksizes=[1,2,4,6,8]
vars=['Time', 'MR','Count_NaN','Total','GS3']

ii=jj=kk=0
for ii in range(len(DMs)):
    ## DM에 따라 파일을 호출함. DM이 높을 수록 앞뒤에 잘리는 날짜들이 있음
    if DMs[ii]==1:
        file_list = np.array(recursive_file(path_source_dir, pattern="*DM_*.nc"))
    else:
        file_list = np.array(recursive_file(path_source_dir, pattern="*DM"+str(DMs[ii])+"*.nc"))

    ## 파일명 중 필요한 것만 필터링
    file_list = file_list[np.array(['res100' not in name for name in file_list])]
    file_list = file_list[np.array(['Dokdo' not in name for name in file_list])]
    ## 파일명에서 Date를 추출
    date_list = [file_list[ii].split('\\')[-1].split('_')[3].split('.nc')[0] for ii in range(len(file_list))]
    ## DM이 여러 개라서 중복을 제거
    date_list = np.unique(date_list)

    ## 커널 사이즈에 따라서...
    for jj in range(len(ksizes)):
        DF = np.zeros((len(date_list), len(vars)))
        DF[:] = np.nan
        DF = pd.DataFrame(DF, columns=vars)
        for kk in range(len(file_list)):
            path_file=file_list[kk]
            DF['MR'][kk], DF['Count_NaN'][kk], DF['Total'][kk]=\
                visible_DM_kernel(path_file=path_file, MaskIn=MaskIn, ksize=(ksizes[jj], ksizes[jj]))
            DF['Time'][kk] = str(date_list[kk])
        date_start = DF.Time[DF.index[0]]
        date_end = DF.Time[DF.index[-1]]

        GF_Ratio = GF(DF['MR'], 3, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
        DF['GS3'] = GF_Ratio

        path_output_dir = 'E:/20_Product/GOCI1/CHL/MissingRatio/DM_Kernel'
        os.makedirs(path_output_dir, exist_ok=True)

        path_out_file = path_output_dir + '/' + 'MR_DM' + str(DMs[ii]) + '_Kernel' + str(ksizes[jj])
        DF.to_csv(path_out_file + '.csv', index=False)

        DF.Time=pd.to_datetime(DF.Time)
        plt.figure(2, figsize=(23, 7))
        plt.plot(DF['Time'], DF['MR'], label='Orig')
        plt.plot(DF['Time'], DF['GS3'], label='GS3')
        plt.xlabel('DateTime')
        plt.ylabel('MR')
        plt.grid()
        plt.legend()

        plt.title('MR GOCI1 DM' + str(DMs[ii]) + ' Kernel' + str(ksizes[jj]) + ' ' + date_start + ' ' + date_end)
        plt.tight_layout()
        plt.savefig(path_out_file + '.png')
        plt.clf()
        del DF


'''
############################
가로축 Spatial, 세로축 Temporal
결측률 비교 테이블 생성. 월평균
############################
'''

from Lib.lib_os import *
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

path_input_dir='E:/20_Product/GOCI1/CHL/MissingRatio/DM_Kernel'
path_output_dir=path_input_dir+'/Result'
os.makedirs(path_output_dir, exist_ok=True)
file_list=np.array(recursive_file(path_input_dir,pattern='*DM*kernel*.csv'))

## Column List
Name_list=[file_list[ii].split('\\')[-2] for ii in range(len(file_list))]
Name_list=[Name_list[ii].replace('kernel','K') for ii in range(len(file_list))]

## Month List
date_start='2018-03-04'
date_end='2021-03-24'
date_range=list(pd.date_range(date_start,date_end,freq='31D').astype(str))
month_list=[date_range[ii][:-3] for ii in range(len(date_range))]

## DM List
# ii=0
DM_list_FN = np.array([file_list[ii].split('\\')[-1].split('_')[1] for ii in range(len(file_list))])
DM_list = np.unique(DM_list_FN)

## Kernel List
kernel_list_FN=np.array([file_list[ii].split('\\')[-1].split('_')[2][:-4] for ii in range(len(file_list))])
kernel_list=np.unique(kernel_list_FN)

## Month, DM, Kernel의 3D 행렬
MDK=np.zeros((len(month_list),len(kernel_list), len(DM_list)))
MDK[:]=np.nan

## Missing Ratio(MR) 표 만들기: 가로축 Spatial, 세로축 Temporal
for hh in range(len(month_list)):
    ## 파일을 월별로 만들기 때문에 가장 반복구조 최상위에 위치
    monthly = np.zeros((len(DM_list), len(kernel_list)))
    monthly[:] = np.nan
    monthly = pd.DataFrame(monthly, columns=kernel_list)

    for ii in range(len(kernel_list)): # 만들어질 테이블의 col 방향 kernel
        for jj in range(len(DM_list)): # 만들어질 테이블의 row 방향 DM
            cond_file=np.array([kernel_list[ii]==kernel_list_FN])&np.array([DM_list[jj]==DM_list_FN]) # DM과 kernel 조건
            tgt_file=file_list[cond_file[0]][0] # 조건에 맞는 파일 검색
            DFallperiod = pd.read_csv(tgt_file) # 파일 읽기
            DFallperiod.Time = pd.to_datetime(DFallperiod.Time) # Datetime 변환

            ## Monthly 작업
            cond_1month=month_list[hh]==DFallperiod['Time'].dt.strftime('%Y-%m') # 해당 Month 내 일자들 index
            month_mean=np.mean(DFallperiod['MR'][cond_1month]) # 1개월 평균계산
            monthly[kernel_list[ii]][jj]=month_mean # 표에 입력
    MDK[hh,:,:]=np.array(monthly)
    monthly.index=DM_list # DM 문자열을 index에 입력
    monthly.to_csv(path_output_dir+'/MR_'+month_list[hh]+'.csv', index=True) # 저장



## DM에 따른 MR 시계열 그래프
plt.figure(figsize=(10,4))
from datetime import datetime as dt
import matplotlib.pyplot as plt
month_list_dt=np.array([dt.strptime(month_list[jj],'%Y-%m') for jj in range(len(month_list))])
for ii in range(len(DM_list)):
    mr=MDK[:,ii,0]
    plt.plot(month_list_dt,mr,label=DM_list[ii])
plt.ylim(0,1)
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('UTC', fontsize=12)
plt.ylabel('MR', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Monthly Mean MR by DM')
plt.tight_layout()



## Kernel에 따른 MR 시계열 그래프
plt.figure(figsize=(10,4))
from datetime import datetime as dt
import matplotlib.pyplot as plt
month_list_dt=np.array([dt.strptime(month_list[jj],'%Y-%m') for jj in range(len(month_list))])
for ii in range(len(kernel_list)):
    mr=MDK[:,0,ii]
    plt.plot(month_list_dt,mr,label=kernel_list[ii])
plt.ylim(0,1)
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('UTC', fontsize=12)
plt.ylabel('MR', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Monthly Mean MR by Kernel(with DM1)')
plt.tight_layout()