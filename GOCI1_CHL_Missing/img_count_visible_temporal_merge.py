'''
GOCI 가용자료 정도를 파악하기 위함
1. 매일의 GOCI1 자료 갯수
2. 파일마다의 구름비율(결측률)
3. 홀수일마다의 결측률
해양빅데이터센터
전호군
업뎃: 2021.12.07 # 관측외 영역 제외 기능 추가
초안: 2021.10.06
'''

import os, sys
from Lib.lib_GOCI1 import *
from Lib.lib_os import *
import pandas as pd
import matplotlib.pyplot as plt

def count(file_list, date_start, date_end):
    date_list_str=list(pd.date_range(date_start,date_end).strftime('%Y%m%d').astype(str))
    date_list_dt=pd.date_range(date_start,date_end)

    count=[]
    for ii in range(len(date_list_str)):
        tp1=file_list[np.array(['GA_'+date_list_str[ii] in file_name for file_name in file_list])]
        if not len(tp1)==0:
            count.append(len(tp1))
        else:
            count.append(np.nan)

    plt.figure(figsize=(11,7))
    plt.plot(date_list_dt,count)
    plt.ylabel('Count')
    plt.xlabel('Date')
    plt.title('# of Daily GOCI1 '+date_start+'~'+date_end)
    plt.grid()
    path_out_file=path_output_dir+'/'+'Count_GOCI1'+date_start+'-'+date_end
    plt.savefig(path_out_file+'.png')
    plt.close()

    ## DF 저장
    pd.DataFrame({'Date':date_list_str,'Count':count}).to_csv(path_out_file+'.csv', index=False)


def Visible_Hourly(file_list, date_start, date_end, LM, NonObsArea):
    from scipy.ndimage import gaussian_filter as GF

    count_nan=[]
    date=[]
    for ii in range(len(file_list)):
        # ii=8300
        file_name=file_list[ii].split('/')[-1]
        print(file_name)
        ## Date
        date.append(file_name.split('_')[-1][:-8])
        ## Chl
        CHL=read_GOCI1_Chl(file_list[ii])['CHL']
        ## NaN
        CHL[NonObsArea|LM] = 0
        nan=np.sum(np.isnan(CHL))
        count_nan.append(nan)

    DF=pd.DataFrame({'Time':date, 'Count_NaN':count_nan})
    DF['Time']=pd.to_datetime(DF['Time'])
    ttl=np.sum(~(NonObsArea|LM))
    DF['MR']=DF['Count_NaN']/ttl
    DF['Total']=ttl
    DF=DF[['Time', 'MR', 'Count_NaN', 'Total']].sort_values(by='Time')
    # DF['sma_NaNRatio']=DF['NaNRatio'].rolling(window=8*30, win_type='gaussian', center=True).mean(std=0.1)
    # DF['sma_MR']=DF['MR'].rolling(window=30).mean()
    GF_Ratio = GF(DF['MR'], 3, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    DF['GS3'] = GF_Ratio

    path_out_file = path_output_dir + '/' + 'MR_GOCI1_HR_' + date_start + '-' + date_end
    DF.to_csv(path_out_file+'.csv', index=False)

    plt.figure(figsize=(23,7))
    plt.plot(DF['Time'],DF['MR'], label='Orig')
    plt.plot(DF['Time'], DF['GS3'], label='GS3')
    plt.xlabel('DateTime')
    plt.ylabel('MR')
    plt.grid()
    plt.legend()
    plt.title('MR GOCI1 HR ' + date_start + ' ' + date_end)
    plt.tight_layout()
    plt.savefig(path_out_file+'.png')
    plt.close()


def Visible_DM(file_list, date_start, date_end, LM, NonObsArea):
    import netCDF4 as nc
    from scipy.ndimage import gaussian_filter as GF

    count_nan=[]
    date=[]
    for ii in range(len(file_list)):
    # for ii in range(0,10):
        # ii=0
        file_name=file_list[ii].split('/')[-1]
        print(file_name)
        ## Date
        date.append(file_name.split('_')[-1][:-3])
        ## Chl
        CHL=np.array(nc.Dataset(file_list[ii])['chl'])
        ## NaN
        CHL[NonObsArea|LM] = 0
        nan=np.sum(np.isnan(CHL))
        count_nan.append(nan)

    DF = pd.DataFrame({'Time': date, 'Count_NaN': count_nan})
    DF['Time'] = pd.to_datetime(DF['Time'])
    ttl = np.sum(~(NonObsArea | LM))
    DF['MR'] = DF['Count_NaN'] / ttl
    DF['Total']=ttl
    DF=DF[['Time', 'MR', 'Count_NaN', 'Total']].sort_values(by='Time')
    # DF['sma_NaNRatio']=DF['NaNRatio'].rolling(window=8*30, win_type='gaussian', center=True).mean(std=0.1)
    # DF['sma_MR']=DF['MR'].rolling(window=10).mean()
    GF_Ratio = GF(DF['MR'], 3, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    DF['GS3'] = GF_Ratio


    path_out_file = path_output_dir + '/' + 'MR_GOCI1_DM1_' + date_start + '-' + date_end
    DF.to_csv(path_out_file+'.csv', index=False)

    plt.figure(figsize=(23,7))
    plt.plot(DF['Time'],DF['MR'], label='Orig')
    plt.plot(DF['Time'], DF['GS3'], label='GS3')
    plt.xlabel('DateTime')
    plt.ylabel('MR')
    plt.grid()
    plt.legend()
    plt.title('MR GOCI1 DM1 ' + date_start + ' ' + date_end)
    plt.tight_layout()
    plt.savefig(path_out_file+'.png')
    plt.close()


def Visible_MultiDM(file_list, date_start, date_end, LM, NonObsArea, days=3):
    import netCDF4 as nc
    from scipy.ndimage import gaussian_filter as GF


    count_nan=[]
    date=[]

    for ii in range(int(np.floor(days/2)),int(len(file_list)-np.floor(days/2))):
        ## Date
        date.append(file_list[ii].split('/')[-1].split('_')[-1][:-3])
        ## Dataset
        CHL=np.zeros((days, LM.shape[0], LM.shape[1]))

        ## 과거 현재 미래의 자료를 읽어서 쌓기
        kk=0
        for jj in range(int(ii-np.floor(days/2)),int(ii+np.floor(days/2))+1):
            file_name=file_list[jj].split('/')[-1]
            print(file_name)
            ## Chl
            CHL[kk,:,:]=np.array(nc.Dataset(file_list[jj])['chl'])
            kk+=1

        # 누적자료 합성
        CHL=np.nanmean(CHL, axis=0)
        ## NaN
        CHL[NonObsArea|LM] = 0
        nan=np.sum(np.isnan(CHL))
        count_nan.append(nan)

    DF = pd.DataFrame({'Time': date, 'Count_NaN': count_nan})
    DF['Time'] = pd.to_datetime(DF['Time'])
    ttl = np.sum(~(NonObsArea | LM))
    DF['MR'] = DF['Count_NaN'] / ttl
    DF['Total'] = ttl
    DF=DF[['Time', 'MR', 'Count_NaN', 'Total']].sort_values(by='Time')
    GF_Ratio=GF(DF['MR'], 3, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    DF['GS3']=GF_Ratio
    path_out_file = path_output_dir + '/' + 'MR_GOCI1_DM_'+str(days)+'_' + date_start + '-' + date_end
    DF.to_csv(path_out_file+'.csv', index=False)

    plt.figure(figsize=(23,7))
    plt.plot(DF['Time'],DF['MR'], label='Orig')
    plt.plot(DF['Time'],DF['GS3'], label='GS3')
    plt.ylim(0, 1)

    plt.xlabel('Time[KST]')
    plt.ylabel('MR')
    plt.grid()
    plt.legend()
    plt.legend(loc='upper right')

    plt.tight_layout()

    plt.title('MR GOCI1 DM'+str(days)+' ' + date_start + ' ' + date_end)
    plt.tight_layout()
    plt.savefig(path_out_file+'.png')
    plt.close()




## Land Maksing
LM=read_GOCI1_LM()
LM=LM>=1 # 해안선과 육지를 모두 Maksing
print('LandArea:'+str(round(np.sum(LM)/(LM.shape[0]*LM.shape[1]),4)))

## NonObsArea 관측외 영역
flag=read_GOCI1_flag()
NonObsArea=flag.copy()
NonObsArea[NonObsArea>-2.6e+36]=np.nan
NonObsArea=~np.isnan(NonObsArea)
print('NonObsArea:'+str(round(np.sum(NonObsArea)/(NonObsArea.shape[0]*NonObsArea.shape[1]),4)))

'''
Vislble Hourly
'''

## Hourly 경로 설정
path_coord = 'E:/02_Satellite/GOCI1/Coordinates'
path_source_dir= 'E:/02_Satellite/GOCI1/CHL'
path_output_dir= 'E:/20_Product/GOCI1/CHL/MissingRatio'
os.makedirs(path_output_dir, exist_ok=True)


## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=np.array(recursive_file(path_source_dir, pattern="*.he5"))
file_list=file_list[np.array(['res100' not in name for name in file_list])]
file_list=file_list[np.array(['Dokdo' not in name for name in file_list])]

date_start='2018-03-04'
date_end='2021-03-24'

Visible_Hourly(file_list, date_start, date_end, LM, NonObsArea)


'''
Visible DM, MultipleDM
'''
## DM 경로 설정
path_coord = 'E:/02_Satellite/GOCI1/Coordinates'
path_source_dir= 'E:/20_Product/CHL_DM'
path_output_dir= 'E:/20_Product/GOCI1/CHL/MissingRatio'
os.makedirs(path_output_dir, exist_ok=True)


## 특정 디렉토리의 하부폴더내까지 포함하여 모든 파일경로를 반환
file_list=np.array(recursive_file(path_source_dir, pattern="*DM_*.nc"))
file_list=file_list[np.array(['res100' not in name for name in file_list])]
file_list=file_list[np.array(['Dokdo' not in name for name in file_list])]

date_start='2018-03-04'
date_end='2021-03-24'

## Visible DM
Visible_DM(file_list, date_start, date_end, LM, NonObsArea)

## Visible Multiple DM
list_days=[3, 5, 7, 9]
for days in list_days:
    Visible_MultiDM(file_list, date_start, date_end, LM, NonObsArea, days=9)



'''
############################
Graph
hourly, 1,3,5,7,9DM
############################
'''
from scipy.ndimage import gaussian_filter as GF
from Lib.lib_os import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_input_dir='E:/20_Product/GOCI1/CHL/MissingRatio'
list_file=recursive_file(path_input_dir,pattern='*.csv')[:6]
HR1=pd.read_csv(list_file[5])
DM1=pd.read_csv(list_file[0])
DM3=pd.read_csv(list_file[1])
DM5=pd.read_csv(list_file[2])
DM7=pd.read_csv(list_file[3])
DM9=pd.read_csv(list_file[4])

DF_list=[HR1, DM1, DM3, DM5, DM7, DM9]
DF_Name=['HR1', 'DM1', 'DM3', 'DM5', 'DM7', 'DM9']

date_start='2018-03-04'
date_end='2021-03-26'

## Gaussian Smoothing 적용하고 CSV저장
for ii in range(len(DF_list)):
    # ii=0
    if ii==0:
        sigma=3 # hourly만 30개
    else:
        sigma=3 # 1,3,5,7,9은 11개

    # date_start = DF_list[ii]['Time'][DF_list[ii].index[0]][:10]
    # date_end = DF_list[ii]['Time'][DF_list[ii].index[-1]][:10]

    GS=GF(DF_list[ii]['MR'], sigma=3, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0)

    DF_list[ii]['GS'+str(sigma)]=GS

    DF_list[ii]['Time']=pd.to_datetime(DF_list[ii]['Time'])

    path_output_name=path_input_dir+'/MR_GOCI1_CHL_'+DF_Name[ii]+'_'+date_start+'_'+date_end
    DF_list[ii].to_csv(path_output_name+'.csv', index=False)




## 만들어진 DF를 이용해 그림 따로 저장
plt.figure(figsize=(10, 4))

sigma=3
for ii in range(len(DF_list)):
    plt.plot(DF_list[ii]['Time'],DF_list[ii]['MR'], label='Orig', alpha=1)
    plt.plot(DF_list[ii]['Time'],DF_list[ii]['GS'+str(sigma)], label='Gaussian'+str(sigma), alpha=1)
    plt.xlabel('Time[KST]')
    plt.ylabel('MR')
    plt.ylim(0,1)
    plt.legend(loc='upper right')
    plt.grid()
    plt.title(DF_Name[ii])
    plt.tight_layout()
    path_output_name=path_input_dir+'/MR_GOCI1_CHL_'+DF_Name[ii]+'_'+date_start+'_'+date_end
    plt.savefig(path_output_name+'.png')
    plt.clf()


## GS 그래프를 모두어 그리기
plt.figure(figsize=(10, 4))
sigma=3
for ii in range(len(DF_list)):
    DF_list[ii]['Time'] = pd.to_datetime(DF_list[ii]['Time'])
    plt.plot(DF_list[ii]['Time'],DF_list[ii]['GS'+str(sigma)], label=DF_Name[ii], linestyle='-', linewidth=1, alpha=1)
plt.xlabel('Time[KST]')
plt.ylabel('MR')
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.grid()
# plt.title('n-days mean gaussian smooth')
plt.tight_layout()
path_output_name=path_input_dir+'/MR_GOCI1_CHL_nDM'+'_'+date_start+'_'+date_end
plt.savefig(path_output_name + '.png')



