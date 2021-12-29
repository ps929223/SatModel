'''
Coprenicus와 조사원 조위관측소의 해류 자료를 비교하는 코드
- 연 단위 비교
제작: 전호군
부서: 해양빅데이터센터
초안: 2021.09.27
'''


import netCDF4 as nc
import pandas as pd
import datetime as dt
import os, sys
sys.path.append('E:/programming/Dokdo')
import Lib.Map as Map
from Lib.lib_Geo import *
from Lib.lib_os import *
from Lib.lib_math import *
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import gridspec


def norm(src,vmin,vmax):
    vnorm=(src-vmin)/(vmax-vmin)
    return vnorm

def read_Model_DS(path_Model, datatype):
    '''
    path_file=
    datatype='Wind'
    '''
    ds = nc.Dataset(path_Model)
    # vars = list(ds.variables)

    # 시간변수 계산
    time = np.array(ds.variables['time'][:])
    if datatype == 'CUR':
        time_since = dt.datetime.strptime('1950-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        time = np.array([time_since + dt.timedelta(hours=int(ii)) for ii in time])
        lat = np.array(ds.variables['latitude'][:])
        lon = np.array(ds.variables['longitude'][:])
        uo = np.array(ds.variables['uo'][:])  # eastward_sea_water_velocity
        vo = np.array(ds.variables['vo'][:])  # northward_sea_water_velocity
    elif datatype =='Wind':
        time_since = dt.datetime.strptime('1900-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        time = np.array([time_since + dt.timedelta(hours=int(ii)) for ii in time])
        lat = np.array(ds.variables['lat'][:])
        lon = np.array(ds.variables['lon'][:])
        uo = np.array(ds.variables['eastward_wind'][:])  # eastward_sea_water_velocity
        vo = np.array(ds.variables['northward_wind'][:])  # northward_sea_water_velocity

    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    dataset={'time':time, 'mesh_lon': mesh_lon, 'mesh_lat':mesh_lat, 'uo':uo, 'vo':vo}

    return dataset

def extract_Model_data(target_time,target_coord,Model_DS, guard_cell=10):
    '''
    # target_time = '2020-01-03 00:00:00'
    # target_time = str(dataset['time'][0])
    # target_time = str(target_times[0])
    # guard_cell=0
    '''

    # 데이터 읽기
    mesh_lon=Model_DS['mesh_lon']
    mesh_lat=Model_DS['mesh_lat']
    uo = Model_DS['uo']
    vo = Model_DS['vo']
    time=Model_DS['time']

    # 시간변수 계산
    dt_target_time = dt.datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S')

    # 타겟시각의 인덱스
    idx_target_time=int(np.where(dt_target_time==time)[0])

    # 타겟시각의 uo,vo
    if len(uo.shape)==4:
        uo=uo[idx_target_time,0,:,:]
        vo=vo[idx_target_time,0,:,:]
    elif len(uo.shape)==3:
        uo = uo[idx_target_time, :, :]
        vo = vo[idx_target_time, :, :]

    # 거리계산
    dist_x=target_coord[0]-mesh_lon
    dist_y=target_coord[1]-mesh_lat
    dist_xy=np.abs(dist_x**2+dist_y**2)

    # 최근접적 X, Y 인덱스
    idx_y, idx_x=np.where(np.min(dist_xy)==dist_xy)
    idx_y, idx_x=int(idx_y), int(idx_x)

    # 데이터 추출
    ext_mesh_lon=mesh_lon[idx_y-guard_cell:idx_y+guard_cell+1,idx_x-guard_cell:idx_x+guard_cell+1]
    ext_mesh_lat=mesh_lat[idx_y-guard_cell:idx_y+guard_cell+1,idx_x-guard_cell:idx_x+guard_cell+1]

    ext_uo=uo[idx_y-guard_cell:idx_y+guard_cell+1,idx_x-guard_cell:idx_x+guard_cell+1]
    ext_vo=vo[idx_y-guard_cell:idx_y+guard_cell+1,idx_x-guard_cell:idx_x+guard_cell+1]

    try:
        ext_dataset = {'time':time[idx_target_time],'mesh_lon':ext_mesh_lon,'mesh_lat':ext_mesh_lat, 'uo':ext_uo, 'vo':ext_vo}
    except:
        ext_dataset = {'time': time, 'mesh_lon': ext_mesh_lon, 'mesh_lat': ext_mesh_lat, 'uo': ext_uo, 'vo': ext_vo}

    return ext_dataset


def interp(ext_dataset,SF=2):
    time = ext_dataset['time']
    mesh_lon = ext_dataset['mesh_lon']
    mesh_lat = ext_dataset['mesh_lat']
    uo = ext_dataset['uo']
    vo = ext_dataset['vo']

    lin_lon=cv2.resize(mesh_lon, None, fx=SF, fy=SF, interpolation=cv2.INTER_LINEAR)
    lin_lat=cv2.resize(mesh_lat, None, fx=SF, fy=SF, interpolation=cv2.INTER_LINEAR)

    cub_uo=cv2.resize(uo, None, fx=SF, fy=SF, interpolation=cv2.INTER_CUBIC)
    cub_vo=cv2.resize(vo, None, fx=SF, fy=SF, interpolation=cv2.INTER_CUBIC)

    intp_dataset={'time':time,'mesh_lon':lin_lon,'mesh_lat':lin_lat, 'uo':cub_uo, 'vo':cub_vo}

    return intp_dataset


# import matplotlib.pyplot as plt
# plt.figure()
# plt.subplot(1,2,1)
# plt.pcolor(ext_dataset['mesh_lon'],ext_dataset['mesh_lat'],ext_dataset['uo'])
# plt.scatter(target_coord[0], target_coord[1], facecolors='w', edgecolors='k')
# plt.subplot(1,2,2)
# plt.pcolor(intp_dataset['mesh_lon'],intp_dataset['mesh_lat'],intp_dataset['uo'])
# plt.scatter(target_coord[0], target_coord[1], facecolors='w', edgecolors='k')


def read_Buoy_month_txt(path_file):
    '''
    path_file='D:/04_Observation/Ulleung/data_2021_DT_DT_3_202108_KR.txt'
    '''
    df=pd.read_csv(path_file, sep='\t') # seperator: tab
    df = df.drop(columns='Unnamed: 14')
    df.columns=['Time', 'TideHt(cm)', 'SWTemp(℃)', 'Salinity(PSU)', 'SigWVHt(m)', 'SigWVP(sec)',
       'MaxWVHt(m)', 'MaxWVP(sec)', 'WindSpd(m/s)', 'WindDir(16points)', 'WindDir(deg)', 'AirTemp(℃)',
       'AirPress(hPa)', 'Visibility(m)']

    df.Time = pd.to_datetime(df.Time)
    df.Time = df.Time - dt.timedelta(hours=9) # UTC로 보정

    return df

def DM_Buoy(df):
    df['new_date'] = [d.date() for d in df['Time']]
    df['new_time'] = [d.time() for d in df['Time']]
    df['WindSpd(m/s)'].replace('-', np.nan, inplace=True)
    df['WindSpd(m/s)']=df['WindSpd(m/s)'].astype(float)
    df['WindDir(deg)'].replace('-', np.nan, inplace=True)
    df['WindDir(deg)'] = df['WindDir(deg)'].astype(float)*3.1415/180

    new_Dict=dict(df.groupby('new_date').mean())
    return new_Dict


def extract_model_point_dataset(target_times,target_coord,Model_DS,SF=5):
    '''
    target_times
    target_coord
    dataset=Model_DS
    SF=5
    '''
    DF=[]
    # ii=0;SF=5
    for ii in range(len(target_times)):
        ext_dataset=extract_Model_data(str(target_times[ii]),target_coord,Model_DS, guard_cell=10)
        intp_dataset=interp(ext_dataset, SF)
        ext_pt_ds=extract_Model_data(str(target_times[ii]),target_coord,intp_dataset, guard_cell=0)
        DF.append(ext_pt_ds)
    DF = pd.DataFrame(DF)

    for ii in range(len(DF)):
        DF.mesh_lon[ii] = DF.mesh_lon[ii][0][0]
        DF.mesh_lat[ii] = DF.mesh_lat[ii][0][0]
        DF.uo[ii] = DF.uo[ii][0][0]
        DF.vo[ii] = DF.vo[ii][0][0]
    DF_dict = dict(DF)
    DF = pd.DataFrame(DF_dict)
    return DF


def read_ObsCenter_month_txt(path_file):
    '''
    해양관측부이
    path_file='D:/04_Observation/ObsCenter/Ulleung_NE/Monthly/data_2021_TW_KG_KG_0101_2021_KR.txt'
    '''
    df=pd.read_csv(path_file, skiprows=[0,1,2], sep='\t') # seperator: tab
    df = df.drop(columns='Unnamed: 17')
    df.columns=['Time', 'SurfOCVel(cm/s)', 'SurfOCDir(16points)', 'SurfOCDir(deg)', 'SWTemp(℃)', 'Salinity(PSU)',
       'SigWVHt(m)', 'SigWVP(sec)', 'MaxWVHt(m)',
       'MaxWVP(sec)', 'WVDir(16points)', 'WVDir(deg)', 'WindSpd(m/s)',
       'WindDir(16points)', 'WindDir(deg)', 'AirTemp(℃)', 'AirPress(hPa)']

    df.Time = pd.to_datetime(df.Time)
    df.Time = df.Time - dt.timedelta(hours=9) # UTC로 보정

    return df


def read_ObsCenter_year_csv(path_file):
    '''
    해양관측부이
    path_file='E:/04_Observation/KHOA_MarineBuoy/NamhaeEast_2020.csv'
    '''
    df=pd.read_csv(path_file) # seperator: tab
    df.Time = pd.to_datetime(df.Time)
    df.Time = df.Time - dt.timedelta(hours=9) # UTC로 보정
    return df

def DM_ObsCenter_month_txt(df):
    import Lib.lib_Geo as Geo

    df['new_date'] = [d.date() for d in df['Time']]
    df['new_time'] = [d.time() for d in df['Time']]

    df['WindSpd(m/s)'].replace('-', np.nan, inplace=True)
    df['WindSpd(m/s)']=df['WindSpd(m/s)'].astype(float)
    df['WindDir(deg)'].replace('-', np.nan, inplace=True)
    df['WindDir(deg)'] = df['WindDir(deg)'].astype(float)*3.1415/180

    df['SurfOCVel(m/s)'].replace('-', np.nan, inplace=True)
    df['SurfOCVel(m/s)'] = df['SurfOCVel(cm/s)'].astype(float)
    df['SurfOCDir(deg)'].replace('-', np.nan, inplace=True)
    df['SurfOCDir(deg)'] = df['SurfOCDir(deg)'].astype(float) * 3.1415 / 180

    df['SurfOC_U(m/s)'],df['SurfOC_V(m/s)']=gyrospd2uv(df['SurfOCDir(deg)'],df['SurfOCVel(m/s)'])
    df['SurfOC_U(m/s)'],df['SurfOC_V(m/s)']=df['SurfOC_U(m/s)']/100,df['SurfOC_V(m/s)']/100

    new_Dict=df.groupby('new_date').mean()
    new_Dict['SurfOCVel(m/s)']=new_Dict['SurfOCVel(cm/s)']/100 # cm into m
    new_Dict['SurfOCDir(deg)']=np.remainder(new_Dict['SurfOCDir(deg)']*180/3.1415,360)


    return new_Dict



def DM_ObsCenter_year_csv(ObsCenter_DF):
    import Lib.lib_Geo as Geo
    ObsCenter_DF['new_date'] = [d.date() for d in ObsCenter_DF['Time']]
    ObsCenter_DF['new_time'] = [d.time() for d in ObsCenter_DF['Time']]

    ObsCenter_DF['WindVel(m/s)'].replace('-', np.nan, inplace=True)
    ObsCenter_DF['WindVel(m/s)']=ObsCenter_DF['WindVel(m/s)'].astype(float)
    ObsCenter_DF['WindDir(deg)'].replace('-', np.nan, inplace=True)

    ObsCenter_DF['SurfCurVel'].replace('-', np.nan, inplace=True)
    ObsCenter_DF['SurfCurVel(m/s)'] = ObsCenter_DF['SurfCurVel'].astype(float) / 100
    ObsCenter_DF['SurfCurDir(deg)'].replace('-', np.nan, inplace=True)

    ObsCenter_DF['SurfCur_U(m/s)'],ObsCenter_DF['SurfCur_V(m/s)']=Geo.gyrospd2uv(ObsCenter_DF['SurfCurDir(deg)'],ObsCenter_DF['SurfCurVel(m/s)'])
    ObsCenter_DF['WindVel_U(m/s)'],ObsCenter_DF['WindVel_V(m/s)']=Geo.gyrospd2uv(ObsCenter_DF['WindDir(deg)'],ObsCenter_DF['WindVel(m/s)'])

    new_Dict=ObsCenter_DF.groupby('new_date').mean()

    return new_Dict


def all_process(date_range, path_Model, path_Obs, station_name, datatype):
    '''
    date_range=['2020-01-01','2020-12-31']
    station_name = 'UlleungNE'
    path_Model = 'E:/01_Model/CMEMS/GLOBAL-ANALYSIS-FORECAST-PHY-001-024/UV/NamhaeDonghae/GLO-ANAL-FCST-PHY-001-024-UV_daily_2020.nc'
    # path_Model='E:/01_Model/CMEMS/WIND_GLO_WIND_L4_NRT_OBSERVATIONS_012_004/NamhaeDonghae/CERSAT-GLO-BLENDED_WIND_L4-V6-OBS_2020.nc'
    path_Obs = 'E:/04_Observation/KHOA_MarineBuoy/NamhaeEast_2020.csv'
    datatype='Wind' # 'CUR'
    '''
    import pandas as pd
    import Lib.Map as Map
    import Lib.lib_stat as stat
    ### DateRange
    if datatype=='CUR':
        target_times = pd.date_range(date_range[0] + ' 12:00:00', date_range[1] + ' 12:00:00', freq='D')
    elif datatype=='Wind':
        target_times = pd.date_range(date_range[0] + ' 00:00:00', date_range[1] + ' 00:00:00', freq='D')
    target_coord = Map.station_psn()[station_name]

    ### Read Model
    Model_DS = read_Model_DS(path_Model, datatype=datatype)
    Model_DF = extract_model_point_dataset(target_times, target_coord, Model_DS, SF=1)

    ### Read Obs
    # ObsCenter_DF = read_ObsCenter_month_txt(path_Cur_Obs)
    # ObsCenter_DF = pd.DataFrame(DM_ObsCenter_month_txt(ObsCenter_DF))
    ObsCenter_DF = read_ObsCenter_year_csv(path_Obs)
    ObsCenter_DF = pd.DataFrame(DM_ObsCenter_year_csv(ObsCenter_DF))


    cond=np.array([str(dt.datetime.strptime(date_range[0], '%Y-%m-%d')-dt.timedelta(days=1))[:10]==str(date) for date in ObsCenter_DF.index])
    ObsCenter_DF = ObsCenter_DF[~cond]
    ###

    path_save_dir='E:/20_Product/Compare/'+datatype+'/'+station_name
    os.makedirs(path_save_dir,exist_ok=True)
    path_save_csv = path_save_dir+'/'+datatype+'_'+station_name+'_DF_' + date_range[0][:4] + '.csv'
    path_save_png1 = path_save_dir+'/'+datatype+'_'+station_name+'_Scat_' + date_range[0][:4] + '.png'
    path_save_png2 = path_save_dir+'/'+datatype+'_'+station_name+'_Tren_' + date_range[0][:4] + '.png'
    path_save_png3 = path_save_dir+'/'+datatype+'_'+station_name+'_Hist_' + date_range[0][:4] + '.png'

    DF = pd.DataFrame([])
    DF['Time'] = np.array(Model_DF['time'])

    ObsCenter_vars={'CUR':'SurfCur','Wind':'WindVel'}

    DF[datatype+'_U_Obs'] = np.array(ObsCenter_DF[ObsCenter_vars[datatype]+'_U(m/s)'])
    DF[datatype+'_V_Obs'] = np.array(ObsCenter_DF[ObsCenter_vars[datatype]+'_V(m/s)'])
    DF[datatype+'_U_Model'] = np.array(Model_DF['uo'])
    DF[datatype+'_V_Model'] = np.array(Model_DF['vo'])

    DF.to_csv(path_save_csv, index=False)


    try:
        U_MSE = stat.rmse(np.array(DF[datatype+'_U_Model'].astype('f')),np.array(DF[datatype+'_U_Obs'].astype('f')))
    except:
        U_MSE = np.nan

    try:
        V_MSE = stat.rmse(DF[datatype+'_V_Model'].astype('f'),DF[datatype+'_V_Obs'].astype('f'))
    except:
        V_MSE = np.nan

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(nrows=1,  # row 몇 개
                           ncols=2,  # col 몇 개
                           height_ratios=[1],
                           width_ratios=[1, 1])

    limsVal={'CUR':(-1,1),'Wind':(-20,20)}
    PsnText={'CUR':(-0.75, 0.75),'Wind':(-15,15)}
    ax0 = plt.subplot(gs[0])
    ax0.scatter(DF[datatype+'_U_Model'], DF[datatype+'_U_Obs'], s=1, facecolors=None, edgecolors='tab:blue')
    plt.xlim(limsVal[datatype][0],limsVal[datatype][1])
    plt.ylim(limsVal[datatype][0],limsVal[datatype][1])
    plt.xlabel(datatype+'_U_Model[m/s]')
    plt.ylabel(datatype+'_U_Obs[m/s]')
    plt.title(datatype+'_U Vector')
    plt.grid()
    plt.text(PsnText[datatype][0],PsnText[datatype][1], 'RMSE: %.3f' % (U_MSE))
    ax1 = plt.subplot(gs[1])
    ax1.scatter(DF[datatype+'_V_Model'], DF[datatype+'_V_Obs'], s=1, facecolors=None, edgecolors='tab:blue')
    plt.xlim(limsVal[datatype][0],limsVal[datatype][1])
    plt.ylim(limsVal[datatype][0],limsVal[datatype][1])
    plt.xlabel(datatype+'_V_Model[m/s]')
    plt.ylabel(datatype+'_V_Obs[m/s]')
    plt.title(datatype+'_V Vector')
    plt.grid()
    plt.text(PsnText[datatype][0],PsnText[datatype][1], 'RMSE: %.3f' % (V_MSE))
    plt.savefig(path_save_png1)
    plt.close()

    plt.figure(2, figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(DF['Time'], DF[datatype+'_U_Model'], s=3, label=datatype+'_U_Model')
    plt.scatter(DF['Time'], DF[datatype+'_U_Obs'], s=3,label=datatype+'_U_Obs')
    plt.ylim(limsVal[datatype][0],limsVal[datatype][1])
    plt.xlabel('Time[UTC]')
    plt.ylabel(datatype+'_U[m/s]')
    plt.title(datatype+'_U Vector')
    plt.grid()
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.scatter(DF['Time'], DF[datatype+'_V_Model'], s=3, label=datatype+'_V_Model')
    plt.scatter(DF['Time'], DF[datatype+'_V_Obs'], s=3, label=datatype+'_V_Obs')
    plt.ylim(limsVal[datatype][0],limsVal[datatype][1])
    plt.xlabel('Time[UTC]')
    plt.ylabel(datatype+'_V[m/s]')
    plt.title(datatype+'_V Vector')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(path_save_png2)
    plt.close()

    plt.figure(3, figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.hist(DF[datatype+'_U_Model'] - DF[datatype+'_U_Obs'], bins=np.linspace(limsVal[datatype][0],limsVal[datatype][1],20))
    plt.ylabel('Freq.')
    plt.title('Difference of Current U (Model-Obs) [m/s]')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.hist(DF[datatype+'_V_Model'] - DF[datatype+'_V_Obs'], bins=np.linspace(limsVal[datatype][0],limsVal[datatype][1],20))
    plt.ylabel('Freq.')
    plt.title('Difference of Current V (Model-Obs) [m/s]')
    plt.grid()
    plt.savefig(path_save_png3)
    plt.close()

    return U_MSE, V_MSE


''' Test Code '''

DIR_MSES=[]
Vel_MSES=[]

import Lib.Map as Map
from Lib.lib_os import *

## Buoy위치(KHOA, KMA포함)
station_psn=Map.station_psn()
## Buoy이름(KHOA, KMA포함)
station_list=list(station_psn.keys())
## Buoy이름 각각
KMA_station=Map.station_list()['KMA']
KHOA_station=Map.station_list()['KHOA']
stations=KMA_station+KHOA_station

target_year = 2018  # int
for ii in range(len(stations)):
    # ii=0
    station_name = stations[ii]

    ## station 이름에 따른 관측자료경로지정
    if station_name in KMA_station:
        path_Obs_dir = 'E:/04_Observation/KMA_MarineBuoy'
    elif station_name in KHOA_station:
        path_Obs_dir = 'E:/04_Observation/KHOA_MarineBuoy'

    ## 연도에 따른 모델경로지정
    if target_year in range(2018,2020):
        path_Cur_Model_dir = 'E:/01_Model/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/UV/NamhaeDonghae'
        path_Wind_Model_dir = 'E:/01_Model/CMEMS/WIND_GLO_WIND_L4_REP_OBSERVATIONS_012_006/NamhaeDonghae'
    else:
        path_Cur_Model_dir = 'E:/01_Model/CMEMS/GLOBAL-ANALYSIS-FORECAST-PHY-001-024/UV/NamhaeDonghae'
        path_Wind_Model_dir = 'E:/01_Model/CMEMS/WIND_GLO_WIND_L4_NRT_OBSERVATIONS_012_004/NamhaeDonghae'

    path_Obs = recursive_file(path_Obs_dir, station_name + '*' + str(target_year) + '*')[0]
    path_Cur_Model=recursive_file(path_Cur_Model_dir,'*'+str(target_year)+'*')[0]
    path_Wind_Model = recursive_file(path_Wind_Model_dir, '*' + str(target_year) + '*')[0]

    date_range=[str(target_year)+'-01-01',str(target_year)+'-12-31']
    try:
        CU_MSE, CV_MSE=all_process(date_range, path_Cur_Model, path_Obs, station_name, datatype='CUR')
    except:
        None
    try:
        WU_MSE, WV_MSE=all_process(date_range, path_Wind_Model, path_Obs, station_name, datatype='Wind')
    except:
        None

