'''
Coprenicus와 조사원 GOCI의 Chl 자료를 비교하는 코드
- 연 단위 비교
제작: 전호군
부서: 해양빅데이터센터
초안: 2021.09.29
'''


import netCDF4 as nc
import pandas as pd
import datetime as dt
import os, sys
sys.path.append('D:/programming/Dokdo')
import Lib.Map as Map
from Lib.Lib_Geo import *
from Lib.lib_GOCI1 import *
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

def read_CHL_Model_DS(path_file):
    '''
    path_file='D:/01_Model/CMEMS/GLOBAL_REANALYSIS_BIO_001_029/NamhaeDonghae\\GLO-REANAL-BIO-001-029_2019.nc'
    '''
    ds = nc.Dataset(path_file)
    # vars = list(ds.variables)

    # 시간변수 계산
    time = np.array(ds.variables['time'][:])
    time_since = ds.variables['time'].units[-19:]
    time_since=dt.datetime.strptime(time_since, '%Y-%m-%d %H:%M:%S')
    time=np.array([time_since+dt.timedelta(hours=int(ii)) for ii in time])

    lat = np.array(ds.variables['latitude'][:])
    lon = np.array(ds.variables['longitude'][:])
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)

    CHL = np.array(ds.variables['chl'][:])  # CHL

    dataset={'time':time, 'mesh_lon': mesh_lon, 'mesh_lat':mesh_lat, 'CHL':CHL}

    return dataset

def extract_CHL_Model_data(target_time,target_coord,dataset, guard_cell=10):
    '''
    # target_time = '2019-01-03 12:00:00'
    # target_time = str(dataset['time'][0])
    # target_time = str(target_times[ii])
    # target_coord = [131.552583, 38.007361] # 'Ulleung_NE'
    # dataset = ext_dataset
    # dataset = intp_dataset
    # guard_cell=0
    '''

    ## 데이터 읽기
    time = dataset['time']
    mesh_lon=dataset['mesh_lon']
    mesh_lat=dataset['mesh_lat']
    CHL = dataset['CHL']

    ## 시간변수 계산
    # dt_target_time=target_times[ii]
    # target_time = str(target_times[ii])
    dt_target_time = dt.datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S')

    # 거리계산
    dist_x = target_coord[0] - mesh_lon
    dist_y = target_coord[1] - mesh_lat
    dist_xy = np.abs(dist_x ** 2 + dist_y ** 2)

    # 최근접적 X, Y 인덱스
    idx_y, idx_x = np.where(np.min(dist_xy) == dist_xy)
    idx_y, idx_x = int(idx_y), int(idx_x)

    ## 타겟시각의 인덱스
    if not len(np.where(dt_target_time==time)[0])==0:
        idx_target_time=int(np.where(dt_target_time==time)[0])

        # 타겟시각의 CHL
        if len(CHL.shape) == 4:
            CHL = CHL[idx_target_time, 0, :, :]
    else:
        CHL = np.ones((CHL.shape[-2],CHL.shape[-1]))*np.nan

    ext_CHL = CHL[idx_y - guard_cell:idx_y + guard_cell + 1, idx_x - guard_cell:idx_x + guard_cell + 1]
    ext_mesh_lon = mesh_lon[idx_y - guard_cell:idx_y + guard_cell + 1, idx_x - guard_cell:idx_x + guard_cell + 1]
    ext_mesh_lat = mesh_lat[idx_y - guard_cell:idx_y + guard_cell + 1, idx_x - guard_cell:idx_x + guard_cell + 1]

    try:
        ext_dataset = {'time': dt_target_time, 'mesh_lon': ext_mesh_lon, 'mesh_lat': ext_mesh_lat, 'CHL': ext_CHL}
    except:
        ext_dataset = {'time': dt_target_time, 'mesh_lon':ext_mesh_lon, 'mesh_lat':ext_mesh_lat, 'CHL':ext_CHL}


    return ext_dataset


def interp(ext_dataset,SF=2):
    time = ext_dataset['time']
    mesh_lon = ext_dataset['mesh_lon']
    mesh_lat = ext_dataset['mesh_lat']
    CHL = ext_dataset['CHL']

    lin_lon=cv2.resize(mesh_lon, None, fx=SF, fy=SF, interpolation=cv2.INTER_LINEAR)
    lin_lat=cv2.resize(mesh_lat, None, fx=SF, fy=SF, interpolation=cv2.INTER_LINEAR)

    cub_CHL=cv2.resize(CHL, None, fx=SF, fy=SF, interpolation=cv2.INTER_CUBIC)

    intp_dataset={'time':time,'mesh_lon':lin_lon,'mesh_lat':lin_lat, 'CHL':cub_CHL}

    return intp_dataset


def extract_model_point_dataset(target_times, target_coord,dataset,SF=5):
    '''
    target_coord=Map.station_psn()['Ulleung_NE']
    dataset = CHL_Model_DS
    dataset = dataset
    '''

    DF=[]
    # ii=3;SF=5
    for ii in range(len(target_times)):
        ext_dataset=extract_CHL_Model_data(str(target_times[ii]),target_coord,dataset, guard_cell=10)
        intp_dataset=interp(ext_dataset, SF)
        ext_pt_ds=extract_CHL_Model_data(str(target_times[ii]),target_coord,intp_dataset, guard_cell=0)
        DF.append(ext_pt_ds)
    DF = pd.DataFrame(DF)

    for ii in range(len(DF)):
        DF.CHL[ii] = DF.CHL[ii][0][0]
    DF_dict = dict(DF)
    DF = pd.DataFrame(DF_dict)
    return DF


def read_GOCI1_in_date_range(target_year, target_coord, path_CHL_GOCI1_dir):
    '''
    date_range = ['2018-01-01', '2018-12-31']
    target_coord=Map.station_psn()['Ulleung_NE']
    path_CHL_GOCI1_dir='E:/CHL'
    '''

    ## 경위도
    mesh_lon, mesh_lat = read_GOCI1_coordnates()

    ## 최근접위치
    idx = find_nearst_idx(mesh_lon,mesh_lat,target_coord[0], target_coord[1])

    # 파일목록
    list_file=recursive_file(path_CHL_GOCI1_dir, '*GOCI*GA_'+str(target_year)+'*.he5')

    time=[];CHL=[]
    for ii in range(len(list_file)):
        print(list_file[ii])
        time.append(read_GOCI1_Chl(list_file[ii])['time'])
        CHL.append(float(read_GOCI1_Chl(list_file[ii])['CHL'][idx[0],idx[1]]))

    ext_dataset={'Time':pd.to_datetime(time), 'CHL':CHL}

    return ext_dataset



def DM_GOCI(ext_dataset):
    ## 빈공간 생산
    dates = pd.DataFrame([])
    dates['Time'] = pd.date_range(date_range[0],date_range[1], freq='1H').astype(str)
    dates['Time'] = pd.to_datetime(dates['Time'])
    dates['new_date'] = [d.date() for d in dates['Time']]
    dates['new_time'] = [d.time() for d in dates['Time']]

    ## ext_dataset 변환
    ext_dataset=pd.DataFrame(ext_dataset)
    ext_dataset['new_date'] = [d.date() for d in ext_dataset['Time']]
    ext_dataset['new_time'] = [d.time() for d in ext_dataset['Time']]

    ##
    dd=pd.concat([dates, ext_dataset], axis=0)

    new_dataset=dd.groupby('new_date').mean()

    return new_dataset


def draw(DF, path_save_dir, target_year):

    try:
        CHL_MSE = rmse(DF['CHL_Model'].astype(float),DF['CHL_GOCI'].astype(float))
    except:
        CHL_MSE = np.nan

    path_save_png1 = path_save_dir + '/CHL_ULGNE_Scatt_' + str(target_year) + '.png'
    path_save_png2 = path_save_dir + '/CHL_ULGNE_Trend_' + str(target_year) + '.png'
    path_save_png3 = path_save_dir + '/CHL_ULGNE_Histo_' + str(target_year) + '.png'

    import matplotlib.pyplot as plt

    plt.figure(1, figsize=(7,6))
    plt.scatter(DF['CHL_Model'], DF['CHL_GOCI'], s=1, facecolors=None, edgecolors='tab:blue')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel('CHL_Model')
    plt.ylabel('CHL_GOCI')
    plt.title('CHL[mg m^-3]')
    plt.grid()
    plt.text(0.15, 1.85, 'RMSE: %.3f' % (CHL_MSE))
    plt.savefig(path_save_png1)
    plt.clf()


    plt.figure(2, figsize=(7,6))
    plt.plot(DF['Time'], DF['CHL_Model'], label='CHL_Model')
    plt.scatter(DF['Time'], DF['CHL_GOCI'], label='CHL_GOCI', c='tab:orange', s=3)
    plt.ylim(0,2)
    plt.xlabel('Time[UTC]')
    plt.ylabel('CHL[mg m^-3]')
    plt.title('CHL[mg m^-3]')
    plt.grid()
    plt.legend(loc='upper center')
    plt.savefig(path_save_png2)
    plt.clf()

    plt.figure(3, figsize=(7, 6))
    plt.hist(DF['CHL_Model'] - DF['CHL_GOCI'], bins=np.linspace(-1.5,1.5,20))
    plt.ylabel('Freq.')
    plt.title('Difference of CHL (Model-Obs) [mg m^3]')
    plt.grid()
    plt.savefig(path_save_png3)
    plt.clf()

    return CHL_MSE


def all_process(date_range,path_CHL_Model, path_CHL_GOCI1_dir):
    '''
    date_range = ['2019-01-01', '2019-12-31']
    path_CHL_Model = 'D:/01_Model/CMEMS/GLOBAL_REANALYSIS_BIO_001_029/NamhaeDonghae\\GLO-REANAL-BIO-001-029_2019.nc'
    path_CHL_GOCI1_dir = 'E:/CHL'
    '''
    ### DateRange
    target_times = pd.date_range(date_range[0] + ' 12:00:00', date_range[1] + ' 12:00:00', freq='D')
    target_coord = Map.station_psn()['Ulleung_NE']

    ### Read Model
    dataset = read_CHL_Model_DS(path_CHL_Model)
    CHL_Model_DF = extract_model_point_dataset(target_times, target_coord, dataset, SF=5)

    ### Read GOCI
    CHL_GOCI_DS = read_GOCI1_in_date_range(target_year, target_coord, path_CHL_GOCI1_dir)
    CHL_GOCI_DF = pd.DataFrame(DM_GOCI(CHL_GOCI_DS))
    cond=np.array([str(dt.datetime.strptime(date_range[0], '%Y-%m-%d')-dt.timedelta(days=1))[:10]==str(date) for date in CHL_GOCI_DF.index])
    CHL_GOCI_DF = CHL_GOCI_DF[~cond]
    ###



    path_save_dir='D:/20_Product/Compare/CHL/Ulleung_NE/Yearly'
    os.makedirs(path_save_dir,exist_ok=True)

    DF = pd.DataFrame([])
    DF['Time'] = np.array(CHL_Model_DF['time'])
    DF['CHL_GOCI'] = np.array(CHL_GOCI_DF['CHL'])
    DF['CHL_Model'] = np.array(CHL_Model_DF['CHL'])
    path_save_csv = path_save_dir+'/CHL_ULGNE_' + date_range[0][:7] + '.csv'
    DF.to_csv(path_save_csv, index=False)

    # DF = pd.read_csv(path_save_dir+'/CHL_ULGNE_2020-01.csv')
    # DF['Time']=pd.to_datetime(DF['Time'])
    draw(DF, path_save_dir, target_year=2020)



''' Test Code '''

target_year=2020 # int
path_CHL_GOCI1_dir = 'E:/CHL'

if target_year in range(2018,2020):
    path_CHL_Model_dir = 'D:/01_Model/CMEMS/GLOBAL_REANALYSIS_BIO_001_029/NamhaeDonghae'
else:
    path_CHL_Model_dir = 'D:/01_Model/CMEMS/GLOBAL_ANALYSIS_FORECAST_BIO_001_028/CHL/NamhaeDonghae'

path_CHL_Model=recursive_file(path_CHL_Model_dir,'*'+str(target_year)+'*')[0]


date_range=[str(target_year)+'-01-01',str(target_year)+'-12-31']
CHL_MSE=all_process(date_range, path_CHL_Model, path_CHL_GOCI1_dir)