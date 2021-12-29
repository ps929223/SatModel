import pandas as pd

from Lib.lib_CMEMS import *
from Lib.lib_os import *
import cv2
import matplotlib.pyplot as plt

def get_UV_lonlat_for_target_coord_date(ds, target_date, coord):
    '''
    ds = ds_Cur
    get_UV_lonlat(ds, target_date, coord)
    target_date='2020-08-01'
    coord = [131.4, 132.6, 36.9, 37.6]
    '''
    date_idx=[target_date in name for name in ds['time'].astype(str)]
    uo=np.nanmean(ds['uo'][date_idx], axis=0)
    uo=uo.reshape(uo.shape[-2],uo.shape[-1])
    vo=np.nanmean(ds['vo'][date_idx], axis=0)
    vo = vo.reshape(vo.shape[-2], vo.shape[-1])

    mesh_lon, mesh_lat =ds['mesh_lon'], ds['mesh_lat']
    lons, lats = np.unique(mesh_lon) , np.unique(mesh_lat)
    idx_lon = np.where((coord[0] < lons) & (lons < coord[1]))[0]
    idx_lat = np.where((coord[2] < lats) & (lats < coord[3]))[0]
    mesh_lon1 = mesh_lon[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    mesh_lat1 = mesh_lat[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    uo1 = uo[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    vo1 = vo[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]

    ds={'time':target_date, 'mesh_lon':mesh_lon1, 'mesh_lat':mesh_lat1,
        'uo':uo1, 'vo':vo1}
    return ds


def get_UV_lonlat_for_target_coord(ds, coord):
    '''
    coord = [131.4, 132.6, 36.9, 37.6]
    '''

    uo, vo =ds['uo'], ds['vo']
    mesh_lon, mesh_lat =ds['mesh_lon'], ds['mesh_lat']

    lons, lats = np.unique(mesh_lon) , np.unique(mesh_lat)
    idx_lon = np.where((coord[0] < lons) & (lons < coord[1]))[0]
    idx_lat = np.where((coord[2] < lats) & (lats < coord[3]))[0]

    mesh_lon1 = mesh_lon[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    mesh_lat1 = mesh_lat[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    uo1 = uo[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]-1:idx_lon[-1]+1]
    vo1 = vo[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]-1:idx_lon[-1]+1]

    ds={'time':ds['time'], 'mesh_lon':mesh_lon1, 'mesh_lat':mesh_lat1,
        'uo':uo1, 'vo':vo1}
    return ds


def get_SST_lonlat_for_target_coord_date(ds, target_date, coord):
    '''
    ds = ds_SST
    get_SST_lonlat(ds, target_date, coord)
    target_date='2020-08-01'
    coord = [131.4, 132.6, 36.9, 37.6]
    '''
    date_idx=[target_date in name for name in ds['time'].astype(str)]
    sst=np.nanmean(ds['SST'][date_idx], axis=0)

    mesh_lon, mesh_lat =ds['mesh_lon'], ds['mesh_lat']
    lons, lats = np.unique(mesh_lon) , np.unique(mesh_lat)
    idx_lon = np.where((coord[0] < lons) & (lons < coord[1]))[0]
    idx_lat = np.where((coord[2] < lats) & (lats < coord[3]))[0]
    mesh_lon1 = mesh_lon[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    mesh_lat1 = mesh_lat[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    sst1 = sst[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]

    ds={'time':target_date, 'mesh_lon':mesh_lon1, 'mesh_lat':mesh_lat1,
        'SST':sst1}
    return ds



def get_SST_lonlat_for_target_coord(ds, coord):
    '''
    coord = [131.4, 132.6, 36.9, 37.6]
    '''

    sst =ds['SST']
    mesh_lon, mesh_lat =ds['mesh_lon'], ds['mesh_lat']

    lons, lats = np.unique(mesh_lon) , np.unique(mesh_lat)
    idx_lon = np.where((coord[0] < lons) & (lons < coord[1]))[0]
    idx_lat = np.where((coord[2] < lats) & (lats < coord[3]))[0]

    mesh_lon1 = mesh_lon[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    mesh_lat1 = mesh_lat[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    sst1 = sst[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]-1:idx_lon[-1]+1]

    ds={'time':ds['time'], 'mesh_lon':mesh_lon1, 'mesh_lat':mesh_lat1,
        'SST':sst1}
    return ds



def get_SLA_lonlat_for_target_coord_date(ds, target_date, coord):
    '''
    ds = ds_SLA
    get_SLA_lonlat(ds, target_date, coord)
    target_date='2020-08-01'
    coord = [131.4, 132.6, 36.9, 37.6]
    '''
    date_idx=[target_date in name for name in ds['time'].astype(str)]
    sla=np.nanmean(ds['SLA'][date_idx], axis=0)

    mesh_lon, mesh_lat =ds['mesh_lon'], ds['mesh_lat']
    lons, lats = np.unique(mesh_lon) , np.unique(mesh_lat)
    idx_lon = np.where((coord[0] < lons) & (lons < coord[1]))[0]
    idx_lat = np.where((coord[2] < lats) & (lats < coord[3]))[0]
    mesh_lon1 = mesh_lon[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    mesh_lat1 = mesh_lat[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    sla1 = sla[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]

    ds={'time':target_date, 'mesh_lon':mesh_lon1, 'mesh_lat':mesh_lat1,
        'SLA':sla1}
    return ds


def get_SLA_lonlat_for_target_coord(ds, coord):
    '''
    coord = [131.4, 132.6, 36.9, 37.6]
    '''

    sla =ds['SLA']
    mesh_lon, mesh_lat =ds['mesh_lon'], ds['mesh_lat']

    lons, lats = np.unique(mesh_lon) , np.unique(mesh_lat)
    idx_lon = np.where((coord[0] < lons) & (lons < coord[1]))[0]
    idx_lat = np.where((coord[2] < lats) & (lats < coord[3]))[0]

    mesh_lon1 = mesh_lon[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    mesh_lat1 = mesh_lat[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]:idx_lon[-1]+1]
    sla1 = sla[idx_lat[0]:idx_lat[-1]+1, idx_lon[0]-1:idx_lon[-1]+1]

    ds={'time':ds['time'], 'mesh_lon':mesh_lon1, 'mesh_lat':mesh_lat1,
        'SLA':sla1}
    return ds


def interp_UV(ext_dataset,dims=(100,100)):
    # ext_dataset=intp_Wind
    time = ext_dataset['time']
    mesh_lon = ext_dataset['mesh_lon']
    mesh_lat = ext_dataset['mesh_lat']
    uo = ext_dataset['uo']
    vo = ext_dataset['vo']

    lons, lats = np.unique(mesh_lon), np.unique(mesh_lat)
    lons = np.linspace(lons[0],lons[-1],dims[1])
    lats = np.linspace(lats[0],lats[-1],dims[0])

    mesh_lon, mesh_lat = np.meshgrid(lons, lats)

    cub_uo=cv2.resize(uo, dims, interpolation=cv2.INTER_CUBIC)
    cub_vo=cv2.resize(vo, dims, interpolation=cv2.INTER_CUBIC)

    intp_dataset={'time':time,'mesh_lon':mesh_lon,'mesh_lat':mesh_lat, 'uo':cub_uo, 'vo':cub_vo}

    return intp_dataset


def interp_SST(ext_dataset,dims=(100,100)):
    # ext_dataset=intp_SST
    time = ext_dataset['time']
    mesh_lon = ext_dataset['mesh_lon']
    mesh_lat = ext_dataset['mesh_lat']
    SST = ext_dataset['SST']

    lons, lats = np.unique(mesh_lon), np.unique(mesh_lat)
    lons = np.linspace(lons[0],lons[-1],dims[1])
    lats = np.linspace(lats[0],lats[-1],dims[0])

    mesh_lon, mesh_lat = np.meshgrid(lons, lats)

    cub_sst=cv2.resize(SST, dims, interpolation=cv2.INTER_CUBIC)

    intp_dataset={'time':time,'mesh_lon':mesh_lon,'mesh_lat':mesh_lat, 'SST':cub_sst}

    return intp_dataset


def interp_SLA(ext_dataset,dims=(100,100)):
    # ext_dataset=intp_SST
    time = ext_dataset['time']
    mesh_lon = ext_dataset['mesh_lon']
    mesh_lat = ext_dataset['mesh_lat']
    SLA = ext_dataset['SLA']

    lons, lats = np.unique(mesh_lon), np.unique(mesh_lat)
    lons = np.linspace(lons[0],lons[-1],dims[1])
    lats = np.linspace(lats[0],lats[-1],dims[0])

    mesh_lon, mesh_lat = np.meshgrid(lons, lats)

    cub_sla=cv2.resize(SLA, dims, interpolation=cv2.INTER_CUBIC)

    intp_dataset={'time':time,'mesh_lon':mesh_lon,'mesh_lat':mesh_lat, 'SLA':cub_sla}

    return intp_dataset

def ds_to_flattenCSV(ds,fname):
    import pandas as pd
    import os
    keys=list(ds.keys())
    DF=pd.DataFrame([])
    for ii in range(1, len(keys)):
        DF[keys[ii]]=ds[keys[ii]].flatten()
    path_dir='E:/20_Product/CHL_Krig/env'
    os.makedirs(path_dir,exist_ok=True)
    DF.to_csv(path_dir+'/'+fname, index=False)


def create_KAGIS_env_dataset(date_str):
    '''
    한국지리정보학회(KAGIS) 발표자료용 데이터셋 생성
    Chl, SST, SLA, WIND, CUR Substract, Upsampling 자료를 생산 함
    '''

    ## 자료의 경로
    path_input_wind = 'E:/01_Model/CMEMS/WIND_GLO_WIND_L4_NRT_OBSERVATIONS_012_004/' \
                    'NamhaeDonghae/CERSAT-GLO-BLENDED_WIND_L4-V6-OBS_2020.nc'
    path_input_cur = 'E:/01_Model/CMEMS/GLOBAL-ANALYSIS-FORECAST-PHY-001-024/UV/' \
                     'NamhaeDonghae/GLO-ANAL-FCST-PHY-001-024-UV_daily_2020.nc'
    path_input_sst= 'E:/01_Model/CMEMS/SST_GLO_SST_L4_REP_OBSERVATIONS_010_024/' \
                          'Donghae/C3S-GLO-SST-L4-REP-OBS-SST_2020.nc'
    path_input_sla= 'E:/01_Model/CMEMS/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/' \
                    'Donghae/dataset-duacs-nrt-global-merged-allsat-phy-l4_2020.nc'

    ## Substract를 위한 대상영역 설정
    coord = [131.0, 133, 36.7, 37.8]

    ds_Wind = read_Wind_NC(path_input_wind)
    ds_Cur = read_Cur_NC(path_input_cur)
    ds_SST = read_SST_NC(path_input_sst, grid_x=100, grid_y=100)
    ds_SLA = read_SLA_NC(path_input_sla, grid_x=100, grid_y=100)

    ## 원자료 출력
    # plt.pcolor(ds_SST['mesh_lon'], ds_SST['mesh_lat'], ds_SST['SST'][0,:,:]-273.5, vmin=-2, vmax=20, cmap='jet')
    # plt.grid()
    # plt.colorbar()

    ## Subset
    # date_str='2020-06-01'
    ds_Wind=get_UV_lonlat_for_target_coord_date(ds_Wind, date_str, coord)
    ds_Cur=get_UV_lonlat_for_target_coord_date(ds_Cur, date_str, coord)
    ds_SST=get_SST_lonlat_for_target_coord_date(ds_SST, date_str, coord)
    ds_SLA=get_SLA_lonlat_for_target_coord_date(ds_SLA, date_str, coord)

    ## Subset 출력
    # plt.pcolor(ds_SST['mesh_lon'], ds_SST['mesh_lat'], ds_SST['SST']-273.5, vmin=15, vmax=20, cmap='jet')
    # plt.grid()
    # plt.colorbar()

    intp_Wind=interp_UV(ds_Wind,(1000,1000))
    intp_Cur=interp_UV(ds_Cur,(1000,1000))
    intp_SST=interp_SST(ds_SST,(1000,1000))
    intp_SLA=interp_SLA(ds_SLA,(1000,1000))

    coord = [131.4, 132.6, 36.9, 37.6]

    intp_Wind=get_UV_lonlat_for_target_coord(intp_Wind,coord)
    intp_Cur=get_UV_lonlat_for_target_coord(intp_Cur,coord)
    intp_SST=get_SST_lonlat_for_target_coord(intp_SST,coord)
    intp_SLA=get_SLA_lonlat_for_target_coord(intp_SLA,coord)

    intp_Wind=interp_UV(intp_Wind,(100,100))
    intp_Cur=interp_UV(intp_Cur,(100,100))
    intp_SST=interp_SST(intp_SST,(100,100))
    intp_SLA=interp_SLA(intp_SLA,(100,100))

    ## 보간자료 출력
#    plt.pcolor(intp_SST['mesh_lon'], intp_SST['mesh_lat'], intp_SST['SST']-273.5, vmin=15, vmax=20, cmap='jet')
#    plt.grid()
#    plt.colorbar()

    ds_to_flattenCSV(intp_Wind,'Dokdo_'+date_str+'_Wind.csv')
    ds_to_flattenCSV(intp_Cur,'Dokdo_'+date_str+'_Cur.csv')
    ds_to_flattenCSV(intp_SST,'Dokdo_'+date_str+'_SST.csv')
    ds_to_flattenCSV(intp_SLA,'Dokdo_'+date_str+'_SLA.csv')


def create_KAGIS_ML_dataset(path_out_dir):
    import numpy as np
    import pandas as pd

    import os

    dir_ML_ds=path_out_dir+'/MLds'
    os.makedirs(dir_ML_ds, exist_ok=True)
    ORM_list=np.array(recursive_file(path_out_dir+'/CHL_Orig','Dokdo*_CHL_ORM.csv'))
    Mask_list=np.array(recursive_file(path_out_dir+'/CHL_Orig','Dokdo*_CHL_Mask.csv'))
    date_list=[Mask_list[ii].split('\\')[-1].split('_')[1] for ii in range(len(Mask_list))]

    CUR_list=np.array(recursive_file(path_out_dir+'/env','Dokdo*Cur.csv'))
    SLA_list=np.array(recursive_file(path_out_dir+'/env','Dokdo*SLA.csv'))
    SST_list=np.array(recursive_file(path_out_dir+'/env','Dokdo*SST.csv'))
    Wind_list=np.array(recursive_file(path_out_dir+'/env','Dokdo*Wind.csv'))
    Cloud=np.genfromtxt(path_out_dir+'/grid/Dokdo_cloud_2020-08-01.csv', delimiter=',')
    Cloud=Cloud==1

    # ii=0
    for ii in range(len(date_list)):
        ORM = pd.read_csv(ORM_list[ii])
        CUR = pd.read_csv(CUR_list[ii])
        SLA = pd.read_csv(SLA_list[ii])
        SST = pd.read_csv(SST_list[ii])
        Wind = pd.read_csv(Wind_list[ii])

        ## 해수온도는 화씨에서 섭씨로 변경 -273.15
        new_DF=pd.concat([SST['SST']-273.15,SLA['SLA'], Wind['uo'], Wind['vo'],
                          CUR['uo'], CUR['vo'], CUR['mesh_lon'], CUR['mesh_lat']], axis=1)
        new_DF.columns=['SST', 'SLA', 'Wind_U', 'Wind_V', 'CUR_U', 'CUR_V', 'mesh_lon', 'mesh_lat']


        ## CHL과 환경요소자료 자료 매칭
        new_DF2=pd.concat([ORM['chl-a'],new_DF], axis=1)


        ## CHL과 환경값의 상관성
        corr=new_DF2.corr(method='pearson')
        path_pearson_dir=path_out_dir+'/pearson'
        os.makedirs(path_pearson_dir, exist_ok=True)
        corr.to_csv(path_pearson_dir+'/pearson_'+date_list[ii]+'.csv')

        ## Train/Test
        Train=new_DF2[~Cloud.flatten()]
        Train.to_csv(dir_ML_ds+'/Train_'+date_list[ii]+'.csv', index=False)
        Test=new_DF2[Cloud.flatten()]
        Test.to_csv(dir_ML_ds+'/Test_'+date_list[ii]+'.csv', index=False)

        ## Train/Test Data 가시화
        # plt.figure(figsize=(12,4))
        # plt.subplot(1,2,1)
        # plt.scatter(Train.mesh_lon, Train.mesh_lat, c=Train.SST, s=8, vmin=15, vmax=20, cmap='jet')
        # plt.xlim(131.4, 132.6)
        # plt.ylim(36.9, 37.6)
        # plt.colorbar()
        # plt.grid()

        # plt.subplot(1, 2, 2)
        # plt.scatter(Test.mesh_lon, Test.mesh_lat, c=Test.SST, s=8, vmin=15, vmax=20, cmap='jet')
        # plt.xlim(131.4, 132.6)
        # plt.ylim(36.9, 37.6)
        # plt.colorbar()
        # plt.grid()

        ## Test Data 가시화
