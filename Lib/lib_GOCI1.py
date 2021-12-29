import numpy as np
import h5py
import pandas as pd
import os, sys
import spectral
import datetime as dt
import matplotlib.pyplot as plt


def read_GOCI1_coordnates():
    import spectral
    '''
    미리 저장된 GOCI1 좌표를 읽음
    '''
    path_coord = 'E:/02_Satellite/GOCI1/Coordinates'
    mesh_lon=spectral.open_image(path_coord+'/lons.hdr')
    mesh_lon=mesh_lon[:,:].reshape(5685,5567)
    mesh_lat=spectral.open_image(path_coord+'/lats.hdr')
    mesh_lat=mesh_lat[:,:].reshape(5685,5567)
    return mesh_lon, mesh_lat

def read_GOCI1_flag():
    import spectral
    '''
    미리 저장된 GOCI1 Flag를 읽음
    '''
    path_flag = 'E:/02_Satellite/GOCI1/flag/Flag.hdr'
    flag=spectral.open_image(path_flag)
    flag=flag[:,:].reshape(5685,5567)
    return flag

def read_GOCI1_Chl(path_file):
    '''
    path_file = 'E:/02_Satellite/GOCI1/CHL/2018/03/COMS_GOCI_L2A_GA_20180316051640.CHL.he5'
    '''
    ## 파일 읽기
    import h5py
    import numpy as np
    # import matplotlib.pyplot as plt
    data = h5py.File(path_file, mode='r')
    data['HDFEOS/POINTS'].keys()
    CHL =np.array(data['HDFEOS/GRIDS/Image Data/Data Fields/CHL Image Pixel Values'])
    time = path_file.split('/')[-1].split('_')[4].split('.')[0]
    time = str(dt.datetime.strptime(time, '%Y%m%d%H%M%S'))[:19]
    CHL[CHL==-999]=np.nan
    unit = 'mg/m-3'
    dataset={'time':time, 'CHL':CHL, 'unit':unit}
    return dataset

def find_nearst_idx(mesh_lon, mesh_lat, target_lon, target_lat):
    '''
    meshgrid의 경위도에서 내가 원하는 경위도와 가장 가까운 위치의 index 반환
    target_lon = 131.8666  # Dokdo
    target_lat = 37.23972  # Dokdo
    '''

    # 유클리드 거리 계산
    dist=np.sqrt((mesh_lon-target_lon)**2+(mesh_lat-target_lat)**2)
    idx=np.where(dist==dist.min())
    return idx


def extract_for_specific_row(data, mesh_lon, target_row):
    '''
    데이터 및 경도 매트릭스로부터 특정 row에 대한 데이터 및 경도 자료를 추출
    target_row = 2419
    '''
    ext_data=data[target_row,:]
    ext_lon=mesh_lon[target_row,:]
    dataset={'ext_lon':ext_lon,'ext_data':ext_data}
    return dataset

def crop_outside_nan(data):
    '''
    GOCI1은 외곽에 NaN이 많아서 NaN을 지운 것만 얻고자 함
    '''

    r_idx = ~np.isnan(np.nanmean(data, axis=1))
    c_idx = ~np.isnan(np.nanmean(data, axis=0))

    ext_data = data[r_idx, :]
    ext_data = ext_data[:, c_idx]
    return ext_data

def read_GOCI1_LM():
    '''
    path_input_hdr='D:/02_Satellite/GOCI1/GOCI1_Landmask/GDPS_MASK.hdr'
    path_input_img='D:/02_Satellite/GOCI1/GOCI1_Landmask/GDPS_MASK.in'
    '''
    path_input_hdr = 'E:/02_Satellite/GOCI1/GOCI1_Landmask/GDPS_MASK.hdr'
    path_input_img = 'E:/02_Satellite/GOCI1/GOCI1_Landmask/GDPS_MASK.in'
    import numpy as np
    from spectral.io import envi
    import matplotlib.pyplot as plt
    img = envi.open(path_input_hdr, path_input_img)
    img = img[:,:,0]
    img = img.reshape(img.shape[0],img.shape[1]) # 2: Land, 0: Sea, 130: Coastline
    return img
