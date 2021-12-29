import netCDF4 as nc
import numpy as np
import os, sys
os.environ['GDAL_CONFIG']=r'C:/Program Files/GDAL'

def run_script(path_dir, script):
    '''
    path_dir='D:/02_Satellite/GOCI2/CHL/2021-09-08'
    file_name='GK2B_GOCI2_L2_20210908_001530_LA_S004_Chl.nc'
    path_file='D:/02_Satellite/GOCI2/CHL/2021-09-08/GK2B_GOCI2_L2_20210908_001530_LA_S004_Chl.nc'
    '''
    import os
    os.chdir(path_dir)
    with open('script.bat','w') as f:
        f.write('\n'.join(script))
    os.system('script.bat')

def convert_NC_into_GTIFF(path_dir, path_file):
    script=['gdal_translate -ot Float32 -of GTiff -projwin_srs EPSG:4326 NETCDF:"'+
            path_file+'":geophysical_data/Chl '+
            path_file[:-3]+'.tif']
    run_script(path_dir, script)


def read_GOCI2_Chl(path_file):
    '''
    path_dir = 'D:/02_Satellite/GOCI2/CHL/2021-09-08'
    file_name = 'GK2B_GOCI2_L2_20210908_001530_LA_S004_Chl.nc'
    list_file = os.listdir(path_dir)
    path_file = path_dir + '/' + file_name
    '''
    ds = nc.Dataset(path_file)
    data=np.array(ds['geophysical_data']['Chl'][:])
    obs_time=ds.observation_end_time
    unit=ds['geophysical_data']['Chl'].units
    data[data==-999]=np.nan
    flag=ds['geophysical_data']['flag']
    mesh_lat=np.array(ds['navigation_data']['latitude'][:])
    mesh_lon=np.array(ds['navigation_data']['longitude'][:])
    coord=[ds.geospatial_lon_min, ds.geospatial_lon_max, ds.geospatial_lat_min, ds.geospatial_lat_max]
    dataset={'mesh_lon':mesh_lon,'mesh_lat':mesh_lat,'data':data,'flag':flag, 'coord':coord, 'unit':unit, 'obs_time':obs_time}
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
    target_row = 592
    '''
    ext_data=data[target_row,:]
    ext_lon=mesh_lon[target_row,:]
    dataset={'ext_lon':ext_lon,'ext_data':ext_data}
    return dataset


def read_GOCI2_LM():
    path_input_hdr = 'D:/02_Satellite/GOCI2/GOCI2_LandMask/GOCI2_Land_LA_S0004.hdr'
    path_input_img = 'D:/02_Satellite/GOCI2/GOCI2_LandMask/GOCI2_Land_LA_S0004.img'
    import numpy as np
    from spectral.io import envi
    import matplotlib.pyplot as plt
    img = envi.open(path_input_hdr, path_input_img)
    img = img[:,:,0]
    img = img.reshape(img.shape[0],img.shape[1]) # 2: Land, 0: Sea, 1: Coastline
    np.unique(img)
    return img