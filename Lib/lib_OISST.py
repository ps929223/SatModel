import netCDF4 as nc
import numpy as np
import datetime as dt



def read_SST_NC(path_file):
    '''
    path_file='D:/01_Model/OISST_onlyAVHRR/oisst-avhrr-v02r01.20200801.nc'
    OISST는 독도영역에 대해 CMEMS 보다 해상도가  낮음
    '''

    ## 데이터 읽기
    data = nc.Dataset(path_file, mode = 'r')

    ## 시간변수 계산
    time = np.array(data.variables['time'][:])
    time_since = data['time'].units[-19:]
    time_since = dt.datetime.strptime(time_since, '%Y-%m-%d %H:%M:%S')
    time = time_since + dt.timedelta(days=int(time[0]))

    ## 경위도
    lon = data.variables['lon'][:]
    lat = data.variables['lat'][:]

    ## SST
    SST = np.array(data.variables['sst'][0, 0, ::])
    mesh_lon, mesh_lat = np.meshgrid(lon,lat)

    dataset={'time': str(time), 'mesh_lon':mesh_lon, 'mesh_lat': mesh_lat, 'SST': SST}
    return dataset