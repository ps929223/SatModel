import netCDF4 as nc
import numpy as np
import datetime as dt

def resize2DArray(array, grid_y, grid_x):
    try:
        import cv2
        array = cv2.resize(array, dsize=(grid_x, grid_y))
    except:
        import PIL.Image as Image
        img = Image.fromarray(array)
        array=np.array(img.resize((grid_x, grid_y), Image.BICUBIC))
    return array

def resize3DArray(array, grid_z, grid_y, grid_x):
    import cv2
    newArray=np.zeros((grid_z,grid_y,grid_x))
    newArray[:]=np.nan
    array1=np.array(array)
    array1[array>10e5]=np.nan
    for ii in range(grid_z):
        newArray[ii,:,:]=cv2.resize(array1[0,:,:], dsize=(grid_x, grid_y))
    return newArray

def read_CHL4km_NC(path_file, grid_y, grid_x):
    '''
    path_file='D:/01_Model/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP/2020/07/20200701_d-ACRI-L4-CHL-MULTI_4KM-GLO-DT.nc'
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

    ## Resampling
    lon = np.linspace(lon[0], lon[-1], grid_x)
    lat = np.linspace(lat[0], lat[-1], grid_y)

    ## CHL
    CHL = data.variables['CHL'][0, ::]
    CHL = resize2DArray(CHL, grid_y=grid_y, grid_x=grid_x)
    mesh_lon, mesh_lat = np.meshgrid(lon,lat)

    dataset={'time': time, 'mesh_lon':mesh_lon, 'mesh_lat': mesh_lat, 'CHL': CHL}
    return dataset


def read_BIO_NC(path_file, grid_y, grid_x):
    '''
    path_file='E:/01_Model/CMEMS/GLOBAL_ANALYSIS_FORECAST_BIO_001_028/CHL/Donghae/GLO-ANAL-FCST-BIO-001-028_2020.nc'
    grid_y=195
    grid_x=201
    '''
    import netCDF4 as nc
    import numpy as np
    import datetime as dt

    ## 데이터 읽기
    data = nc.Dataset(path_file, mode = 'r')

    ## 시간변수 계산
    time = np.array(data.variables['time'][:])
    time_since = data['time'].units[-19:]
    time_since = dt.datetime.strptime(time_since, '%Y-%m-%d %H:%M:%S')
    time = np.array([time_since + dt.timedelta(hours=int(time[ii])) for ii in range(len(time))])

    ## 경위도
    lon = data.variables['longitude'][:]
    lat = data.variables['latitude'][:]

    ## Resampling
    lon = np.linspace(lon[0], lon[-1], grid_x)
    lat = np.linspace(lat[0], lat[-1], grid_y)

    ## BIO
    chl = data.variables['chl'][:]
    grid_z,_,r,c= chl.shape
    chl = chl.reshape(grid_z,r,c)
    chl = resize3DArray(chl, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    fe = data.variables['fe'][:]
    fe = fe.reshape(grid_z,r,c)
    fe = resize3DArray(fe, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    no3 = data.variables['no3'][:]
    no3 = no3.reshape(grid_z,r,c)
    no3 = resize3DArray(no3, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    nppv = data.variables['nppv'][:]
    nppv = nppv.reshape(grid_z,r,c)
    nppv = resize3DArray(nppv, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    o2 = data.variables['o2'][:]
    o2 = o2.reshape(grid_z,r,c)
    o2 = resize3DArray(o2, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    ph = data.variables['ph'][:]
    ph = ph.reshape(grid_z,r,c)
    ph = resize3DArray(ph, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    phyc = data.variables['phyc'][:]
    phyc = phyc.reshape(grid_z,r,c)
    phyc = resize3DArray(phyc, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    po4 = data.variables['po4'][:]
    po4 = po4.reshape(grid_z,r,c)
    po4 = resize3DArray(po4, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    si = data.variables['si'][:]
    si = si.reshape(grid_z,r,c)
    si = resize3DArray(si, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    spco2 = data.variables['spco2'][:]
    spco2 = spco2.reshape(grid_z,r,c)
    spco2 = resize3DArray(spco2, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    mesh_lon, mesh_lat = np.meshgrid(lon,lat)
    mesh_lon.shape

    dataset={'time': time, 'mesh_lon':mesh_lon, 'mesh_lat': mesh_lat, 'chl': chl, 'fe':fe,
             'no3': no3, 'nppy': nppv, 'o2': o2, 'ph': ph, 'phyc': phyc, 'po4':po4, 'si':si,
             'spco2':spco2}
    return dataset




def read_Wind_NC(path_file):
    '''
    path_file='D:/01_Model/CMEMS/WIND_GLO_WIND_L4_NRT_OBSERVATIONS_012_004/NamhaeDonghae/CERSAT-GLO-BLENDED_WIND_L4-V6-OBS_2020.nc'
    '''
    ## NC 읽기
    ds = nc.Dataset(path_file)
    # vars = list(ds.variables)

    ## 시간변수 계산
    time = np.array(ds.variables['time'][:])
    time_since = dt.datetime.strptime('1900-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    time = np.array([time_since + dt.timedelta(hours=int(ii)) for ii in time])

    ## 경위도 meshgrid
    lat = np.array(ds.variables['lat'][:])
    lon = np.array(ds.variables['lon'][:])
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)

    ## UV vector
    uo = np.array(ds.variables['eastward_wind'][:])  # eastward_wind_velocity
    vo = np.array(ds.variables['northward_wind'][:])  # northward_wind_velocity

    ## 하나로 묶기
    dataset = {'time': time, 'mesh_lon': mesh_lon, 'mesh_lat': mesh_lat, 'uo': uo, 'vo': vo}

    return dataset



def read_Cur_NC(path_file):
    '''
    path_file=path_Cur_Model
    '''
    ds = nc.Dataset(path_file)
    # vars = list(ds.variables)

    # 시간변수 계산
    time = np.array(ds.variables['time'][:])
    time_since=dt.datetime.strptime('1950-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    time=np.array([time_since+dt.timedelta(hours=int(ii)) for ii in time])

    lat = np.array(ds.variables['latitude'][:])
    lon = np.array(ds.variables['longitude'][:])
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)

    uo = np.array(ds.variables['uo'][:])  # eastward_sea_water_velocity
    vo = np.array(ds.variables['vo'][:])  # northward_sea_water_velocity

    dataset={'time':time, 'mesh_lon': mesh_lon, 'mesh_lat':mesh_lat, 'uo':uo, 'vo':vo}

    return dataset



def read_SST_NC(path_file, grid_y, grid_x):
    '''
    path_file='E:/01_Model/CMEMS/SST_GLO_SST_L4_REP_OBSERVATIONS_010_024/Donghae/C3S-GLO-SST-L4-REP-OBS-SST_2020.nc'
    grid_x=.25
    grid_y=.25
    '''
    import netCDF4 as nc
    import datetime as dt

    ## 데이터 읽기
    data = nc.Dataset(path_file, mode = 'r')

    ## 시간변수 계산
    time = np.array(data.variables['time'][:])
    time_since = data['time'].units[-19:]
    time_since = dt.datetime.strptime(time_since, '%Y-%m-%d %H:%M:%S')
    time=np.array([time_since+dt.timedelta(seconds=int(ii)) for ii in time])

    ## 경위도
    lon = data.variables['lon'][:]
    lat = data.variables['lat'][:]

    ## Resampling
    lon = np.linspace(lon[0], lon[-1], grid_x)
    lat = np.linspace(lat[0], lat[-1], grid_y)

    ## SST
    SST = np.array(data.variables['analysed_sst'])
    SST[SST==-32768]=10e20
    SST=SST-273.15  ## Kelvin>>섭씨
    grid_z,r,c=SST.shape
    SST = resize3DArray(SST, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    mesh_lon, mesh_lat = np.meshgrid(lon,lat)

    dataset={'time': time, 'mesh_lon':mesh_lon, 'mesh_lat': mesh_lat, 'SST': SST}
    return dataset




def read_SLA_NC(path_file, grid_y, grid_x):
    '''
    path_file='E:/01_Model/CMEMS/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/Donghae/dataset-duacs-nrt-global-merged-allsat-phy-l4_2020.nc'
    '''

    ## 데이터 읽기
    data = nc.Dataset(path_file, mode = 'r')

    ## 시간변수 계산
    time = np.array(data.variables['time'][:])
    time_since = data['time'].units[-19:]
    time_since = dt.datetime.strptime(time_since, '%Y-%m-%d %H:%M:%S')
    time=np.array([time_since+dt.timedelta(days=int(ii)) for ii in time])

    ## 경위도
    lon = data.variables['longitude'][:]
    lat = data.variables['latitude'][:]

    ## Resampling
    lon = np.linspace(lon[0], lon[-1], grid_x)
    lat = np.linspace(lat[0], lat[-1], grid_y)

    ## SLA
    SLA = np.array(data.variables['sla'])
    grid_z,r,c=SLA.shape
    SLA = resize3DArray(SLA, grid_z=grid_z, grid_y=grid_y, grid_x=grid_x)

    mesh_lon, mesh_lat = np.meshgrid(lon,lat)

    dataset={'time': time, 'mesh_lon':mesh_lon, 'mesh_lat': mesh_lat, 'SLA': SLA}
    return dataset