'''
1년 Denisty Map NC 자료를 읽어서 연간 DensityMap을 NC형태로 저장
2021.09.10
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

import os,sys
import numpy as np

## 호군 로컬PC
path_input='D:/20_Product/VBD/NC/DensityMonthly'
path_output='D:/20_Product/VBD/NC/DensityYearly'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def make_Yearly_NC(year,area,resolution):
    '''
    지난해 12개월 Denisty Map NC 자료를 읽어서 연간 DensityMap을 NC형태로 저장
    make_Yearly_NC(year,resolution)
    year='2020'
    area='Donghae' # Changwon
    resolution=0.05

    '''
    import netCDF4 as nc
    import Map
    sector=Map.sector()
    coord=sector[area]


    path_year=path_input+'/'+year
    list_file=np.array(os.listdir(path_year))
    list_nc=list_file[np.array(['.nc' in name for name in list_file])]
    list_area=list_nc[np.array([area in name for name in list_nc])]
    list_res=list_area[np.array([str(resolution) in name for name in list_area])]

    tp=nc.Dataset(os.path.join(path_year,list_res[0]))
    meshlon = np.array(tp['meshlon'])
    meshlat = np.array(tp['meshlat'])
    density=np.zeros((12,tp.dimensions['lat'].size,tp.dimensions['lon'].size),float) # 12개월 밀도맵 집

    for ii in range(len(list_res)):
        tp = nc.Dataset(os.path.join(path_year, list_res[ii]))
        density[ii,:,:]=np.array(tp['density']) # 12개월 밀도맵 Stack

    density=np.sum(density,axis=0) # 12개월 밀도맵 합해서 1개 Matrix 생성


    '''
    2. NC에 할당
    '''

    dir = path_output
    os.makedirs(dir,exist_ok=True)
    fn_path=dir+'/ShipDensity'+'_'+area+'_'+year+'_'+str(resolution)+'.nc'
    ds = nc.Dataset(fn_path, 'w', format='NETCDF4')

    lat=ds.createDimension('lat', meshlat.shape[1])
    lon=ds.createDimension('lon', meshlat.shape[0])
    density_x=ds.createDimension('density_x', meshlat.shape[0])
    density_y = ds.createDimension('density_y', meshlat.shape[1])

    ds.title='Ship Density '+area+'-'+year+' Res:'+str(resolution)

    ds.area=area
    ds.resolution=resolution

    lat = ds.createVariable('meshlat', np.float32, ('lon','lat'))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ds.createVariable('meshlon', np.float32, ('lon','lat'))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'

    # 프로젝션 변수
    crs = ds.createVariable('VBD_map_projection', np.int32)
    crs.long_name = 'VBD Density Grid Projection'
    crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs.EPSG_code = "EPSG:4326"
    crs.latitude_of_projection_origin = np.min(meshlat)
    crs.longitude_of_projection_origin = np.min(meshlon)
    crs.semi_major_axis = 6378137.0  # WGS84
    crs.semi_minor_axis = 6356752.5  # WGS84
    crs.spatial_resolution = resolution

    # Define a 3D variable to hold the data
    Density = ds.createVariable('density',np.int,('density_x','density_y')) # note: unlimited dimension is leftmost
    Density.units = 'ship' # ship
    Density.standard_name = '# of ships' # this is a CF standard name

    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    lat[:] = np.flipud(meshlat)
    lon[:] = meshlon
    Density[:]=np.flipud(density)
    ds.close()


'처리코드'
areas=['Donghae','Changwon']
years=['2017','2018','2019','2020']
res=[0.05]

for area in areas:
    for year in years:
        for resolution in res:
            # area='Donghae'
            # year='2017'
            # resolution=0.1
            make_Yearly_NC(year,area,resolution)
