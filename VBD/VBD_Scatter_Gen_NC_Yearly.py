'''
Monthly VBD NC 자료를 읽어서 Monthly VBD NC 생성
2021.09.10
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

import netCDF4 as nc
import os, sys
import numpy as np

'입력/출력, 소스코드 path 설정'

# 호군 로컬 PC
global path_NC, path_out_NC
path_NC='D:/20_Product/VBD/NC/ScatterMonthly'
path_out_NC='D:/20_Product/VBD/NC/ScatterYearly'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def read_onemonth_NC(path_file):
    # -*- coding: utf-8 -*-
    # path_file='D:/20_Product/VBD/NC/MonthlyScatter/2016/ShipScatter_EastAsia_2016-07.nc'
    import Map

    ds = nc.Dataset(path_file)

    lat = np.array(ds.variables['lat'])
    lon = np.array(ds.variables['lon'])

    Date_Mscan = np.array(ds.variables['Date_Mscan'])

    return lon, lat, Date_Mscan

def make_yearly_NC(lons, lats, Date_Mscans, year, area):
    import netCDF4 as nc
    dir = path_out_NC + '/' + year
    os.makedirs(dir, exist_ok=True)
    fn = dir + '/ShipScatter' + '_' + area + '_' + year + '.nc'
    ds = nc.Dataset(fn, 'w', format='NETCDF4')

    lat = ds.createDimension('lat', len(lats))
    lon = ds.createDimension('lon', len(lons))
    Date_Mscan = ds.createDimension('Date_Mscan', len(Date_Mscans))

    ds.title = 'Ship Scatter ' + area + '-' + year

    ds.area = area

    lat = ds.createVariable('lat', np.float32, ('lat'))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ds.createVariable('lon', np.float32, ('lon'))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    scan = ds.createVariable('Date_Mscan', np.str, ('Date_Mscan'))
    scan.units = 'Scan yyyy-mm-dd HH:MM:SS.S'
    scan.long_name = 'Scan Time'

    # 프로젝션 변수
    crs = ds.createVariable('VBD_map_projection', np.int32)
    crs.long_name = 'VBD Scatter Projection'
    crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs.EPSG_code = "EPSG:4326"
    # crs.latitude_of_projection_origin = min(lats)
    # crs.longitude_of_projection_origin = min(lons)
    crs.semi_major_axis = 6378137.0  # WGS84
    crs.semi_minor_axis = 6356752.5  # WGS84

    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    lat[:] = lats
    lon[:] = lons
    scan[:] = Date_Mscans
    ds.close()



'처리코드'
years=['2017','2018','2019','2020']
areas=['Donghae', 'Changwon']

# year='2017'
# area='Donghae'

for year in years:
    path_year=path_NC+'/'+year
    file_list=np.array(os.listdir(path_year))
    file_list=file_list[np.array(['.nc' in name for name in file_list])]
    # area='PIF'
    for area in areas:
        print(year + ' ' + area)
        area_file_list = file_list[np.array([area in name for name in file_list])]
        lons=np.array([])
        lats=np.array([])
        Date_Mscans=np.array([])
        if len(area_file_list)!= 0:
            for file_name in area_file_list:
                # file_name='ShipScatter_EastAsia_2016-08.nc'
                path_file = path_year +'/'+ file_name
                lon, lat, Date_Mscan=read_onemonth_NC(path_file)
                lons = np.concatenate([lons, lon])
                lats = np.concatenate([lats, lat])
                Date_Mscans = np.concatenate([Date_Mscans, Date_Mscan])
            make_yearly_NC(lons, lats, Date_Mscans, year, area)