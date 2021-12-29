'''
1개월치 VBD 자료를 읽어서 DensityMap을 NC형태로 저장
초기 2021.09.01
Auth: Hokun Jeon
Marne Bigdata Center
KIOST
'''


'라이브러리 호출'
import numpy as np
import os, sys
import pandas as pd
import gzip
import datetime as dt

## 호군 로컬PC
path_VBD = 'D:/03_MarineTraffic/VBD'
path_out_NC='D:/20_Product/VBD/NC/DensityMonthly'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def days2month(path_month):
    '''
    VBD daily data(*.csv.gz)를 한달치를 읽어 전체 lonlat을 반환

    lonlat=days2month(path_month)
    path_month = 'D:/03_MarineTraffic/VBD/2018/M02'

    daily 데이터가 한달치씩 폴더 안에 있어야 함. 이 때 월 폴더명은 숫자앞에 'M'이 기입되어 있음
    daily 데이터 형식은 csv.gz 이어야 함
    '''

    day_list=np.array(os.listdir(path_month))
    day_list=day_list[np.array(['csv.gz' in name for name in day_list])] # gz만 추출

    lons = [];
    lats = [];
    for ii in range(len(day_list)):
        tp = pd.read_csv(gzip.open(os.path.join(path_month,day_list[ii]))) # gz 압축해제 후, DF 읽기
        lons = lons + list(tp.Lon_DNB)
        lats = lats + list(tp.Lat_DNB)
    lonlat=[lons, lats]
    return lonlat

def make_month_NC(lonlat,year,month,area,resolution):
    '''
    days2month 함수에서 반환한 lonlat을 이용해 NC를 생성
    make_month_NC(lonlat,year,month,area,resolution)
    year='2020'
    month='03'
    area='Donghae' # Changwon
    resolution=0.01
    lonlat=days2month(path_month)
    path_month = 'D:/03_MarineTraffic/VBD/2018/M02'

    daily 데이터가 한달치씩 폴더 안에 있어야 함. 이 때 월 폴더명은 숫자앞에 'M'이 기입되어 있음
    daily 데이터 형식은 csv.gz 이어야 함
    '''

    import Map
    sector=Map.sector()
    # area='Donghae'
    coord=sector[area]

    '''
    1. NC에 할당할 데이터 생성
    '''

    lons = np.array(lonlat[0])
    lats = np.array(lonlat[1])

    if coord[1]>180:
        lons[lons<0]=lons[lons<0]+360 # 표시하고자하는 값이 Int'nal Time Zone을 포함하는 경우
    cond = ((coord[0] <= lons) & (lons <= coord[1])) & ((coord[2] <= lats) & (lats <= coord[3]))  # 범위 내 자료 조건
    lons = lons[cond]
    lats = lats[cond]
    lat_bins = np.linspace(coord[2], coord[3], int((coord[3] - coord[2])/resolution + 1))
    lon_bins = np.linspace(coord[0], coord[1], int((coord[1] - coord[0])/resolution + 1))
    density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    meshlon, meshlat=np.meshgrid(lon_bins, lat_bins)



    # HMT Mobile 시스템을 위해 격자중심좌표로 변경하고, 수평수직 끝단 열은 지움
    meshlon, meshlat=meshlon+resolution/2, meshlat+resolution/2
    meshlon = meshlon[:-1,:-1]
    meshlat = meshlat[:-1, :-1]

    '''
    2. NC에 할당
    '''
    import netCDF4 as nc

    dir = path_out_NC+'/'+year
    os.makedirs(dir,exist_ok=True)
    fn_path=dir+'/ShipDensity'+'_'+area+'_'+ year +'-'+month+'_'+str(resolution)+'.nc'
    ds = nc.Dataset(fn_path, 'w', format='NETCDF4')

    lat=ds.createDimension('lat', len(lat_bins)-1)
    lon=ds.createDimension('lon', len(lon_bins)-1)
    density_x=ds.createDimension('density_x', len(lon_bins)-1)
    density_y = ds.createDimension('density_y', len(lat_bins)-1)

    ds.title='Ship Density '+area+'-'+ year+'-'+month+' Res:'+str(resolution)

    ds.area=area
    ds.resolution=resolution

    lat = ds.createVariable('meshlat', np.float32, ('lat','lon'))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ds.createVariable('meshlon', np.float32, ('lat','lon'))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'

    # 프로젝션 변수
    crs = ds.createVariable('VBD_map_projection', np.int32)
    crs.long_name = 'VBD Density Grid Projection'
    crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    crs.EPSG_code = "EPSG:4326"
    crs.latitude_of_projection_origin = min(lats)
    crs.longitude_of_projection_origin = min(lons)
    crs.semi_major_axis = 6378137.0  # WGS84
    crs.semi_minor_axis = 6356752.5  # WGS84
    crs.spatial_resolution = resolution

    # Define a 3D variable to hold the data
    Density = ds.createVariable('density',np.int,('density_y','density_x')) # note: unlimited dimension is leftmost
    Density.units = 'ship' # ship
    Density.standard_name = '# of ships' # this is a CF standard name

    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    lat[:] = np.flipud(meshlat)
    lon[:] = meshlon
    Density[:]=np.flipud(density)
    ds.close()


'Batch시에만 사용'
areas=['Donghae', 'Changwon']
years=['2017','2018','2019','2020']
res=[0.05]
# area='Donghae'
# year='2017'
for area in areas:
    for year in years:
        for resolution in res:
            month_list=os.listdir(os.path.join(path_VBD,year))
            # 해당 연도의 월별 자료 생산
            for ii in range(len(month_list)):
                path_month=os.path.join(path_VBD,year,month_list[ii])
                if not len(os.listdir(path_month))==0:
                    print(path_month)
                    lonlat=days2month(path_month)
                    month=month_list[ii].split('M')[1] # 앞에 M을 때어내고 월'을 나타내는 숫자만 남기기
                    make_month_NC(lonlat, year, month, area,resolution)