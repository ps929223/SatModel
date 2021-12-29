import netCDF4 as nc
import numpy as np
import Lib.Map as Map
import matplotlib.pyplot as plt
import datetime as dt

def draw_ADT(path_file, area, target_date):
    '''
    path_file='D:/01_Model/CMEMS/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046_Donghae/SEALEVEL_adt_2020-01-01_2021-01-01.nc'
    area='Donghae'
    target_date='2020-05-06'
    '''


    ds = nc.Dataset(path_file)
    vars=list(ds.variables)

    lat=np.array(ds.variables['latitude'][:])
    lon=np.array(ds.variables['longitude'][:])
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    time = np.array(ds.variables['time'][:])
    date_start = ds.variables['time'].units.replace('days since ','')
    date_start = dt.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')
    day_diff= (dt.datetime.strptime(target_date, '%Y-%m-%d')-date_start).days
    date_idx=int(np.where(day_diff==time)[0])

    adt=np.array(ds.variables['adt'][date_idx,:,:])

    # 동해 범위좌표
    coord=Map.sector()['Donghae']
    # 독도 점좌표
    dokdo = Map.dokdo_psn()

    # 지도 가시화
    plt.figure(3, figsize=(5, 5))
    m=Map.making_map(coord, map_res='i', grid_res=1)
    xx, yy = m(mesh_lon, mesh_lat)
    pm=m.pcolor(xx,yy,adt, cmap='jet')
    cb = m.colorbar()
    cb.set_label('[' + Unit + ']', rotation=270)

    # 독도 표시
    x, y = m(dokdo[0], dokdo[1])
    m.scatter(x, y, facecolor='w', edgecolor='k', s=80, marker='o')

    plt.title('ADT '+target_date)