'''
CHL Obs를 가시화
Auth: Hokun Jeon
2021.09.13
Marine Bigdata Center
KIOST
'''

import numpy as np
import os, sys
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

'Setting path'
path_input_dir='D:/01_Model/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP'
path_output_dir= 'D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP'
path_Lib='D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def CHL_map(path_input_file, map_res, grid_res, area):
    '''
    path_input_file='D:/01_Model/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP/2020/07/20200701_d-ACRI-L4-CHL-MULTI_4KM-GLO-DT.nc'
    map_res='h'
    grid_res=1
    area= 'Donghae'
    '''

    import Map

    # 파일명에서 정보 추출
    year=path_input_file.split('/')[4]
    month=path_input_file.split('/')[5]
    file_name= path_input_file.split('/')[6]



    # 데이터 읽기
    data = Dataset(path_input_file, mode = 'r')
    coord = Map.sector()[area]
    lon = data.variables['lon'][:]
    lat = data.variables['lat'][:]
    CHL = np.log10(data.variables['CHL'][0, ::])
    mesh_lon, mesh_lat = np.meshgrid(lon,lat)

    # 지도 가시화
    plt.figure(3, figsize=(5, 5))
    m=Map.making_map(coord, map_res, grid_res)
    xx, yy = m(mesh_lon, mesh_lat)

    # 정보 가시화
    # pm=m.pcolormesh(xx, yy, np.array(CHL), cmap=plt.cm.RdBu_r, vmin=-2, vmax=2)
    pm = m.pcolormesh(xx, yy, np.array(CHL), cmap='jet', vmin=-2, vmax=2)
    cb = m.colorbar()
    cb.set_label('Concentration of Chlorophyll[mg/m-3]', size=10)
    plt.title(file_name[:-3])

    # 독도 표시
    dokdo = Map.dokdo_psn()
    x, y = m(dokdo[0], dokdo[1])
    m.scatter(x, y, facecolor='w', edgecolor='k', s=80, marker='o')

    # 이미지 저장
    save_dir = path_output_dir + '/' + year + '/' + month
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + '/' + file_name[:-3] + '.png')
    plt.clf()


years=['2020','2021']
map_res='h'
grid_res=1
area='Donghae'
# ii=1;jj=2;kk=0
error_file=[]
for ii in range(len(years)):
    path_year=path_input_dir+'/'+years[ii]
    months=os.listdir(path_year)
    for jj in range(len(months)):
        path_month=path_year+'/'+months[jj]
        list_file=np.array(os.listdir(path_month))
        for kk in range(len(list_file)):
            path_input_file=path_month+'/'+list_file[kk]
            try:
                CHL_map(path_input_file, map_res, grid_res, area)
            except:
                print('Error: '+path_input_file)
                error_file.append(path_input_file)