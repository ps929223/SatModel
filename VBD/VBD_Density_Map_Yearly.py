'''
VBD Density NC 자료를 읽어서 DensityMap을 가시화
2021.09.10
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

import netCDF4 as nc
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import datetime as dt


'입력/출력, 소스코드 path 설정'

## 호군 로컬PC
path_NC='D:/20_Product/VBD/NC/DensityYearly'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def read_yearly_NC(path_file):
    # path_file='D:/20_Product/VBD/NC/DensityYearly/ShipDensity_PIF_2018_0.25.nc'

    ds = nc.Dataset(path_file)

    mesh_lat = np.array(ds.variables['meshlat'])
    mesh_lon = np.array(ds.variables['meshlon'])

    density = np.array(ds.variables['density'])
    density = np.log2(density)
    density[np.isinf(density)]=np.nan

    area=ds.area

    cmap1 = plt.cm.get_cmap('jet')
    # cmap1.set_under(color='white')

    fig1 = plt.figure(2, figsize=(13, 9))
    sector=Map.sector()
    coord=sector[area]
    m = Map.making_map(coord, 1)
    ticks=[1, 20, 10, 50, 100, 500, 1000]
    clim = (0, 1)
    xx, yy=m(mesh_lon,mesh_lat)
    plt.pcolormesh(xx, yy, density, cmap=cmap1, alpha=.8, vmin=np.log(ticks[0]), vmax=np.log(ticks[-1]))
    # Projection이 'cyl'인 경우 x,y 대신 mesh_lon, mesh_lat를 사용함
    plt.clim(np.log2(ticks[0]), np.log2(ticks[-1])) # 범위 제한을 해서 Colorbar가 최대값을 표시하지 못하는 현상을 막음
    cb = plt.colorbar(shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%1.1f')
    cb.set_ticks(list(np.log2(ticks)))
    tick_labels = list(np.array(ticks).astype(int).astype(str))
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=20)
    cb.set_label('Number of Ships', size=18)
    # make image bigger:
    # plt.gcf().set_size_inches(12, 12)

    title=ds.title
    plt.title(title, fontsize=30)
    # plt.pause(2)
    plt.tight_layout()

    # area='EastAsia'
    # last_year='2020'
    # month='03'
    dir = path_NC
    os.makedirs(dir,exist_ok=True)
    year=title.split('-')[1][:4]
    pathout=dir+'/ShipDensity'+'_'+ds.area+'_'+year+'_'+str(ds.resolution)+'.png'
    plt.savefig(pathout)
    plt.clf()


areas=['Donghae','Changwon']
years=['2017','2018','2019','2020']
res=[0.05]

'Batch 처리'
file_list=np.array(os.listdir(path_NC))
nc_list=file_list[np.array(['.nc' in name for name in file_list])]
for area in areas:
    area_list=nc_list[np.array([area in name for name in nc_list])]
    for year in years:
        year_list=area_list[np.array([year in name for name in area_list])]
        for resolution in res:
            res_list = year_list[np.array([str(resolution) in name for name in year_list])][0]
            path_file=os.path.join(path_NC,res_list)
            read_yearly_NC(path_file)