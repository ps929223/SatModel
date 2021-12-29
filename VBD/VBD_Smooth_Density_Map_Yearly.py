'''
DensityMap Smoothing
2021.09.10
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import numpy as np


'입력/출력, 소스코드 path 설정'

## 호군 로컬PC
path_input='D:/20_Product/VBD/NC/DensityYearly'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)

def read_yearly_NC(path_file):
    # -*- coding: utf-8 -*-
    # path_file='D:/20_Product/VBD/NC/DensityYearly/ShipDensity_Donghae_2020_0.25.nc'
    import Map
    from scipy.ndimage import gaussian_filter

    ds = nc.Dataset(path_file)

    mesh_lat = np.array(ds.variables['meshlat'])
    mesh_lon = np.array(ds.variables['meshlon'])

    density = np.array(ds.variables['density'])
    GFTR = gaussian_filter(density, sigma=2)  # Gausian Filter Blur
    GFTR=GFTR.astype(float)
    GFTR[GFTR==0]=np.nan

    area=ds.area

    sector=Map.sector()



    cmap1 = plt.cm.get_cmap('jet')
    cmap1.set_over(color='w', alpha=None)
    cmap1.set_under(color='w', alpha=None)

    coord = sector[area]
    fig1, ax = plt.subplots(figsize=(13, 10))

    m = Map.making_map(area, 5)

    if ds.resolution == 0.25:
        ticks=[1, 5, 10, 20, 40, 60, 80, 100]
    elif ds.resolution == 0.1:
        ticks=[1, 5, 10, 20, 40, 60, 80, 100]


    # m.readshapefile(path_code + '/' + 'world_eez_v8_2014/world_eez_v8_2014', 'world_eez_v8_2014_0_360', linewidth=0.5)
    pc= m.pcolormesh(mesh_lon, mesh_lat, GFTR, cmap=cmap1, alpha=10, vmin=ticks[0], vmax=ticks[-1])
    norm = mpl.colors.BoundaryNorm(ticks, cmap1.N, extend='both')
    cb= fig1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap1), orientation='vertical', label="Number of ships",
                      shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%2i')

    # plt.clim(1, np.log10(ticks[-1]))  # 범위 제한을 해서 Colorbar가 최대값을 표시하지 못하는 현상을 막음
    cb.set_ticks(list(ticks))
    tick_labels = list(np.array(ticks).astype(int).astype(str))
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=20)
    cb.set_label('Number of Ships', size=18)

    title=ds.title
    plt.title(title, fontsize=30)
    plt.tight_layout()

    last_year=title.split('-')[1][:4]

    dir = path_input
    os.makedirs(dir,exist_ok=True)

    pathout=dir+'/ShipSmoothDensity'+'_'+ds.area+'_'+last_year+'_'+str(ds.resolution)+'.png'
    plt.savefig(pathout)
    plt.close()

'Batch 생산'
years=['2017','2018','2019','2020']
areas=['Donghae','Changwon']
res=[0.05]

'Batch 처리'
file_list=np.array(os.listdir(path_input))
nc_list=file_list[np.array(['.nc' in name for name in file_list])]
for area in areas:
    area_list=nc_list[np.array([area in name for name in nc_list])]
    for year in years:
        year_list=area_list[np.array([year in name for name in area_list])]
        for resolution in res:
            res_list = year_list[np.array([str(resolution) in name for name in year_list])][0]
            path_file=os.path.join(path_input,res_list)
            read_yearly_NC(path_file)