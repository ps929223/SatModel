# -*- coding: utf-8 -*-
'''
Density Map Smoothing
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

# 호군 로컬 PC
path_NC='D:/20_Product/VBD/NC/DensityMonthly'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def read_month_NC(path_file, map_res, grid_res):
    # path_file='D:/20_Product/VBD/NC/DensityMonthly/2018/ShipDensity_PIF_2018-02_0.25.nc'
    import Map
    from scipy.ndimage import gaussian_filter

    ds = nc.Dataset(path_file)

    mesh_lat = np.array(ds.variables['meshlat'])
    mesh_lon = np.array(ds.variables['meshlon'])

    density = np.array(ds.variables['density'])
    GFTR = gaussian_filter(density, sigma=2)  # Gausian Filter Blur

    help(gaussian_filter)
    GFTR=GFTR.astype(float)
    GFTR[GFTR==0]=np.nan

    area=ds.area

    sector=Map.sector()


    cmap1 = plt.cm.get_cmap('jet')
    cmap1.set_over(color='w', alpha=None)
    cmap1.set_under(color='w', alpha=None)
    coord = sector[area]
    fig1, ax = plt.subplots(figsize=(13, 10))

    m = Map.making_map(coord, map_res, grid_res)

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
    month=title.split('-')[2][:2]

    dir = path_NC+'/'+last_year
    os.makedirs(dir,exist_ok=True)

    pathout=dir+'/ShipSmoothDensity'+'_'+ds.area+'_'+last_year+'-'+month+'_'+str(ds.resolution)+'.png'
    plt.savefig(pathout)
    plt.close()

'Batch 생산'
years=['2017','2018','2019','2020']
areas=['Donghae']
# year='2020'

for year in years:
    path_year=path_NC+'/'+year
    file_list=np.array(os.listdir(path_year))
    file_list=file_list[np.array(['.nc' in name for name in file_list])]
    # area='PIF'
    for area in areas:
        file_list = file_list[np.array([area in name for name in file_list])]
        for ii in range(len(file_list)):
            path_file = path_year +'/'+ file_list[ii]
            read_month_NC(path_file, map_res, grid_res)