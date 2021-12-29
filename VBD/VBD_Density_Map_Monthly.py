'''
VBD Density NC 자료를 읽어서 Monthly 가시화
2021.09.01
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

import netCDF4 as nc
import matplotlib.pyplot as plt
import os, sys
import numpy as np


# 호군 로컬 PC
path_NC='D:/20_Product/VBD/NC/DensityMonthly'
path_Lib = 'D:/programming/SatModel/Lib'
sys.path.append(path_Lib)


def read_month_NC(path_file, map_res1, grid_res1):
    # path_file='D:/20_Product/VBD/NC/DensityMonthly/2018/ShipDensity_Donghae_2018-02_0.05.nc'
    # map_res1='h'
    # grid_res1=1
    import Map

    ds = nc.Dataset(path_file)

    mesh_lat = np.array(ds.variables['meshlat'])
    mesh_lon = np.array(ds.variables['meshlon'])

    density = np.array(ds.variables['density'])
    # density[np.isinf(density)]=np.nan

    area=ds.area


    cmap1 = plt.cm.get_cmap('jet')
    # cmap1.set_under(color='white')

    fig1 = plt.figure(2, figsize=(13, 9))
    sector=Map.sector()
    # area='Changwon'
    coord=sector[area]
    # map_res1='i'
    # grid_res1=0.05
    m = Map.making_map(coord, map_res1, grid_res1)
    ticks=[1, 5, 10, 50, 100, 500]
    clim = (0, 1)
    xx, yy=m(mesh_lon,mesh_lat)
    plt.pcolormesh(xx, yy, density, cmap='gist_yarg_r', alpha=.8, vmin=np.log(ticks[0]), vmax=np.log(ticks[-1]))
    # plt.pcolormesh(xx, yy, density, cmap=cmap1, alpha=.8, vmin=np.log(ticks[0]), vmax=np.log(ticks[-1]))
    # Projection이 'cyl'인 경우 x,y 대신 mesh_lon, mesh_lat를 사용함
    # plt.clim(np.log(ticks[0]),np.log(ticks[0])]) # 범위 제한을 해서 Colorbar가 최대값을 표시하지 못하는 현상을 막음
    cb = plt.colorbar(shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%1.1f')


    dokdo = Map.dokdo_psn()
    x, y = m(dokdo[0], dokdo[1])
    m.scatter(x, y, facecolor='w', edgecolor='k', s=80, marker='o')

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
    year=title.split('-')[1][:4]
    month=title.split('-')[2][:2]

    dir = path_NC+'/'+year
    os.makedirs(dir,exist_ok=True)

    pathout=dir+'/ShipDensity'+'_'+ds.area+'_'+year+'-'+month+'_'+str(ds.resolution)+'.png'

    # plt.savefig('%s' %pathout, bbox_inches='tight', transparent=True, pad_inches=0, format = 'png', dpi = 300)

    # plt.savefig('%s' % pathout, transparent=True, pad_inches=0, format='png', dpi=300)
    # plt.savefig('%s' % pathout, format='png', dpi=300)
    # plt.savefig('%s' % pathout, dpi=300)
    # plt.savefig(pathout, dpi=300)
    plt.savefig(pathout)
    fig1.clf()


'처리코드'
years=['2020']
areas=['Donghae']

map_res = {'Donghae':'h','Changwon':'f'}
grid_res = {'Donghae':1,'Changwon':0.01}

for year in years:
    path_year=path_NC+'/'+year
    file_list=np.array(os.listdir(path_year))
    file_list=file_list[np.array(['.nc' in name for name in file_list])]
    # area='PIF'
    for area in areas:
        file_list = file_list[np.array([area in name for name in file_list])]
        for file_name in file_list:
            path_file = path_year +'/'+ file_name
            read_month_NC(path_file, map_res[area], grid_res[area])