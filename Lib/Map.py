'''
지도생성
2021.09.01
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

## 지도에 넣을 섹터
def sector():
    sector = {'Korea': [119, 135, 31, 44], 'Busan':[128.5, 129.5, 34.5, 35.5],
              'EastAsia': [117, 143, 23, 49], 'Changwon': [128.2, 128.9, 34.8, 35.2],
              'Donghae': [128, 138, 34, 43.7], 'Dokdo': [131.4, 132.6, 36.9, 37.6],
              'DokdoIs':[131+51/60+22/60**2, 131+52/60+45/60**2, 37+14/60+8/60**2, 37+15/60+3/60**2],
              'NamhaeDonghae':[125.7, 133, 32, 39], 'Busan':[128.63,129.68, 34.66, 35.25],
              'Ulsan':[129.3, 129.57, 35.3, 35.6], 'Sokcho2Hosan':[128.61,130.2,36.95,38.38]}
    return sector

def dokdo_psn():
    coord=[131+52/60, 37+14/60+23/60**2]
    return coord

def station_psn():
    psn={'UlleungNE':[131.552583, 38.007361], 'UlleungNW':[130.601194, 37.742722],
         'NamhaeEast':[128.419027, 34.222472], 'KoreaStrait':[129.121250, 34.9190], 'JejuSouth':[126.965861, 32.090416],
         'JejuStrait':[126.5905, 33.700111], 'Donghae':[129.95, 37.48056], 'Geoje': [128.9, 34.7667],
         'Geomun':[127.50127222, 34.00135], 'Pohang':[129.78333333, 36.35], 'Seogwipo':[127.016, 33.13],
         'Tongyeong':[128.21, 34.4],'Uljin':[129.866, 36.9],'Ulleung':[131.1144, 37.4554],'Ulsan':[129.833, 35.35]}
    return psn

def station_list():
    stations={'KHOA':['UlleungNE','UlleungNW','NamhaeEast','KoreaStrait','JejuSouth','JejuStrait'],
              'KMA':['Donghae','Geoje','Geomun','Pohang','Seogwipo','Tongyeong','Uljin','Ulleung','Ulsan']}
    return stations

def mycmap(cm_name):
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    import numpy as np
    # cm_name='viridis'
    mycmap = cm.get_cmap(cm_name, 12)
    vs = np.linspace(0,1,10)

    ii=0
    final_map=np.zeros((0,4))
    for ii in range(len(vs)-1):
        if ii!=len(vs)-2:
            tp = np.linspace(mycmap(vs[ii]), mycmap(vs[ii + 1]), 5)
            final_map=np.concatenate([final_map, tp], axis=0)
        else:
            residual=255-final_map.shape[0]
            tp = np.linspace(mycmap(vs[ii]), mycmap(vs[ii + 1]), residual)
            final_map = np.concatenate([final_map, tp], axis=0)
    final_map=ListedColormap(final_map)

    return final_map

def gen_sector_mesh(sector_name,n_row, n_col):
    import Lib.Map as Map
    import numpy as np
    coord=Map.sector()[sector_name]

    lons=np.linspace(coord[0],coord[1],n_row)
    lats=np.linspace(coord[2],coord[3],n_col)
    mesh_lon, mesh_lat = np.meshgrid(lons,lats)
    return mesh_lon, mesh_lat

def making_map(coord, map_res, grid_res):
    '''
    지도 그리기
    :param coord: 좌표 최소최대값 [minlon, maxlon, minlat, maxlat]
    :param map_res: map resolution # c,l,i,h,f
    :param grid_res: 지도 내 격자 [단위: deg]
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    m = Basemap(projection='merc', lat_0=(coord[2] + coord[3]) / 2, lon_0=(coord[0] + coord[1]) / 2,
                resolution=map_res,
                llcrnrlon=coord[0], urcrnrlon=coord[1], llcrnrlat=coord[2], urcrnrlat=coord[3])
    m.drawcoastlines(color='#5a6b5c')
    m.drawcountries()
    # m.drawmapboundary(fill_color='#99ffff') # 바다색
    m.drawmapboundary(fill_color='#ffffff') # 바다색
    m.fillcontinents(color='#dbdbdb', lake_color='#ffffff')
    # m.fillcontinents(color='#010101', lake_color='#010101')
    # m.drawmapboundary(fill_color='#000000') # 바다색
    m.drawparallels(np.arange(coord[2], coord[3], grid_res),color='gray',dashes=[1,3],labels=[1,0,0,1],fontsize=10)
    meridians=m.drawmeridians(np.arange(coord[0], coord[1], grid_res),color='gray',dashes=[1,3],labels=[1,0,0,1],fontsize=10)
    for ii in meridians:
        try:
            meridians[ii][1][0].set_rotation(45)
            meridians[ii][1][1].set_rotation(45)
        except:
            pass
    plt.show()
    return m

#
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
#
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
#
# plt.show()