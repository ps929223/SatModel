import netCDF4 as nc
import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.path.append('D:/programming/Dokdo')


def norm(z):
    vmin,vmax=np.percentile(z, [0,100])
    z=(z-vmin)/(vmax-vmin)
    return z

def plot_vars(ds, vars, area, data_type, mesh_lon, mesh_lat):
    # ii=3
    for ii in range(len(vars)):
        # 변수추출
        SN = ds.variables[vars[ii]].standard_name
        Unit = ds.variables[vars[ii]].units

        if data_type !='bio':
            SF = ds.variables[vars[ii]].scale_factor
            z = np.array(ds.variables[vars[ii]][:]) * SF
        else:
            z = np.array(ds.variables[vars[ii]][:])

        if np.std(norm(z)) > 0.3:
            thred=(np.max(z)-np.min(z))/2
            z[z>thred]=np.nan

        # 값설정
        vmin, vmax = np.nanpercentile(z, [5, 95])
        if len(z.shape) > 2:
            z = z.reshape(mesh_lon.shape)
        if Unit in ['m']: # 단위가 길이일 때 최소값은 0
            vmin = 0

        # 가시화
        sect = Map.sect()
        # area='Donghae'
        coord = sect[area]
        m = Map.making_map(coord, 1)
        xx, yy = m(mesh_lon, mesh_lat)
        # cmap=Map.mycmap('jet')
        cmap='jet'
        pm = m.pcolor(xx, yy, z, vmin=vmin, vmax=vmax, cmap=cmap)
        cb = m.colorbar()
        cb.set_label('['+Unit+']', rotation=270)
        plt.title(SN)
        dokdo = Map.dokdo_psn()
        x, y = m(dokdo[0], dokdo[1])
        m.scatter(x, y, facecolor='w', edgecolor='k', s=80, marker='o')
        save_dir = 'CMEMS/' + data_type
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + '/' + SN + '.png')
        plt.clf()


def draw_CMEMS(path_file, area, data_type):
    ds = nc.Dataset(path_file)
    vars=list(ds.variables)

    lat=np.array(ds.variables['latitude'][:])
    lon=np.array(ds.variables['longitude'][:])
    time=np.array(ds.variables['time'][:])
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)

    if data_type=='phy':
        ## vector
        uo = np.array(ds.variables['uo'][:]) # eastward_sea_water_velocity
        vo = np.array(ds.variables['vo'][:]) # northward_sea_water_velocity
        usi = np.array(ds.variables['usi'][:]) # eastward_sea_ice_velocity
        vsi = np.array(ds.variables['vsi'][:]) # northward_sea_ice_velocity

        ## scalar
        vars = list(set(vars) - set(['latitude', 'longitude', 'time', 'depth', 'usi', 'vsi', 'siconc', 'sithick','uo','vo']))
        plot_vars(ds, vars, area, data_type, mesh_lon, mesh_lat)

    elif data_type=='wav':
        ## vector
        VSDX = np.array(ds.variables['VSDX'][:]) # sea_surface_wave_stokes_drift_x_velocity
        VSDY = np.array(ds.variables['VSDY'][:]) # sea_surface_wave_stokes_drift_y_velocity
        ## scalar
        vars = list(set(vars) - set(['latitude', 'longitude', 'time', 'VSDX', 'VSDY']))
        plot_vars(ds, vars, area, data_type, mesh_lon, mesh_lat)

    elif data_type=='bio':
        ## vector 없음
        ## scalar
        vars = list(set(vars) - set(['latitude', 'longitude', 'time','depth']))
        plot_vars(ds, vars, area, data_type, mesh_lon, mesh_lat)


path_CMEMS= 'D:/01_Model/CMEMS'
folder_types= np.array(os.listdir(path_CMEMS))
types=folder_types[2]
date_list=np.array(os.listdir(path_CMEMS+'/'+types))
file_list=np.array(os.listdir(path_CMEMS+'/'+types+'/'+date_list[0]))
path_file=path_CMEMS+'/'+types+'/'+date_list[0]+'/'+file_list[0]

data_type=file_list[0].split('-')[3]
plt.figure(figsize=(7,7))
draw_CMEMS(path_file,'Donghae', data_type)


