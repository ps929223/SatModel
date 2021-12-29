# -*- coding: utf-8 -*-
'''
VIIRS DNB 500m 격자자료를 읽어서 가시화
초안: 2021.09.10
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

'라이브러리 호출'
import h5py
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import datetime as dt

# '입력/출력, 소스코드 path 설정'
## 호군 로컬PC
path_H5 = ''
path_PNG = ''
path_code = ''
path_Lib = ''
sys.path.append(path_Lib)


def VIIRS_Map(list_path_VIIRS, area):
    import Map
    sector = Map.sector()
    # for i in range(len(list_path_VIIRS)):
    # list_path_VIIRS=list_path_VIIRS[0]
    path_In = path_VIIRS + '/' + list_path_VIIRS
    data = h5py.File(path_In,'r+')
    data.keys()
    data['HDFEOS'].keys()
    data['HDFEOS']['ADDITIONAL'].keys()
    data['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'].keys()
    data['HDFEOS']['GRIDS'].keys()
    data['HDFEOS']['GRIDS']['VNP_Grid_DNB'].keys()
    data['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields'].keys()
    data['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['QF_Cloud_Mask'][:]
    data['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['DNB_At_Sensor_Radiance_500m'][:]
    data['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Granule'][:]
    QF_DNB=data['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['QF_DNB'][:]
    DNB=data['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['DNB_At_Sensor_Radiance_500m'][:]
    minDNB,maxDNB=np.quantile(DNB,[0, 1])

    latitudes = data.variables['latitude'][::-1]
    longitudes = data.variables['longitude'][:].tolist()
    slpin = 0.01 * data.variables['PRMSL_meansealevel'][:].squeeze()
    # Dataset.close()

    uin = data.variables['UGRD_10maboveground'][:].squeeze()
    vin = data.variables['VGRD_10maboveground'][:].squeeze()

    slp = np.zeros((slpin.shape[0], slpin.shape[1] + 1), np.float)
    slp[:, 0:-1] = slpin[::-1];
    slp[:, -1] = slpin[::-1, 0]

    u = np.zeros((uin.shape[0], uin.shape[1] + 1), np.float64)
    u[:, 0:-1] = uin[::-1];
    u[:, -1] = uin[::-1, 0]

    v = np.zeros((vin.shape[0], vin.shape[1] + 1), np.float64)
    v[:, 0:-1] = vin[::-1];
    v[:, -1] = vin[::-1, 0]

    longitudes.append(360.);
    longitudes = np.array(longitudes)
    lons, lats = np.meshgrid(longitudes, latitudes)

    plt.figure(3, figsize=(30, 30))

    m = Map.making_map(area)

    x, y = m(lons, lats)
    # area='GuineaBay'
    gab = 4
    clevs = {'Korea': (1016, 1040, gab), 'Sec1': (960, 1060, gab), 'Sec2': (960, 1060, gab), 'Sec3': (960, 1060, gab),
             'Sec4': (960, 1060, gab), 'Zone1': (960, 1060, gab), 'PIF': (990, 1040, gab),
             'Falkland': (960, 1060, gab), 'EastAsia': (1016, 1040, gab), 'GuineaBay': (990, 1040, gab)}
    set_ticks = np.arange(clevs[area][0], clevs[area][1] + gab, gab)

    yy = np.arange(0, y.shape[0], 4)
    xx = np.arange(0, x.shape[1], 4)
    points = np.meshgrid(yy, xx)
    CS1 = m.contour(x, y, slp, np.arange(clevs[area][0], clevs[area][1], clevs[area][2]), linewidths=0.5, colors='k',
                    animated=True)
    CS2 = m.contourf(x, y, slp, np.arange(clevs[area][0], clevs[area][1], clevs[area][2]), cmap=plt.cm.RdBu_r,
                     animated=True)
    barbs = m.barbs(x[points], y[points], u[points], v[points], pivot='middle', barbcolor='#333333')
    cb = m.colorbar(CS2, "right", size="5%", pad="2%")
    cb.set_ticks(set_ticks)
    cb.set_label('hPa', fontsize=40)
    cb.ax.tick_params(labelsize=30)

    now = datetime.now() - timedelta(days=1)
    nowDate = now.strftime('%Y%m%d')
    title = 'Wind&AirPress ' + area + ' ' + nowDate[:4] + '-' + nowDate[4:6] + '-' + nowDate[6:8] + '-' + time

    save_path = path_PNG + '/' + area

    os.makedirs(save_path, exist_ok=True)
    P = list_path_VIIRS.split('P')[1][:3]
    save_filename = save_path + '/' + 'Wind_AirPress' + '_' + area + '_' + nowDate[:4] + '-' + nowDate[
                                                                                               4:6] + '-' + nowDate[
                                                                                                            6:8] + '-' + time + '_P' + P

    plt.title(title, fontsize=60)
    plt.savefig('%s.png' % save_filename, bbox_inces='tight', transparent=True, pad_inches=0, format='png', dpi=300)
    # plt.savefig('%s.png' %save_filename, transparent=True, pad_inches=0, format = 'png', dpi = 300)
    plt.clf()


## 자료생산
yesterday = str(dt.datetime.today() - dt.timedelta(days=1))  # 어제 날짜
yes_yyyy = yesterday[0:4]  # 오늘날짜 연
yes_mm = yesterday[5:7]  # 오늘날짜 월
yes_dd = yesterday[8:10]  # 오늘날짜 일

path_VIIRS = path_H5 + '/' + yes_yyyy + '-' + yes_mm + '-' + yes_dd+'/'
# path_VIIRS = 'E:/Satellite/VIIRS/2021-04-06-00'

list_path_VIIRS = os.listdir(path_VIIRS)  # 원본폴더의 파일목록 반환
list_path_VIIRS = [s for s in list_path_VIIRS if '.h5' in s]  # nc file list만 반환

VIIRS_Map(list_path_VIIRS[0], 'Donghae')  # 첫번째 파일만 읽어서 가시화
