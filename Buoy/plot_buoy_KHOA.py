import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Lib.lib_os import *

path_input_dir='E:/04_Observation/KHOA_MarineBuoy'
path_output_dir=path_input_dir+'/plot'
os.makedirs(path_output_dir, exist_ok=True)

area=['JejuSouth','JejuStrait','KoreaStrait','NamhaeEast','UlleungNE','UlleungNW']
items=['SurfCurVel', 'SurfCurDir(deg)',
       'SeaTemp', 'PSU', 'SigWaveHt', 'SigWavePeriod', 'MaxWaveHt',
       'MaxWavePeriod', 'WaveDir(deg)', 'WindVel(m/s)',
       'WindDir(deg)', 'AirTemp(℃)', 'AirPress(hPa)']

for ii in range(len(area)):
    # ii=0
    file_list=np.array(recursive_file(path_input_dir,area[ii]+'*.csv'))
    data=pd.DataFrame()
    for jj in range(len(file_list)):
        # jj=0
        tp=pd.read_csv(file_list[jj])
        tp.keys()
        data=pd.concat([data, tp], axis=0)

    plt.figure(figsize=(16, 30))
    period_str=file_list[0][-8:-4]+'-'+file_list[-1][-8:-4]
    plt.suptitle('KHOA '+ area[ii] +' '+ period_str)
    data['Time'] = pd.to_datetime(data['Time'])

    for jj in range(len(items)):
        str_cond=data[items[jj]]=='-'
        data[items[jj]][str_cond]=np.nan
        data[items[jj]]=data[items[jj]].astype(float)
        # jj=0
        plt.subplot(len(items),1,jj+1)
        plt.plot(data['Time'],data[items[jj]],label=items[jj])
        plt.legend(loc='upper left')
        plt.grid()
    plt.tight_layout()
    plt.savefig(path_output_dir+'/KHOA_'+area[ii]+'_'+period_str)
    plt.close()


