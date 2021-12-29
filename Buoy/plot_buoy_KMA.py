import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Lib.lib_os import *

path_input_dir='E:/04_Observation/KMA_MarineBuoy'
path_output_dir=path_input_dir+'/plot'
os.makedirs(path_output_dir, exist_ok=True)

area=['Donghae','Geoje','Geomun','Pohang','Seogwipo','Tongyeong','Uljin','Ulleung','Ulsan']
items=['WindVel(m/s)', 'WindDir(deg)', 'GUSTVel(m/s)',
       'AirPress(hPa)', 'Humid(%)', 'AirTemp(°C)', 'SeaTemp(°C)',
       'MaxWaveHt(m)', 'SigWaveHt(m)', 'AvgWaveHt(m)', 'WavePeriod(sec)',
       'WaveDir(deg)']

for ii in range(len(area)):
    # ii=1
    file_list=np.array(recursive_file(path_input_dir,area[ii]+'*.csv'))
    data=pd.DataFrame()
    for jj in range(len(file_list)):
        # jj=0
        tp=pd.read_csv(file_list[jj])
        data=pd.concat([data, tp], axis=0)

    plt.figure(figsize=(16, 30))
    period_str=file_list[0][-8:-4]+'-'+file_list[-1][-8:-4]
    plt.suptitle('KMA '+ area[ii] +' '+ period_str)
    for jj in range(len(items)):
        plt.subplot(len(items),1,jj+1)
        data['Time']=pd.to_datetime(data['Time'])
        plt.plot(data['Time'],data[items[jj]],label=items[jj])
        plt.legend(loc='upper left')
        plt.grid()
    plt.tight_layout()
    plt.savefig(path_output_dir+'/KMA_'+area[ii]+'_'+period_str)
    plt.close()


