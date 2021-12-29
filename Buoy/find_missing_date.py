import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Lib.lib_os import *

path_input_dir='E:/04_Observation/KHOA_MarineBuoy'
path_output_dir=path_input_dir+'/missdate'
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

    miss_date=pd.DataFrame()
    for jj in range(len(items)):
        str_cond=data[items[jj]]=='-'
        tt=data['Time'][str_cond]
        td = pd.DataFrame(pd.to_datetime(data['Time']).dt.strftime('%Y-%m-%d').unique())
        td.to_csv(path_output_dir+'/missdate_'+area[ii]+'_'+items[jj].split('(')[0]+'.csv', index=False)
