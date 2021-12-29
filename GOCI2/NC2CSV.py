'''
NC를 읽어서 CSV로
'''
import cv2
import netCDF4 as nc
import os, sys
sys.path.append('D:/programming/Dokdo')
from Lib.lib_os import *
from Lib.lib_GOCI1 import *
import numpy as np
import pandas as pd
from tqdm import tqdm

path_dir='E:/20_Product/CHL_DM'

file_list=np.array(recursive_file(path_dir,pattern='*.nc'))
file_list=np.array(file_list)[np.array(['_res100' in name for name in file_list])]

for ii in tqdm(range(len(file_list))):
    # ii=0
    data=nc.Dataset(file_list[ii])
    chl = data['chl'][:].flatten()
    lm = data['lm'][:].flatten()
    noa = data['NonObsArea'][:].flatten()
    lon = data['lon'][:].flatten()
    lat = data['lat'][:].flatten()
    DF = pd.DataFrame({'lon':lon, 'lat':lat, 'chl':chl, 'lm':lm, 'noa':noa})
    DF.to_csv(file_list[ii][:-3]+'.csv', index=False)