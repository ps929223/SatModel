import os
import h5py
import numpy as np

path_flag = 'D:/02_Satellite/GOCI1/flag'
os.makedirs(path_flag, exist_ok=True)
path_file = 'D:/02_Satellite/GOCI1/L2C/COMS_GOCI_L2C_GA_20201217001646.he5'


data = h5py.File(path_file, mode='r')
tt=data['HDFEOS/GRIDS/Image Data/Data Fields/FLAG Image Pixel Values'][:]
len(np.unique(tt))

path_out=path_flag+'/'+path_file.split('/')[-1][:-4]+'.csv'
