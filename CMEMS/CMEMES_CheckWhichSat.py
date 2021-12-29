import netCDF4 as nc
import pandas as pd
import numpy as np
from Lib.lib_os import *
path_dir='E:/01_Model/CMEMS/OCEANCOLOUR_GLO_CHL_L4_REP_OBSERVATIONS_009_082/Donghae'


file_list=np.array(recursive_file(path_dir,'*.nc'))

years=[]
sensors=[]
for ii in range(len(file_list)):
    years.append(file_list[ii][:-3].split('_')[-1])
    # ii=0
    sensors.append(nc.Dataset(file_list[ii]).sensor_name)