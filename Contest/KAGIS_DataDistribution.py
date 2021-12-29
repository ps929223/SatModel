

from Lib.lib_os import *
import numpy as np


path_dir='D:/30_Conference/2021-11_KAGIS(JEJU)/data/CHL_Orig'
file_list=np.array(recursive_file(path_dir,'*Mask.csv'))

from tqdm import tqdm
import pandas as pd
import Lib.lib_KAGIS_draw as Kd

for ii in tqdm(range(len(file_list))):
    # ii=0
    target_date_str=file_list[ii].split('\\')[-1].split('_')[1]
    data=pd.read_csv(file_list[ii])
    Kd.QQ(data,target_date_str)
    Kd.Hist(data,target_date_str)
