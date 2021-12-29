
'''
데이터 분포를 확인 QQ, Histogram
'''

from Lib.lib_os import *
import numpy as np
import os

path_dir='E:/20_Product/CHL_Krig/CHL_Orig'
path_output_dir='E:/20_Product/CHL_Krig'
QQ_dir=path_output_dir+'/QQ'
Hist_dir=path_output_dir+'/Hist'

os.makedirs(QQ_dir, exist_ok=True)
os.makedirs(Hist_dir, exist_ok=True)

file_list=np.array(recursive_file(path_dir,'*Mask.csv'))

from tqdm import tqdm
import pandas as pd
import Lib.lib_draw as Kd
import matplotlib.pyplot as plt

for ii in tqdm(range(len(file_list))):
    # ii=0
    target_date_str=file_list[ii].split('\\')[-1].split('_')[1]
    data=pd.read_csv(file_list[ii])
    Kd.QQ(data['chl-a'])
    plt.title(target_date_str)
    plt.savefig(QQ_dir+'/QQ_'+target_date_str+'.png')
    plt.clf()
    Kd.Hist(data['chl-a'])
    plt.title(target_date_str)
    plt.savefig(Hist_dir+'/Hist_'+target_date_str+'.png')
    plt.clf()
