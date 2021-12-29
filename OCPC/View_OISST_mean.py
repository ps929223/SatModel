import pandas as pd
import os, sys
sys.path.append('D:/programming/SatModel/Lib')
from Lib.lib_os import *
import matplotlib.pyplot as plt

path_input_dir='D:/01_Model/KIO_OCPC'
path_files=recursive_file(path_input_dir,'*SST*.csv')
path_output_dir='D:/20_Product/KIO_OCPC/meanOISST'
os.makedirs(path_output_dir, exist_ok=True)

def draw(path_file):
    '''
    path_file='D:/01_Model/KIO_OCPC\\OISST_mean_202105.csv'
    '''
    data=pd.read_csv(path_file)
    ## 시간 Column 형태 변경
    data=data.rename(columns={'Unnamed: 0':'Month'})
    data=data[data['Month'].notna()] # nan행 제거
    tab_color=['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray']
    MM=['YS_monthlymean', 'ECS_monthlymean', 'ES_monthlymean', 'ASK_monthlymean']
    CL=['YS_climatology', 'ECS_climatology', 'ES_climatology', 'ASK_climatology']

    plt.figure(1, figsize=(10,5))
    for jj in range(len(MM)):
        plt.plot(data.Month, data[MM[jj]], c=tab_color[jj], label=MM[jj])
        plt.plot(data.Month, data[CL[jj]], linestyle='dashed', c=tab_color[jj], label=CL[jj])
    plt.legend()
    plt.grid()
    plt.title('Monthly Mean SST ('+ data.Month[0] + '~' + data.Month[data.index[-1]] +')')
    plt.xlabel('Date')
    plt.ylabel('Temperature ($^\circ$C)')
    plt.savefig(path_output_dir+'/'+path_file.split('\\')[-1][:-3]+'png')
    plt.clf()


for ii in range(len(path_files)):
    draw(path_files[ii])