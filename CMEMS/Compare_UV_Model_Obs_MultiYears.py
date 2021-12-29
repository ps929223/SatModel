'''
연별 생성된 Coprenicus와 조사원 해양관측부이의 해류 자료를 비교하는 코드
- 월 단위 비교
제작: 전호군
부서: 해양빅데이터센터
초안: 2021.09.28
최근업뎃: 2021.11.15
'''

import pandas as pd
import numpy as np
from Lib.lib_os import *
from Lib.lib_math import *
import cv2
from matplotlib import gridspec
import os


def draw(DF, datatype):

    try:
        U_MSE = rmse(DF[datatype+'_U_Model'],DF[datatype+'_U_Obs'])
    except:
        U_MSE = np.nan

    try:
        V_MSE = rmse(DF[datatype+'_V_Model'],DF[datatype+'_V_Obs'])
    except:
        V_MSE = np.nan


    limsVal={'CUR':(-1,1),'Wind':(-20,20)}
    PsnText={'CUR':(-0.75, 0.75),'Wind':(-15,15)}

    import matplotlib.pyplot as plt

    fig = plt.figure(1,figsize=(12,5))
    gs = gridspec.GridSpec(nrows=1,  # row 몇 개
                           ncols=2,  # col 몇 개
                           height_ratios=[1],
                           width_ratios=[1, 1])
    ax0 = plt.subplot(gs[0])
    ax0.scatter(DF[datatype+'_U_Model'], DF[datatype+'_U_Obs'], s=1, facecolors=None, edgecolors='tab:blue')
    plt.xlim(limsVal[datatype][0], limsVal[datatype][1])
    plt.ylim(limsVal[datatype][0], limsVal[datatype][1])
    plt.xlabel(datatype+'_U_Model[m/s]')
    plt.ylabel(datatype+'_U_Obs[m/s]')
    plt.title(datatype+'_U Vector')
    plt.grid()
    plt.text(PsnText[datatype][0],PsnText[datatype][1], 'RMSE: %.3f' % (U_MSE))
    ax1 = plt.subplot(gs[1])
    ax1.scatter(DF[datatype+'_V_Model'], DF[datatype+'_V_Obs'], s=1, facecolors=None, edgecolors='tab:blue')
    plt.xlim(limsVal[datatype][0], limsVal[datatype][1])
    plt.ylim(limsVal[datatype][0], limsVal[datatype][1])
    plt.xlabel(datatype+'_V_Model[m/s]')
    plt.ylabel(datatype+'_V_Obs[m/s]')
    plt.title(datatype+'_V Vector')
    plt.grid()
    plt.text(PsnText[datatype][0], PsnText[datatype][1], 'RMSE: %.3f' % (V_MSE))
    path_save_png = path_save_dir + '/' + datatype + '_' + areas[ii] + '_Scatter_' + '2018-2020.png'
    plt.savefig(path_save_png)
    plt.close()

    plt.figure(2, figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(pd.to_datetime(DF['Time']), DF[datatype+'_U_Model'], s=3, label=datatype+'_U_Model')
    plt.scatter(pd.to_datetime(DF['Time']), DF[datatype+'_U_Obs'], s=3, label=datatype+'_U_Obs')
    plt.ylim(limsVal[datatype][0], limsVal[datatype][1])
    plt.xlabel('Time[UTC]')
    plt.ylabel(datatype+'_U[m/s]')
    plt.title(datatype+'_U Vector')
    plt.grid()
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.scatter(pd.to_datetime(DF['Time']), DF[datatype+'_V_Model'], s=3, label=datatype+'_V_Model')
    plt.scatter(pd.to_datetime(DF['Time']), DF[datatype+'_V_Obs'], s=3, label=datatype+'_V_Obs')
    plt.ylim(limsVal[datatype][0], limsVal[datatype][1])
    plt.xlabel('Time[UTC]')
    plt.ylabel(datatype+'_V[m/s]')
    plt.title(datatype+'_V Vector')
    plt.grid()
    plt.legend(loc='upper right')
    path_save_png = path_save_dir + '/' + datatype + '_' + areas[ii] + '_Trend_' + '2018-2020.png'
    plt.savefig(path_save_png)
    plt.close()


    plt.figure(3, figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.hist(DF[datatype+'_U_Model'] - DF[datatype+'_U_Obs'],
             bins=np.linspace(limsVal[datatype][0],limsVal[datatype][1],20))
    plt.ylabel('Freq.')
    plt.title('Difference of '+datatype+' U (Model-Obs) [m/s]')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.hist(DF[datatype+'_V_Model'] - DF[datatype+'_V_Obs'],
             bins=np.linspace(limsVal[datatype][0],limsVal[datatype][1],20))
    plt.ylabel('Freq.')
    plt.title('Difference of '+datatype+' V (Model-Obs) [m/s]')
    plt.grid()
    path_save_png = path_save_dir + '/' + datatype + '_' + areas[ii] + '_Hist_' + '2018-2020.png'
    plt.savefig(path_save_png)
    plt.close()



''' Test Code '''

datatype='Wind'

path_dir='E:/20_Product/Compare/'+datatype
list_file=recursive_file(path_dir,"*.csv")

areas=np.unique([list_file[ii].split('\\')[-1].split('_')[1] for ii in range(len(list_file))])

for ii in range(len(areas)):
    path_save_dir=path_dir+'/'+areas[ii]+'/MultiYears'
    os.makedirs(path_save_dir, exist_ok=True)
    my_file = recursive_file(path_dir, datatype+'*'+areas[ii]+'*.csv')
    DF=pd.DataFrame([])
    for path_file in my_file:
        tp=pd.read_csv(path_file)
        DF=pd.concat([DF, tp], axis=0)
    path_save_csv = path_save_dir + '/'+datatype+'_'+areas[ii]+'_'+'2018-2020.csv'
    DF.to_csv(path_save_csv, index=False)
    draw(DF, datatype)
