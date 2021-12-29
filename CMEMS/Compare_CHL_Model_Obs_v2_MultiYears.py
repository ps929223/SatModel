'''
연별 생성된 Coprenicus와 조사원 GOCI의 Chl 자료를 비교하는 코드
- 월 단위 비교
제작: 전호군
부서: 해양빅데이터센터
초안: 2021.09.30
'''


import pandas as pd
import numpy as np
from Lib.lib_os import *
from Lib.lib_math import *
import cv2
from matplotlib import gridspec
import matplotlib.pyplot as plt
import os


def draw(DF):
    #import scipy

    DF['Time']=pd.to_datetime(DF['Time'])
    DF['sma_CHL_GOCI'] = DF['CHL_GOCI'].rolling(window=10,min_periods=2).mean()
    # scipy.stats.gaussian_kde(DF['CHL_GOCI'][DF['CHL_GOCI'].notnull()], bw_method=2, weights=None)
    try:
        CHL_MSE = rmse(DF['CHL_Model'],DF['CHL_GOCI'])
    except:
        CHL_MSE = np.nan


    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7,6))
    plt.scatter(DF['CHL_Model'], DF['CHL_GOCI'], s=1, facecolors=None, edgecolors='tab:blue')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel('CHL_Model')
    plt.ylabel('CHL_GOCI')
    plt.title('CHL[mg m^-3]')
    plt.grid()
    plt.text(1, 19, 'RMSE: %.3f' % (CHL_MSE))
    plt.savefig(path_save_png1)
    plt.clf()

    plt.figure(2, figsize=(9, 6))
    plt.plot(DF['Time'], DF['CHL_Model'], label='CHL_Model')
    plt.scatter(DF['Time'], DF['CHL_GOCI'], label='CHL_GOCI', c='tab:orange', s=3)
    plt.plot(DF['Time'], DF['sma_CHL_GOCI'], label='sma_CHL_GOCI', c='tab:green')
    plt.ylim(0,4)
    plt.xlabel('Time[UTC]')
    plt.ylabel('CHL[mg m^-3]')
    plt.title('CHL[mg m^-3]')
    plt.grid()
    plt.legend(loc='upper center')
    plt.savefig(path_save_png2)
    plt.clf()

    plt.figure(3, figsize=(7, 6))
    plt.hist(DF['CHL_Model'] - DF['CHL_GOCI'], bins=np.linspace(-1.5,1.5,20))
    plt.ylabel('Freq.')
    plt.title('Difference of CHL (Model-Obs) [mg m^-3]')
    plt.grid()
    plt.savefig(path_save_png3)
    plt.clf()



''' Test Code '''

path_dir='D:/20_Product/Compare/CHL/Ulleung_NE/Yearly'
list_file=recursive_file(path_dir,"*.csv")
path_save_dir='D:/20_Product/Compare/CHL/Ulleung_NE/MultiYears'
os.makedirs(path_save_dir, exist_ok=True)
path_save_csv=path_save_dir+'/CHL_ULGNE_2018-2020.csv'
path_save_png1=path_save_dir+'/CHL_ULGNE_Scatt_2018-2020.png'
path_save_png2=path_save_dir+'/CHL_ULGNE_Trend_2018-2020.png'
path_save_png3=path_save_dir+'/CHL_ULGNE_Histo_2018-2020.png'

DF=pd.DataFrame([])
for path_file in list_file:
    tp=pd.read_csv(path_file)
    DF=pd.concat([DF, tp], axis=0)

DF.to_csv(path_save_csv, index=False)
draw(DF)

