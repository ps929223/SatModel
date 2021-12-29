'''
동해 Chl ML예측을 위한 데이터셋 생성
관련 lib 파일
lib_KAGIS_createML_DS.py
'''
import numpy as np
import pandas as pd
from Lib.lib_os import *
import Lib.lib_KAGIS_createML_DS as Kcm
from tqdm import tqdm

path_out_dir='E:/20_Product/CHL_Krig'

date_list=np.array(recursive_file(path_out_dir+'/CHL_Orig','Dokdo*_CHL_Mask.csv'))
date_list=[date_list[ii].split('\\')[-1].split('_')[1] for ii in range(len(date_list))]


## 환경요소만 추출
for ii in range(len(date_list)):
    print(date_list[ii])
    Kcm.create_KAGIS_env_dataset(date_list[ii])

## Chl과 환경요소가 더해진 Train/Test 생성
Kcm.create_KAGIS_ML_dataset(path_out_dir)