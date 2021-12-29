
import pandas as pd
from Lib.lib_os import *
import numpy as np
path_pearson_dir = 'D:/30_학술대회/2021-11 한국지리정보학회(제주)/data/ORM_IQRscore/pearson'

file_list=recursive_file(path_pearson_dir, '*.csv')
date_list=[file_list[ii].split('\\')[-1].split('_')[1].split('.')[0] for ii in range(len(file_list))]

tp=pd.read_csv(file_list[0])
data=np.zeros((len(tp),len(file_list)))
data[:]=np.nan

for ii in range(len(file_list)):
    tp = pd.read_csv(file_list[ii])
    data[:,ii]=tp['chl-a']

data=data.T
DF=pd.DataFrame(data, columns=tp.columns[1:], index=date_list)

import seaborn as sns
import matplotlib.pyplot as plt
# plt.figure(figsize=(2,4))
sns.boxplot(data=DF[['SST', 'SLA', 'Wind_U', 'Wind_V', 'CUR_U', 'CUR_V']], orient='h')
plt.grid()
plt.xlabel('R')
plt.tight_layout()