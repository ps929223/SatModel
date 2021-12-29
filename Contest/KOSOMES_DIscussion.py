import pandas as pd
from Lib.lib_os import *
import numpy as np
import matplotlib.pyplot as plt

path_perform_dir='D:/30_Conference/2021-11_KOSOMES(PUS)/data/perform'
file_list=np.array(recursive_file(path_perform_dir,'*.csv'))

Data=pd.DataFrame()
for ii in range(len(file_list)):
    tp=pd.read_csv(file_list[ii])
    Data=pd.concat([Data, tp], axis=0)


Data['date']=pd.to_datetime(Data['date'])
plt.figure(figsize=(12,4))
plt.plot(Data['date'],Data['accuracy'], linestyle='-', label='Accuracy', alpha=1)
plt.plot(Data['date'],Data['f1_score'], linestyle='-', label='f1_score', alpha=1)
plt.plot(Data['date'], [Data['accuracy'].mean()]*len(Data), label='Ave.Accuracy', alpha=.7)
plt.plot(Data['date'], [Data['f1_score'].mean()]*len(Data), label='Ave.f1_score', alpha=.7)

# plt.plot(Data['date'],Data['precision'], linestyle='--', label='precision', alpha=.7)
# plt.plot(Data['date'],Data['recall'], linestyle='--', label='recall', alpha=.7)
plt.ylim(0,1)
plt.legend()
plt.grid()
plt.tight_layout()