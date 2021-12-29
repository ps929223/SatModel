'''
결과물 가시화
'''

### 경로설정 및 라이브러리

from Lib.lib_os import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path_dir_DF='D:/30_Conference/2021-11_KOSOMES(PUS)/data/Perform'


### 일일 점수
file_list=recursive_file(path_dir_DF,'*2020-*.csv')

DF=pd.DataFrame()
for ii in range(len(file_list)):
    tp=pd.read_csv(file_list[ii])
    DF=pd.concat([DF, tp], axis=0)

DF=DF[DF.accuracy!=min(DF.accuracy)]
DF.date[DF.accuracy==min(DF.accuracy)]

DF['date']=pd.to_datetime(DF['date'])
plt.figure(figsize=(10,3))
plt.scatter(DF.date, DF.accuracy, label='Accuracy', s=7, alpha=.7)
plt.scatter(DF.date, DF.f1_score, label='F1 score', s=7, alpha=.7)
plt.grid()
plt.legend(loc='lower right')
plt.ylim(0.2,1.1)
plt.tight_layout()

np.nanmean(DF.accuracy)
np.nanmean(DF.f1_score)

### 월 점수
file_list=recursive_file(path_dir_DF,'*2020.csv')

DF=pd.DataFrame()
for ii in range(len(file_list)):
    tp=pd.read_csv(file_list[ii])
    DF=pd.concat([DF, tp], axis=0)


DF['month']=pd.to_datetime(DF['month'])
plt.figure(figsize=(10,3))
plt.plot(DF.month, DF.accuracy, 'o-', label='Accuracy')
plt.plot(DF.month, DF.f1_score, 'o-', label='F1 score')
plt.grid()
plt.legend(loc='lower right')
plt.ylim(0.2,1.1)
plt.tight_layout()

DF['f1_score'].mean()



## Decision Tree 그려내기
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np

path_DF='D:/30_Conference/2021-11_KOSOMES(PUS)/data/DF/thred0/DF2020-01.csv'
DF=pd.read_csv(path_DF)
DF=DF.dropna()
features = DF.drop(columns=['date', 'lon', 'lat', 'vbd', 'Class'])
classes = DF['Class']

np.nanmax(DF.vbd)

train_input, test_input, train_target, test_target \
    = train_test_split(features, classes, test_size=.3, random_state=42, shuffle=True)

dt=DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=list(features.keys()), class_names=np.unique(test_target),
          fontsize=12)

