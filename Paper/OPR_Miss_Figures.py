
'''
GOCI-I 영역
'''
import Lib.lib_GOCI1 as G1
import Lib.Map as Map
import matplotlib.pyplot as plt
import numpy as np
meshlon, meshlat=G1.read_GOCI1_coordnates()
coord=[np.min(meshlon), np.max(meshlon), np.min(meshlat), np.max(meshlat)]
coord_Bound=[np.floor(coord[0])-5, np.ceil(coord[1])+5, np.floor(coord[2])-5, np.ceil(coord[3])+5]

square=np.array([[coord[0],coord[2]],
                 [coord[1],coord[2]],
                 [coord[1],coord[3]],
                 [coord[0],coord[3]],
                 [coord[0],coord[2]]])


m=Map.making_map(coord=coord_Bound,map_res='h',grid_res=5)
plt.tight_layout()
xx, yy=m(square[:,0],square[:,1])
m.plot(xx,yy, c='tab:blue', label='Coverage')
m.scatter(*m(130,36), c='tab:purple', s=100, label='Center')
plt.legend(loc='upper right')


'''
Observation Count
'''
import pandas as pd
import numpy as np

path_csv='E:/20_Product\GOCI1\CHL\MissingRatio\LM이전/Count_GOCI12018-03-04-2021-03-26.csv'
DF=pd.read_csv(path_csv)
np.histogram(DF.Count, bins=np.array([7,8,9,10,11,12,13,14,15,100]))
len(DF.Date[~DF.Count.isna()])

'''
Time-series Missing Ratio Graph
'''
# img_count_visible.py 참고



'''
############################
Table Missing Ratio
hourly, 1,3,5,7,9DM
############################
'''
from scipy.ndimage import gaussian_filter as GF
from Lib.lib_os import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_input_dir='E:/20_Product/GOCI1/CHL/MissingRatio'
list_file=recursive_file(path_input_dir,pattern='*.csv')[:6]
HR1=pd.read_csv(list_file[5])
DM1=pd.read_csv(list_file[0])
DM3=pd.read_csv(list_file[1])
DM5=pd.read_csv(list_file[2])
DM7=pd.read_csv(list_file[3])
DM9=pd.read_csv(list_file[4])

DF_list=[HR1, DM1, DM3, DM5, DM7, DM9]
DF_Name=['HR1', 'DM1', 'DM3', 'DM5', 'DM7', 'DM9']

date_start='2018-03-04'
date_end='2021-03-26'


### Overall mean and variation of missing ratio
col_name=['mean','variation']
DF=np.zeros((len(DF_Name),len(col_name)))
DF[:]=np.nan
DF=pd.DataFrame(DF, columns=col_name)

for ii in range(len(DF_list)):
    DF['mean'][ii]=DF_list[ii].NaNRatio.mean()
    DF['variation'][ii]=DF_list[ii].NaNRatio.var()

DF.index=DF_Name
DF.to_csv(path_input_dir+'/DMn_allMeanVars_'+date_start+'_'+date_end+'.csv', float_format='%.3f')


### Monthly mean and variation of missing ratio

month_list=list(pd.date_range(date_start, date_end, freq='1M').strftime('%Y-%m').astype(str))
DF_mean=np.zeros((len(month_list),len(DF_Name)))
DF[:]=np.nan
DF_mean=pd.DataFrame(DF_mean, columns=DF_Name)
DF_var=DF_mean.copy()

for ii in range(len(DF_list)):
    # ii=0
    tp=DF_list[ii]
    for jj in range(len(month_list)):
        # jj=0
        cond_month=np.array([month_list[jj] in month for month in tp.Time])
        DF_mean[DF_Name[ii]][jj]=tp[cond_month].NaNRatio.mean()
        DF_var[DF_Name[ii]][jj]=tp[cond_month].NaNRatio.var()

DF_mean.index=month_list
DF_var.index=month_list
DF_mean.to_csv(path_input_dir+'/DMn_mean_'+date_start+'_'+date_end+'.csv', float_format='%.2f')
DF_var.to_csv(path_input_dir+'/DMn_var_'+date_start+'_'+date_end+'.csv', float_format='%.4f')



'''
############################
Table Autocorrelation
hourly, 1,3,5,7,9DM
############################
'''
import pandas as pd
import numpy as np

path_input_dir='E:/20_Product/GOCI1/CHL/Autocorrelation'
file_name='AC_indices_Cloud.csv'
DF=pd.read_csv(path_input_dir+'/'+file_name)

date_start='2018-03-04'
date_end='2021-03-26'
month_list=np.array(pd.date_range(date_start, date_end, freq='1M').strftime('%Y-%m').astype(str))


### Overall mean and variation of missing ratio
AC_name=['Moran','Geary','Getis']


for ii in range(len(AC_name)):
    # ii=0
    col_list=DF.columns[np.array([AC_name[ii] in name for name in DF.columns])]
    DF_mean = pd.DataFrame()
    DF_var = pd.DataFrame()
    for jj in range(len(col_list)):
        # jj=0
        DF_mean[col_list[jj]] = [np.nan] * len(month_list)
        DF_var[col_list[jj]] = [np.nan] * len(month_list)
        for kk in range(len(month_list)):
            # kk=0
            cond_month=np.array([month_list[kk] in date for date in DF.Date])
            DF_mean[col_list[jj]][kk] = DF[col_list[jj]][cond_month].mean()
            DF_var[col_list[jj]][kk] = DF[col_list[jj]][cond_month].var()
    DF_mean.index=month_list
    DF_mean.to_csv(path_input_dir + '/DMn_mean_Monthly_' + AC_name[ii]+'_'+date_start+'_'+date_end+'.csv',
                   float_format='%.2f')
    DF_var.index=month_list
    DF_var.to_csv(path_input_dir + '/DMn_var_Monthly_' + AC_name[ii] + '_' + date_start + '_' + date_end + '.csv',
                   float_format='%.4f')


Col_Names=list(DF_meanvar.columns)
for ii in range(len(Col_Names)):
    Col_Names[ii]=Col_Names[ii].replace('MoranI', 'MI')
    Col_Names[ii]=Col_Names[ii].replace('GearyC', 'GC')
    Col_Names[ii]=Col_Names[ii].replace('GetisGZ', 'GG')
DF_meanvar.columns=Col_Names
DF_meanvar.to_csv(path_input_dir+'/AC_MeanVarMonthly_'+date_start+'_'+date_end+'.csv', float_format='%.4f')

### Monthly mean and variation of missing ratio

month_list=list(pd.date_range(date_start, date_end, freq='1M').strftime('%Y-%m').astype(str))
DF_mean=pd.DataFrame()
DF_var=DF_mean.copy()


for ii in range(len(AC_name)):
    # ii=0
    col_list=DF.columns[np.array([AC_name[ii] in name for name in DF.columns])]
    DF_mean = pd.DataFrame()
    DF_var = pd.DataFrame()
    for jj in range(len(col_list)):
        # jj=0
        DF_mean[col_list[jj]] = [np.nan] * len(month_list)
        DF_var[col_list[jj]] = [np.nan] * len(month_list)
        for kk in range(len(month_list)):
            # kk=0
            cond_month=np.array([month_list[kk] in date for date in DF.Date])
            DF_mean[col_list[jj]][kk] = DF[col_list[jj]][cond_month].mean()
            DF_var[col_list[jj]][kk] = DF[col_list[jj]][cond_month].var()

DF_mean.index=month_list
DF_var.index=month_list
DF_mean.to_csv(path_input_dir+'/DMn_mean_'+date_start+'_'+date_end+'.csv', float_format='%.2f')
DF_var.to_csv(path_input_dir+'/DMn_var_'+date_start+'_'+date_end+'.csv', float_format='%.4f')