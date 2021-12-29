'''
한국지리정보학회
위성 Chl-a 공간보간 및 예측 방법 연구
전호군
관련 lib 파일: lib_KAGIS_prediction.py
업뎃: 2021.10.31
초안: 2021.10.08
https://medium.com/@nikhil.munna001/kriging-implementation-in-python-with-source-code-541b9b5c35f
'''

'''
###########################
## Ordinary Kriging
###########################
'''

from Lib.lib_GOCI1 import *
from Lib.lib_os import *
import Lib.lib_KAGIS_prediction as Kp
import Lib.lib_KAGIS_draw as Kd

from tqdm import tqdm
import Lib.Map as Map
import Lib.lib_Geo as Geo


path_out_dir='D:/30_Conference/2021-11_KAGIS(JEJU)/data'
coord = Map.sector()['Dokdo']
dokdo_psn= Map.dokdo_psn()

target_date_str='2020-08-01'

# ORM_list=recursive_file('D:/30_Conference/2021-11_KAGIS(JEJU)/data/ORM_IQRscore/CHL_Orig','*ORM.csv')
Mask_list=recursive_file('D:/30_Conference/2021-11_KAGIS(JEJU)/data/ORM_IQRscore/CHL_Orig','*Mask.csv')
Train_list=recursive_file('D:/30_Conference/2021-11_KAGIS(JEJU)/data/MLds','Train*.csv')
Test_list=recursive_file('D:/30_Conference/2021-11_KAGIS(JEJU)/data/MLds','Test*.csv')
ddm_lon, ddm_lat=Kp.gen_Dokdo_mesh()
ranges=np.zeros(len(Train_list))
ranges[:]=np.nan
sills=ranges.copy()
nuggets=ranges.copy()
RMSEs=ranges.copy()
MADs=ranges.copy()
date_list=ranges.copy().astype(str)

for ii in tqdm(range(len(Train_list))):
    # ii=0
    target_date_str=Train_list[ii].split('\\')[-1].split('_')[1]
    date_list[ii]=target_date_str
    print(target_date_str)
    # ORM=pd.read_csv(Train_list[ii])
    # Mask=pd.read_csv(Mask_list[ii])
    # cond=ORM['chl-a']==Mask['chl-a']
    # Train=ORM[cond]
    # Test=ORM[~cond]
    Train=pd.read_csv(Train_list[ii])[['mesh_lon','mesh_lat','chl-a']]
    Train.columns=['lon','lat','chl-a']
    Train=Train[~Train['chl-a'].isna()]
    Test=pd.read_csv(Test_list[ii])[['mesh_lon','mesh_lat','chl-a']]
    Test.columns=['lon','lat','chl-a']
    Test=Test[~Test['chl-a'].isna()]

    V=Kp.variogram_SKG(Train, target_date_str)
    ranges[ii], sills[ii], nuggets[ii]= V.parameters
    print('Variogram.. Done')

    krig_chl=Kp.krig_SKG(ddm_lon, ddm_lat, V, target_date_str)
    print('Kriging.. Done')

    # krig_pykrig(ddm_lon, ddm_lat, krig_train)
    RMSEs[ii], MADs[ii]=Kp.compare_result2test(ddm_lon,ddm_lat,krig_chl,Test,target_date_str)
    print('Comparison.. Done')

    # ## 구름 가려진 부분과 Test 그림 비교
    # import matplotlib.pyplot as plt
    #
    # # jj=0
    # idx_ys=[];idx_xs=[];
    # for jj in range(len(Test)):
    #     idx_y, idx_x=Geo.find_nearst_idx(ddm_lon,ddm_lat,np.array(Test.lon)[jj],np.array(Test.lat)[jj])
    #     idx_ys.append(int(idx_y))
    #     idx_xs.append(int(idx_x))
    # Krig_Draw_DF=np.zeros((len(idx_ys),3))
    # Krig_Draw_DF[:]=np.nan
    # Krig_Draw_DF=pd.DataFrame(Krig_Draw_DF, columns=['lon','lat','chl-a'])
    # for jj in range(len(Krig_Draw_DF)):
    #     Krig_Draw_DF.lon[jj]=ddm_lon[idx_ys[jj],idx_xs[jj]]
    #     Krig_Draw_DF.lat[jj]=ddm_lat[idx_ys[jj],idx_xs[jj]]
    #     Krig_Draw_DF['chl-a'][jj]=krig_chl[idx_ys[jj],idx_xs[jj]]
    #
    # cmap1 = plt.cm.get_cmap('turbo')
    # # cmap1.set_over(color='w', alpha=None)
    # cmap1.set_under(color='w', alpha=None)
    #
    # plt.scatter(Test.lon, Test.lat, c=Test['chl-a'], s=6, vmin=0, vmax=.4, cmap='turbo')
    # plt.scatter(Krig_Draw_DF.lon, Krig_Draw_DF.lat, c=Krig_Draw_DF['chl-a'], s=6, vmin=0, vmax=.4, cmap='turbo')
    #
    # plt.xlabel('Longitude', fontsize=14)
    # plt.ylabel('Latitude', fontsize=14)
    # plt.xlim(131.4, 132.6)
    # plt.ylim(36.9, 37.6)
    # plt.grid()
    # plt.tight_layout()



import pandas as pd
DF=pd.DataFrame({'Date':date_list, 'Range':ranges, 'Sill': sills, 'Nugget': nuggets, 'RMSE':RMSEs, 'MAD':MADs})
DF.to_csv(path_out_dir+'/ResultSummary_Kriging_'+date_list[0]+'-'+date_list[-1]+'.csv', index=False)

### Cloud Ratio
ratio_cloud=np.zeros(len(Mask_list))
ratio_cloud[:]=np.nan
for ii in range(len(Mask_list)):
    # ii=0
    df=pd.read_csv(Mask_list[ii])
    ratio_cloud[ii]=sum(df['chl-a'].isna())/len(df)
DF=pd.DataFrame({'Date':date_list, 'Ratio_cloud':ratio_cloud})
DF.to_csv(path_out_dir+'/Ratio_cloud_'+date_list[0]+'-'+date_list[-1]+'.csv', index=False)



import matplotlib.pyplot as plt
ResultKrig=pd.read_csv(path_out_dir+'/ORM_IQRscore/ResultSummary_Kriging_2020-06-01-2020-08-29.csv')
RatioCloud=pd.read_csv(path_out_dir+'/Ratio_cloud_2020-06-01-2020-08-29.csv')

plt.figure()
plt.scatter(RatioCloud.Ratio_cloud,ResultKrig.RMSE, label='RMSE')
plt.scatter(RatioCloud.Ratio_cloud,ResultKrig.MAD, label='MAD')
plt.ylim(0,.08)
plt.legend()
plt.grid()
plt.xlabel('Cloud to All Ratio')
plt.ylabel('Chl-a Concentration [mg m-3]')


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(2,4))
sns.boxplot(data=ResultKrig[['RMSE','MAD']], orient='v')
plt.ylabel('Chl-a Concentration [mg m-3]')
plt.ylim(0,.08)
plt.grid()
plt.tight_layout()



'''
###########################
## Ridge Regression
###########################
'''
dir_ML_ds = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data/MLds'
path_out_dir='D:/30_Conference/2021-11_KAGIS(JEJU)/data'

from Lib.lib_os import *
import Lib.lib_KAGIS_prediction as Kp
import numpy as np
import pandas as pd
Train_list=recursive_file(dir_ML_ds,'Train*.csv')
Test_list=recursive_file(dir_ML_ds,'Test*.csv')

date_list=[Test_list[ii].split('\\')[-1].split('_')[1].split('.')[0] for ii in range(len(Test_list))]


## 데이터 읽기

RR_score=np.zeros(len(date_list))
RR_score[:]=np.nan
RMSE=RR_score.copy()
MAD=RR_score.copy()

for ii in range(len(date_list)):
    # ii=00
    Train=pd.read_csv(Train_list[ii])
    Test=pd.read_csv(Test_list[ii])
    Train_NA=Train[~np.isnan(Train['chl-a'])] ## NaN이 없는 파일
    Test_NA=Test[~np.isnan(Test['chl-a'])]
    RR_score[ii], RMSE[ii], MAD[ii]= Kp.Ridge(Train_NA, Test, Test_NA, date_list[ii])

DF=pd.DataFrame({'Date':date_list, 'RR_score':RR_score, 'RMSE':RMSE, 'MAD':MAD})
DF.to_csv(path_out_dir+'/ResultSummary_RidgeReg_'+date_list[0]+'-'+date_list[-1]+'.csv', index=False)


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=DF[['RMSE','MAD']], orient='h')
plt.xlabel('Chl-a Concentration [mg m-3]')
plt.grid()
plt.tight_layout()


import matplotlib.pyplot as plt
ResultRR=pd.read_csv(path_out_dir+'/ORM_IQRscore/ResultSummary_RidgeReg_2020-06-01-2020-08-29.csv')
RatioCloud=pd.read_csv(path_out_dir+'/Ratio_cloud_2020-06-01-2020-08-29.csv')

plt.figure()
plt.scatter(RatioCloud.Ratio_cloud,ResultRR.RMSE, label='RMSE')
plt.scatter(RatioCloud.Ratio_cloud,ResultRR.MAD, label='MAD')
plt.legend()
plt.grid()
plt.ylim(0,.08)
plt.xlabel('Cloud to All Ratio')
plt.ylabel('Chl-a Concentration [mg m-3]')


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(2,4))
sns.boxplot(data=ResultRR[['RMSE','MAD']], orient='v')
plt.ylabel('Chl-a Concentration [mg m-3]')
plt.ylim(0,.08)
plt.grid()
plt.tight_layout()






'''
### Random Forest
'''
from Lib.lib_os import *
import Lib.lib_RFReg as RFR
import numpy as np
import pandas as pd
import Lib.lib_KAGIS_draw as Kd

dir_ML_ds = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data/MLds'
path_out_dir='D:/30_Conference/2021-11_KAGIS(JEJU)/data'


Train_list=recursive_file(dir_ML_ds,'Train*.csv')
Test_list=recursive_file(dir_ML_ds,'Test*.csv')

date_list=[Test_list[ii].split('\\')[-1].split('_')[1].split('.')[0] for ii in range(len(Test_list))]

for ii in range(len(Train_list)):
    # ii=0
    Train=pd.read_csv(Train_list[ii])
    Test=pd.read_csv(Test_list[ii])
    Train_NA=Train[~np.isnan(Train['chl-a'])] ## NaN이 없는 파일
    Test_NA=Test[~np.isnan(Test['chl-a'])]

    train_input=Train_NA.drop(columns=['chl-a','mesh_lon','mesh_lat'])
    test_input=Test_NA.drop(columns=['chl-a','mesh_lon','mesh_lat'])
    train_target=Train_NA['chl-a']
    test_target=Test_NA['chl-a']

    best_n, nTreeList, mseOos=RFR.find_best_estimator(train_input, test_input,
                                              train_target, test_target)


    # best_n=210
    regr, pred = RFR.RF(train_input, test_input, train_target, test_target,
           max_depth=5, random_state=531, n_estimators=best_n)


    plt.scatter(Test_NA.mesh_lon,Test_NA.mesh_lat,c=pred,s=8, vmin=0, vmax=.5, cmap='turbo', label='Pred')
    cb = plt.colorbar(label="Chl-a [mg m-3]", shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%.1f')
    plt.title('Pred')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(131.4,132.6)
    plt.ylim(36.9, 37.6)
    plt.grid()
    plt.tight_layout()


    Kd.hist2d(Test_NA['chl-a'],pred)
    plt.tight_layout()

    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import median_absolute_error as mad

    RMSE = mean_squared_error(pred, Test_NA['chl-a'], squared=False)  # False: RMSE, True: MSE
    MAD = mad(pred, np.array(Test_NA['chl-a']))
    # sum(np.isnan(Test_NA['chl-a']))
    # sum(np.isinf(Test_NA['chl-a']))
    # sum(Test_NA['chl-a'])
    # sum(np.isnan(pred.astype(float)))
    # sum(np.isinf(pred.astype(float)))
    # sum(pred.astype(float))