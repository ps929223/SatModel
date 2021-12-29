'''
###############
PCA 선박척수 예측
###############
'''


def PCA_Pred(path_DF_dir,DF_names,cols):
    '''
    path_DF_dir='E:/20_Product/FishingBoatConcentration/data/DF/thred0'
    DF_names=['DF2020-%s.csv' % month for month in month_str]
    cols=['bath','no3','si','sla','chl','sst','ph']
    '''
    import Lib.lib_PCAReg as PR

    ## PCA 설명력 검증
    for ii in range(len(DF_names)):
        # ii=0
        path_DF_file=path_DF_dir+'/'+DF_names[ii]

        DF=pd.read_csv(path_DF_file)
        DF=DF.dropna()
        train_input=DF[cols]
        train_target=DF['vbd']

        pca, best_n, respon = PR.find_best_n_component(
            train_input=train_input
            , train_target=train_target)

        chosen_vars=cols[:int(best_n)]

        train_input, test_input, train_target, test_target \
            = train_test_split(DF[chosen_vars], DF['vbd'], test_size=.3, random_state=42)
        #    =train_test_split(pd.concat([DF['chl']/DF['fe'],DF['bath']/DF['nppy']],axis=1),
        #    DF['vbd'], test_size=.3, random_state=42)




        regr, pred = PR.PCAReg(pca,
                               train_input=train_input,
                               test_input=test_input,
                               train_target=train_target,
                               test_target=test_target)


'''
###############
RF 선박척수 예측
###############
'''

def RFR_month(path_DF_dir,DF_name,cols):
    import Lib.lib_RFReg as RFR
    import Lib.lib_CompareActualPred as CAP
    import os
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    ## RF 선박척수 설명력 검증

    path_perform_dir='E:/20_Product/FishingBoatConcentration/data/perform/monthly'
    os.makedirs(path_perform_dir, exist_ok=True)

    path_DF_file=path_DF_dir+'/'+DF_name

    DF = pd.read_csv(path_DF_file)
    DF = DF.dropna()

    train_input, test_input, train_target, test_target\
        =train_test_split(DF[cols].astype(float), DF['vbd'].astype(int), test_size=.3, random_state=42)

    best_n, nTreeList, mseOos=RFR.find_best_estimator(train_input, test_input, train_target, test_target)

    regr, prediction=RFR.RF(train_input, test_input, train_target, test_target,
       max_depth=10, random_state=531, n_estimators=best_n)

    rmse=CAP.hist2d(actual=test_target, pred=prediction, lims=[1, 8], n_divide=8, vmin=1, vmax=300, log2=True)
    plt.savefig(path_perform_dir+'/'+DF_names[:-4]+'.png')
    plt.clf()
    return rmse

def RFR_date(DF_train,DF_test, cols,target_date_str):
    # target_date_str='2020-01-01'
    import Lib.lib_RFReg as RFR
    import Lib.lib_CompareActualPred as CAP
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    path_perform_dir='E:/20_Product/FishingBoatConcentration/data/perform/daily'
    os.makedirs(path_perform_dir, exist_ok=True)

    DF_train = DF_train.dropna()
    DF_test = DF_test.dropna()

    train_input = DF_train[cols].astype(float)
    train_target = DF_train['vbd'].astype(float)
    test_input = DF_test[cols].astype(float)
    test_target = DF_test['vbd'].astype(float)

    best_n, nTreeList, mseOos = RFR.find_best_estimator(train_input, test_input, train_target, test_target)

    regr, pred = RFR.RF(train_input, test_input, train_target, test_target,
                              max_depth=10, random_state=531, n_estimators=best_n)

    rmse = CAP.hist2d(actual=test_target, pred=pred, lims=[1, 10], n_divide=10, vmin=1, vmax=100, log2=True)
    plt.savefig(path_perform_dir+'/'+target_date_str+'.png')
    plt.clf()

    return rmse


'''
Test Code
'''
import numpy as np
import pandas as pd
import datetime as dt

path_DF_dir='E:/20_Product/FishingBoatConcentration/data/DF/thred0'

month_str=np.linspace(1,12,12)
month_str=np.array(['%02i' % month for month in month_str])
DF_names=np.array(['DF2020-%s.csv' % month for month in month_str])

cols=['bath','no3','si','sla','chl','sst','ph']

target_date=pd.date_range('2020-01-01','2020-12-31')
target_date_strs=target_date.astype(str)

target_month_strs=np.array([target_date_strs[ii].split('-')[1] for ii in range(len(target_date_strs))])

days_before_start=30
days_before_end=5
days_before_diff=days_before_start-days_before_end
rmse_DF=np.zeros((days_before_diff,2))
rmse_DF[:]=np.nan
rmse_DF=pd.DataFrame(rmse_DF, columns=['date','rmse'])

#for ii in range(days_before_start, len(target_date_strs)-days_before_end):
for ii in range(49, len(target_date_strs) - days_before_end):
    # ii=30
    ## Train 데이터 날짜구간 지정
    train_end=(target_date[ii]-dt.timedelta(days=days_before_end)).strftime(format='%Y-%m-%d')
    train_start=(target_date[ii]-dt.timedelta(days=days_before_start)).strftime(format='%Y-%m-%d')
    train_dates=pd.date_range(train_start,train_end).astype(str)
    train_start_DF_name=DF_names[train_start.split('-')[1]==month_str][0]
    train_end_DF_name = DF_names[train_end.split('-')[1] == month_str][0]

    ## Train 데이터 편집
    train_DF1=pd.read_csv(path_DF_dir+'/'+train_start_DF_name)
    train_DF2=pd.read_csv(path_DF_dir+'/'+train_end_DF_name)
    train_pDF=pd.concat([train_DF1, train_DF2], axis=0)
    train_pDF=train_pDF[train_pDF.duplicated()].sort_values(by='date').reset_index(inplace=False)
    train_DF=pd.DataFrame()
    for jj in range(len(train_dates)):
        tp=train_pDF[train_dates[jj]==train_pDF.date]
        train_DF=pd.concat([train_DF, tp], axis=0)
    del(train_DF1, train_DF2, train_pDF)

    ## Test 데이터
    test_DF_name = DF_names[target_date_strs[ii].split('-')[1] == month_str][0]
    test_DF=pd.read_csv(path_DF_dir+'/'+test_DF_name)
    test_DF=test_DF[target_date_strs[ii]==test_DF.date]

    rmse_DF['date'][ii]=target_date_strs
    try:
        rmse_DF['rmse'][ii]=RFR_date(train_DF, test_DF, cols, target_date_strs[ii])
    except:
        continue



for ii in range(len(month_str)):
    # ii=0
    datesInMonth=target_date_strs[target_month_strs==month_str[ii]]
    rmses_daily = np.zeros((len(target_date_strs), 1))
    rmses_daily[:] = np.nan
    for jj in range(len(datesInMonth)):
        # jj=0
        print(datesInMonth[jj])
        rmses_daily[jj]=RFR_date(path_DF_dir, DF_names[ii], cols, datesInMonth[jj])
    rmses_daily=pd.DataFrame(rmses_daily, columns=['RMSE'], index=datesInMonth)
    rmses_daily.to_csv('E:/20_Product/FishingBoatConcentration/data/perform/daily/RMSE_'+DF_names[ii])