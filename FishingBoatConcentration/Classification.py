import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path_DF_dir='E:/20_Product/FishingBoatConcentration/data/DF/thred0'

month_str=np.linspace(1,12,12)
month_str=['%02i' % month for month in month_str]

DF_names=['DF2020-%s.csv' % month for month in month_str]

cols=['bath','chl', 'fe', 'no3', 'nppy', 'o2', 'ph', 'phyc', 'si', 'spco2','sst', 'sla']

'''
#######################
RF 선박 고/중/저밀도 분류
#######################
'''

def set_DF(path_file):
    import pandas as pd
    DF = pd.read_csv(path_file)
    DF.dropna(inplace=True)
    cond_land = DF['bath'] == 999
    DF = DF[~cond_land]

    class_names = list(set(DF.Class))
    class_count = [sum(DF.Class == name) for name in class_names]

    ## 클래스별로 실제 수 그대로 데이터 만듬
    DF_LL = DF[DF.Class == 'LL']
    DF_I = DF[DF.Class == 'II']
    DF_HH = DF[DF.Class == 'HH']

    ## 클래스별로 동일수의 데이터 만듬
    # equal_qty=min(class_count)
    # DF_LL=DF[DF.Class=='LL'][-equal_qty:]
    # DF_I=DF[DF.Class=='I'][-equal_qty:]
    # DF_HH=DF[DF.Class=='HH'][-equal_qty:]

    ## 클리스 병합
    new_DF = pd.concat([DF_LL, DF_I, DF_HH], axis=0)
    new_DF = new_DF.sort_values(by=['date', 'lon', 'lat'])
    return new_DF

def run_classification_date(new_DF,target_date_str):
    # target_date_str='2020-01-01'
    import Lib.lib_RFClass as TE
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    features = new_DF.drop(columns=['date','lon','lat','vbd','Class'])
    classes =new_DF['Class']
    cond_date=new_DF.date==target_date_str
    train_input, test_input, train_target, test_target\
        = features[~cond_date], features[cond_date],classes[~cond_date], classes[cond_date]

    rf=TE.RFC_fit(train_input, train_target)
    pred=rf.predict(test_input)

    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
    label=np.unique(train_target)
    Confusion=pd.DataFrame(confusion_matrix(test_target, pred, labels=label), columns=label, index=label)
    Confusion_dir='D:/30_Conference/2021-11_KOSOMES(PUS)/data/confusion'
    os.makedirs(Confusion_dir, exist_ok=True)
    Confusion.to_csv(Confusion_dir+'/'+target_date_str+'.csv', index=True)
    print(Confusion)
    acc=accuracy_score(test_target,pred)
    f1=f1_score(test_target,pred, average='macro')
    recall=recall_score(test_target,pred, average='macro')
    prec=precision_score(test_target,pred, average='macro')
    print('Acc: %.4f f1: %.4f recall: %.4f prec: %.4f' % (acc, f1, recall, prec))
    return rf, acc, f1, recall, prec



def run_classification_month(new_DF,target_month):
    # target_date_str='2020-01-01'
    import Lib.lib_RFClass as RFC
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    features = new_DF.drop(columns=['date','lon','lat','vbd','Class'])
    # features = features[['no3','si','sst','sla']]
    classes =new_DF['Class']

    train_input, test_input, train_target, test_target\
        =train_test_split(features, classes, test_size=.3, random_state=42, shuffle=True)

    'Test Code'
    rf=RFC.RFC_fit(train_input, train_target)
    pred=rf.predict(test_input)

    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
    label=np.unique(train_target)
    Confusion=pd.DataFrame(confusion_matrix(test_target, pred, labels=label), columns=label, index=label)
    Confusion_dir='D:/30_Conference/2021-11_KOSOMES(PUS)/data/confusion'
    os.makedirs(Confusion_dir, exist_ok=True)
    Confusion.to_csv(Confusion_dir+'/'+target_month+'.csv', index=True)
    print(Confusion)
    acc=accuracy_score(test_target,pred)
    f1=f1_score(test_target,pred, average='macro')
    recall=recall_score(test_target,pred, average='macro')
    prec=precision_score(test_target,pred, average='macro')
    print('Acc: %.4f f1: %.4f recall: %.4f prec: %.4f' % (acc, f1, recall, prec))
    return rf, acc, f1, recall, prec


from Lib.lib_os import *
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
path_DF_dir='E:/20_Product/FishingBoatConcentration/data/DF/thred0'
file_list = np.array(recursive_file(path_DF_dir, 'DF*.csv'))

n_row=[]
data=pd.DataFrame()
for ii in tqdm(range(len(file_list))):
    tp=pd.read_csv(file_list[ii])
    n_row.append(len(tp))
    data=pd.concat([data, tp], axis=0)

date_list_str=data['date'].unique()
n_row=[]
for ii in range(len(date_list_str)):
    cond_date=date_list_str[ii]==data['date']
    n_row.append(sum(cond_date))


### 1개월 테스트
perform = np.zeros((len(file_list), 5))
perform[:] = np.nan
perform = pd.DataFrame(perform, columns=['month', 'accuracy', 'f1_score', 'recall', 'precision'])
for ii in tqdm(range(len(file_list))):
    # ii=0
    new_DF=set_DF(file_list[ii])
    date_lists=np.unique(new_DF.date)
    # 1개월 테스트
    target_month_str=file_list[ii].split('\\')[-1][2:9]
    perform['month'][ii]=target_month_str
    rf, perform['accuracy'][ii], perform['f1_score'][ii],\
    perform['recall'][ii], perform['precision'][ii]\
        =run_classification_month(new_DF, target_month_str)
perform_dir='E:/20_Product/FishingBoatConcentration/data/DF/perform'
os.makedirs(perform_dir, exist_ok=True)
year_str = '2020'
perform.to_csv(perform_dir+'/Perfomance_'+year_str+'.csv', index=False)


### 날짜별 테스트
for ii in tqdm(range(len(file_list))):
    # ii=0
    new_DF=set_DF(file_list[ii])
    date_lists=np.unique(new_DF.date)
    perform=np.zeros((len(date_lists),5))
    perform[:]=np.nan
    perform=pd.DataFrame(perform, columns=['date','accuracy', 'f1_score', 'recall', 'precision'])

    # 날짜별 테스트
    for jj in range(len(date_lists)):
        # jj=0
        perform['date'][jj]=date_lists[jj]
        rf, perform['accuracy'][jj], perform['f1_score'][jj],\
        perform['recall'][jj], perform['precision'][jj]\
            =run_classification_date(new_DF, date_lists[jj])
    perform_dir='E:/20_Product/FishingBoatConcentration/data/DF/perform'
    os.makedirs(perform_dir, exist_ok=True)
    month_str=file_list[ii].split('\\')[-1].split('DF')[1].split('.csv')[0]
    perform.to_csv(perform_dir+'/Perfomance_'+month_str+'.csv', index=False)


'''
NC데이터에서 직접 데이터를 읽어, 예상 선박밀도 지도를 그리기
'''



def read_specific_date_DF(path_nc, specific_date):
    '''
    path_nc='D:/30_Conference/2021-11_KOSOMES(PUS)/data/KOSOMES_0p05.nc'
    specific_date='2020-01-05'
    '''
    import netCDF4 as nc
    import pandas as pd
    import numpy as np

    data=nc.Dataset(path_nc)
    date=data['time'][:]
    mesh_lon, mesh_lat=data['longitude'][:], data['latitude'][:]
    lon_flat, lat_flat = mesh_lon.flatten(), mesh_lat.flatten()

    cols=['chl', 'fe', 'no3', 'nppy', 'o2', 'ph', 'phyc', 'si', 'spco2','sst', 'sla']

    cond_date=date==specific_date
    cond_land = data['bath'][:, :].flatten() == 999
    DF=pd.DataFrame()
    DF['lon'] = data['longitude'][:, :].flatten()
    DF['lat'] = data['latitude'][:, :].flatten()
    DF['bath'] = data['bath'][:].flatten()
    for jj in range(0,len(cols)):
        DF[cols[jj]]=data[cols[jj]][cond_date,:,:].flatten()
    DF=DF[~cond_land]
    DF.dropna()
    return mesh_lon, mesh_lat, DF

def draw_map(DF, text):
    import matplotlib.pyplot as plt
    plt.scatter(DF.lon, DF.lat, s=2, c=DF.Class, vmin=0, vmax=3, cmap='viridis')
    plt.xlim(min(DF.lon), max(DF.lon))
    plt.ylim(min(DF.lat), max(DF.lat))
    cb = plt.colorbar()
    cb.set_ticks([1, 2, 3])
    cb.set_ticklabels(['Low', 'Medium', 'High'])
    cb.ax.tick_params(labelsize=10)
    # cb.set_label('Number of Ships', size=10)
    plt.grid()
    plt.title(text)

def draw_acutal_predict_map(specific_date):
    '''
    specific_date = '2020-01-04'
    '''
    import os
    import netCDF4 as nc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import Lib.lib_TreeEmsemble as TE


    path_nc = 'D:/30_Conference/2021-11_KOSOMES(PUS)/data/KOSOMES_0p05.nc'
    year_month=specific_date[:7]
    path_map_dir='D:/30_Conference/2021-11_KOSOMES(PUS)/data/predMap'
    os.makedirs(path_map_dir,exist_ok=True)
    path_month_DF = 'D:/30_Conference/2021-11_KOSOMES(PUS)/data/DF/thred0/'+year_month+'.csv'
    cols = ['bath','chl', 'fe', 'no3', 'nppy', 'o2', 'ph', 'phyc', 'si', 'spco2', 'sst', 'sla']


    ## Acutal

    data = nc.Dataset(path_nc)
    date = data['time'][:]
    mesh_lon, mesh_lat = data['longitude'][:], data['latitude'][:]
    cond_date = date == specific_date
    cond_land = data['bath'][:, :]
    vbd=data['vbd'][cond_date,:,:].reshape(cond_land.shape)
    Class=np.array(vbd.astype(float))
    q1, q2 = 1,4
    Class[Class==0]=0
    Class[Class==q1]=1
    Class[(q1<Class)&(Class<=q2)]=2
    Class[q2<Class]=3
    Class[cond_land==999]=np.nan

    plt.figure(1, figsize=(9,4))
    plt.subplot(1,2,1)
    plt.pcolor(mesh_lon,mesh_lat,Class, vmin=0, vmax=3, cmap='viridis')
    plt.xlim(np.min(mesh_lon), np.max(mesh_lon))
    plt.ylim(np.min(mesh_lat), np.max(mesh_lat))
    cb = plt.colorbar()
    cb.set_ticks([1, 2, 3])
    cb.set_ticklabels(['Low', 'Medium', 'High'])
    cb.ax.tick_params(labelsize=10)
    # cb.set_label('Number of Ships', size=10)
    plt.grid()
    plt.suptitle('Fishing Density Map '+specific_date)
    plt.title('Actual')

    ## Train data 준비
    path_month_DF = 'D:/30_Conference/2021-11_KOSOMES(PUS)/data/DF/thred0/DF'+year_month+'.csv'
    train_DF = pd.read_csv(path_month_DF)
    cond_date = train_DF.date == specific_date
    train_DF = train_DF[~cond_date]
    train_DF = train_DF.dropna()
    train_input, train_target = train_DF[cols], train_DF['Class']

    ## Test data 준비
    mesh_lon, mesh_lat, test_DF = read_specific_date_DF(path_nc, specific_date)
    test_DF = test_DF.dropna()

    ## Train & Test

    rf = TE.RFC_fit(train_input=train_input, train_target=train_target)

    predict = rf.predict(test_DF[cols])
    predict[predict == 'LL'] = 1
    predict[predict == 'II'] = 2
    predict[predict == 'HH'] = 3

    test_DF['Class'] = predict

    ## Test 자료
    plt.subplot(1, 2, 2)
    draw_map(test_DF, 'Prediction')
    plt.tight_layout()
    plt.savefig(path_map_dir + '/FishMap_' + specific_date + '.png')
    plt.clf()

    del rf



draw_acutal_predict_map('2020-06-24')

