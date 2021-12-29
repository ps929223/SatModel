
def gen_Dokdo_mesh():
    import Lib.Map as Map
    import numpy as np

    coord=Map.sector()['Dokdo']
    lons=np.linspace(coord[0],coord[1],100)
    lats=np.linspace(coord[2],coord[3],100)
    mesh_lon, mesh_lat = np.meshgrid(lons,lats)
    return mesh_lon, mesh_lat



def add_xy_colum(mesh_lon, mesh_lat, DF):
    '''
    DF에 새로운 x, y 행을 추가하여
    개별 Point 클라우드에서 가장 가까운 격자의 idx 를 반환
    DF=krig_test
    DF=np.array(krig_test['chl-a'])
    '''
    import Lib.lib_Geo as Geo
    import numpy as np

    idx_x = []
    idx_y = []
    vmax=len(DF)
    for ii in np.arange(vmax):
        #print(ii)
        # ii=2
        idx = Geo.find_nearst_idx(mesh_lon, mesh_lat, np.array(DF.lon)[ii], np.array(DF.lat)[ii])
        idx_x.append(int(idx[1]))
        idx_y.append(int(idx[0]))

    DF['x'] = idx_x
    DF['y'] = idx_y
    return DF


def variogram_SKG(krig_train,target_date_str):
    import os
    import skgstat as skg
    import numpy as np
    import matplotlib.pyplot as plt

    ''' Variogram 생성 '''
    path_out_dir = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data/Variogram_SKG'
    os.makedirs(path_out_dir, exist_ok=True)
    coords=np.array(krig_train[['lon','lat']]) # valid: train, invalid: test

    V=skg.Variogram(coordinates=coords, values=krig_train['chl-a'], estimator='matheron', model='exponential',
                  dist_func='euclidean', bin_func='even', normalize=False, fit_method='trf',
                  fit_sigma=None, use_nugget=True, maxlag=None, bandwidth='q33', n_lags=20, verbose=False)

    # V=skg.DirectionalVariogram(coordinates=coords, values=krig_train['chl-a'], estimator='matheron', model='exponential',
    #                            dist_func='euclidean', bin_func='even', normalize=False, fit_method='trf',
    #                            fit_sigma=None, directional_model='triangle', azimuth=0, tolerance=45.0,
    #                            bandwidth='q33', use_nugget=False, maxlag=.5, n_lags=10, verbose=False)

    range, sill, nugget=V.parameters
    print('range: %.4f, sil: %.4f, nugget: %.4f' % (range, sill, nugget))
    V.plot()
    # plt.title('range: %.4f, sil: %.4f, nugget: %.4f' % (range, sill, nugget))
    print(V.describe())
    print(V)

    plt.tight_layout()
    plt.savefig(path_out_dir + '/Dokdo_CHL_VariogramSKG_' + target_date_str + '.png')
    plt.clf()

    return V


def krig_SKG(mesh_lon, mesh_lat, V,target_date_str):
    import Lib.lib_KAGIS_draw as Kd
    import skgstat as skg
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    path_out_dir = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data/Variogram_SKG'
    os.makedirs(path_out_dir, exist_ok=True)
    ''' Ordinary Kriging (by ScikitGstat)'''

    coordinates=pd.DataFrame({'lon':mesh_lon.flatten(), 'lat':mesh_lat.flatten()})
    coordinates=np.array(coordinates)

    ok = skg.OrdinaryKriging(V, min_points=5, max_points=15, mode='exact')

    krig_chl = ok.transform(coordinates[:,0], coordinates[:,1]).reshape(mesh_lon.shape)
    Kd.pcolor(mesh_lon,mesh_lat,krig_chl)
    plt.tight_layout()
    plt.savefig(path_out_dir + '/Dokdo_CHL_KrigSKG_' + target_date_str + '.png')
    plt.clf()
    return krig_chl

def krig_pykrig(mesh_lon, mesh_lat, krig_train):
    ''' Ordinary Kriging (by pykrig)'''
    from pykrige.ok import OrdinaryKriging
    import matplotlib.pyplot as plt
    import Lib.lib_KAGIS_draw as Kd
    import numpy as np

    OK = OrdinaryKriging(krig_train['lon'],
        krig_train['lat'],
        krig_train['chl-a'],
        variogram_model="spherical", # gaussian, linear, spherical, exponential
        # variogram_parameters = {'range':0.0586,'sill': 0.0127 , 'nugget' : 0.0193,  'n_lags':20},
        verbose=True,
        enable_plotting=True,
        coordinates_type="euclidean",
        nlags=20)

    plt.figure()
    print('Sill: %.4f, Range: %.4f, Nugget: %.4f'
          % (OK.variogram_model_parameters[0], OK.variogram_model_parameters[1],OK.variogram_model_parameters[2]))

    pred, Sigma2 = OK.execute("grid", np.unique(mesh_lon), np.unique(mesh_lat))
    Kd.pcolor(mesh_lon, mesh_lat, np.array(pred).T)

    print("Predicted_Value:    ", np.array(pred))
    print("Sigma2:   ", np.array(Sigma2))


def compare_result2test(ddm_lon,ddm_lat,krig_chl,Test,target_date_str):
    import Lib.lib_KAGIS_draw as Kd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    ''' Test 데이터와 Kring 결과를 비교 '''
    path_out_dir = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data/KrigCompare'
    os.makedirs(path_out_dir, exist_ok=True)
    # krig_test = Test
    krig_test=add_xy_colum(ddm_lon,ddm_lat,Test)

    ext_krig_chl=[]
    vmax=len(krig_test)
    for ii in np.arange(vmax):
        ext_krig_chl.append(krig_chl[np.array(krig_test.y)[ii],np.array(krig_test.x)[ii]])

    from sklearn.metrics import mean_squared_error
    from statsmodels.robust.scale import mad
    nan_cond=np.isnan(ext_krig_chl) | np.isnan(krig_test['chl-a'])
    RMSE = mean_squared_error(np.array(ext_krig_chl)[~nan_cond], krig_test['chl-a'][~nan_cond], squared=False) # False: RMSE, True: MSE
    MAD = mad(np.array(ext_krig_chl)[~nan_cond]-krig_test['chl-a'][~nan_cond])
    # plt.hist(np.array(ext_krig_chl)[~nan_cond], krig_test['chl-a'][~nan_cond],bin=(np.linspace(0,1,100),np.linspace(0,1,100)))
    print('RMSE: %.4f, MAD: %.4f' % (RMSE, MAD))

    x=np.array(krig_test['chl-a'])
    y=np.array(ext_krig_chl)
    fig= plt.figure(200,figsize=(10,4))
    Kd.hist2d(x,y)
    plt.suptitle('Compare_PredActual_'+target_date_str, fontsize=15)
    plt.tight_layout()
    plt.savefig(path_out_dir+'/PredActual_'+target_date_str+'.png')
    plt.close()

    return RMSE, MAD



def match_feature_target(DF_feature, DF_target):
    '''
    최근접거리를 계산해서 Matching함

    DF_feature = new_DF
    DF_target = CHL
    '''
    import numpy as np
    import pandas as pd
    import Lib.lib_Geo as Geo


    SST=[];SLA=[];Wind_U=[];Wind_V=[];CUR_U=[];CUR_V=[]
    vmax=len(DF_target)
    for ii in np.arange(vmax):
        # ii=0
        idx = Geo.find_nearst_idx(np.array(DF_feature['mesh_lon']), np.array(DF_feature['mesh_lat']),
                              np.array(DF_target.lon)[ii], np.array(DF_target.lat)[ii])[0]
        SST.append(float(DF_feature['SST'][idx]))
        SLA.append(float(DF_feature['SLA'][idx]))
        Wind_U.append(float(DF_feature['Wind_U'][idx]))
        Wind_V.append(float(DF_feature['Wind_V'][idx]))
        CUR_U.append(float(DF_feature['CUR_U'][idx]))
        CUR_V.append(float(DF_feature['CUR_V'][idx]))

    DF_match=pd.DataFrame({'SST':SST, 'SLA':SLA, 'Wind_U':Wind_U, 'Wind_V':Wind_V,'CUR_U':CUR_U, 'CUR_V':CUR_V,
                           'chl-a':DF_target['chl-a'], 'lon':DF_target.lon, 'lat':DF_target.lat})

    return DF_match


def outliers_idx_iqr(data):
    import numpy as np

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3-q1
    lower_bound=q1-(iqr*1.5)
    upper_bound=q3+(iqr*1.5)
    return np.where((data > upper_bound)|(data < lower_bound))

def outliers_bool_iqr(data):
    import numpy as np
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3-q1
    lower_bound=q1-(iqr*1.5)
    upper_bound=q3+(iqr*1.5)
    return (np.array(data) > upper_bound)|(np.array(data) < lower_bound)


def Ridge(Train_NA, Test, Test_NA, date_str):
    import os

    path_alpha_dir = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data/alpha'
    path_MLcompare_dir = 'D:/30_Conference/2021-11_KAGIS(JEJU)/data/MLCompare'
    os.makedirs(path_alpha_dir, exist_ok=True)
    os.makedirs(path_MLcompare_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    import Lib.lib_RidgeReg as RR
    import pandas as pd
    import numpy as np

    ## Train과 Test 세트 구분
    train_input=Train_NA[['SST', 'SLA', 'Wind_U', 'Wind_V', 'CUR_U', 'CUR_V']]
    test_input=Test_NA[['SST', 'SLA', 'Wind_U', 'Wind_V', 'CUR_U', 'CUR_V']]
    train_target=Train_NA['chl-a']
    test_target=Test_NA['chl-a']

    ## Ridge Regression의 시작
    alpha_list=alpha_list = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

    ## 제어값 alpha에 따른 R2 그래프를 작도하고, 그 중 최적 alpha를 찾음
    train_score, test_score, alpha_list, best_alpha \
        = RR.RidgeReg_find_alpha(alpha_list, train_input, test_input, train_target, test_target)
    plt.savefig(path_alpha_dir+'/alphacurve_'+date_str+'.png')
    plt.close()

    ## 최적 alpha를 이용해 Ridge Regression 수행해고 Actual과 Pred 그래프
    Result, ridge, train_scaled, test_scaled, ss = RR.RidgeReg(best_alpha, train_input, test_input, train_target, test_target)
    plt.savefig(path_MLcompare_dir+'/PredAcutalGraph_'+date_str+'.png')
    plt.close()

    ## Actual과 Pred 값
    DF=pd.concat([Test_NA[['mesh_lon','mesh_lat']],Result], axis=1)
    DF.to_csv(path_MLcompare_dir+'/PredAcutal_'+date_str+'.csv', index=False)

    pred=ridge.predict(test_scaled)
    Test_Pred=Test['chl-a']
    Test_Pred[~np.isnan(Test_Pred)]=pred

    ## 실제데이터와 예측데이터 비교
    idx=np.array(test_input.index)
    plt.figure(20,figsize=(12,4))
    plt.subplot(1,2,1)
    plt.scatter(DF.mesh_lon,DF.mesh_lat,c=DF.actual,s=8, vmin=0, vmax=.5, cmap='turbo', label='Actual')
    cb = plt.colorbar(label="Chl-a [mg m-3]", shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%.1f')
    plt.title('Acutal')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(131.4,132.6)
    plt.ylim(36.9, 37.6)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(DF.mesh_lon,DF.mesh_lat,c=DF.pred,s=8, vmin=0, vmax=.5, cmap='turbo', label='Pred')
    cb = plt.colorbar(label="Chl-a [mg m-3]", shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%.1f')
    plt.title('Pred')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(131.4,132.6)
    plt.ylim(36.9, 37.6)
    plt.grid()

    plt.tight_layout()

    plt.savefig(path_MLcompare_dir+'/PredAcutalMap_'+date_str+'.png')
    plt.close()

    ## RMSE
    from sklearn.metrics import mean_squared_error
    RMSE = mean_squared_error(ridge.predict(test_scaled), test_target, squared=False) # False: RMSE, True: MSE

    ## MAD
    from statsmodels.robust.scale import mad
    MAD = mad(ridge.predict(test_scaled)-test_target)

    ## RR 점수
    RR_score=ridge.score(test_scaled, test_target)



    return RR_score, RMSE, MAD




