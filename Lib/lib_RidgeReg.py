def RidgeReg(alpha, train_input, test_input, train_target, test_target):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from sklearn.preprocessing import PolynomialFeatures
    poly=PolynomialFeatures(include_bias=False)
    poly.fit(train_input)
    train_poly=poly.transform(train_input)
    print(train_poly.shape)
    poly.get_feature_names()
    test_poly=poly.transform(test_input)

    from sklearn.preprocessing import StandardScaler
    ss= StandardScaler()
    ss.fit(train_poly)
    train_scaled=ss.transform(train_poly)
    test_scaled=ss.transform(test_poly)

    from sklearn.linear_model import Ridge
    ridge=Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)

    ## 훈련된 Ridge Regression으로 예측하고, 실제 test값과 비교
    Result=pd.DataFrame({'actual':test_target, 'pred':ridge.predict(test_scaled)})

    ## Acutal과 predict의 2D Histgram
    my_cmap = mpl.cm.get_cmap('viridis')
    my_cmap.set_under('w')


    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist2d(test_target,ridge.predict(test_scaled), bins=(100,100), range=[[0, 1],[0, 1]],  cmap=my_cmap, vmin=0.1, vmax=50)
    plt.grid()
    plt.xlabel('Actual [mm m-3]', fontsize=14)
    plt.ylabel('Pred [mm m-3]', fontsize=14)

    plt.subplot(1,2,2)
    plt.hist2d(test_target,ridge.predict(test_scaled), bins=(100,100), range=[[0.1, 0.4],[0.1, 0.4]], cmap=my_cmap, vmin=0.1, vmax=50)
    plt.grid()
    plt.xlabel('Actual [mm m-3]', fontsize=14)

    cb = plt.colorbar(format='%i')
    cb.set_label("Count of Chl-a point", size=14)
    cb.ax.tick_params(labelsize=12)

    plt.tight_layout()

    from sklearn.metrics import mean_squared_error
    from statsmodels.robust.scale import mad
    import numpy as np
    nan_cond = np.isnan(ridge.predict(test_scaled))
    RMSE = mean_squared_error(np.array(ridge.predict(test_scaled))[~nan_cond], test_target[~nan_cond],
                              squared=False)  # False: RMSE, True: MSE
    MAD = mad(np.array(ridge.predict(test_scaled))[~nan_cond] - test_target[~nan_cond])
    print('RMSE: %.4f, MAD: %.4f' % (RMSE, MAD))

    return Result, ridge, train_scaled, test_scaled, ss


def RidgeReg_find_alpha(alpha_list, train_input, test_input, train_target, test_target):
    '''
    alpha_list = [1e3, 1e4, 1e5, 1e6, 1e7]
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    print(train_poly.shape)
    poly.get_feature_names()
    test_poly = poly.transform(test_input)

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(train_poly)
    train_scaled = ss.transform(train_poly)
    test_scaled = ss.transform(test_poly)

    from sklearn.linear_model import Ridge
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)

    train_score = []
    test_score = []
    for alpha in alpha_list:
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_scaled, train_target)
        train_score.append(ridge.score(train_scaled, train_target))
        test_score.append(ridge.score(test_scaled, test_target))


    diff = np.abs(np.array(train_score) - np.array(test_score))
    cond = diff == min(diff)
    alpha_list=np.array(alpha_list)
    best_alpha=alpha_list[cond]
    print('Best Alpha: %i' % best_alpha)


    plt.plot(np.log10(alpha_list), train_score, label='Train')
    plt.plot(np.log10(alpha_list), test_score, label='Test')

    concat_traintest = np.concatenate([train_score, test_score])
    plt.plot(np.log10([best_alpha, best_alpha]), [np.min(concat_traintest), np.max(concat_traintest)], '--',
             label='Best')
    plt.text(np.log10(best_alpha),np.mean([np.min(concat_traintest), np.max(concat_traintest)]),'a= %i' % int(best_alpha))

    plt.grid()
    plt.legend()

    plt.xlabel('alpha [log10]')
    plt.ylabel(r'R$^2$')

    plt.tight_layout

    return train_score, test_score, alpha_list, best_alpha