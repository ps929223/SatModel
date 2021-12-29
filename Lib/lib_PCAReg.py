def find_best_n_component(train_input, train_target):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale
    from sklearn import model_selection
    from sklearn.model_selection import RepeatedKFold
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    # scale predictor variables
    pca = PCA()
    X_reduced = pca.fit_transform(scale(train_input))

    # define cross validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    regr = LinearRegression()
    mse = []

    # # Calculate MSE with only the intercept
    # score = -1 * model_selection.cross_val_score(regr,
    #                                              np.ones((len(X_reduced), 1)), train_target, cv=cv,
    #                                              scoring='neg_mean_squared_error').mean()
    # mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, len(train_input.keys())+1):
        score = -1 * model_selection.cross_val_score(
            regr, X_reduced[:, :i], train_target, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)

    best_n=int(np.where(mse==min(mse))[0]+1)

    # Plot cross-validation results
    plt.plot(mse)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('MSE')
    plt.title('chl-a')

    respon_var=np.round(pca.explained_variance_ratio_, decimals=4) * 100
    cumsum_respon_var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)

    tt=pd.DataFrame({'ResponVAR':respon_var, 'CumSumResponVAR':cumsum_respon_var}, index=list(train_input.keys())).T
    respon=tt[tt.index=='ResponVAR']
    print(tt)
    return pca, best_n, respon


def PCAReg(pca, train_input, test_input, train_target, test_target):
    from sklearn.preprocessing import scale
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from statsmodels.robust.scale import mad
    import matplotlib.pyplot as plt
    import numpy as np

    # scale the training and testing data
    X_reduced_train = pca.fit_transform(scale(train_input))
    X_reduced_test = pca.transform(scale(test_input))[:, :1]

    # train PCR model on training data
    regr = LinearRegression()
    regr.fit(X_reduced_train[:, :1], train_target)

    # calculate RMSE
    pred = regr.predict(X_reduced_test)

    cmap1 = plt.cm.get_cmap('jet')
    cmap1.set_over(color='w', alpha=None)
    cmap1.set_under(color='w', alpha=None)


    plt.figure()
    plt.hist2d(np.array(test_target), pred, bins=(100,100), range=[[0, 30],[0,30]], cmap='jet', vmin=0.01)
    plt.xlabel('Actual')
    plt.ylabel('pred')
    plt.grid()

    RMSE = mean_squared_error(test_target, pred, squared=False)  # False: RMSE, True: MSE
    # MAD = mad(np.array(test_target.astype(float)), pred)
    print('RMSE: %.3f' % RMSE)
    # print('MAD: %.3f' % MAD)

    return regr, pred


''' Test code '''
# pca=find_best_n_component(train_input, train_target)
# regr, pred=PCAReg(pca, train_input, test_input, train_target, test_target)