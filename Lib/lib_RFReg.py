
def find_best_estimator(train_input, test_input, train_target, test_target):
    '''
    https://riverzayden.tistory.com/14
    '''
    from sklearn import ensemble
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import matplotlib.pyplot as plt

    mseOos = []
    nTreeList = range(50, 500, 10)
    for iTrees in nTreeList:
        depth = None
        maxFeat = 4  # 조정해볼 것
        RFModel = ensemble.RandomForestRegressor(n_estimators=iTrees,
                                                     max_depth=depth, max_features=maxFeat,
                                                     oob_score=False, random_state=531)
        RFModel.fit(train_input, train_target)
        # 데이터 세트에 대한 MSE 누적
        prediction = RFModel.predict(test_input)
        mseOos.append(mean_squared_error(test_target, prediction))
    print("MSE")
    print(mseOos)
    #
    # plt.plot(nTreeList, mseOos)
    # plt.xlabel('Number of Trees in Ensemble')
    # plt.ylabel('Mean Squared Error')
    # plt.grid()
    best_n=nTreeList[int(np.where(mseOos==min(mseOos))[0])]
    # plt.scatter(best_n,min(mseOos),edgecolors='r', facecolors='w')
    # plt.text(best_n,min(mseOos),best_n)
    # plt.tight_layout()
    # # 피처 중요도 도표 그리기
    # featureImportance = RFModel.feature_importances_
    #
    #
    # # 가장 높은 중요도 기준으로 스케일링
    # featureImportance = featureImportance / featureImportance.max()
    # sorted_idx = np.argsort(featureImportance)
    # barPos = np.arange(sorted_idx.shape[0]) + .5
    # plt.barh(barPos, featureImportance[sorted_idx], align='center')
    # plt.yticks(barPos, np.array(test_input.keys())[sorted_idx])
    # plt.xlabel('Variable Importance')
    # plt.grid()
    # plt.tight_layout()

    return best_n, nTreeList, mseOos



def RF(train_input, test_input, train_target, test_target,
       max_depth=5, random_state=531, n_estimators=150):
    '''
    https://riverzayden.tistory.com/14
    '''
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import matplotlib.pyplot as plt

    regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state,
                                 n_estimators=n_estimators)
    regr.fit(train_input, train_target)
    prediction = regr.predict(test_input)
    print('MSE: '+ str(mean_squared_error(test_target, prediction, squared=False)))

    return regr, prediction