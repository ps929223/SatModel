

def variogram_SKG(x, y, z):
    import skgstat as skg
    import numpy as np
    import pandas as pd

    ''' Variogram 생성 '''
    x = np.array(x).reshape(len(x), 1)
    y = np.array(y).reshape(len(y), 1)
    coords=np.concatenate([x,y],axis=1)

    V=skg.Variogram(coordinates=coords, values=z, estimator='matheron', model='exponential',
                  dist_func='euclidean', bin_func='even', normalize=False, fit_method='trf',
                  fit_sigma=None, use_nugget=True, maxlag=None, bandwidth='q33', n_lags=20, verbose=False)

    range, sill, nugget=V.parameters
    print('range: %.4f, sil: %.4f, nugget: %.4f' % (range, sill, nugget))

    return V

def OK_SKG(mesh_lon, mesh_lat, V):
    import skgstat as skg

    ''' Ordinary Kriging (by ScikitGstat)'''
    ok = skg.OrdinaryKriging(V, min_points=5, max_points=15, mode='exact')
    krig_chl = ok.transform(mesh_lon.flatten(), mesh_lat.flatten()).reshape(mesh_lon.shape)
    return krig_chl


def compare_pred2actual(pred, actual):
    import numpy as np
    from sklearn.metrics import mean_squared_error, median_absolute_error
    pred=np.array(pred)
    actual=np.array(actual)

    nan_cond=np.isnan(pred) | np.isnan(actual)
    RMSE = mean_squared_error(pred[~nan_cond],actual[~nan_cond], squared=False) # False: RMSE, True: MSE
    MAD = median_absolute_error(pred[~nan_cond],actual[~nan_cond])
    print('RMSE: %.4f, MAD: %.4f' % (RMSE, MAD))

    return RMSE, MAD

