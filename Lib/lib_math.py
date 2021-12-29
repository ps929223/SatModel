import numpy as np
def rmse(predictions, targets):
    '''
    predictions=np.array(DF['CHL_Model'])
    targets=np.array(DF['CHL_GOCI'])
    '''
    cond = ~np.isnan(predictions) & ~np.isnan(targets)
    differences = predictions[cond] - targets[cond]           #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val                                           #get the ^

def norm(src,vmin,vmax):
    vnorm=(src-vmin)/(vmax-vmin)
    return vnorm