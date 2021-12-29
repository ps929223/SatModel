def rmse(pred, actual):
    import numpy as np
    nominator=np.nansum((actual-pred)**2)
    cond=~np.isnan(pred) & ~ np.isnan(actual)
    denominator=sum(cond)
    rmse=np.sqrt(nominator/denominator)
    return rmse

def r2(pred, actual):
    import numpy as np
    nominator=np.nansum((actual-pred)**2)
    denominator = np.nansum((actual-np.nanmean(actual)))
    r2=1-nominator/denominator
    return r2

def r2adjust(r2):
    r2adj=1-(1-r2)*(n-1)/(n-P-1)


def twosampletest(a,b):
    import scipy.stats as stats
    # perform two sample t-test with equal variances
    result=stats.ttest_ind(a=a, b=b, equal_var=True)
    return result