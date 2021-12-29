def thompson_test(x):
    import numpy as np
    nn=len(x)
    mu=np.mean(x)
    ss=np.std(x)
    t1=(x-mu)/ss
    Threshold=(np.sqrt(nn-1)*t1)/(np.sqrt(nn-1-t1**2))
    return Threshold

def nan_helper(y):
    import numpy as np
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def tsclean(y, qmin, qmax):
    '''
    R의 tsclean을 구현
    y=DF[' S37Tmp01']
    '''

    import numpy as np
    from scipy import interpolate
    q01, q09 = np.nanquantile(y, [qmin, qmax])
    U = q09 + 2 * (q09 - q01)
    L = q01 - 2 * (q09 - q01)

    cond = (L < y) & (y < U)
    y[~cond]=np.nan

    return y
