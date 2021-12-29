
def z_score(data):
    import numpy as np
    z=(data-np.nanmean(data))/np.nanstd(data)
    return z

def t_score(data):
    return 10*z_score(data)+50


def ORM_zscore(data,threshold=3):
    import numpy as np
    z = np.abs(z_score(data))
    if threshold > 4 or threshold < 0 :
        print('threhoshold must be between 0.0 and 4.0')
    else:
        cond = z > threshold
        print('Outlier Min: %.2f Max: %.2f' % (np.nanmin(data[cond]),np.nanmax(data[cond])))
        data[cond]=np.nan
        return data

def MAD(data):
    import numpy as np
    return np.median(np.abs(data-np.median(data)))

def ORM_modified_zscore(data):
    import numpy as np
    Mi=0.6745*(data-np.nanmedian(data))/MAD(data)
    cond_out=np.abs(Mi)>3.5
    data[cond_out]=np.nan
    return data

def ORM_IQRscore(data):
    import numpy as np
    q1,q3=np.nanquantile(data,[.25,.75])
    IQR=q3-q1
    cond = (data < (q1-1.5 * IQR)) | (data > (q3+1.5 * IQR))
    data[cond] = np.nan
    return data

def ORM_speckle2D(array):
    # array=n_chl
    # import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter as GF
    import numpy as np
    # Gap Filling 1)
    # import cv2
    # mask=np.isnan(array)
    # filled_array=cv2.inpaint(src=array, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # Gap Filling 2)
    from sklearn.impute import SimpleImputer as SI
    imp=SI(strategy='mean')
    filled_array = imp.fit_transform(array)
    n_array=GF(filled_array, sigma=np.nanstd(filled_array)/3)
    residual=filled_array-n_array
    cond_nan=np.isnan(ORM_IQRscore(residual))
    # cond_nan=np.isnan(ORM_modified_zscore(residual))
    ORM_array=array.copy()
    ORM_array[cond_nan]=np.nan
    return ORM_array



def ORM_IQR(data):
    import numpy as np
    Q1,Q3 = np.nanquantile(data, [0.25, 0.75])
    IQR = Q3 - Q1
    cond_Out= (data < Q1 - 1.5*IQR) |  (Q3 + 1.5*IQR < data)
    data[cond_Out]=np.nan
    return data

def despike(yi, th=1.e-8):
    '''
    Remove spike from array yi, the spike area is where the difference between
    the neigboring points is higher than th.
    '''
    import numpy as np
    y = np.copy(yi) # use y = y1 if it is OK to modify input array
    n = len(y)
    x = np.arange(n)
    c = np.argmax(y)
    d = abs(np.diff(y))
    try:
        l = c - 1 - np.where(d[c-1::-1]<th)[0][0]
        r = c + np.where(d[c:]<th)[0][0] + 1
    except: # no spike, return unaltered array
        return y

    # for fit, use area twice wider then the spike
    if (r-l) <= 3:
        l -= 1
        r += 1
        s = int(round((r-l)/2.))
        lx = l - s
        rx = r + s
        # make a gap at spike area
        xgapped = np.concatenate((x[lx:l],x[r:rx]))
        ygapped = np.concatenate((y[lx:l],y[r:rx]))
        # quadratic fit of the gapped array
        z = np.polyfit(xgapped,ygapped,2)
        p = np.poly1d(z)
        y[l:r] = p(x[l:r])
        return y




