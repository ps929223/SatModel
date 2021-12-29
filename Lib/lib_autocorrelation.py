import cv2
import netCDF4 as nc
import numpy as np


def autocorr(x):
    result = np.correlate(x.flatten(), x.flatten())
    return result[result.size/2:]


def covariance(x, y):
    '''
    https://stackabuse.com/covariance-and-correlation-in-python/
    '''
    # Finding the mean of the series x and y
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)
    # Subtracting mean from the individual elements
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = np.sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))]) # 분자
    denominator = len(x)-1 # 분모
    cov = numerator/denominator
    return cov

def correlation(x, y):
    '''
    https://stackabuse.com/covariance-and-correlation-in-python/
    '''
    # Finding the mean of the series x and y
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y)/float(len(y))
    # Subtracting mean from the individual elements
    sub_x = [i-mean_x for i in x]
    sub_y = [i-mean_y for i in y]
    # covariance for x and y
    numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    # Standard Deviation of x and y
    std_deviation_x = sum([sub_x[i]**2.0 for i in range(len(sub_x))])
    std_deviation_y = sum([sub_y[i]**2.0 for i in range(len(sub_y))])
    # squaring by 0.5 to find the square root
    denominator = (std_deviation_x*std_deviation_y)**0.5 # short but equivalent to (std_deviation_x**0.5) * (std_deviation_y**0.5)
    cor = numerator/denominator
    return cor


def uclid_dist(lon1, lat1, lon2, lat2):
    import numpy as np
    dist=np.sqrt((lon2-lon1)**2+(lat2-lat1)**2)
    return dist


def DistInv(lons, lats):

    w=np.zeros((len(lons),len(lons)))
    w[w==0]=np.nan

    ## 역거리값으로 가중치를 계산
    for ii in range(len(lons)):
        w[:,ii]=1/uclid_dist(lons[ii],lats[ii],lons,lats)
        w[ii,ii]=0

    ## inf는 나중에 연산에 영향 미치므로 nan으로 수정
    w[np.isinf(w)]=np.nan

    return w



def moranI(w,zs):
    import numpy as np

    var = np.nanvar(zs)
    sum_w = np.nansum(w)

    nominator=[]
    # for ii in range(len(zs)):
    #     for jj in range(len(zs)):
    #         nominator.append(w[ii,jj]*(zs[ii]-np.nanmean(zs))*(zs[jj]-np.nanmean(zs)))

    right=(zs-np.mean(zs)).reshape([1,len(zs)])
    left=(zs-np.mean(zs)).reshape([len(zs),1])

    nominator = w * np.dot(left,  right)

    nominator=np.nansum(nominator)

    Is = nominator / var / sum_w
    print('Moran`s I: '+str(round(Is,6)))


    ## null hypothesis -- source: Wiki
    N = len(w)

    # Expectation
    Ei = -1 / (N - 1)  # E(I) under null hypothesis
    # print('Expectation: ' + str(round(Ei, 6)))

    # Variance
    S1=1/2*np.sum((w+w.T)**2)
    S2=np.sum((np.sum(w,axis=0)+np.sum(w,axis=1))**2)
    S3=(1/N*(np.sum((zs-np.mean(zs))**4)))/(1/N*(np.sum((zs-np.mean(zs))**2))**2)
    S4=(N**2-3*N+3)*S1-N*S2+3*np.sum(w)**2
    S5=(N**2-N)*S1-2*N*S2+6*np.sum(w)**2

    Vi=(N*S4-S3*S5)/((N-1)*(N-2)*(N-3)*np.sum(w)**2)-Ei**2
    # print('Variation: ' + str(round(Vi, 6)))
    # Standard deviation
    Si=np.sqrt(Vi)
    # print('Std: ' + str(round(Si, 6)))
    return Is

def gearyC(w,zs):
    import numpy as np

    var = np.nanvar(zs)
    sum_w = np.nansum(w)

    nominator=[]
    # for ii in range(len(zs)):
    #     for jj in range(len(zs)):
    #         nominator.append(w[ii,jj]*(zs[ii]-np.nanmean(zs))*(zs[jj]-np.nanmean(zs)))

    right = np.zeros((len(zs), len(zs)))

    for ii in range(len(zs)):
        right[:, ii] = (zs[ii] - zs) ** 2

    nominator = np.nansum(w * right)

    C = nominator / var / (sum_w*2)
    print('Geary`s C: '+str(round(C,6)))

    return C


def globalGetisG(w,zs):
    import numpy as np

    N=len(zs)
    right=zs.reshape([1,N])
    left=right.T

    G_norm=w * np.dot(left,  right).astype(float)
    G_denorm=np.dot(left,  right).astype(float)

    for ii in range(len(G_norm)):
        G_norm[ii,ii]=np.nan
        G_denorm[ii,ii]=np.nan
        w[ii,ii]=np.nan

    Gstar = np.divide(np.nansum(G_norm),np.nansum(G_denorm))
    E = np.nansum(w)/(N*(N-1))
    V = np.nanmean(np.divide(G_norm, G_denorm)**2) - np.nanmean(np.divide(G_norm, G_denorm))**2

    ZG = (Gstar - E)/V

    print('Global Getis-Ord`s ZG: '+str(round(ZG,6)))

    return ZG


def localGetisG(w,zs):
    import numpy as np

    right=zs.reshape([1,len(zs)])
    left=right.T

    zs_dot = np.dot(left, right)
    for ii in range(len(zs)):
        zs_dot[ii,ii]=np.nan
        w[ii,ii]=np.nan

    nominator = w * zs_dot
    denominator = zs_dot

    Gi=np.nansum(np.divide(nominator,denominator),axis=0)

    N=len(zs)
    Eg=np.sum(w)/(N(N-1))

    print('Local Getis-Ord`s G: '+str(round(G,6)))

    return Gi

