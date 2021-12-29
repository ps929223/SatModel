

def uclid_dist(lon1, lat1, lon2, lat2):
    import numpy as np
    dist=np.sqrt((lon2-lon1)**2+(lat2-lat1)**2)
    return dist

def randomIdx(lons, lats):
    import numpy as np
    d=np.zeros((len(lons),len(lons)))
    d[d==0]=np.nan

    ## 거리 계산
    for ii in range(len(lons)):
        d[:,ii]=uclid_dist(lons[ii],lats[ii],lons,lats)
    meanR=np.nanmean(np.nanmin(d, axis=1))
    S=(np.max(lons)-np.min(lons))*(np.max(lons)-np.min(lons))
    rho=np.divide(1,len(lons)/S)
    RI=2*meanR/np.sqrt(rho)
    print(round(RI,5))
    return RI

from Lib.lib_os import *
path='E:/20_Product/GOCI1/CHL/Autocorrelation/TestImage'



'''
Point Pattern with Test Image
단순한 이미지들로 자기상관관계 지수의 역할 확인
'''
import os
import numpy as np
import pandas as pd
import cv2
path_img_dir='D:/20_Product\GOCI1\CHL\Autocorrelation\TestImage'
file_list=np.array(os.listdir(path_img_dir))
file_list=file_list[np.array(['.png' in name for name in file_list])]

x=np.arange(0,100,1).flatten()
y=x.copy()
x, y = np.meshgrid(x,y)
x= x.flatten()
y= y.flatten()

RI=[]
for ii in range(len(file_list)):
    img=cv2.imread(path_img_dir+'/'+file_list[ii], cv2.IMREAD_GRAYSCALE)
    img=img.flatten()
    cond = img==0
    n_x = x[cond]
    n_y = y[cond]
    # DF=pd.DataFrame({'x':x,'y':y,'z':img}).to_csv(path_img_dir+'/'+file_list[ii][:-4]+'.csv', index=False)
    print('--------------------')
    print(file_list[ii])
    print('--------------------')
    RI.append(randomIdx(n_x, n_y))



