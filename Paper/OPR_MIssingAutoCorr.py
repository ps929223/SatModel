
'''
----------------------------------------------
DM 1,3,5,7,9에 대한 자동상관계수를 구해서 CSV로 저장
----------------------------------------------
'''

import numpy as np
import pandas as pd
import netCDF4 as nc
import Lib.lib_autocorrelation as AC
import matplotlib.pyplot as plt
import os
import Lib.lib_GOCI1 as G1

date_list=pd.date_range('2018-03-08','2021-03-22').astype(str)

n_days=['DM1','DM3','DM5','DM7','DM9']
avoid_Date=['2019-05-08','2019-05-09','2019-05-10','2019-05-11','2019-05-12',
            '2019-05-13','2019-06-22','2021-01-26']


data = nc.Dataset('E:/20_Product/CHL_DM/CHL_DM3/2018/03/GOCI1_DM3_CHL_2018-03-21_res100.nc')
lons, lats, zs, lm=np.array(data['lon']).flatten(), np.array(data['lat']).flatten(), np.array(data['chl']).flatten(), np.array(data['lm']).flatten()
meshlon, meshlat, meshz, meshlm=np.array(data['lon']), np.array(data['lat']), np.array(data['chl']), np.array(data['lm'])
lm = lm == 2 # 2: land, 0: sea

## NonObsArea 관측외 영역
flag=G1.read_GOCI1_flag()
NonObsArea=flag.copy()
NonObsArea[NonObsArea>-2.6e+36]=np.nan
NonObsArea=~np.isnan(NonObsArea)
print('NonObsArea:'+str(round(np.sum(NonObsArea)/(NonObsArea.shape[0]*NonObsArea.shape[1]),4)))

import cv2
NonObsArea = cv2.resize(src=NonObsArea.astype(float), dsize=(100,100), interpolation=cv2.INTER_CUBIC)
NonObsArea[NonObsArea>=0.5]=1
NonObsArea[NonObsArea<0.5]=0
NonObsArea=NonObsArea.astype(bool)

MaskIn=~(NonObsArea.flatten()|lm)

plt.figure(1,figsize=(9,12))
for ii in range(len(date_list)):
    # ii=428
    # ii=0
    print('___________________')
    print(date_list[ii])
    print('___________________')

    if date_list[ii] in avoid_Date:
        print('Skip due to No data for the date '+date_list[ii])
        continue
    else:

        # ii=0
        yyyy = date_list[ii][:4]
        mm = date_list[ii][5:7]
        dd = date_list[ii][8:10]

        path_file=['E:/20_Product/CHL_DM/CHL_DM1/'+yyyy+'/'+mm+'/GOCI1_DM_CHL_'+date_list[ii]+'_res100.nc',
                   'E:/20_Product/CHL_DM/CHL_DM3/'+yyyy+'/'+mm+'/GOCI1_DM3_CHL_'+date_list[ii]+'_res100.nc',
                   'E:/20_Product/CHL_DM/CHL_DM5/'+yyyy+'/'+mm+'/GOCI1_DM5_CHL_'+date_list[ii]+'_res100.nc',
                   'E:/20_Product/CHL_DM/CHL_DM7/'+yyyy+'/'+mm+'/GOCI1_DM7_CHL_'+date_list[ii]+'_res100.nc',
                   'E:/20_Product/CHL_DM/CHL_DM9/'+yyyy+'/'+mm+'/GOCI1_DM9_CHL_'+date_list[ii]+'_res100.nc']

        moran=[];geary=[];getis=[]
        for jj in range(len(path_file)):
            # jj=4
            data=nc.Dataset(path_file[jj])
            zs = np.array(data['chl']).flatten()
            meshz = np.array(data['chl'])

            # ## Case 1: 관측값이 있는 것들의 샘플
            # cond = np.isnan(zs)
            # n_lons = lons[~cond]
            # n_lats = lats[~cond]
            # n_zs = zs[~cond]

            ## Case 2: 관측값 없는 것과 있는 것에 대해 Flag를 달자
            ## MaskOut 마스킹
            n_zs = zs[MaskIn] # MaskIn데이터만 사용
            n_lons = lons[MaskIn]
            n_lats = lats[MaskIn]

            cond = np.isnan(n_zs)
            n_zs[cond] = 1 # 구름
            n_zs[~cond] = 0 # 관측

            missratio=len(n_zs[cond])/len(n_zs)

            ## 역거리가중
            w=AC.DistInv(n_lons,n_lats)
            np.sum(w)
            mI=AC.moranI(w,n_zs)
            gC=AC.gearyC(w,n_zs)
            gG=AC.globalGetisG(w,n_zs)
            moran.append(mI)
            geary.append(gC)
            getis.append(gG)

            DMnumber = path_file[jj].split('/')[-1].split('_')[1]
            meshz[meshlm==2]=50
            cond_nan=np.isnan(meshz)
            meshz[cond_nan] = np.nan
            plt.subplot(3,2,jj+1)
            plt.pcolor(meshlon,meshlat,meshz)
            plt.clim(0,100)
            plt.title('Auto Correlation ' + date_list[ii] + ' '+ DMnumber)
            plt.text(113, 47, ' MR: ' + str(round(missratio, 2)), fontsize=12)
            plt.text(113, 45, ' mI: ' + str(round(mI, 4)), fontsize=12)
            plt.text(113, 43, ' gC: ' + str(round(gC, 4)), fontsize=12)
            plt.text(113, 41, ' gG: ' + str(round(gG, 4)), fontsize=12)

            plt.grid()
            path_out_dir = 'E:/20_Product/GOCI1/CHL/Autocorrelation/AC'
            os.makedirs(path_out_dir, exist_ok=True)

        plt.tight_layout()
        plt.savefig(path_out_dir+'/AC_'+date_list[ii]+'.png')
        plt.clf()


        # DF = pd.DataFrame({'Getis': getis})
        DF=pd.DataFrame({'Moran': moran, 'Geary': geary, 'Getis': getis})
        DF.index=['DM1','DM3','DM5','DM7','DM9']

        DF.to_csv(path_out_dir+'/AC_'+date_list[ii]+'.csv')



'''
----------------------------------------------
날짜별 CSV로 저장한 자기상관계수를 3차원 자료로 읽어서 전체 CSV로 저장
----------------------------------------------
'''

import os
import numpy as np
import pandas as pd
from Lib.lib_os import *
from tqdm import tqdm

path_input_dir='E:/20_Product/GOCI1/CHL/Autocorrelation'
file_list=np.array(recursive_file(path_input_dir, 'AC_*.csv'))
file_list=file_list[np.array(['indices' not in name for name in file_list])]

sample=pd.read_csv(file_list[0])
idx=list(sample['Unnamed: 0'])
col=list(sample.columns[1:])

dataset=np.zeros((len(file_list),len(idx),len(col)))
dataset[dataset==0]=np.nan

date_list=[]
for ii in tqdm(range(len(file_list))):
    # ii=0
    date_list.append(file_list[ii].split('_')[-1].split('.csv')[0])
    dataset[ii,:,:]=np.array(pd.read_csv(file_list[ii]))[:,1:] # 1행의 Index는 제외하였음

import matplotlib.pyplot as plt
n_days=['1','3','5','7','9']
DF=pd.DataFrame({'Date':date_list})
DF['Date']=pd.to_datetime(DF['Date'])

for ii in range(len(n_days)):
    DF['MoranI_DM'+n_days[ii]]=dataset[:,ii,0]
for ii in range(len(n_days)):
    DF['GearyC_DM'+n_days[ii]]=dataset[:,ii,1]
for ii in range(len(n_days)):
    DF['GetisGZ_DM'+n_days[ii]]=dataset[:,ii,2]

DF.to_csv(path_input_dir+'/AC_indices.csv', index=False)



'''
----------------------------------------------
전체 CSV를 가시화
----------------------------------------------
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


target='Cloud' # 'Chl'

path_input_dir='E:/20_Product/GOCI1/CHL/Autocorrelation'
Index_Names=['MoranI','GearyC','GetisGZ']
DF=pd.read_csv(path_input_dir+'/AC_indices_'+target+'.csv')
DF['Date']=pd.to_datetime(DF['Date'])

ylims_Cloud={'MoranI':[0,0.3],'GearyC':[0.7,1.1],'GetisGZ':[0,1.5]}
ylims_Chl={'MoranI':[-0.2,0.6],'GearyC':[0.2,1.4],'GetisGZ':[-1,5]}
n_days=['1','3','5','7','9']

plt.figure(figsize=(10, 4))
for Index_Name in Index_Names:
    for ii in range(len(n_days)):
        plt.scatter(DF['Date'],DF[Index_Name+'_DM'+n_days[ii]], s=15, label='DM'+n_days[ii], alpha=0.7)
        if target=='Cloud':
            plt.ylim(ylims_Cloud[Index_Name])
        elif target=='Chl':
            plt.ylim(ylims_Chl[Index_Name])
    plt.xlabel('Date')
    plt.ylabel('Index '+Index_Name)
    plt.grid()
    plt.title(Index_Name+' '+target)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_input_dir+'/AC_'+Index_Name+'_'+target+'.png')
    plt.clf()


xlims={'Cloud':[-0.1,1.2],'Chl':[-.7,1.7]}

DF=pd.read_csv(path_input_dir+'/AC_indices_'+target+'.csv')
tt=pd.melt(DF)
tt=tt[tt.variable!='Date']
plt.figure(figsize=(10, 4))
sns.boxplot(x='value', y='variable', data=tt)
plt.xlim(xlims[target][0],xlims[target][1])
plt.grid()
plt.tight_layout()
plt.savefig(path_input_dir+'/AC_Boxplot_'+target+'.png')


'''
Autocorrelation with Test Image
단순한 이미지들로 자기상관관계 지수의 역할 확인
'''
import os
import numpy as np
import pandas as pd
import cv2
import Lib.lib_autocorrelation as AC
from Lib.lib_os import *
path_img_dir='E:/20_Product/GOCI1/CHL/Autocorrelation/TestImage'
file_list=recursive_file(path_img_dir, '*.png')

x=np.arange(0,100,1).flatten()
y=x.copy()
x, y = np.meshgrid(x,y)
x= x.flatten()
y= y.flatten()

w=AC.DistInv(x, y)

import matplotlib.pyplot as plt
for ii in range(len(file_list)):
    # ii=5
    img=cv2.imread(file_list[ii], cv2.IMREAD_GRAYSCALE)
    img=img.flatten()

    # DF=pd.DataFrame({'x':x,'y':y,'z':img}).to_csv(path_img_dir+'/'+file_list[ii][:-4]+'.csv', index=False)
    print('--------------------')
    print(file_list[ii])
    print('--------------------')
    # zs=img
    AC.moranI(w, img)
    AC.gearyC(w, img)
    AC.globalGetisG(w, img)

