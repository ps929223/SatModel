import os, sys
sys.path.append('D:/programming/Dokdo')

'''
독도부이 Daily 자료를 읽어 처리
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm


def get_vars(DF):
    DF['KST'] = pd.to_datetime(DF['KST'])

    ## 중복/유사로 인해 불필요하여 제거 필요한 변수
    vars_notneed = ['bId', 'Size', 'yymmddHHMMSS', 'ddmmyy', 'HHMMSS', 'Latitude', 'Longitude'
                    'mmddhhmmss', 'mmddhhmmss', 'mmddhhmmss.1', 'mmddhhmmss.2', 'mmddhhmmss.3',
                    'mmddhhmmss.4', 'mmddhhmmss.5']  # Plot에서 제외할 Column

    ## 불필요 변수 제거
    vars_last = list(DF.columns)
    vars_psn = ['Latitude', 'Longitude']
    vars_adcper = ['a_%02ier' % num for num in np.linspace(1, 40, 40)]  # ADCP Error
    vars_noocean = ['a_pit', 'a_rol', 'a_hdg', 'Com', 'bBP', 'bHR', 'Temp', 'in_V', 'Volt', 'w_date', 'w_time',
                    'QSP2300.1', 'QSP2300', 'L', 'MD', 'No', 'SOG', 'GT']

    vars_target = list(set(vars_last) - set(vars_psn + vars_adcper + vars_noocean + vars_notneed))

    ## 초기에 상하임계값으로 필터링할 변수
    vars_S37Con = ['S37Con%02i' % num for num in np.linspace(1, 6, 6)]  # Conductivity 번호 6개
    vars_S37Dpt = ['S37Dpt%02i' % num for num in np.linspace(1, 6, 6)]  # Depth 번호 6개
    vars_S37Tmp = ['S37Tmp%02i' % num for num in np.linspace(1, 6, 6)]  # Temperature 번호 6개
    vars_a = ['a%02i' % num for num in np.linspace(1, 40, 40)]  # ADCP 수심
    vars_a_vr = ['a_%02ivr' % num for num in np.linspace(1, 40, 40)]  # ADCP 상하유속 40개
    vars_a_ew = ['a_%02iew' % num for num in np.linspace(1, 40, 40)]  # ADCP 동서유속 40개
    vars_a_ns = ['a_%02ins' % num for num in np.linspace(1, 40, 40)]  # ADCP 북남유속 40개



    vars_thred1 = ['37S_C', '37S_T', '1WD', '2WD', 'bP', 'BP', 'BP.1', 'chl', 'HR', 'HR.1', 'Temp.1', 'Temp.2', 'a_tmp',
                   'w_FH', 'w_FP', 'w_MH', 'w_MP', 'w_ZH', 'w_ZP', 'ntu']  # 19개
    vars_thred = vars_thred1 + vars_S37Tmp + vars_S37Con + vars_S37Dpt + vars_a + vars_a_vr + vars_a_ew + vars_a_ns
    lim_thred1 = {'37S_C': (2, 6), '37S_T': (2, 30), '1WD': (0, 360), '2WD': (0, 360),
                  'bP': (990, 1040), 'BP': (990, 1040), 'BP.1': (990, 1040), 'chl': (0, 6), 'HR': (30, 100), 'HR.1': (30, 100),
                  'Temp.1': (5, 30), 'Temp.2': (5, 30), 'a_tmp': (5, 30),
                  'w_FH': (0, 8), 'w_FP': (0, 10), 'w_MH': (0, 8), 'w_MP': (0, 10), 'w_ZH': (0, 8), 'w_ZP': (0, 10),
                  'ntu': (0, 5)}

    lim_S37Con = dict()
    lim_S37Dpt = dict()
    lim_S37Tmp = dict()
    for ii in range(len(vars_S37Tmp)):
        lim_S37Con[vars_S37Con[ii]] = (15, 50)
        lim_S37Dpt[vars_S37Dpt[ii]] = (0.5, 30)
        lim_S37Tmp[vars_S37Tmp[ii]] = (3, 30)

    lim_a = dict()
    lim_a_vr = dict()
    lim_a_ew = dict()
    lim_a_ns = dict()
    for ii in range(len(vars_a_vr)):
        lim_a[vars_a[ii]] = (-100, 0)
        lim_a_vr[vars_a_vr[ii]] = (-100, 100)
        lim_a_ew[vars_a_ew[ii]] = (-750, 750)
        lim_a_ns[vars_a_ns[ii]] = (-750, 750)

    from itertools import chain
    lim_thred = dict(chain.from_iterable((d.items() for d in (lim_thred1, lim_S37Con, lim_S37Dpt, lim_S37Tmp,\
                                                              lim_a, lim_a_vr, lim_a_ew, lim_a_ns))))

    ## Kernel Smoothing 대상 변수
    vars_ksmooth = ['1WG', '1WS', '2WG', '2WS']  # 4개

    return vars_target, vars_thred, lim_thred, vars_ksmooth


def plot_Dokdobuoy(vars_target, DFrmna):
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    plt.figure(figsize=(13, 8))
    for ii in tqdm(range(len(vars_target))):

        plt.subplot(3,1,1)
        plt.scatter(DFrmna['KST'],DFrmna[vars_target[ii]].astype(float),s=2, alpha=0.5)
        plt.title(vars_target[ii])
        plt.grid()

        plt.subplot(3,1,2)
        plt.boxplot(DFrmna[vars_target[ii]][~np.isnan(DFrmna[vars_target[ii]].astype(float))], vert=False)
        plt.grid()

        plt.subplot(3,1,3)
        plt.hist(DFrmna[vars_target[ii]][~np.isnan(DFrmna[vars_target[ii]].astype(float))], bins=400)
        plt.grid()

        plt.tight_layout()

        plt.savefig(path_output_dir+'/'+vars_target[ii]+'.png')

        plt.clf()

sys.path.append('D:/programming/Dokdo')

path_input_dir='E:/Dokdo_DB/독도부이/CSV/Daily'
path_output_dir='E:/20_Product/Buoy/Dokdo'
os.makedirs(path_output_dir, exist_ok=True)

file_list=np.array(os.listdir(path_input_dir))

#### Daily 자료를 병합
DF=pd.DataFrame()
for ii in tqdm(range(2,len(file_list))):
    # ii=2
    path_input=path_input_dir + '/' + file_list[ii]
    tp = pd.read_csv(path_input)
    vars_last = np.array(list(tp.columns)) # tp의 column명
    vars_last=[vars_last[jj].replace(' ','') for jj in range(len(vars_last))] # 공백제거
    tp.columns=vars_last # tp의 column명 재설정
    DF=pd.concat([DF, tp], axis=0) # 병합

DF['KST']='20'+DF['yymmddHHMMSS'].astype(str) # 날짜 앞에 yy붙혀서 date col에 저장
DF=DF.drop(columns='yymmddHHMMSS')
DF['KST']=pd.to_datetime(DF['KST']) # datetime형태로 저장
DF=DF.rename(columns={'a_01vt':'a_01vr'}, inplace=False)

vars_target, vars_thred, lim_thred, vars_ksmooth=get_vars(DF)

DF=DF[vars_target]
cond_nan=DF==-32768
DFna=DF[~cond_nan]
DFna.reset_index(inplace=True)
DFna=DFna.drop(columns=['index'])

## 날짜중복
cond_dup=DFna['KST'].duplicated()
print('Dupilcate: '+str(sum(cond_dup)))
## 날짜중복 제거
DFna=DFna[~cond_dup]

DFna.to_csv(path_output_dir+'/Recent_'+\
              str(DFna.KST[DFna.index[0]])[:10]+'_'+\
              str(DFna.KST[DFna.index[-1]])[:10]+'.csv', index=False)



'''
특정기간에 대해 병합한 자료를 가시화
'''

import pandas as pd

path_input='E:/20_Product/Buoy/Dokdo/Recent_2020-10-16_2021-11-18.csv'
path_output_dir='E:/20_Product/Buoy/Dokdo/Summary/Original'
os.makedirs(path_output_dir, exist_ok=True)
DFna=pd.read_csv(path_input)
DFna['KST']=pd.to_datetime(DFna['KST'])

vars_target, vars_thred, lim_thred, vars_ksmooth=get_vars(DFna)

plot_Dokdobuoy(list(set(vars_target)-set(['KST'])), DFna)



'''
설정한 임계값으로 대상 변수의 이상값 제거
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import Lib.lib_outlier as out

path_input='E:/20_Product/Buoy/Dokdo/Recent_2020-10-16_2021-11-18.csv'
path_output_dir='E:/20_Product/Buoy/Dokdo/Summary/Thred'
os.makedirs(path_output_dir, exist_ok=True)

DFna=pd.read_csv(path_input)

vars_target, vars_thred, lim_thred, vars_ksmooth=get_vars(DFna)

import matplotlib.pyplot as plt
## 상하한선 변수들 처리
for ii in range(len(vars_thred)):
    # ii=0
    tp=DFna[vars_thred[ii]]
    lims=lim_thred[vars_thred[ii]]
    cond_valid= (lims[0] < tp) & (tp < lims[1])
    tp[~cond_valid] = np.nan
    DFna[vars_thred[ii]]= tp

DFna.to_csv(path_output_dir+'/Recent_'+\
              str(DFna.KST[DFna.index[0]])[:10]+'_'+\
              str(DFna.KST[DFna.index[-1]])[:10]+'_thred.csv', index=False)

plot_Dokdobuoy(list(set(vars_target)-set(['KST'])),DFna)




'''
전체변수의 Kernel 이용한 제거
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import Lib.lib_outlier as out

path_input='E:/20_Product/Buoy/Dokdo/Summary/Thred/Recent_2020-10-16_2021-11-18_thred.csv'
path_output_dir='E:/20_Product/Buoy/Dokdo/Summary/kernel'
os.makedirs(path_output_dir, exist_ok=True)

DFthred=pd.read_csv(path_input)
DFthred['KST']=pd.to_datetime(DFthred['KST'])

vars_target, vars_thred, lim_thred, vars_ksmooth=get_vars(DFthred)

vars_target2=list(set(vars_target)-set(['KST']))

DFkernel=DFthred.copy()
from scipy.ndimage import gaussian_filter1d

for ii in tqdm(range(len(vars_target2))):
    # kernel=DFkernel[vars_target2[ii]].rolling(window=100, min_periods=10, win_type='gaussian',center=True).mean(std=100)
    kernel=gaussian_filter1d(DFkernel[vars_target2[ii]], sigma=np.nanstd(DFkernel['37S_C'])*1.5)
    rr=DFthred[vars_target2[ii]]-kernel
    ORM_rr=out.ORM_IQR(rr)
    cond_nan=np.isnan(ORM_rr)
    DFkernel[vars_target2[ii]][cond_nan]=np.nan

    ## 상한/하한
    # kernel = gaussian_filter1d(DFkernel['37S_C'], sigma=np.nanstd(DFkernel['37S_C'])*1.5)
    # plt.scatter(DFthred['KST'],DFkernel['37S_C'], s=1, alpha=.5, label='Original')
    # plt.plot(DFkernel['KST'], kernel, alpha=.5, label='G.Smooth', c='red')
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.xlabel('KST')
    # plt.ylabel('Conductivity[mS/cm]')

    ## 잔차제거
    # rr = DFkernel['37S_C'] - kernel
    # ORM_rr=out.ORM_IQR(rr)
    # cond_diff=ORM_rr!=rr
    # plt.scatter(DFkernel['KST'],rr, s=1, alpha=.5, label='Normal')
    # plt.scatter(DFkernel['KST'][cond_diff],rr[cond_diff], s=1, c='r', alpha=1, label='Outlier')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('KST')
    # plt.ylabel('Residual[mS/cm]')
    # plt.ylim(-0.05,0.05)

    ## 최종결과
    # plt.scatter(DFkernel['KST'],DFkernel['37S_C'],s=1)
    # plt.grid()
    # plt.xlabel('KST')
    # plt.ylabel('Conductivity[mS/cm]')



DFkernel.to_csv(path_output_dir+'/Recent_'+\
              str(DFkernel['KST'][DFkernel.index[0]])[:10]+'_'+\
              str(DFkernel['KST'][DFkernel.index[-1]])[:10]+'_kernel.csv', index=False)

plot_Dokdobuoy(list(set(vars_target2)-set(['KST'])),DFkernel)


'''
변수마다 결측수 계산
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import Lib.lib_outlier as out

path_input='E:/20_Product/Buoy/Dokdo/Summary/kernel/Recent_2020-10-16_2021-11-18_kernel.csv'
path_output_dir='E:/20_Product/Buoy/Dokdo/Summary/misscount'
os.makedirs(path_output_dir, exist_ok=True)
DF=pd.read_csv(path_input)
DF['KST']=pd.to_datetime(DF['KST'])
vars_target, vars_thred, lim_thred, vars_ksmooth=get_vars(DF)

vars_target2=list(set(vars_target)-set(['KST']))

TimeInt=np.arange(len(DF))

misscount=pd.DataFrame()
from datetime import timedelta as td
for ii in range(len(vars_target2)):
    cond_nan=np.isnan(DF[vars_target2[ii]])
    tt=np.array(TimeInt[~cond_nan])
    tt1=np.hstack([0, tt[:-1]])
    diff=tt-tt1
    count, xx=np.histogram(diff,
                           bins=np.array([0,3,6,144, 144*2,144*4,144*7,144*14,144*30,144*600])) # 단위: 10분
    xx=xx[1:]
    misscount['duration']=[np.nan]*len(xx)
    misscount['10min']=[np.nan]*len(xx)
    misscount[vars_target2[ii]] = count
    misscount['duration'] = xx * td(minutes=10)
    misscount['10min'] = xx

misscount.to_csv(path_output_dir + '/Recent_' + \
                str(DF.KST[DF.index[0]])[:10] + '_' + \
                str(DF.KST[DF.index[-1]])[:10] + '_'+ \
                'misscount.csv', index=False)


'''
변수별 비교
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path_DForig='E:/20_Product/Buoy/Dokdo/Recent_2020-10-16_2021-11-18.csv'
path_DFthred='E:/20_Product/Buoy/Dokdo/Summary/thred/Recent_2020-10-16_2021-11-18_thred.csv'
path_DFkernel='E:/20_Product/Buoy/Dokdo/Summary/kernel/Recent_2020-10-16_2021-11-18_kernel.csv'

DForig=pd.read_csv(path_DForig)
DForig['KST']=pd.to_datetime(DForig['KST'])
DFthred=pd.read_csv(path_DFthred)
DFthred['KST']=pd.to_datetime(DFthred['KST'])
DFkernel=pd.read_csv(path_DFkernel)
DFkernel['KST']=pd.to_datetime(DFkernel['KST'])

vars_target, vars_thred, lim_thred, vars_ksmooth=get_vars(DForig)

vars_target2=list(set(vars_target)-set(['KST']))
var='37S_C'

plt.figure()
plt.subplot(1,3,1)
plt.scatter(DForig['KST'],DForig[var], s=1, alpha=.5)
plt.hlines(lim_thred[var][0],xmin=DForig['KST'][DForig.index[0]],xmax=DForig['KST'][DForig.index[-1]], colors='r')
plt.hlines(lim_thred[var][1],xmin=DForig['KST'][DForig.index[0]],xmax=DForig['KST'][DForig.index[-1]], colors='r')
plt.grid()
plt.xlabel('KST')
plt.ylabel('Conductivity[mS/cm]')

plt.subplot(1,3,2)
plt.scatter(DFthred['KST'],DFthred[var], s=1, alpha=1, label='Original', c='tab:blue')
plt.scatter(DFkernel['KST'],DFkernel[var], s=1, alpha=1, label='G.Smooth', c='tab:red')
plt.legend()
plt.ylim(0,8)
plt.grid()
plt.xlabel('KST')
plt.ylabel('Conductivity[mS/cm]')

plt.subplot(1,3,3)
plt.scatter(DFkernel['KST'],DFkernel[var], s=1, alpha=.5)
plt.ylim(0,8)
plt.grid()

plt.tight_layout()

DFset=[DForig, DFthred, DFkernel]
labels=['Orig','Thred','Kernel']
for ii in range(len(DFset)):
    plt.scatter(DFset[ii]['KST'], DFset[ii][var], s=1, alpha=0.5, labels=labels[ii])


