'''
Copernicus와 VBD자료의 상관성
2021.12.17
'''

import netCDF4 as nc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

VBD_datetime=pd.read_csv('E:/20_Product/FishingBoatConcentration/data/VBDdatetime.csv')
path_source_dir='E:/20_Product/FishingBoatConcentration/data'
file_name='FBC_0p05.nc'
path_corr_dir='E:/20_Product/FishingBoatConcentration/data/corr'
path_scatter_dir='E:/20_Product/FishingBoatConcentration/data/scatter'
path_hist_dir='E:/20_Product/FishingBoatConcentration/data/hist'
path_data_dir='E:/20_Product/FishingBoatConcentration/data'

data=nc.Dataset(path_source_dir+'/'+file_name)

date = np.array(data['time'])
mesh_lon = np.array(data['longitude'][:])
mesh_lat = np.array(data['latitude'][:])
bath = np.array(data['bath'][:])
time = np.array(data['time'][:])
vbd = np.array(data['vbd'][:])
chl = np.array(data['chl'][:])
fe = np.array(data['fe'][:])
no3 = np.array(data['no3'][:])
nppy = np.array(data['nppy'][:])
o2 = np.array(data['o2'][:])
ph = np.array(data['ph'][:])
phyc = np.array(data['phyc'[:]])
po4 = np.array(data['po4'][:])
si = np.array(data['si'][:])
spco2 = np.array(data['spco2'][:])
sst = np.array(data['sst'][:])
sla = np.array(data['sla'][:])

others = {'chl':chl, 'fe':fe, 'no3':no3, 'nppy':nppy, 'o2':o2,
          'ph':ph, 'phyc':phyc, 'po4':po4, 'si':si, 'spco2':spco2,
          'sst':sst, 'sla':sla}
others_key=list(others.keys())

lims= {'vbd':[0,20], 'bath':[-1128, 0], 'chl':[0,2.5], 'fe':[0,0.004], 'no3':[0,5], 'nppy':[0,50], 'o2':[200,280],
          'ph':[8,8.2], 'phyc':[1,10], 'si':[2,12], 'spco2':[28,50],
          'sst':[12,28], 'sla':[-.1,.4]}

month_str=np.linspace(1,12,12)
month_str=['%02i' % month for month in month_str]

thred_ships=0 # 선박척수 임계값

grid_y, grid_x=bath.shape

for mm in range(len(month_str)):
    # mm=0
    date_M = np.unique(time[np.array(['2020-%s-' % month_str[mm] in name for name in time])])
    corr_mat=np.zeros((len(date_M),len(others)+1,len(others)+1))
    corr_mat[:]=np.nan

    p_date=[]
    p_lon = []
    p_lat = []
    p_bath = []
    p_vbd = []
    p_chl = []
    p_fe = []
    p_no3 = []
    p_nppy = []
    p_o2 = []
    p_ph = []
    p_phyc = []
    p_po4 = []
    p_si = []
    p_spco2 = []
    p_sst = []
    p_sla = []

    DF=pd.DataFrame()
    for ii in range(len(date_M)):
        # ii=0
        cond_date=date_M[ii]==time
        theday_vbd=vbd[cond_date,:,:]
        true_idx=np.where(cond_date==True)[0][0]
        for jj in range(sum(cond_date)):
            # jj=0
            now_vbd=theday_vbd[jj,:,:].reshape(grid_y,grid_x)
            cond_psn = now_vbd > thred_ships  # 선박 10척 이상인 셀만 취급
            p_date.extend([date_M[ii]] * len(mesh_lon[cond_psn].flatten()))
            p_lon.extend(mesh_lon[cond_psn].flatten())
            p_lat.extend(mesh_lat[cond_psn].flatten())
            p_bath.extend(bath[cond_psn].flatten())
            p_vbd.extend(now_vbd[cond_psn].flatten())
            p_chl.extend(chl[true_idx, cond_psn].flatten())
            p_fe.extend(fe[true_idx, cond_psn].flatten())
            p_no3.extend(no3[true_idx, cond_psn].flatten())
            p_nppy.extend(nppy[true_idx, cond_psn].flatten())
            p_o2.extend(o2[true_idx, cond_psn].flatten())
            p_ph.extend(ph[true_idx, cond_psn].flatten())
            p_phyc.extend(phyc[true_idx, cond_psn].flatten())
            p_si.extend(si[true_idx, cond_psn].flatten())
            p_spco2.extend(spco2[true_idx, cond_psn].flatten())
            p_sst.extend(sst[true_idx, cond_psn].flatten())
            p_sla.extend(sla[true_idx, cond_psn].flatten())

    DF = pd.DataFrame({'date':p_date, 'lon':p_lon, 'lat': p_lat, 'vbd':p_vbd, 'bath': p_bath, 'chl':p_chl, 'fe':p_fe,
                       'no3':p_no3, 'nppy':p_nppy, 'o2':p_o2, 'ph':p_ph, 'phyc':p_phyc, 'si':p_si, 'spco2':p_spco2,
                       'sst':p_sst, 'sla':p_sla})
    DF_keys=list(DF.keys())

    # ## Cell 선박척수 Histogram: Threshold 결정에 참고
    # xx=np.quantile(np.log10(DF.vbd), np.linspace(0,1,20))
    # plt.hist(np.log10(DF.vbd),bins=xx)
    # plt.xticks(ticks=xx, labels=np.quantile(DF.vbd, np.linspace(0,1,20)).astype(int))
    # plt.grid()
    # plt.xlabel('VBD')
    # plt.ylabel('Frequency')
    # os.makedirs(path_hist_dir, exist_ok=True)
    # plt.savefig(path_hist_dir+'/hist_cell_'+date_M[0][:7]+'.png')
    # plt.close()

    DF3=pd.DataFrame()
    ## 2D hist
    for ii in range(4,len(DF.columns)):
        # ii=5
        tp=DF[DF_keys[ii]]
        cond_nan=np.isnan(tp)
        v_vbd=np.array(p_vbd)[~cond_nan]
        v_tp=tp[~cond_nan]
        bins_vbd=np.floor(np.linspace(lims['vbd'][0],lims['vbd'][-1],20))
        bins_item=np.linspace(lims[DF_keys[ii]][0],lims[DF_keys[ii]][-1],20)

        DF2 = pd.DataFrame()
        # for jj in range(len(bins_vbd)-1):
        #    # jj=0
        #    cond_range1=(bins_vbd[jj] < DF.vbd) & (DF.vbd < bins_vbd[jj+1]) # 환경정보 도수분포표의 한 단계 범위
        #    if sum(cond_range1)!=0:
        #        q1, q2 = np.nanquantile(tp[cond_range1], [.5, .9])  # 뽑힌 VBD 값들의 상위 80% 100%
        #        cond_range2=(q1 < tp) & (tp < q2) # 전체 중에서 위 조건 만족하는 범위
        #        DF2=pd.concat([DF2, DF[cond_range2]], axis=0)


        for jj in range(len(bins_item)-1):
            # jj=4
            cond_range1=(bins_item[jj] < tp) & (tp < bins_item[jj+1]) # 환경정보 도수분포표의 한 단계 범위
            vbd_range=DF.vbd[cond_range1] # 해당 파트의 VBD를 뽑음
            if len(vbd_range)!=0:
                ## quantile 기준
                # q1, q2=np.nanquantile(vbd_range, [.75, .95]) # 뽑힌 VBD 값들의 상위 80% 100%
                # cond_range2=(q1 < DF.vbd) & (DF.vbd < q2) # 전체 중에서 위 조건 만족하는 범위
                # cond = cond_range1 & cond_range2 # 위 조건 1,2를 동시에 만족
                # DF2=pd.concat([DF2, DF[cond]], axis=0)
                # ## 지정 선박척수 기준
                vmax=np.max(vbd_range)
                vmin=vmax-3
                cond_range2 = (vmin <= DF.vbd) & (DF.vbd <= vmax)
                cond=cond_range1 & cond_range2
                DF2=pd.concat([DF2, DF[cond]], axis=0)

        DF3 = pd.concat([DF3, DF2], axis=0)

        hist, vbd_edge, tp_edge = \
            np.histogram2d(DF2.vbd,DF2[DF_keys[ii]], bins=(bins_vbd,bins_item))


        hist[hist==0]=np.nan
        plt.figure(2)
        plt.pcolor(vbd_edge,tp_edge, hist.T, vmin=0, vmax=25)
        plt.colorbar()
        plt.xlim(lims['vbd'])
        plt.ylim(lims[DF_keys[ii]])
        plt.xlabel('VBD')
        plt.ylabel(DF_keys[ii])
        plt.grid()
        path_hist_dir = path_data_dir+'/hist/thred' + str(thred_ships)
        os.makedirs(path_hist_dir, exist_ok=True)
        plt.savefig(path_hist_dir+'/hist_VBD_'+DF_keys[ii]+'_'+date_M[0][:7]+'.png')
        plt.clf()



    #
    ## Correlation 계산
    ## Correlation
    corr=DF3.drop(columns=['date','lon','lat']).corr(method='pearson')
    corr.index=corr.columns
    path_corr_dir=path_data_dir+'/corr/thred'+str(thred_ships)
    os.makedirs(path_corr_dir, exist_ok=True)
    corr.to_csv(path_corr_dir+'/corr_'+date_M[0][:7]+'.csv', index=True)
    #
    ## Correlation +/-
    plt.figure(3, figsize=(10,8))
    plt.pcolor(np.array(corr), vmin=-1, vmax=1, cmap='RdBu_r')
    ticks = np.arange(0.5,len(corr.keys())+0.5,1)
    plt.xticks(ticks=ticks, labels=corr.keys(), fontsize=12)
    plt.yticks(ticks=ticks, labels=corr.keys(), fontsize=12)
    corr_f = np.array(corr).flatten()
    mesh_grid=np.meshgrid(ticks, ticks)
    for kk in range(len(mesh_grid[0].flatten())):
        # kk=0
        plt.text(mesh_grid[0].flatten()[kk]-0.3,mesh_grid[1].flatten()[kk]-0.1,round(corr_f[kk],1), fontsize=12, va='center')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path_corr_dir+'/corr_'+date_M[0][:7]+'.png')
    plt.clf()
    #
    ## Correlation abs
    plt.figure(4, figsize=(10,8))
    r2=DF3.drop(columns=['date','lon','lat']).corr(method='pearson')**2
    plt.pcolor(np.abs(np.array(r2)), vmin=0, vmax=1, cmap='RdBu_r')
    plt.xticks(ticks=ticks, labels=corr.keys(), fontsize=12)
    plt.yticks(ticks=ticks, labels=corr.keys(), fontsize=12)
    r2_f = np.array(r2).flatten()
    for kk in range(len(mesh_grid[0].flatten())):
        # kk=0
        plt.text(mesh_grid[0].flatten()[kk] - 0.3, mesh_grid[1].flatten()[kk] - 0.1, round(r2_f[kk], 1), fontsize=12,
                 va='center')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path_corr_dir+'/r2_'+date_M[0][:7]+'.png')
    plt.clf()
    #
    #
    ## Scatter
    plt.figure(5)
    for ii in range(3,len(DF.columns)):
        plt.scatter(p_vbd,DF[DF_keys[ii]], alpha=.7, s=8)
        plt.xlim(lims['vbd'])
        plt.ylim(lims[DF_keys[ii]])
        plt.xlabel('VBD')
        plt.ylabel(DF_keys[ii])
        plt.grid()
        path_scatter_dir = path_data_dir+'/scatter/thred' + str(thred_ships)
        os.makedirs(path_scatter_dir, exist_ok=True)
        plt.savefig(path_scatter_dir+'/scatter_VBD_'+DF_keys[ii]+'_'+date_M[0][:7]+'.png')
        plt.clf()

    DF3=DF3.sort_index()

    dup_cond=DF3.index.duplicated()
    DF3=DF3[~dup_cond]
    DF3=DF3.reset_index(inplace=False)
    DF3=DF3.drop(columns='index')



    # DF['Class']=np.nan

    # ## 5 classes
    # cond_LL= DF.vbd < 3
    # cond_L = (3 <= DF.vbd) & (DF.vbd < 5)
    # cond_I = (5 <= DF.vbd) & (DF.vbd < 10)
    # cond_H = (10 <= DF.vbd) & (DF.vbd < 15)
    # cond_HH = (15 <= DF.vbd)
    # DF['Class'][cond_LL] = 'LL'
    # DF['Class'][cond_L] = 'L'
    # DF['Class'][cond_I] = 'I'
    # DF['Class'][cond_H] = 'H'
    # DF['Class'][cond_HH] = 'HH'

    # ## 2 classes
    # q1 = np.median(DF3.vbd)
    # cond_LL= DF3.vbd < q1
    # cond_HH = q1 <= DF3.vbd
    # DF3['Class']=np.nan
    # DF3['Class'][cond_LL] = 'LL'
    # DF3['Class'][cond_HH] = 'HH'

    # 3 classes
    # q1, q2 = np.nanquantile(DF3.vbd, [.333, .666])
    q1, q2 = 1, 4
    cond_LL= DF3.vbd <= q1
    cond_II= (q1 < DF3.vbd) & (DF3.vbd < q2)
    cond_HH = q2 < DF3.vbd
    DF3['Class']=np.nan
    DF3['Class'][cond_LL] = 'LL'
    DF3['Class'][cond_II] = 'II'
    DF3['Class'][cond_HH] = 'HH'



    ## DF 저장
    path_DF_dir=path_data_dir+'/DF/thred'+str(thred_ships)
    os.makedirs(path_DF_dir, exist_ok=True)
    DF3.to_csv(path_DF_dir+'/DF'+date_M[0][:7]+'.csv', index=False)

