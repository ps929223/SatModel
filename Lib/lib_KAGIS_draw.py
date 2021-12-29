
def pcolor(mesh_lon, mesh_lat, chl):
    '''
    mesh_lon, mesh_lat, chl 입력하면 그림
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    fig= plt.figure(4,figsize=(8,6))

    cmap1 = plt.cm.get_cmap('turbo')
    # cmap1.set_over(color='w', alpha=None)
    cmap1.set_under(color='w', alpha=None)

    plt.pcolor(mesh_lon, mesh_lat, chl, vmin=0.001, vmax=.4, cmap=cmap1) # gist_ncar, nipy_spectral, gnuplot2
    cb = plt.colorbar(format='%.2f')
    cb.set_label("Chl-a [mg m-3]", size=14)
    ticks=np.linspace(0, .4, 11)
    cb.set_ticks(list(ticks))
    tick_labels = ['%.2f' % num for num in ticks]
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=12)
    plt.grid()
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.xlim(131.4,132.6)
    plt.ylim(36.9,37.6)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tight_layout()

    import Lib.Map as Map
    coord=Map.dokdo_psn()
    plt.scatter(coord[0], coord[1], facecolors='w', edgecolors='r', s=100, marker='s', label='Dokdo')
    plt.legend(loc='upper right')


def QQ(DF,target_date_str):
    path_out_dir = 'D:/30_학술대회/2021-11 한국지리정보학회(제주)/data/QQ'
    import os
    os.makedirs(path_out_dir, exist_ok=True)
    ## QQ plot
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    plt.figure(2, figsize=(6,5))
    stats.probplot(DF['chl-a'], plot=plt)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_out_dir + '/Dokdo_CHL_QQ_' + target_date_str + '.png')
    plt.close()


def Hist(DF,target_date_str):
    path_out_dir = 'D:/30_학술대회/2021-11 한국지리정보학회(제주)/data/Hist'
    import os
    os.makedirs(path_out_dir, exist_ok=True)
    ## QQ plot
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(10,8))
    plt.hist(DF['chl-a'], bins=20)
    plt.xlabel('Chl-a Concentration [mg m-3]', fontsize=14)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_out_dir + '/Dokdo_CHL_Hist_' + target_date_str + '.png')
    plt.close()


def scatter(DF,target_date_str):
    '''
    DF를 넣으면 lon, lat, chl-a 행를 인식하여 그림을 그려줌
    '''

    import Lib.Map as Map
    import matplotlib.pyplot as plt
    import numpy as np

    path_out_dir = 'D:/30_학술대회/2021-11 한국지리정보학회(제주)/data'

    fig= plt.figure(3,figsize=(8,6))
    plt.scatter(DF.lon, DF.lat, c=DF['chl-a'], s=1, vmin=0, vmax=.4, cmap='turbo') # gist_ncar, nipy_spectral, gnuplot2
    cb = plt.colorbar(format='%.2f')
    cb.set_label("Chl-a [mg m-3]", size=14)
    ticks=np.linspace(0, .4, 11)
    cb.set_ticks(list(ticks))
    tick_labels = ['%.2f' % num for num in ticks]
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    coord=Map.dokdo_psn()
    plt.scatter(coord[0], coord[1], facecolors='w', edgecolors='r', s=100, marker='s', label='Dokdo')
    plt.legend(loc='upper right')

    plt.grid()
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.xlim(131.4,132.6)
    plt.ylim(36.9,37.6)

    plt.tight_layout()
    plt.savefig(path_out_dir + '/Dokdo_CHL_Scatter_' + target_date_str + '.png')
    plt.close()


def hist2d(x,y):
    import matplotlib.pyplot as plt
    import numpy as np

    cmap1 = plt.cm.get_cmap('viridis')
    # cmap1.set_over(color='w', alpha=None)
    cmap1.set_under(color='w', alpha=None)

    plt.subplot(1,2,1)
    z, x_edge, y_edge = np.histogram2d(
        x, y, bins=(np.linspace(0, 1, 200), np.linspace(0, 1, 200)))
    pc = plt.pcolor(x_edge, y_edge, z, vmin=0.1, vmax=50)
    plt.xlabel('Acutal', fontsize=14)
    plt.ylabel('Pred', fontsize=14)
    plt.grid()
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    z, x_edge, y_edge = np.histogram2d(
        x, y, bins=(np.linspace(0, .4, 100), np.linspace(0, .4, 100)))

    plt.subplot(1, 2, 2)
    pc = plt.pcolor(x_edge, y_edge, np.log10(z), vmin=0.1, vmax=np.log10(50))
    plt.xlabel('Acutal', fontsize=14)
    plt.grid()
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    # norm = mpl.colors.BoundaryNorm(ticks, cmap1.N, extend='both')
    # cb = fig1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap1), orientation='vertical', label="Number of ships",
    #                    shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%2i')
    cb = plt.colorbar(format='%i')
    ticks = [0, 2, 5, 10, 20, 50]
    tick_labels = list(np.array(ticks).astype(int).astype(str))
    cb.set_label("Count of Chl-a point", size=14)
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=12)