
def pcolor(mesh_lon, mesh_lat, z, cmap, vmin, vmax, ticks):
    '''
    mesh_lon, mesh_lat, chl 입력하면 그림
    '''
    import matplotlib.pyplot as plt
    fig= plt.figure(4,figsize=(8,6))

    cmap1 = plt.cm.get_cmap(cmap)
    cmap1.set_under(color='w', alpha=None)

    plt.pcolor(mesh_lon, mesh_lat, z, vmin=vmin, vmax=vmax, cmap=cmap1) # gist_ncar, nipy_spectral, gnuplot2
    cb = plt.colorbar(format='%.2f')
    cb.set_label("Chl-a [mg m-3]", size=14)
    cb.set_ticks(list(ticks))
    tick_labels = ['%.2f' % num for num in ticks]
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=12)
    plt.grid()
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.xlim(mesh_lon.min(),mesh_lon.max())
    plt.ylim(mesh_lat.min(),mesh_lat.max())
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tight_layout()



def QQ(data):
    ## QQ plot
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    plt.figure(2, figsize=(6,5))
    stats.probplot(data, plot=plt)
    plt.grid()


def Hist(data):
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(10,8))
    plt.hist(data, bins=20)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid()

def scatter(lon, lat, z, cmap, vmin, vmax, ticks, dot_size, marker):
    '''
    DF를 넣으면 lon, lat, chl-a 행를 인식하여 그림을 그려줌
    '''

    import matplotlib.pyplot as plt
    import numpy as np

    fig= plt.figure(3,figsize=(8,6))
    plt.scatter(lon, lat, c=z, s=dot_size, marker=marker, vmin=vmin, vmax=vmax, cmap=cmap) # gist_ncar, nipy_spectral, gnuplot2
    cb = plt.colorbar(format='%.2f')
    cb.set_label("Chl-a [mg m-3]", size=14)
    cb.set_ticks(list(ticks))
    tick_labels = ['%.2f' % num for num in ticks]
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.grid()
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.xlim(min(lon),max(lon))
    plt.ylim(min(lat),max(lat))
    plt.tight_layout()



def hist2d(x,y,vmin, vmax, ticks):
    import matplotlib.pyplot as plt
    import numpy as np

    cmap1 = plt.cm.get_cmap('viridis')
    # cmap1.set_over(color='w', alpha=None)
    cmap1.set_under(color='w', alpha=None)

    plt.subplot(1,2,1)
    z, x_edge, y_edge = np.histogram2d(
        x, y, bins=(np.linspace(0, 1, 200), np.linspace(0, 1, 200)))
    pc = plt.pcolor(x_edge, y_edge, z, vmin=0.1, vmax=50)
    plt.xlabel('Actual', fontsize=14)
    plt.ylabel('Pred', fontsize=14)
    plt.grid()
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    z, x_edge, y_edge = np.histogram2d(
        x, y, bins=(np.linspace(0, .4, 100), np.linspace(0, .4, 100)))

    plt.subplot(1, 2, 2)
    pc = plt.pcolor(x_edge, y_edge, np.log10(z), vmin=vmin, vmax=np.log10(vmax))
    plt.xlabel('Actual', fontsize=14)
    plt.grid()
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    # norm = mpl.colors.BoundaryNorm(ticks, cmap1.N, extend='both')
    # cb = fig1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap1), orientation='vertical', label="Number of ships",
    #                    shrink=0.5, aspect=20, fraction=0.2, pad=0.02, format='%2i')
    cb = plt.colorbar(format='%i')
    tick_labels = list(np.array(ticks).astype(int).astype(str))
    # cb.set_label("Count of Chl-a point", size=14)
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=12)