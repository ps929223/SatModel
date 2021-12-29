
def hist2d(actual,pred,lims, n_divide, vmin, vmax, log2=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_squared_error

    cmap1 = plt.cm.get_cmap('viridis')
    # cmap1.set_over(color='w', alpha=None)
    cmap1.set_under(color='w', alpha=None)

    if log2==True:
        z, x_edge, y_edge = np.histogram2d(
            actual, pred, bins=(np.linspace(lims[0], lims[1], n_divide), np.linspace(lims[0], lims[1], n_divide)))
        pc = plt.pcolor(x_edge, y_edge, np.log2(z), vmin=np.log2(vmin), vmax=np.log2(vmax))
        cb = plt.colorbar(format='%i')
        ticks = np.linspace(np.log2(vmin), np.log2(vmax), 5)
        tick_labels = list(np.round(2**ticks).astype(int).astype(str))
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels)
    else:
        z, x_edge, y_edge = np.histogram2d(
            actual, pred, bins=(np.linspace(lims[0], lims[1], n_divide), np.linspace(lims[0], lims[1], n_divide)))
        pc = plt.pcolor(x_edge, y_edge, z, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(format='%i')
        ticks = np.linspace(vmin, vmax, 5)
        tick_labels = list(ticks.astype(int).astype(str))
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels)

    plt.title('n='+str(np.nansum(z).astype(int)))
    plt.xlabel('Acutal', fontsize=14)
    plt.ylabel('Pred', fontsize=14)
    plt.grid()
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    rmse=mean_squared_error(actual, pred, squared=False)
    print('RMSE: '+ str(rmse))
    return rmse

