import matplotlib.pyplot as plt

def performance(x, y, ens, metric='AUC', savename=None):
    """Plot the performance of base classifiers and ensembles.
    
    Plot the true vs. the inferred performance of each base classifier. Along
    the y-axis plot the performance of each inputted ensemble.

    Args:
    x : (M,) ndarray
        True performance
    y : (M, )ndarray
        Inferred performance
    ens : python dict
        {ensemble name : ensemble performance, ...}
    metric : str
        Either AUC or BA
    """
    plt.figure(figsize=(4.5, 3))
    plt.plot(x, y, 'o', mfc='none', mew=2, ms=7.5, label='Base Classifiers')
    ax = plt.gca()
    xlims = ax.get_xlim()
    for w in ens:
        plt.plot(xlims, [ens[w], ens[w]], ':', label=w)
    plt.plot(xlims, xlims, ':', color='k', alpha=0.5)

    ax.set_position([0.16, 0.175, 0.45, 0.8])
    plt.xlabel('True {}'.format(metric), fontsize=15)
    plt.ylabel('Inferred {}'.format(metric), fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if savename is not None:
        plt.savefig(savename, fmt=savename.split('.')[-1])
    plt.show(block=False)
