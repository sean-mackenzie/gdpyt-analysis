import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# formatting
from utils import fit

plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'
fontP = FontProperties()
fontP.set_size('medium')


def plot_scatter(dficts, pids, xparameter='frame', yparameter='z', min_cm=0.5, z0=0, take_abs=False, fit_data=False,
                 fit_function=None):
    """ Plot particle ID displacement across all frames """

    if isinstance(pids, int):
        pids = [pids]

    if fit_function == 'line':
        fit_func = fit.line
    elif fit_function == 'parabola':
        fit_func = fit.parabola

    fig, ax = plt.subplots(figsize=(7.25, 4.25))
    cscatter = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dficts.keys())*len(pids))))

    names = []
    for name, df in dficts.items():
        names.append(name)

        # filter dataframe
        boolean_series = df.id.isin(pids)
        df = df[boolean_series]
        df = df[df['cm'] > min_cm]

        # z-offset
        df.z = df.z - z0

        # sort by x-parameter
        df = df.sort_values(by=xparameter)

        for pid in pids:

            dfp = df[df['id'] == pid]

            # plot
            cs = next(cscatter)
            ax.scatter(dfp[xparameter], dfp[yparameter], color=cs)
            ax.plot(dfp[xparameter], dfp[yparameter], color=cs, linestyle='dotted', alpha=0.5)

        if fit_data:
            # fit the function
            popt, pcov, _ = fit.fit(df[xparameter], df[yparameter], fit_function=fit_func)

            consts = [np.round(ppt, 2) for ppt in popt]
            if fit_function == 'line':
                fit_label = r'$\mathcal{f}\/\/$' + '({}, {})'.format(consts[0], consts[1]) + r'$=Ax+B$'
            elif fit_function == 'parabola':
                fit_label = r'$\mathcal{f}\/\/$' + '({}, {}, {})'.format(consts[0], consts[1], consts[2]) + r'$=Ax^2+Bx+C$'

            # one standard deviation errors
            perr = np.sqrt(np.diag(pcov))

            # plot fitted function
            xfit = np.linspace(0, df[xparameter].max(), 100)
            ax.plot(xfit, fit_func(xfit, *popt), color='black', linewidth=3, alpha=0.9, linestyle='--', zorder=1.1,
                    label=fit_label + '\n' + r'$\sigma_{A} =$ ' + str(np.round(2 * perr[0], 3)))

    ax.set_xlabel(xparameter, fontsize=18)
    ax.set_ylabel(yparameter, fontsize=18)
    ax.grid(alpha=0.125)

    if fit_data:
        ax.legend(prop=fontP, loc='best', fancybox=True, shadow=False)

    return fig, ax