
# imports
import numpy as np
from scipy.interpolate import griddata

from utils import fit
import analyze

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# formatting
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'
fontP = FontProperties()
fontP.set_size('medium')


def plot_scatter(dficts, xparameter='y', yparameter='z', min_cm=0.5, z0=0, take_abs=False):
    """
    Plot all data (xparameter, yparameter) as scatter points with different colors.

    :param dficts:
    :param xparameter:
    :param yparameter:
    :param min_cm:
    :param z0:
    :return:
    """

    fig, ax = plt.subplots(figsize=(7.25, 4.25))
    cscatter = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dficts.keys()))))

    for name, df in dficts.items():

        # filter dataframe
        df = df[df['cm'] > min_cm]

        # sort by x-parameter and get x- and y-arrays for plotting
        if xparameter is None or xparameter == 'index':
            x = df.index
        else:
            df = df.sort_values(by=xparameter)
            x = df[xparameter]
        y = df[yparameter] - z0

        # take absolute value
        if take_abs:
            y = np.abs(y)

        # plot
        cs = next(cscatter)
        ax.scatter(x, y, color=cs)

    ax.set_xlabel(xparameter, fontsize=18)
    ax.set_ylabel(yparameter, fontsize=18)
    ax.grid(alpha=0.125)
    ax.legend(dficts.keys(), prop=fontP, title=r'$dz$ (mm)', loc='upper right', fancybox=True, shadow=False)

    return fig, ax


def plot_errorbars(dfbicts, xparameter='index', yparameter='z', min_cm=0.5, z0=0):
    """
    Plot all data (xparameter, yparameter) as scatter points with different colors.

    :param dficts:
    :param xparameter:
    :param yparameter:
    :param min_cm:
    :param z0:
    :return:
    """

    fig, ax = plt.subplots(figsize=(7.25, 4.25))
    cscatter = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dfbicts.keys()))))
    cerror = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dfbicts.keys()))))

    for item in dfbicts.items():
        # get name and dataframe (for readability)
        name = item[0]
        df = item[1]

        # filter dataframe
        df = df[df['cm'] > min_cm]

        # sort by x-parameter and get x- and y-arrays for plotting
        if xparameter is None or xparameter == 'index':
            x = df.index
        else:
            df = df.sort_values(by=xparameter)
            x = df[xparameter]
        y = df[yparameter] - z0

        # plot
        cs = next(cscatter)
        ax.errorbar(x, y, yerr=df.z_std * 2, fmt='o', color=cs, ecolor=next(cerror), elinewidth=1, capsize=2, alpha=0.25)
        ax.scatter(x, y, color=cs)

    ax.set_xlabel(xparameter, fontsize=18)
    ax.set_ylabel(yparameter, fontsize=18)
    ax.set_ylim([-10, 100])
    ax.grid(alpha=0.125)
    ax.legend(dfbicts.keys(), prop=fontP, title=r'$dz$ (mm)', loc='upper left', fancybox=True, shadow=False)

    return fig, ax


def plot_fit_and_scatter(fit_function, dficts, xparameter='index', yparameter='z', min_cm=0.5, z0=0, auto_format=False):
    """
    Plot fitted curve and data (xparameter, yparameter) as scatter points with different colors.

    :param dficts:
    :param xparameter:
    :param yparameter:
    :param min_cm:
    :param z0:
    :return:
    """

    fig, ax = plt.subplots(figsize=(7.25, 4.25))
    cscatter = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dficts.keys()))))

    for name, df in dficts.items():

        # drop NaN's
        df = df.dropna(axis=0, subset=[yparameter])

        # filter dataframe
        df = df[df['cm'] > min_cm]

        # sort by x-parameter and get x- and y-arrays for plotting
        if xparameter is None or xparameter == 'index':
            x = df.index
        else:
            df = df.sort_values(by=xparameter)
            x = df[xparameter]
        y = df[yparameter] - z0

        # plot scatter points
        cs = next(cscatter)
        ax.scatter(x, y, color=cs)

        # fit the function
        popt, pcov, fit_func = fit.fit(x, y, fit_function=fit_function)

        # plot fitted function
        xfit = np.linspace(0, x.max(), 100)
        ax.plot(xfit, fit_function(xfit, popt[0], popt[1], popt[2]), color=cs, linewidth=3, alpha=0.9)

    ax.set_xlabel(xparameter, fontsize=18)
    ax.set_ylabel(yparameter, fontsize=18)
    ax.grid(alpha=0.125)
    if auto_format:
        ax.legend(dficts.keys(), prop=fontP, title=r'$dz$ (mm)', loc='upper left', fancybox=True, shadow=False)

    return fig, ax


def plot_dfbicts_local(dfbicts, parameters='rmse_z', h=1, colors=None, linestyles=None, show_legend=False, scale=1.0):

    if isinstance(parameters, str):
        parameter = parameters
        parameterr = None
    elif isinstance(parameters, list):
        parameter = parameters[0]
        parameterr = parameters[1]

    if isinstance(scale, (int, float)):
        scalex, scaley = scale, scale
    else:
        scalex, scaley = scale[0], scale[1]

    fig, ax = plt.subplots(figsize=(4.125*scalex, 2.375*scaley))

    if isinstance(colors, list):
        cscatter = iter(colors)
        cscatterr = iter(colors)
    elif colors == 'Blues':
        cscatter = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
        cscatterr = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
    else:
        cscatter = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
        cscatterr = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))

    if isinstance(linestyles, list):
        lstyle = iter(linestyles)
    else:
        lstyle = iter('-' for i in range(dfbicts.keys()))

    if parameter == 'rmse_z':
        for item in dfbicts.items():
            cs = next(cscatter)
            ls = next(lstyle)
            ax.plot(item[1].index, item[1][parameter] / h, color=cs, linestyle=ls, linewidth=3)
            ax.scatter(item[1].index, item[1][parameter] / h, s=50, color=cs, alpha=0.5)
    else:
        for item in dfbicts.items():
            cs = next(cscatter)
            ls = next(lstyle)
            ax.plot(item[1].index, item[1][parameter], color=cs, linestyle=ls, linewidth=3)
            ax.scatter(item[1].index, item[1][parameter], s=50, color=cs, alpha=0.5)

    if parameterr is not None:
        ax2 = ax.twinx()
        for item in dfbicts.items():
            css = next(cscatterr)
            ax2.plot(item[1].index, item[1][parameterr], color=css, linestyle='--', linewidth=2, alpha=0.75)

    if h != 1 and parameter == 'rmse_z':
        ax.set_ylabel(r'$\sigma_{z}\left(z\right)$ / h', fontsize=18)
    else:
        ax.set_ylabel(parameter, fontsize=18)

    ax.set_xlabel('z ($\mu m$)', fontsize=18)
    ax.grid(alpha=0.25)

    if show_legend:
        ax.legend(dfbicts.keys(), prop=fontP, title=r'$\sigma$', loc='upper left', fancybox=True, shadow=False)

    if parameterr is not None:
        return fig, ax, ax2
    else:
        return fig, ax


def plot_dfbicts_global(dfbicts, parameter='rmse_z', xlabel='parameter', h=1, print_values=True):
    fig, ax = plt.subplots(figsize=(8.25, 4.75))

    names = dfbicts.keys()
    means = np.array([m[parameter].mean() for m in dfbicts.values()])

    if print_values:
        print(names)
        print('mean sigma_z: {}'.format(means / h))

    if parameter == 'rmse_z' and h != 1:
        ax.plot(names, means / h, color='tab:blue', linewidth=3)
        ax.scatter(names, means / h, s=50, color='tab:blue', alpha=0.5)
    else:
        ax.plot(names, means, color='tab:blue', linewidth=3)
        ax.scatter(names, means, s=50, color='tab:blue', alpha=0.5)

    if h != 1 and parameter == 'rmse_z':
        ax.set_ylabel(r'$\sigma_{z}\left(z\right)$ / h', fontsize=18)
    else:
        ax.set_ylabel(parameter, fontsize=18)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.grid(alpha=0.25)

    if print_values:
        if parameter == 'rmse_z' and h != 1:
            ax.set_title('IDs: {},'.format(names) + r'$\sigma_{z}$: ' + '{}'.format(means / h))
        else:
            ax.set_title('IDs: {},'.format(names) + '{}: {}'.format(parameter, means))

    return fig, ax

def plot_scatter_3d(df, fig=None, ax=None, elev=5, azim=-40):
    """

    :param df: dataframe with 'x', 'y', and 'z' columns
    :param fig: figure
    :param ax: axes to plot on
    :param elev: the elevation angle in the z-plane.
    :param azim: the azimuth angle in the x-y plane.
    :return:
    """

    if fig is None:
        fig = plt.figure(figsize=(6, 6))

    if ax is None:
        ax = fig.add_subplot(projection='3d')

    ax.scatter(df.x, df.y, df.z, marker='o', c=df.z)

    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)
    ax.view_init(elev, azim)

    return fig, ax

def plot_heatmap(df, fig=None, ax=None):

    # drop NaNs
    dfc = df.dropna(axis=0, subset=['z'])

    # move x, y, z series to numpy arrays
    x = dfc.x.to_numpy()
    y = dfc.y.to_numpy()
    z = dfc.z.to_numpy()

    # get spatial coordinate extents
    xspace = np.max(x) - np.min(x)
    yspace = np.max(y) - np.min(y)
    zspace = np.max(z) - np.min(z)

    # contour surface levels: 1 level = 1 micron
    lvls_surface = int(np.round(zspace + 1))
    lvls_lines = int(lvls_surface / 5)

    # -----------------------
    # Interpolation on a grid
    # -----------------------
    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.
    ngridx = int(xspace)
    ngridy = int(yspace)

    # Create grid values first.
    xi = np.linspace(np.min(x), np.max(x), ngridx)
    yi = np.linspace(np.min(y), np.max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # plot level surfaces
    cntr = ax.contourf(xi, yi, zi, levels=lvls_surface, cmap="RdBu_r")

    # plot level lines
    ax.contour(xi, yi, zi, levels=lvls_lines, linewidths=0.5, colors='gray')

    # plot data points
    ax.scatter(x, y, c=z, cmap="RdBu_r")

    cbar = fig.colorbar(cntr, ax=ax)
    cbar.ax.set_title(r'$\delta z$')
    ax.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel(r'$y$', fontsize=18)

    return fig, ax