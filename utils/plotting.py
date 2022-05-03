
# imports
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Akima1DInterpolator
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version

from utils import fit, modify

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import collections, colors, transforms

# formatting
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'
fontP = FontProperties()
fontP.set_size('medium')

plt.style.use(['science', 'ieee', 'std-colors'])
# plt.style.use(['science', 'scatter'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)



def plot_scatter(dficts, xparameter='y', yparameter='z', min_cm=0.5, z0=0, take_abs=False,
                 figsize=(6, 4), scattersize=2):
    """
    Plot all data (xparameter, yparameter) as scatter points with different colors.

    :param dficts:
    :param xparameter:
    :param yparameter:
    :param min_cm:
    :param z0:
    :return:
    """

    fig, ax = plt.subplots(figsize=figsize)
    #cscatter = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dficts.keys()))))

    for name, df in dficts.items():

        # filter dataframe
        if min_cm:
            df = df[df['cm'] > min_cm]

        # sort by x-parameter and get x- and y-arrays for plotting
        if xparameter is None or xparameter == 'index':
            x = df.index
        else:
            df = df.sort_values(by=xparameter)
            x = df[xparameter]

        y = df[yparameter]

        if z0:
            y = y - z0

        # take absolute value
        if take_abs:
            y = np.abs(y)

        # plot
        #cs = next(cscatter)
        ax.scatter(x, y, s=scattersize)

    # ax.set_xlabel(xparameter, fontsize=18)
    # ax.set_ylabel(yparameter, fontsize=18)
    # ax.grid(alpha=0.125)
    # ax.legend(dficts.keys(), prop=fontP, title=r'$dz$ (mm)', loc='upper right', fancybox=True, shadow=False)

    return fig, ax


def plot_mean(dficts, xparameter='y', yparameter='z', min_cm=0.5, z0=0, take_abs=False, fit_function=None):
    """
    Plot all data (xparameter, yparameter) as scatter points with different colors.

    :param dficts:
    :param xparameter:
    :param yparameter:
    :param min_cm:
    :param z0:
    :return:
    """
    cscatter = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dficts.keys()))))
    cerror = iter(cm.Spectral(np.linspace(0.95, 0.2, len(dficts.keys()))))

    fig, ax = plt.subplots(figsize=(7.25, 4.25))

    means = []
    for name, df in dficts.items():

        # filter dataframe
        df = df[df['cm'] > min_cm]

        y = df[yparameter] - z0

        # take absolute value
        if take_abs:
            y = np.abs(y)

        yerr = np.std(y)
        y = np.mean(y)
        means.append(y)

        # plot
        cs = next(cscatter)
        ax.errorbar(name, y, yerr=yerr * 2, fmt='o', color=cs, ecolor=next(cerror), elinewidth=3, capsize=4, alpha=0.75)
        ax.scatter(name, y, color=cs)

    ax.set_xlabel(xparameter, fontsize=18)
    ax.set_ylabel(yparameter, fontsize=18)
    ax.grid(alpha=0.125)
    ax.legend(dficts.keys(), prop=fontP, title=r'$dz$ (mm)', loc='upper left', fancybox=True, shadow=False)

    # fit the function
    if fit_function is not None:
        names = list(dficts.keys())
        popt, pcov, fit_func = fit.fit(names, means, fit_function=fit_function)

        # plot fitted function
        xfit = np.linspace(0, np.max(names), 100)
        ax.plot(xfit, fit_function(xfit, *popt), color='black', linewidth=2, linestyle='--', alpha=0.5)

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

    for name, df in dfbicts.items():

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
        ax.errorbar(x, y, yerr=df.z_std * 2, fmt='o', color=cs, ecolor=next(cerror), elinewidth=1, capsize=2, alpha=0.75)
        ax.scatter(x, y, color=cs)

    ax.set_xlabel(xparameter, fontsize=18)
    ax.set_ylabel(yparameter, fontsize=18)
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


def plot_dfbicts_local(dfbicts, parameters='rmse_z', h=1, colors=None, linestyles=None, show_legend=False, scale=None,
                       scatter_on=True, scatter_size=10,
                       label_dict=None,
                       ylabel=None, xlabel=None, semilogx=False, nrows=None, ncols=None):
    """
    Notes:
        1. Plots the dataframe index on x-axis.
        2. If only one parameter is passed (len(parameters) == 1), then no ax2 is returned.

    :param dfbicts:
    :param parameters:
    :param h:
    :param colors:
    :param linestyles:
    :param show_legend:
    :param scale:
    :param scatter_on:
    :param scatter_size:
    :param ylabel:
    :param xlabel:
    :return:
    """

    # format figure
    if isinstance(colors, list):
        colors = colors
        cscatter = None
        cscatterr = None
    elif colors == 'Blues':
        cscatter = iter(cm.Blues(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
        cscatterr = iter(cm.Blues(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
    elif colors == 'inferno':
        cscatter = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
        cscatterr = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
    else:
        # get colors from cycler
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if len(dfbicts) > len(colors):
            colors_repeats = colors + colors
            colors = colors_repeats[:len(dfbicts)]

        cscatter = None
        cscatterr = None

    if isinstance(linestyles, list):
        lstyle = iter(linestyles)
    else:
        lstyle = iter('-' for i in list(dfbicts.keys()))

    if not scale:
        if nrows:
            fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True)
        elif ncols:
            fig, [ax, ax2] = plt.subplots(ncols=2)
        else:
            fig, ax = plt.subplots()
    else:
        if isinstance(scale, (int, float)):
            scalex, scaley = scale, scale
        else:
            scalex, scaley = scale[0], scale[1]

        fig, ax = plt.subplots()
        size_x_inches, size_y_inches = fig.get_size_inches()
        size_x_pixels, size_y_pixels = fig.get_size_inches() * fig.dpi
        plt.close(fig)

        if nrows:
            fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * scalex, size_y_inches * scaley))
        elif ncols:
            fig, [ax, ax2] = plt.subplots(ncols=2, figsize=(size_x_inches * scalex, size_y_inches * scaley))
        else:
            fig, ax = plt.subplots(figsize=(size_x_inches*scalex, size_y_inches*scaley))

    # organize data
    if (isinstance(parameters, str)) or (isinstance(parameters, list) and len(parameters) == 1):
        parameter = parameters
        parameterr = None
        parameter3 = None
        parameter4 = None
    elif isinstance(parameters, list) and len(parameters) == 2:
        parameter = parameters[0]
        parameterr = parameters[1]
        parameter3 = None
        parameter4 = None
    elif isinstance(parameters, list) and len(parameters) == 3:
        parameter = parameters[0]
        parameterr = parameters[1]
        parameter3 = parameters[2]
        parameter4 = None
    elif isinstance(parameters, list) and len(parameters) == 4:
        parameter = parameters[0]
        parameterr = parameters[1]
        parameter3 = parameters[2]
        parameter4 = parameters[3]

    if parameter == 'rmse_z':
        for item, clr in zip(dfbicts.items(), colors):

            if cscatter is not None:
                cs = next(cscatter)
                ls = next(lstyle)
                ax.plot(item[1].index, item[1][parameter] / h)
                if scatter_on:
                    ax.scatter(item[1].index, item[1][parameter] / h)
            else:
                if label_dict:
                    lbl = label_dict[item[0]]['label']
                else:
                    lbl = None

                ls = next(lstyle)

                if scatter_on:
                    ax.scatter(item[1].index, item[1][parameter] / h, color=clr, s=scatter_size)
                    ax.plot(item[1].index, item[1][parameter] / h, color=clr, linestyle=ls, label=lbl)
                else:
                    ax.plot(item[1].index, item[1][parameter] / h, color=clr, label=lbl, linestyle=ls)

    else:
        for item, clr in zip(dfbicts.items(), colors):

            if cscatter is not None:
                ax.plot(item[1].index, item[1][parameter])
                if scatter_on:
                    ax.scatter(item[1].index, item[1][parameter])
            else:
                if label_dict:
                    lbl = label_dict[item[0]]['label']
                else:
                    lbl = None

                ls = next(lstyle)

                if semilogx:
                    ax.semilogx(item[1].index, item[1][parameter] / h)
                else:
                    if scatter_on:
                        ax.scatter(item[1].index, item[1][parameter] / h, s=scatter_size, color=clr)
                        ax.plot(item[1].index, item[1][parameter] / h, color=clr, linestyle=ls, label=lbl)
                    else:
                        ax.plot(item[1].index, item[1][parameter] / h, color=clr, linestyle=ls, label=lbl)

    if parameterr is not None:
        if not nrows:
            ax2 = ax.twinx()
        for item, clr in zip(dfbicts.items(), colors):
            if nrows:
                ax2.plot(item[1].index, item[1][parameterr], color=clr)
            else:
                ax2.plot(item[1].index, item[1][parameterr], color=clr, linestyle='--')

            if parameter3 is not None:
                ax2.plot(item[1].index, item[1][parameter3], color=clr, linestyle=':')

                if parameter4 is not None:
                    ax2.plot(item[1].index, item[1][parameter4], color=clr, linestyle='-.')

    if ylabel:
        ax.set_ylabel(ylabel)
    elif h != 1 and parameter == 'rmse_z':
        ax.set_ylabel(r'$\sigma_{z}\left(z\right) / h$')
    elif parameter == 'rmse_z':
        ax.set_ylabel(r'$\sigma_{z}\left(z\right)$')
    else:
        ax.set_ylabel(parameter)

    if xlabel:
        if nrows:
            ax2.set_xlabel(xlabel)
        else:
            ax.set_xlabel(xlabel)
    else:
        if nrows:
            ax2.set_xlabel('z ($\mu m$)')
        else:
            ax.set_xlabel('z ($\mu m$)')

    ax.grid(alpha=0.25)
    if nrows:
        ax2.grid(alpha=0.25)

    if show_legend:
        ax.legend(dfbicts.keys(), title=r'$\sigma$')

    if parameterr is not None:
        return fig, ax, ax2
    else:
        return fig, ax


def plot_dfbicts_global(dfbicts, parameters='rmse_z', xlabel='parameter', h=1, print_values=False,
                        scale=None, fig=None, ax=None, ax2=None, ax2_ylim=None, color=None, scatter_size=10,
                        smooth=False, ylabel=None):

    if fig is None and ax is None:

        if not scale:
            fig, ax = plt.subplots()
        else:
            if isinstance(scale, (int, float)):
                scalex, scaley = scale, scale
            else:
                scalex, scaley = scale[0], scale[1]

            fig, ax = plt.subplots()
            size_x_inches, size_y_inches = fig.get_size_inches()
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(size_x_inches * scalex, size_y_inches * scaley))

        if ax2 is None and isinstance(parameters, list) and len(parameters) > 1:
            ax2 = ax.twinx()

    # organize data
    if isinstance(parameters, str) or len(parameters) == 1:
        parameter = parameters
        parameterr = None
        parameterrr = None

        names = dfbicts.keys()
        means = np.array([m[parameter].mean() for m in dfbicts.values()])

        sort_by_name = sorted(list(zip(names, means)), key=lambda x: x[0])
        names = [x[0] for x in sort_by_name]
        means = np.array([x[1] for x in sort_by_name])

    elif isinstance(parameters, list) and len(parameters) == 2:
        parameter = parameters[0]
        parameterr = parameters[1]
        parameterrr = None

        names = dfbicts.keys()
        means = np.array([m[parameter].mean() for m in dfbicts.values()])
        means_prr = np.array([m[parameterr].mean() for m in dfbicts.values()])

        sort_by_name = sorted(list(zip(names, means, means_prr)), key=lambda x: x[0])
        names = [x[0] for x in sort_by_name]
        means = np.array([x[1] for x in sort_by_name])
        means_prr = np.array([x[2] for x in sort_by_name])

    elif isinstance(parameters, list) and len(parameters) == 3:
        parameter = parameters[0]
        parameterr = parameters[1]
        parameterrr = parameters[2]

        names = dfbicts.keys()
        means = np.array([m[parameter].mean() for m in dfbicts.values()])
        means_prr = np.array([m[parameterr].mean() for m in dfbicts.values()])
        means_prrr = np.array([m[parameterrr].mean() for m in dfbicts.values()])

        sort_by_name = sorted(list(zip(names, means, means_prr, means_prrr)), key=lambda x: x[0])
        names = [x[0] for x in sort_by_name]
        means = np.array([x[1] for x in sort_by_name])
        means_prr = np.array([x[2] for x in sort_by_name])
        means_prrr = np.array([x[3] for x in sort_by_name])

    else:
        raise ValueError("parameters must be a string or a list of strings")

    # smooth data
    if smooth:
        names = np.array(names)
        names_interp = np.linspace(np.min(names), np.max(names), 500)
        means_interp = Akima1DInterpolator(names, means)(names_interp)
        means = means_interp

        if parameterr:
            means_prr_interp = Akima1DInterpolator(names, means_prr)(names_interp)
            means_prr = means_prr_interp

        if parameterrr:
            means_prrr_interp = Akima1DInterpolator(names, means_prrr)(names_interp)
            means_prrr = means_prrr_interp

        names = names_interp

    # plot figure
    if parameter == 'rmse_z' and h != 1:
        ax.plot(names, means / h, color=color)
        if scatter_size:
            ax.scatter(names, means / h, s=scatter_size, color=color)
    else:
        ax.plot(names, means, color=color)
        if scatter_size:
            ax.scatter(names, means, s=scatter_size, color=color)

    if parameter == 'rmse_z':
        ax.set_ylabel(r'$\overline{\sigma_{z}} / h$')
    elif ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(parameter)

    if parameterr is not None and parameterrr is None:
        ax2.plot(names, means_prr, linestyle='--', color=color)
        ax2.set_ylim(ax2_ylim)
    elif parameterrr is not None:
        ax2.plot(names, means_prr, color=color, linestyle='--')
        ax2.plot(names, means_prrr, color=color, linestyle=':')
        ax2.set_ylim(ax2_ylim)

    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)

    # print results
    if print_values:
        print(names)
        print('{}: {}'.format(parameter, means / h))
        if parameterr:
            print('{}: {}'.format(parameterr, means_prr))

    return fig, ax, ax2


def plot_dfbicts_list_global(dfbicts_list, parameters='rmse_z', xlabel='parameter', h=1, print_values=False,
                             scale=None, colors=None, ax2_ylim=None, scatter_size=10, smooth=False, ylabel=None):
    # format figure
    if not colors:
        # get colors from cycler
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if not scale:
        fig, ax = plt.subplots()
    else:
        if isinstance(scale, (int, float)):
            scalex, scaley = scale, scale
        else:
            scalex, scaley = scale[0], scale[1]

        fig, ax = plt.subplots()
        size_x_inches, size_y_inches = fig.get_size_inches()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(size_x_inches * scalex, size_y_inches * scaley))

    if isinstance(parameters, list) and len(parameters) > 1:
        ax2 = ax.twinx()
    else:
        ax2 = None

    for dfbicts, color in zip(dfbicts_list, colors):
        fig, ax, ax2 = plot_dfbicts_global(dfbicts, parameters, xlabel, h, print_values,
                                           scale=scale, fig=fig, ax=ax, ax2=ax2, ax2_ylim=ax2_ylim,
                                           color=color, scatter_size=scatter_size, smooth=smooth, ylabel=ylabel)

    return fig, ax, ax2


def plot_scatter_z_color(dficts, xparameter='x', yparameter='y', zparameter='z', min_cm=0.5, z0=0, take_abs=False):
    """
    Plot all data (xparameter, yparameter, zparameter) as scatter points with z-parameter as colors.
    """

    for name, df in dficts.items():

        ax = plt.subplot()

        # filter dataframe
        df = df[df['cm'] > min_cm]

        # get x and y values
        x = df[xparameter]
        y = df[yparameter]

        # adjust for z-offset
        z = df[zparameter] - z0

        # take absolute value
        if take_abs:
            z = np.abs(z)

        # plot
        data = ax.scatter(x, y, c=z)

        ax.set_xlabel(xparameter, fontsize=18)
        ax.set_ylabel(yparameter, fontsize=18)
        ax.set_title(name, fontsize=18)
        ax.grid(alpha=0.125)

        # color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        plt.colorbar(data, cax=cax)

        plt.show()

    plt.close('all')


# ---------------------------------   DATAFRAMES   ---------------------------------------------------------------------


def plot_scatter_3d(df, fig=None, ax=None, elev=5, azim=-40, color=None, alpha=0.75):
    """

    :param df: dataframe with 'x', 'y', and 'z' columns
    :param fig: figure
    :param ax: axes to plot on
    :param elev: the elevation angle in the z-plane.
    :param azim: the azimuth angle in the x-y plane.
    :return:
    """

    if not fig:
        fig = plt.figure(figsize=(6, 6))

    if not ax:
        ax = fig.add_subplot(projection='3d')

    if isinstance(df, list):
        x, y, z = df
    else:
        x, y, z = df.x, df.y, df.z

    if color is None:
        color = z

    ax.scatter(x, y, z, marker='o', c=color, alpha=alpha)

    ax.view_init(elev, azim)

    return fig, ax


def plot_scatter_3d_multi_angle(df, z_param='z'):

    fig = plt.figure(figsize=(6.5, 5))
    for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        sc = ax.scatter(df.x, df.y, df[z_param], c=df[z_param])
        ax.view_init(5, v)
        ax.patch.set_alpha(0.0)
        if i == 2:
            plt.colorbar(sc, shrink=0.5)
            ax.get_xaxis().set_ticks([])
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        elif i == 4:
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        else:
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.get_zaxis().set_ticklabels([])
    plt.suptitle('title', y=0.875)
    plt.subplots_adjust(hspace=-0.1, wspace=0.15)

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


# -------------------------------------   ARRAYS   ---------------------------------------------------------------------


def scatter_xy_color_z(df, param_z):
    fig, ax = plt.subplots()
    sc = ax.scatter(df.x, df.y, c=df[param_z], s=3)
    plt.colorbar(sc, shrink=0.75)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig


def scatter_z_by_xy(df, z_params):
    if not isinstance(z_params, list):
        z_params = [z_params]

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches*2, size_y_inches))

    for z_param in z_params:
        ax[0].scatter(df.x, df[z_param], s=3)
        ax[1].scatter(df.y, df[z_param], s=3, label=z_param)

    ax[0].set_xlabel('x')
    ax[0].set_ylabel('z')
    ax[1].set_xlabel('y')
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return fig, ax


def plot_fitted_plane_and_points(df, dict_fit_plane):

    param_z = dict_fit_plane['z_f']
    rmse, r_squared = dict_fit_plane['rmse'], dict_fit_plane['r_squared']
    tilt_x, tilt_y = dict_fit_plane['tilt_x_degrees'], dict_fit_plane['tilt_y_degrees']
    px, py, pz = dict_fit_plane['px'], dict_fit_plane['py'], dict_fit_plane['pz']
    normal = dict_fit_plane['normal']
    d = dict_fit_plane['d']

    fig = plt.figure(figsize=(6.5, 5))

    for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):

        ax = fig.add_subplot(2, 2, i, projection='3d')
        sc = ax.scatter(df.x, df.y, df[param_z], c=df[param_z], s=1)
        ax.plot_surface(px, py, pz, alpha=0.4, color='red')
        ax.view_init(5, v)
        ax.patch.set_alpha(0.0)
        if i == 2:
            plt.colorbar(sc, shrink=0.5)
            ax.get_xaxis().set_ticks([])
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        elif i == 4:
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        else:
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.get_zaxis().set_ticklabels([])

    # title
    plt.suptitle('RMSE: {}, '.format(np.round(rmse, 3)) +
                 r'$R^2$' + ': {}'.format(np.round(r_squared, 3)) + '\n' +
                 r'$(\theta_{x}, \theta_{y})=$' + ' ({}, {} deg.)'.format(np.round(tilt_x, 3), np.round(tilt_y, 3)))
    # deprecated title
    """plt.suptitle(r"$0 = n_x x + n_y y + n_z z - d$" + "= {}x + {}y + {}z - {} \n"
                                                      "(x, y: pixels; z: microns)".format(np.round(normal[0], 5),
                                                                                        np.round(normal[1], 5),
                                                                                        np.round(normal[2], 5),
                                                                                        np.round(d, 5)),
                 y=0.875)"""

    plt.subplots_adjust(hspace=-0.1, wspace=0.15)

    return fig


def scatter_3d_and_surface(x, y, z, func, func_params, fit_params, cmap='RdBu', grid_resolution=30, view='multi'):

    # setup data points for calculating surface model
    model_x_data = np.linspace(min(x), max(x), grid_resolution)
    model_y_data = np.linspace(min(y), max(y), grid_resolution)

    # create coordinate arrays for vectorized evaluations
    X, Y = np.meshgrid(model_x_data, model_y_data)

    # calculate z-coordinate of array
    if func_params == ['x', 'y']:
        Z = func(np.array([X, Y]), *fit_params)
    elif func_params == 'y':
        Z = func(Y, *fit_params)
    elif func_params == 'x':
        Z = func(X, *fit_params)
    else:
        raise ValueError('function parameters not understood.')

    # plot
    if view == 'multi':
        fig = plt.figure(figsize=(12, 10))
        for i, v in zip(np.arange(1, 5), [315, 0, 225, 90]):

            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(x, y, z, c=z, s=0.5, alpha=0.75)
            ps = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.25)
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        ps = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.5)
        ax.scatter(x, y, z, s=1, color='black', alpha=0.5)

        if view == 'y':
            ax.view_init(5, 0)
        elif view == 'x':
            ax.view_init(5, 90)

        plt.colorbar(ps, fraction=0.015, pad=0.08)

    return fig, ax


def scatter_3d_and_spline(x, y, z, bispl, cmap='RdBu', grid_resolution=25, view='multi'):

    # setup data points for calculating surface model
    model_x_data = np.linspace(min(x), max(x), grid_resolution)
    model_y_data = np.linspace(min(y), max(y), grid_resolution)

    # create coordinate arrays for vectorized evaluations
    X, Y = np.meshgrid(model_x_data, model_y_data)
    Z = bispl.ev(X, Y)

    # plot
    if view == 'multi':
        fig = plt.figure(figsize=(12, 10))
        for i, v in zip(np.arange(1, 5), [315, 0, 225, 90]):

            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(x, y, z, c=z, s=0.5, alpha=0.75)
            ps = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.25)
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        ps = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.5)
        ax.scatter(x, y, z, s=1, color='black', alpha=0.5)

        if view == 'y':
            ax.view_init(5, 0)
        elif view == 'x':
            ax.view_init(5, 90)

        plt.colorbar(ps, fraction=0.015, pad=0.08)

    return fig, ax


def scatter_hist(x, y, fig, color=None, colormap='coolwarm', scatter_size=1, kde=True, distance_from_mean=10):

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.075, hspace=0.075)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    if color is not None:
        ax.scatter(x, y, c=color, cmap=colormap, s=scatter_size)
    else:
        ax.scatter(x, y, s=scatter_size)

    # vertical and horizontal lines denote the mean value
    ax.axvline(np.mean(x), ymin=0, ymax=0.5, color='black', linestyle='--', linewidth=0.25, alpha=0.25)
    ax.axhline(np.mean(y), xmin=0, xmax=0.5, color='black', linestyle='--', linewidth=0.25, alpha=0.25)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    nx, binsx, patchesx = ax_histx.hist(x, bins=bins, zorder=2.5)
    ny, binsy, patchesy = ax_histy.hist(y, bins=bins, orientation='horizontal', zorder=2.5)

    # kernel density estimation
    if kde:
        x_plot = np.linspace(np.mean(x) - distance_from_mean, np.mean(x) + distance_from_mean, 1000)
        y_plot = np.linspace(np.mean(y) - distance_from_mean, np.mean(y) + distance_from_mean, 1000)

        x = x[:, np.newaxis]
        x_plot = x_plot[:, np.newaxis]
        kde_x = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(x)
        log_dens_x = kde_x.score_samples(x_plot)
        scale_to_max = np.max(nx) / np.max(np.exp(log_dens_x))
        #ax_histx.fill(x_plot[:, 0], np.exp(log_dens_x) * scale_to_max, fc='lightsteelblue', zorder=2)
        ax_histx.fill_between(x_plot[:, 0], 0, np.exp(log_dens_x) * scale_to_max, fc='lightsteelblue', zorder=2)

        y = y[:, np.newaxis]
        y_plot = y_plot[:, np.newaxis]
        kde_y = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(y)
        log_dens_y = kde_y.score_samples(y_plot)
        scale_to_max = np.max(ny) / np.max(np.exp(log_dens_y))
        ax_histy.fill_betweenx(y_plot[:, 0], 0, np.exp(log_dens_y) * scale_to_max, fc='lightsteelblue', zorder=2)

    return fig, ax, ax_histx, ax_histy


def plot_violin(data, positions, density_directions, facecolors, edgecolor, clrs, qlrs,
                axis_quartiles=0, widths=0.5, bw_method=None,
                plot_median=True, plot_quartile=False, plot_whiskers=False,
                median_marker='_', median_marker_size=25,
                fig=None, ax2=None):

    if not fig:
        fig, ax2 = plt.subplots()

    #ax2.set_title('Customized violin plot')
    parts = ax2.violinplot(data, showmeans=False, showmedians=False, showextrema=False, half_violin=True, widths=widths,
                           positions=positions, density_direction=density_directions, points=100, bw_method=bw_method)

    for ind, pc in enumerate(parts['bodies']):
        pc.set_facecolor(facecolors[ind])
        if edgecolor is None:
            pc.set_edgecolor(facecolors[ind])
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=axis_quartiles)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = positions
    if plot_median:
        ax2.scatter(inds, medians, c=clrs, marker=median_marker, s=median_marker_size, zorder=3)
    if plot_quartile:
        ax2.vlines(inds, quartile1, quartile3, colors=qlrs, linestyle='-', lw=2, alpha=1)
    if plot_whiskers:
        ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # set style for the axes
    """labels = [30, 45]
    for ax in [ax2]:
        set_axis_style(ax, labels)"""

    """if density_directions[0] > 0:
        plt.show()
    j = 1"""

    return fig, ax2


def plot_arrays_on_one_axis(x, ys):
    fig, ax = plt.subplots()

    for y in ys:
        ax.plot(x, y)

    return fig, ax


def plot_arrays_on_two_subplots(x1, y1s, x2, y2s, y12s=None, y22s=None, rows_or_columns='rows', sharex=False,
                                sharey=False, smooth=False):

    if rows_or_columns == 'rows':
        if sharex:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2)
    else:
        if sharey:
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2)

    for y1 in y1s:
        ax1.plot(x1, y1)

    if y12s is not None:
        ax12 = ax1.twinx()
        for y12 in y12s:
            ax12.plot(x1, y12, linestyle='--')
    else:
        ax12 = None

    for y2 in y2s:
        ax2.plot(x2, y2)

    if y22s is not None:
        ax22 = ax2.twinx()
        for y22 in y22s:
            ax22.plot(x2, y22, linestyle='--')
    else:
        ax22 = None

    plt.tight_layout()

    return fig, ax1, ax2, ax12, ax22


# ------------------------------------- PARTICLE DISTRIBUTIONS   -------------------------------------------------------

def plot_dist_errorbars(dficts, xparameter='index', yparameter='z'):
    """
    Error bar plot particle distribution of 'yparameter' as 'xparameter'.
    """

    fig, ax = plt.subplots(figsize=(7.25, 4.25))

    for name, df in dficts.items():

        # plot
        """ax1.scatter(dfg_bid.id, dfg_bid.z_corr, s=5, label=_id)

        _id_z_corr = dfg_bid.z_corr.mean()
        ax2.errorbar(_id, dfg_bid.z_corr.mean(), yerr=dfg_bid.z_corr.std(), fmt='o', elinewidth=3, capsize=4,
                     alpha=0.75)
        ax2.scatter(_id, dfg_bid.z_corr.mean(), s=5)"""
        pass


    return fig, ax


# ---------------------------------   ONE-OFF PLOTTING FUNCTIONS   -----------------------------------------------------


def plot_multi_optimal_cm_via_percent_change_diff(df, split_column='index', true_percent=False, smooth_plots=False):

    x = df.cm_threshold.to_numpy()
    y_sigma = df.rmse_z.to_numpy()

    if true_percent:
        y_percent = df.true_percent_meas.to_numpy()
    else:
        y_percent = df.percent_meas.to_numpy()

    pc_sigma = (y_sigma[0] - y_sigma) / y_sigma[0]
    pc_percent = (y_percent[0] - y_percent) / y_percent[0]
    pc_diff = pc_sigma - pc_percent

    fig, ax1, ax2, ax12, ax22 = plot_arrays_on_two_subplots(x1=x, y1s=[y_sigma], x2=x, y2s=[pc_diff],
                                                            y12s=[y_percent], y22s=None,
                                                            rows_or_columns='rows', sharex=True, smooth=smooth_plots)

    best_cm = x[np.argmax(pc_diff)]
    ax1.axvline(x=best_cm, ymin=0, ymax=1, color='red', linestyle=':', alpha=0.125)
    ax2.axvline(x=best_cm, ymin=0, ymax=1, color='red', linestyle=':', alpha=0.125)
    ax2.scatter(best_cm, np.max(pc_diff), s=5, color='red', alpha=0.5)

    ax1.set_ylabel(r'$\sigma_{z} / h$')
    ax12.set_ylabel(r'$\phi$')
    ax1.grid(alpha=0.125)
    ax2.set_xlabel(r'$c_m$')
    ax2.set_ylabel(r'$\overline{\sigma_{z}}/\overline{\phi}$')
    ax2.grid(alpha=0.125)

    plt.tight_layout()

    return fig, ax1, ax2, ax12, ax22


def plot_single_optimal_cm_via_percent_change_diff(df, true_percent=False, smooth_plots=False):

    x = df.cm_threshold.to_numpy()
    y_sigma = df.rmse_z.to_numpy()

    if true_percent:
        y_percent = df.true_percent_meas.to_numpy()
    else:
        y_percent = df.percent_meas.to_numpy()

    pc_sigma = (y_sigma[0] - y_sigma) / y_sigma[0]
    pc_percent = (y_percent[0] - y_percent) / y_percent[0]
    pc_diff = pc_sigma - pc_percent

    fig, ax1, ax2, ax12, ax22 = plot_arrays_on_two_subplots(x1=x, y1s=[y_sigma], x2=x, y2s=[pc_diff],
                                                            y12s=[y_percent], y22s=None,
                                                            rows_or_columns='rows', sharex=True, smooth=smooth_plots)

    best_cm = x[np.argmax(pc_diff)]
    ax1.axvline(x=best_cm, ymin=0, ymax=1, color='red', linestyle=':', alpha=0.125)
    ax2.axvline(x=best_cm, ymin=0, ymax=1, color='red', linestyle=':', alpha=0.125)
    ax2.scatter(best_cm, np.max(pc_diff), s=5, color='red', alpha=0.5)

    ax1.set_ylabel(r'$\sigma_{z} / h$')
    ax12.set_ylabel(r'$\phi$')
    ax1.grid(alpha=0.125)
    ax2.set_xlabel(r'$c_m$')
    ax2.set_ylabel(r'$\overline{\sigma_{z}}/\overline{\phi}$')
    ax2.grid(alpha=0.125)

    plt.tight_layout()

    return fig, ax1, ax2, ax12, ax22


def plot_normalized_sigma_by_percent(df, smooth_plots=False):

    x = df.cm_threshold.to_numpy()
    y_sigma = df.rmse_z.to_numpy()
    y_true = df.true_percent_meas.to_numpy()
    y_percent = df.percent_meas.to_numpy()

    norm_true = (y_sigma / y_sigma[0]) / (y_true / y_true[0])
    norm_percent = (y_sigma / y_sigma[0]) / (y_percent / y_percent[0])

    ys = [norm_true, norm_percent]
    fig, ax = plot_arrays_on_one_axis(x, ys)

    ax.set_xlabel(r'$c_m$')
    ax.set_ylabel(r'$\tilde{\sigma}_{z}/\tilde{\phi}$')
    ax.grid(alpha=0.125)
    ax.legend([r'$\phi$', r'$\phi_{ID}$'], loc='lower left')

    plt.tight_layout()

    return fig, ax


def plot_3d_scatter_and_plane(df, z_param, p_xyz, fit_plane_params, x_param='x', y_param='y'):

    # get dataframe points
    x = df[x_param].to_numpy()
    y = df[y_param].to_numpy()
    z = df[z_param].to_numpy()

    # get plane points
    px, py, pz = p_xyz[0], p_xyz[1], p_xyz[2]
    d, normal = fit_plane_params[3], fit_plane_params[4]

    fig = plt.figure(figsize=(6.5, 5))

    for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):

        ax = fig.add_subplot(2, 2, i, projection='3d')

        sc = ax.scatter(x, y, z, c=z, s=1)
        ax.plot_surface(px, py, pz, alpha=0.4, color='black')

        ax.view_init(5, v)
        ax.patch.set_alpha(0.0)
        if i == 2:
            plt.colorbar(sc, shrink=0.5)
            ax.get_xaxis().set_ticks([])
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        elif i == 4:
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        else:
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.get_zaxis().set_ticklabels([])

    plt.suptitle(r"$0 = n_x x + n_y y + n_z z - d$" + "= {}x + {}y + {}z - {} \n"
                                                      "(x, y: pixels; z: microns)".format(np.round(normal[0], 3),
                                                                                        np.round(normal[1], 3),
                                                                                        np.round(normal[2], 3),
                                                                                        np.round(d, 3)),
                 y=0.875)
    plt.subplots_adjust(hspace=-0.1, wspace=0.15)

    return fig


def plot_theoretical_gaussian_diameter(z_range, theoretical_diameter_params_path, zf_at_zero, mag_eff):

    # prepare diameter function
    diameter_params = pd.read_excel(theoretical_diameter_params_path, index_col=0)

    if zf_at_zero is True:
        zf = 0
    else:
        zf = diameter_params.loc[['zf_from_nsv']]['mean'].values[0]

    if 'pop_c1' in diameter_params.columns:
        c1 = diameter_params.loc[['pop_c1']]['mean'].values[0]
        c2 = diameter_params.loc[['pop_c2']]['mean'].values[0]
    else:
        c1 = diameter_params.loc[['c1']]['mean'].values[0]
        c2 = diameter_params.loc[['c2']]['mean'].values[0]

    def theoretical_diameter_function(z):
        return mag_eff * np.sqrt(c1 ** 2 * (z - zf) ** 2 + c2 ** 2)

    gauss_diameter = theoretical_diameter_function(z_range)

    fig, ax = plt.subplots()
    ax.plot(z_range, gauss_diameter)

    return fig, ax


def plot_intrinsic_aberrations(dict_intrinsic_aberrations, cubic=True, quartic=True, plot_type='scatter'):
    zs = dict_intrinsic_aberrations['dfai'].zs
    cms = dict_intrinsic_aberrations['dfai'].cms
    zfit = dict_intrinsic_aberrations['zfit']

    fig, ax = plt.subplots()

    if plot_type == 'scatter':
        ax.scatter(zs, cms, s=1, alpha=0.0625, label='data')
    elif plot_type == 'errorbar':
        ax.scatter(zs, cms, s=1, alpha=0.0625, label='data')

    if cubic:
        cmfit_cubic = dict_intrinsic_aberrations['cmfit_cubic']
        ax.plot(zfit, cmfit_cubic, color='black', linewidth=0.5, alpha=0.75, label='cubic')

    if quartic:
        cmfit_quartic = dict_intrinsic_aberrations['cmfit_quartic']
        ax.plot(zfit, cmfit_quartic, color='black', linewidth=0.5, alpha=0.75, linestyle='--', label='quartic')

    ax.grid(alpha=0.125)
    ax.set_ylim([-0.15, 0.15])

    return fig, ax


def plot_calib_stack_self_similarity(df, min_percent_layers=0.75):
    min_layers = df.layers.max() * min_percent_layers

    df = df[df['layers'] > min_layers]
    dfgm = df.groupby('z').mean().reset_index()
    dfgstd = df.groupby('z').std().reset_index()

    fig, ax = plt.subplots()
    ax.scatter(dfgm.z, dfgm.cm, s=2)
    ax.errorbar(dfgm.z, dfgm.cm, yerr=dfgstd.cm, fmt='o', ms=1, elinewidth=0.25, capsize=1, alpha=0.5)
    ax.plot(dfgm.z, dfgm.cm, color='gray', alpha=0.35)

    return fig, ax


def plot_particle_to_particle_similarity(df, min_particles_per_frame=10):

    dfpp = modify.groupby_stats(df, group_by='frame', drop_columns=['image', 'template'])
    dfpp = dfpp[dfpp['z_counts'] > min_particles_per_frame]
    dfpp = dfpp.sort_values('z')

    fig, ax = plt.subplots()
    ax.errorbar(dfpp.z, dfpp.cm, yerr=dfpp.cm_std, fmt='o', ms=1, elinewidth=0.5, capsize=2)
    ax.plot(dfpp.z, dfpp.cm, color='gray', alpha=0.5)

    return fig, ax


# ---------------------------------   HELPER FUNCTIONS   -----------------------------------------------------


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(ticks=list(np.arange(1, len(labels) + 1)), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])