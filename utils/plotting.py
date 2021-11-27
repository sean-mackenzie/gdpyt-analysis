
# imports
import numpy as np
from scipy.interpolate import griddata

from utils import fit
import analyze

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                       scatter_on=True, scatter_size=10):

    # format figure
    if isinstance(colors, list):
        cscatter = iter(colors)
        cscatterr = iter(colors)
    elif colors == 'Blues':
        cscatter = iter(cm.Blues(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
        cscatterr = iter(cm.Blues(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
    elif colors == 'inferno':
        cscatter = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
        cscatterr = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts.keys()))))
    else:
        cscatter = None
        cscatterr = None

    if isinstance(linestyles, list):
        lstyle = iter(linestyles)
    else:
        lstyle = iter('-' for i in list(dfbicts.keys()))

    if not scale:
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

        fig, ax = plt.subplots(figsize=(size_x_inches*scalex, size_y_inches*scaley))

    # organize data
    if isinstance(parameters, str):
        parameter = parameters
        parameterr = None
    elif isinstance(parameters, list) and len(parameters) > 1:
        parameter = parameters[0]
        parameterr = parameters[1]

    if parameter == 'rmse_z':
        for item in dfbicts.items():

            if cscatter is not None:
                cs = next(cscatter)
                ls = next(lstyle)
                ax.plot(item[1].index, item[1][parameter] / h)
                if scatter_on:
                    ax.scatter(item[1].index, item[1][parameter] / h)
            else:
                ax.plot(item[1].index, item[1][parameter] / h)
                if scatter_on:
                    ax.scatter(item[1].index, item[1][parameter] / h, s=scatter_size)
    else:
        for item in dfbicts.items():

            if cscatter is not None:
                cs = next(cscatter)
                ls = next(lstyle)
                ax.plot(item[1].index, item[1][parameter])
                if scatter_on:
                    ax.scatter(item[1].index, item[1][parameter])
            else:
                ax.plot(item[1].index, item[1][parameter] / h)
                if scatter_on:
                    ax.scatter(item[1].index, item[1][parameter] / h, s=scatter_size)

    if parameterr is not None:
        ax2 = ax.twinx()
        for item in dfbicts.items():
            if cscatterr is not None:
                css = next(cscatterr)
                ax2.plot(item[1].index, item[1][parameterr])
            else:
                ax2.plot(item[1].index, item[1][parameterr], linestyle='--')

    if h != 1 and parameter == 'rmse_z':
        ax.set_ylabel(r'$\sigma_{z}\left(z\right) / h$')
    else:
        ax.set_ylabel(parameter)

    ax.set_xlabel('z ($\mu m$)')
    ax.grid(alpha=0.25)

    if show_legend:
        ax.legend(dfbicts.keys(), title=r'$\sigma$')

    if parameterr is not None:
        return fig, ax, ax2
    else:
        return fig, ax


def plot_dfbicts_global(dfbicts, parameters='rmse_z', xlabel='parameter', h=1, print_values=False,
                        scale=None, fig=None, ax=None, ax2=None, ax2_ylim=None, color=None, scatter_size=10):

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
    if isinstance(parameters, str):
        parameter = parameters
        parameterr = None

        names = dfbicts.keys()
        means = np.array([m[parameter].mean() for m in dfbicts.values()])

        sort_by_name = sorted(list(zip(names, means)), key=lambda x: x[0])
        names = [x[0] for x in sort_by_name]
        means = np.array([x[1] for x in sort_by_name])

    elif isinstance(parameters, list) and len(parameters) > 1:
        parameter = parameters[0]
        parameterr = parameters[1]

        names = dfbicts.keys()
        means = np.array([m[parameter].mean() for m in dfbicts.values()])
        means_prr = np.array([m[parameterr].mean() for m in dfbicts.values()])

        sort_by_name = sorted(list(zip(names, means, means_prr)), key=lambda x: x[0])
        names = [x[0] for x in sort_by_name]
        means = np.array([x[1] for x in sort_by_name])
        means_prr = np.array([x[2] for x in sort_by_name])
    else:
        raise ValueError("parameters must be a string or a list of strings")

    # plot figure
    if parameter == 'rmse_z' and h != 1:
        ax.plot(names, means / h)
        ax.scatter(names, means / h, s=scatter_size)
    else:
        ax.plot(names, means)
        ax.scatter(names, means, s=scatter_size)

    if h != 1 and parameter == 'rmse_z':
        ax.set_ylabel(r'$\sigma_{z} / h$')
    else:
        ax.set_ylabel(parameter)

    if parameterr is not None:
        ax2.plot(names, means_prr, linestyle='--')
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
                             scale=None, colors=None, ax2_ylim=None, scatter_size=10):
    # format figure
    if isinstance(colors, list):
        cscatter = iter(colors)
        cscatterr = iter(colors)
    elif colors == 'Blues':
        cscatter = iter(cm.Blues(np.linspace(0.1, 0.9, len(dfbicts_list))))
        cscatterr = iter(cm.Blues(np.linspace(0.1, 0.9, len(dfbicts_list))))
    elif colors == 'inferno':
        cscatter = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts_list))))
        cscatterr = iter(cm.inferno(np.linspace(0.1, 0.9, len(dfbicts_list))))
    else:
        cscatter = None
        cscatterr = None

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

    for dfbicts in dfbicts_list:
        fig, ax, ax2 = plot_dfbicts_global(dfbicts, parameters, xlabel, h, print_values,
                                           scale=scale, fig=fig, ax=ax, ax2=ax2, ax2_ylim=ax2_ylim,
                                           color=None, scatter_size=scatter_size)

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

    if not color:
        color = df.z

    ax.scatter(df.x, df.y, df.z, marker='o', c=color, alpha=alpha)

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