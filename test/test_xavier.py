# test bin, analyze, and plot functions
import itertools
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, CloughTocher2DInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

import filter
import analyze
from correction import correct
from utils import fit, functions, bin, io, plotting, modify, plot_collections
from utils.plotting import lighten_color

# A note on SciencePlots colors
"""
Blue: #0C5DA5
Green: #00B945
Red: #FF9500
Orange: #FF2C00

Other Colors:
Light Blue: #7BC8F6
Paler Blue: #0343DF
Azure: #069AF3
Dark Green: #054907
"""

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])  # 'ieee', 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# A. EVALUATE STAGE TILT ON CALIBRATION COORDS

path_results = '/Users/mackenzie/Downloads/interp3d/results'
path_figs = path_results

# file paths
if path_figs is not None:
    path_calib_surface = path_results + '/calibration-surface'
    if not os.path.exists(path_calib_surface):
        os.makedirs(path_calib_surface)

# read coords
dfc = pd.read_excel('/Users/mackenzie/Downloads/interp3d/Data2.xlsx')
ppx = 'temp'
ppy = 'conc'
ppz = 'curr'
# ---



# fit spline to 'raw' data
bispl_raw, rmse = fit.fit_3d_spline(x=dfc[ppx],
                                    y=dfc[ppy],  # raw: dfc.y, mirror-y: img_yc * 2 - dfc.y
                                    z=dfc[ppz],
                                    kx=1,
                                    ky=2)

# function to plot spline
def scatter_3d_and_spline(x, y, z, bispl,
                          cmap='RdBu', grid_resolution=25, view='multi', units=r'$(pixels)$',
                          bispl_z_offset=0, zlim_range=None,
                          scatter_size=1, scatter_cmap='cool', scatter_alpha=0.8, surface_alpha=0.3):
    # setup data points for calculating surface model
    model_x_data = np.linspace(min(x), max(x), grid_resolution)
    model_y_data = np.linspace(min(y), max(y), grid_resolution)

    # create coordinate arrays for vectorized evaluations
    X, Y = np.meshgrid(model_x_data, model_y_data)
    Z = bispl.ev(X, Y) + bispl_z_offset

    if zlim_range is not None:
        zlim = [np.mean(Z) - zlim_range, np.mean(Z) + zlim_range]

    # plot
    if view == 'multi':
        fig = plt.figure(figsize=(12, 10))
        for i, v in zip(np.arange(1, 5), [315, 0, 225, 90]):

            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(x, y, z, c=z, s=scatter_size, cmap=scatter_cmap, alpha=scatter_alpha)
            ps = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=surface_alpha)
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y$' + ' ' + units)
                # ax.set_yticks([-1200, -900, -600])
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x$' + ' ' + units)
                # ax.set_xticks([400, 800, 1200])
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x$' + ' ' + units)
                # ax.set_xticks([400, 800, 1200])
                ax.set_ylabel(r'$y$' + ' ' + units)
                # ax.set_yticks([-1200, -900, -600])
                ax.get_zaxis().set_ticklabels([])

            if zlim_range is not None:
                ax.set_zlim3d(zlim)
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

    plt.show()

# call spline plotting function
scatter_3d_and_spline(x=dfc[ppx], y=dfc[ppy], z=dfc[ppz], bispl=bispl_raw,
                          cmap='RdBu', grid_resolution=25, view='multi', units=r'$(pixels)$',
                          bispl_z_offset=0, zlim_range=None,
                          scatter_size=1, scatter_cmap='cool', scatter_alpha=0.8, surface_alpha=0.3)

# ---

# ---


# fit plane (x, y, z units: pixels)
points_pixels = np.stack((dfc[ppx], dfc[ppy], dfc[ppz])).T
px_pixels, py_pixels, pz_pixels, popt_pixels = fit.fit_3d_plane(points_pixels)
d, normal = popt_pixels[3], popt_pixels[4]

# calculate fit error
fit_results = functions.calculate_z_of_3d_plane(dfc[ppx], dfc[ppy], popt=popt_pixels)
rmse_plane, r_squared = fit.calculate_fit_error(fit_results, data_fit_to=dfc[ppz].to_numpy())

dict_fit_plane = {
                  'rmse': rmse_plane, 'r_squared': r_squared,
                  'popt_pixels': popt_pixels,
                  'px': px_pixels, 'py': py_pixels, 'pz': pz_pixels,
                  'd': d, 'normal': normal,
                  }


# param_z = dict_fit_plane['z_f']
rmse, r_squared = dict_fit_plane['rmse'], dict_fit_plane['r_squared']
# tilt_x, tilt_y = dict_fit_plane['tilt_x_degrees'], dict_fit_plane['tilt_y_degrees']
px, py, pz = dict_fit_plane['px'], dict_fit_plane['py'], dict_fit_plane['pz']
normal = dict_fit_plane['normal']
d = dict_fit_plane['d']

fig = plt.figure(figsize=(6.5, 5))

for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):

    ax = fig.add_subplot(2, 2, i, projection='3d')
    sc = ax.scatter(dfc[ppx], dfc[ppy], dfc[ppz], c=dfc[ppz], s=1)
    ax.plot_surface(px, py, pz, alpha=0.4, color='red')
    ax.view_init(5, v)
    ax.patch.set_alpha(0.0)
    if i == 2:
        plt.colorbar(sc, shrink=0.5)
        ax.get_xaxis().set_ticks([])
        ax.set_ylabel('Concentration')
        ax.set_zlabel('Current')
    elif i == 4:
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel('Temperature')
        ax.set_zlabel('Current')
    else:
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Concentration')
        ax.get_zaxis().set_ticklabels([])

# title
plt.suptitle('RMSE: {}, '.format(np.round(rmse, 3)) +
             r'$R^2$' + ': {}'.format(np.round(r_squared, 3)))

plt.subplots_adjust(hspace=-0.1, wspace=0.15)
plt.show()

j = 1