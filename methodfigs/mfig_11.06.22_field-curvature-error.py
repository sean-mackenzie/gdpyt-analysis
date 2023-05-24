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
# 1. Setup

# setup file paths
fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
     'results-07.29.22-idpt-tmg/tests/spct_soft-baseline_1/coords/test-coords/publication-test-coords/' \
     'min_cm_0.5_z_is_z-corr-tilt/post-processed/' \
     'test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'

dfc_raw = pd.read_excel(fp)

dfc_z = dfc_raw[dfc_raw['z_true'].abs() < 10]
dfc_z = dfc_z[dfc_z['error_corr_tilt'].abs() < 4]
z_trues = dfc_z.z_true.unique()

for z_true in z_trues:
    dfc = dfc_z[dfc_z['z_true'] == z_true]

    # fit spline to 'raw' data
    kx = 2
    ky = 2
    bispl, rmse = fit.fit_3d_spline(x=dfc.x,
                                    y=dfc.y,  # raw: dfc.y, mirror-y: img_yc * 2 - dfc.y
                                    z=dfc['error_corr_tilt'],
                                    kx=kx,
                                    ky=ky)

    fig, ax = plotting.scatter_3d_and_spline(dfc.x,
                                             dfc.y,
                                             dfc['error_corr_tilt'],
                                             bispl,
                                             cmap='RdBu',
                                             grid_resolution=30,
                                             view='multi')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_zlabel(r'$error_{z} \: (\mu m)$')
    plt.suptitle('fit RMSE = {}'.format(np.round(rmse, 3)))
    path_figs = '/Users/mackenzie/Desktop/sm-test'
    plt.savefig(path_figs + '/zt={}_fit-spline-to-error_kx{}_ky{}_after-tilt-correction.png'.format(np.round(z_true, 1), kx, ky))
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(dfc.r, dfc.error_corr_tilt, 'o')
    ax.set_xlabel('r')
    ax.set_ylabel('error-z')
    plt.savefig(path_figs + '/zt={}_plot-error_kx{}_ky{}_after-tilt-correction.png'.format(np.round(z_true, 1), kx, ky))
    plt.close()