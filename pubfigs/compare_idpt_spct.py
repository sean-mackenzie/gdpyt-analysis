# imports

import os
from os.path import join
import numpy as np
import pandas as pd

from utils import bin, fit, functions, io, plotting
from utils.plotting import lighten_color
from correction import correct
import analyze

import matplotlib.pyplot as plt

# formatting
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

# ------------------------------------------- COMPARE IDPT TO SPCT -----------------------------------------------------

# file paths

# read
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses'
fp_dfc_idpt = join(base_dir, 'results-05.03.22_10X-idpt-FINAL/coords/calib-coords/'
                             'calib_idpt_stats_02.06.22_membrane_characterization_calib_gen_cal.xlsx')
fp_dfc_spct = join(base_dir, 'results-07.03.22_10X-spct/coords/calib-coords/'
                             'calib_spct_stats_02.06.22_membrane_characterization_calib_gen-cal.xlsx')

# save
path_results = join(base_dir, 'shared-results')

path_results = join(path_results, 'sampling-density')
if not os.path.exists(path_results):
    os.makedirs(path_results)

# ---

# --- --- EXPERIMENTAL

microns_per_pixels = 3.2
image_size = 512
area = (image_size * microns_per_pixels) ** 2

# ---

# --- --- READ COORDS

dfi = pd.read_excel(fp_dfc_idpt)
dfs = pd.read_excel(fp_dfc_spct)

# ---

# --- --- PROCESSING

param_z = 'z_true'



# ---

# --- --- PLOT

# groupby z and plot results at every z
plot_groupby_z = False

if plot_groupby_z:

    # setup
    param_z = 'z_true'
    plot_columns = ['mean_dx_microns']
    ms = 4
    save_file_type = '.svg'

    # processing
    dfigz = dfi.groupby(param_z).count().reset_index()
    dfsgz = dfs.groupby(param_z).count().reset_index()

    # lateral sampling frequency
    dfigz['mean_dx_microns'] = np.sqrt(area / dfigz.id)
    dfsgz['mean_dx_microns'] = np.sqrt(area / dfsgz.id)

    # plot
    for pc in plot_columns:
        fig, ax = plt.subplots()
        ax.plot(dfigz[param_z] - 140, dfigz[pc], ms=ms, label='IDPT')
        ax.plot(dfsgz[param_z] - 140, dfsgz[pc], ms=ms, label='SPCT')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\overline{ \delta x} \: (\mu m)$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_results + '/compare1_{}_by_{}{}'.format(pc, param_z, save_file_type))
        plt.close()

    # ---

# ---

# bin by z and plot smoothed results at z bins
plot_bin_by_z = True

if plot_bin_by_z:

    # setup
    param_z = 'z_true'
    column_to_count = 'id'
    z_bins = int(len(dfi[param_z].unique()) / 3)
    plot_columns = ['mean_dx_microns']
    ms = 4
    save_file_type = '.svg'

    # ---

    # processing
    dfi = dfi.groupby(param_z).count().reset_index()
    dfs = dfs.groupby(param_z).count().reset_index()

    # lateral sampling frequency
    dfi['mean_dx_microns'] = np.sqrt(area / dfi.id)
    dfs['mean_dx_microns'] = np.sqrt(area / dfs.id)

    dfim, dfistd = bin.bin_generic(dfi,
                                   column_to_bin=param_z,
                                   column_to_count=column_to_count,
                                   bins=z_bins,
                                   round_to_decimal=1,
                                   return_groupby=True,
                                   )
    dfsm, dfsstd = bin.bin_generic(dfs,
                                   column_to_bin=param_z,
                                   column_to_count=column_to_count,
                                   bins=z_bins,
                                   round_to_decimal=1,
                                   return_groupby=True,
                                   )

    # ---

    # plot
    for pc in plot_columns:
        fig, ax = plt.subplots()
        ax.plot(dfim[param_z] - 140, dfim[pc], '-o', ms=ms, label='IDPT')
        ax.plot(dfsm[param_z] - 140, dfsm[pc], '-o', ms=ms, label='SPCT')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\overline{ \delta x} \: (\mu m)$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_results + '/compare_{}_by_binned-{}{}'.format(pc, param_z, save_file_type))
        plt.close()

    # ---

# ---


print("Analysis completed without errors.")