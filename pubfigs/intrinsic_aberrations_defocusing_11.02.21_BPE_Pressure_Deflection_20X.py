# imports
from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import bin, fit, functions, io, plotting, plot_collections
from utils.plot_collections import plot_spct_stats_compare_ids_by_along_param
from utils.plotting import lighten_color
from correction import correct
import analyze

"""
Blue: #0C5DA5
Green: #00B945
"""

plt.style.use(['science', 'ieee', 'std-colors'])
sci_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'

# --- structure data

# filpaths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.02.21-BPE_Pressure_Deflection_20X/' \
           'analyses/results-04.26.22_spct_stack-id-on-bpe'

# filepaths
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# --- experimental details
method = 'spct'
save_figs = True
show_figs = True

# read calibration coords
dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method=method)

# --- analysis

# filter 1: NaNs
df = dfcstats.dropna()

# filter 2: min num frames
dfcounts = df.groupby('id').count().reset_index()
remove_ids = dfcounts[dfcounts['z_corr'] < dfcounts.z_corr.max() * 0.7].id.unique()
df = df[~df.id.isin(remove_ids)]

# "filter" 3: reduce columns for simplicity
df = df[['frame', 'id', 'x', 'y', 'r', 'z_corr',
         'gauss_xc', 'gauss_yc', 'gauss_sigma_x', 'gauss_sigma_y', 'gauss_sigma_x_y',
         ]]

columns_to_bin = ['x', 'y']
image_length = 512
num_x_bins, num_y_bins = 3, 5
bins_x = np.rint(np.linspace(image_length / (2 * num_x_bins),
                             image_length - image_length / (2 * num_x_bins), num_x_bins),
                 )
bins_y = np.rint(np.linspace(image_length / (2 * num_y_bins),
                             image_length - image_length / (2 * num_y_bins), num_y_bins)
                 )

bins = [bins_x, bins_y]
plot_columns = ['gauss_dxc', 'gauss_dyc', 'gauss_sigma_x_y']
column_to_plot_along = 'z_corr'
round_to_decimals = [1, 1]

# ----------------- SPCT (IDs sliced along X, Y, Z)
for low_level_bins_to_plot in range(len(bins[1])):
    plot_spct_stats_compare_ids_by_along_param(df, columns_to_bin, bins, low_level_bins_to_plot,
                                               plot_columns, column_to_plot_along, round_to_decimals,
                                               save_figs, path_figs, show_figs)




print("Completed")