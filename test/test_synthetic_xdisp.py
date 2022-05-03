# test synthetic particles with x-displacement

from os.path import join

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd

import filter
import analyze
from correction import correct
from utils import io, plotting, modify, details

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation'
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# synthetic
path_synthetic_xdisp = join(base_dir, 'test_coords/synthetic/static-xdisp')
save_id_synthetic_xdisp = 'synthetic_xdisp'

# setup I/O
sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'
drop_columns = ['stack_id', 'z_true', 'x', 'y', 'max_sim', 'error']
results_drop_columns = ['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true', 'y_true']

# setup - binning
column_to_bin_and_assess = 'z_true'
bins = 20
mean_bins = 1
h_synthetic = 1  # (actual value == 100)
round_z_to_decimal = 5
z_range = [-65.001, 35.001]
min_cm = 0.5

save_figs_synthetic_xdisp = True
show_figs_synthetic_xdisp = True

# ---------------------------------
# 1. read .xlsx files to dictionary
dficts = io.read_dataframes(path_synthetic_xdisp, sort_strings, filetype, drop_columns=None)

# filter out the baseline image that isn't displaced
dficts = filter.dficts_filter(dficts, keys=['z_true'], values=[-15.0], operations=['notequalto'],
                              copy=True, only_keys=None, return_filtered=False)

dficts = filter.dficts_filter(dficts, keys=['error'], values=[[-10.0, 10.0]], operations=['between'],
                              copy=True, only_keys=None, return_filtered=False)

# ----------------------------------------------------------------------------------------
# Calculate uncertainty for SPCs

# 3. calculate local z-uncertainty
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range,
                                             round_z_to_decimal, dficts_ground_truth=None)

# plot setup
ylim_synthetic_xdisp = [-0.05, 5]
scale_fig_dim_legend_outside = [1.2, 1]

# 4. plot methods comparison local results
if save_figs_synthetic_xdisp or show_figs_synthetic_xdisp:
    label_dict = {key: {'label': key} for key in list(dfbicts.keys())}
    parameter = ['rmse_z', 'cm']
    fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameters=parameter, h=h_synthetic, label_dict=label_dict,
                                          scale=scale_fig_dim_legend_outside)
    ax.set_ylim(ylim_synthetic_xdisp)
    ax.set_ylabel(r'$\sigma_{z}(z)\: (\mu m)$')
    ax2.set_ylabel(r'$c_{m}$')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 0.1, 0, 1), title=r'$p_{ID,\: calib}$')
    plt.tight_layout()
    if save_figs_synthetic_xdisp:
        plt.savefig(join(path_figs, save_id_synthetic_xdisp + '_spcs_local_rmse_z_and_cm.png'))
    if show_figs_synthetic_xdisp:
        plt.show()
    plt.close(fig)