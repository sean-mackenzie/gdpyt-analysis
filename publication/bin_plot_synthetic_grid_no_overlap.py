# test bin, analyze, and plot functions
from os.path import join
import ast
import numpy as np
import pandas as pd
import analyze
from utils import io, bin, plotting, modify
import filter
from tracking import plotting as trackplot

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# ------------------------------------------------
# formatting
plt.style.use(['science', 'ieee', 'std-colors'])
dpi = 100
scale_fig_dim = [1, 1]
legend_loc = 'upper left'

# ------------------------------------------------
# read files
datasets = ['synthetic grid no overlap random z nl1']
save_ids = ['grid no overlap random z']

test_id = 0
dataset = datasets[test_id]
save_id = save_ids[test_id]

# read .xlsx result files to dictionary
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/{}'.format(dataset)
read_path_name = join(base_path, 'test_coords')
path_name = join(base_path, 'figs')
save_path_name = join(base_path, 'results')
settings_sort_strings = ['settings_id', '_coords_']
test_sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'

# ------------------------------------------------

# bin data for uncertainty assessment
column_to_bin_and_assess = 'z_true'
bins = 1
round_z_to_decimal = 5

# filters for binning
h = 80
z_range = [-40.001, 40.001]
min_cm = 0.9
min_cm2 = 0.9

# split dict by key
inspect_gdpyt_by_key = 1.0
inspect_spc_by_key = 11.0

# assess convergence
inspect_convergence_of_keys = np.arange(10.0, 19.0, 1)
convergence_increments = 1

# -----------------------
# i/o
save_id = save_id + ' cm={}'.format(min_cm)
results_drop_columns = ['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true', 'y_true']

# -----------------------
# formatting figures

# compare static and spc
labels_compare = ['IDPT', 'SPCT']  # [0, 1, 2, 3, 4, 5, 6, 7, 8]  # ['GDPyT', 'GDPT']
colors_compare = None  # ['tab:blue', 'darkblue']
# compare all
ylim_compare_all = [-0.01, 0.1]
ylim_percent_true_measured_compare_all = [0, 105]
ylim_percent_measured_compare_all = [0, 105]
ylim_cm_compare_all = [min_cm, 1.01]
# compare filtered
ylim_compare = [-0.01, 0.1]
ylim_percent_true_measured_compare = [0, 105]
ylim_percent_measured_compare = [0, 105]
ylim_cm_compare = [min_cm, 1.01]

# local
labels_local = labels_compare
colors_local = None
linestyles = None
ylim_gdpyt = [-0.01, 0.15]
ylim_spc = [-0.01, 0.25]
ylim_percent_true_measured_gdpyt = [0, 105]
ylim_percent_true_measured_spc = [0, 105]

# global
labels_global = ['GDPyT', 'GDPT']  # [0, 1, 2, 3, 4, 5, 6, 7, 8]  # ['GDPyT', 'GDPT']  #
# colors_global = ['tab:blue', 'darkblue']
ylim_global = [-0.01, 0.1]
ylim_percent_true_measured_global = [0, 105]
ylim_percent_measured_global = [0, 105]
ylim_cm_global = [min_cm, 1.01]

# convergence assessment
ylim_global_convergence = None  # static: [0.0095, 0.0105], spc: [0.08, 0.0915]
xlabel_for_convergence = r'$N (\#\:of\:images)$'

# end setup
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# read files

dficts = io.read_files('df', read_path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])
dfsettings = io.read_files('dict', read_path_name, settings_sort_strings, filetype, startswith=settings_sort_strings[0],
                           columns=['parameter', 'settings'], dtype=str)
dficts_ground_truth = io.read_ground_truth_files(dfsettings)

# ----------------------------------------------------------------------------------------------------------------------
# start analysis

# ---------------------------------------------------------------
# filter out particle ID's > 170 (because likely due to image segmentation problems)
dficts = filter.dficts_filter(dfictss=dficts, keys=['id'], values=[100], operations=['lessthan'], copy=True)

# get dictionary of only gdpt
"""
filter_keys = 9
gdpt_ids = [xk for xk, xv in dficts.items() if xk > filter_keys]
gdpt_dfs = [xv for xk, xv in dficts.items() if xk > filter_keys]
dficts_spc = {gdpt_ids[i]: gdpt_dfs[i] for i in range(len(gdpt_ids))}

# stack spc dataframes
df_avg = modify.stack_dficts_by_key(dficts_spc, drop_filename=True)
dficts_avg = {1.0: dficts[1.0],
              11.0: df_avg}

# stack spc ground truth dataframes
gdpt_gt_ids = [xk for xk, xv in dficts_ground_truth.items() if xk > filter_keys]
gdpt_gt_dfs = [xv for xk, xv in dficts_ground_truth.items() if xk > filter_keys]
dficts_gt_spc = {gdpt_gt_ids[i]: gdpt_gt_dfs[i] for i in range(len(gdpt_gt_ids))}

df_ground_truth_avg = modify.stack_dficts_by_key(dficts_gt_spc, drop_filename=True)
dficts_ground_truth_avg = {1.0: dficts_ground_truth[1.0],
                           11.0: df_ground_truth_avg}
"""
# ---------------------------------------------------------------
# analyze by method: bin by number of bins

# calculate local rmse_z
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range,
                                             round_z_to_decimal, dficts_ground_truth=dficts_ground_truth)

# calculate mean measurement results and export to excel
"""
"""
dfm = analyze.calculate_bin_measurement_results(dfbicts, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
io.export_df_to_excel(dfm, path_name=join(save_path_name, save_id+'_measurement_results'),
                      include_index=True, index_label='test_id', filetype='.xlsx', drop_columns=results_drop_columns)

# plot methods comparison local results
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
ax.legend(labels_compare, loc=legend_loc)  # , loc='upper left', fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
ax2.set_ylabel(r'$\phi\left(z\right)$')
ax2.set_ylim(ylim_percent_true_measured_compare)
ax.legend(labels_compare, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_true_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax2.set_ylim(ylim_percent_measured_compare)
ax.legend(labels_compare, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim(ylim_cm_compare)
ax.legend(labels_compare, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_cm.png'))
plt.show()

# ---------------------------------------------------------------
# analyze convergence of rmse_z wrt # of frames
"""
for inspect_convergence_of_key in inspect_convergence_of_keys:

    dficts_cumlative, dficts_ground_truth_cumlative = modify.split_dficts_cumulative_series(dficts, dficts_ground_truth,
                                                                                            series_parameter='frame',
                                                                                            increments=convergence_increments,
                                                                                            key=inspect_convergence_of_key)
    # calculate local rmse_z
    dfbicts_cumlative = analyze.calculate_bin_local_rmse_z(dficts_cumlative, column_to_bin_and_assess, bins, min_cm, z_range,
                                                 round_z_to_decimal, dficts_ground_truth=dficts_ground_truth_cumlative)

    # plot global uncertainty - gdpyt
    fig, ax, ax2 = plotting.plot_dfbicts_global(dfbicts_cumlative, parameters='rmse_z', xlabel=xlabel_for_convergence, h=h,
                                                scale=scale_fig_dim)
    if ylim_global_convergence:
        ax.set_ylim(ylim_global_convergence)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id+'_convergence_key{}_global_rmse_z.png'.format(inspect_convergence_of_key)))
    plt.show()

    # calculate mean measurement results and export to excel
    dfm_cumlative = analyze.calculate_bin_measurement_results(dfbicts_cumlative, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
    io.export_df_to_excel(dfm_cumlative, path_name=join(save_path_name, save_id+'_measurement_convergence_key{}'.format(inspect_convergence_of_key)),
                          include_index=True, index_label='test_id', filetype='.xlsx', drop_columns=results_drop_columns)
"""
# ---------------------------------------------------------------