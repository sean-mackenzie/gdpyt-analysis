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
scale_fig_dim = [1, 1]
scale_fig_dim_outside_x_legend = [1.25, 1]
legend_loc = 'best'

# ------------------------------------------------
# read files
datasets = ['synthetic grid overlap random z nl1']
save_ids = ['grid overlap random z']

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

# split dataframe by parameters/values
column_to_split = 'x'
splits = np.array([93.0, 189.0, 284.0, 380.0, 475.0, 571.0, 666.0, 762.0, 858.0, 930])
round_x_to_decimal = 0

# bin data for uncertainty assessment
column_to_bin_and_assess = 'z_true'
bins = 40
round_z_to_decimal = 5

# filters for binning
h = 80
z_range = [-40.001, 40.001]
min_cm = 0.5

# assess convergence
inspect_convergence_of_key = 1.0
convergence_increments = 3

# identify data
keys = [60, 5, 10, 15, 20, 25, 30, 35, 40, 50]  # center-to-center overlap spacing

# split dict by key
inspect_gdpyt_by_key = 1.0
inspect_spc_by_key = 11.0
filter_keys = 0  # Note: all values greater than this will be collected

# -----------------------
# i/o
save_id = save_id + ' cm={}'.format(min_cm)
results_drop_columns = ['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true', 'y_true']

# -----------------------
# formatting figures

# compare static and spc
labels_compare = ['GDPyT', 'GDPT']
colors_compare = None
# compare all
ylim_compare_all = [-0.0005, 0.205]
ylim_percent_true_measured_compare_all = [0, 105]
ylim_percent_measured_compare_all = [0, 105]
ylim_cm_compare_all = [min_cm, 1.01]
# compare filtered
ylim_compare = [-0.0005, 0.15]
ylim_percent_true_measured_compare = [0, 105]
ylim_percent_measured_compare = [0, 105]
ylim_cm_compare = [min_cm, 1.01]

# local
labels_local = [lbl for lbl in keys if lbl > filter_keys]
labels_local.sort()
colors_local = None
linestyles = ['-', '--']
ylim_gdpyt = [-0.0005, 0.1]
ylim_spc = [-0.005, 0.5]
ylim_percent_true_measured_gdpyt = [0, 105]
ylim_percent_true_measured_spc = [0, 105]

# global
labels_global = ['GDPyT', 'GDPT']
colors_global = None
xlabel_for_keys = r'$\delta x (pix)$'
ylim_global = [-0.0005, 0.3105]
ylim_percent_true_measured_global = [0, 101]
ylim_percent_measured_global = [0, 101]
ylim_cm_global = [min_cm, 1.01]

# convergence assessment
ylim_global_convergence = None  # static: [0.0095, 0.0105], spc: [0.08, 0.0915]
xlabel_for_convergence = r'$N (\#\:of\:images)$'

# end setup
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# read files

dficts = io.read_files('df', read_path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])
dfsettings = io.read_files('dict', read_path_name, settings_sort_strings, filetype, startswith=settings_sort_strings[0], columns=['parameter', 'settings'], dtype=str)
dficts_ground_truth = io.read_ground_truth_files(dfsettings)

# ----------------------------------------------------------------------------------------------------------------------
# start analysis

# ---------------------------------------------------------------
# filter out particle ID's > 170 (because likely due to image segmentation problems)
dficts = filter.dficts_filter(dfictss=dficts, keys=['id'], values=[170], operations=['lessthan'], copy=True)


# ---------------------------------------------------------------
# analyze convergence of rmse_z wrt # of frames
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

# ---------------------------------------------------------------
# analyze by method: bin by number of bins

# calculate local rmse_z
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range,
                                             round_z_to_decimal, dficts_ground_truth=dficts_ground_truth)

# calculate mean measurement results and export to excel
dfm = analyze.calculate_bin_measurement_results(dfbicts, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
io.export_df_to_excel(dfm, path_name=join(save_path_name, save_id+'_measurement_results'),
                      include_index=True, index_label='test_id', filetype='.xlsx', drop_columns=results_drop_columns)

# plot methods comparison local results
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
#ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
ax2.set_ylabel(r'$\phi\left(z\right)$')
ax2.set_ylim(ylim_percent_true_measured_compare)
#ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_true_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax2.set_ylim(ylim_percent_measured_compare)
#ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare_all)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim(ylim_cm_compare)
#ax.legend(labels_compare, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_all_local_rmse_z_and_cm.png'))
plt.show()

# ---------------------------------------------------------------
# analyze by column: bin by list
plt.style.use(['science', 'ieee', 'muted'])

# -----------------------------
# split dataframes into subsets by column

# dataframes for inspection
dfgdpyt = dficts[inspect_gdpyt_by_key]
dfspc = dficts[inspect_spc_by_key]

dfsplicts_gdpyt = modify.split_df_and_merge_dficts(dfgdpyt, keys, column_to_split, splits, round_x_to_decimal)
dfsplicts_spc = modify.split_df_and_merge_dficts(dfspc, keys, column_to_split, splits, round_x_to_decimal)

# split ground truth dataframes into subsets
dfgt_gdpyt = dficts_ground_truth[inspect_gdpyt_by_key]
dfgt_spc = dficts_ground_truth[inspect_spc_by_key]

dfsplicts_gt_gdpyt = modify.split_df_and_merge_dficts(dfgt_gdpyt, keys, column_to_split, splits, round_x_to_decimal)
dfsplicts_gt_spc = modify.split_df_and_merge_dficts(dfgt_spc, keys, column_to_split, splits, round_x_to_decimal)

# -----------------------------
# local z-uncertainty (z)

# calculate local rmse_z
dfbicts_gdpyt = analyze.calculate_bin_local_rmse_z(dfsplicts_gdpyt, column_to_bin_and_assess, bins, min_cm, z_range, round_z_to_decimal, dficts_ground_truth=dfsplicts_gt_gdpyt)
dfbicts_spc = analyze.calculate_bin_local_rmse_z(dfsplicts_spc, column_to_bin_and_assess, bins, min_cm, z_range, round_z_to_decimal, dficts_ground_truth=dfsplicts_gt_spc)

# get dictionary of only gdpyt
gdpyt_ids = [xk for xk, xv in dfbicts_gdpyt.items() if xk > filter_keys]
gdpyt_dfs = [xv for xk, xv in dfbicts_gdpyt.items() if xk > filter_keys]
dfbicts_gdpyt = {gdpyt_ids[i]: gdpyt_dfs[i] for i in range(len(gdpyt_ids))}
# get dictionary of only gdpt
gdpt_ids = [xk for xk, xv in dfbicts_spc.items() if xk > filter_keys]
gdpt_dfs = [xv for xk, xv in dfbicts_spc.items() if xk > filter_keys]
dfbicts_spc = {gdpt_ids[i]: gdpt_dfs[i] for i in range(len(gdpt_ids))}

# compare local
dfbicts_compare = modify.stack_dficts([dfbicts_gdpyt, dfbicts_spc], [1.0, 11.0])

# ----------------------------------------------------------
# plot local - compare methods on filtered data
"""
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts_compare, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare)
ax.legend(labels_compare, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_filtered_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_compare, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare)
ax2.set_ylabel(r'$\phi\left(z\right)$')
ax2.set_ylim(ylim_percent_true_measured_compare)
ax.legend(labels_compare, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_filtered_local_rmse_z_and_true_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_compare, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare)
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax2.set_ylim(ylim_percent_measured_compare)
ax.legend(labels_compare, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_filtered_local_rmse_z_and_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_compare, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_compare)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim(ylim_cm_compare)
ax.legend(labels_compare, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_filtered_local_rmse_z_and_cm.png'))
plt.show()
"""
# ----------------------------------------------------------
# plot local - gdpyt
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_gdpyt)
ax.legend(labels_local, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim_outside_x_legend)
ax.set_ylim(ylim_gdpyt)
ax2.set_ylabel(r'$\phi\left(z\right)$')
ax2.set_ylim(ylim_percent_true_measured_gdpyt)
ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_local_rmse_z_and_true_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_gdpyt)
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax2.set_ylim([0, 101])
ax.legend(labels_local, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_local_rmse_z_and_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, scale=scale_fig_dim)
ax.set_ylim(ylim_gdpyt)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim([min_cm, 1.01])
ax.legend(labels_local, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_local_rmse_z_and_cm.png'))
plt.show()

# plot local - SPC
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim_outside_x_legend)
ax.set_ylim(ylim_spc)
ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim_outside_x_legend)
ax.set_ylim(ylim_spc)
ax2.set_ylabel(r'$\phi\left(z\right)$')
ax2.set_ylim(ylim_percent_true_measured_spc)
ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z_and_true_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim_outside_x_legend)
ax.set_ylim(ylim_spc)
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax2.set_ylim([0, 101])
ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z_and_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h, scale=scale_fig_dim_outside_x_legend)
ax.set_ylim(ylim_spc)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim([min_cm, 1.01])
ax.legend(labels_local, loc='upper left', bbox_to_anchor=(1.2, 1), fancybox=True, shadow=False, ncol=1)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_local_rmse_z_and_cm.png'))
plt.show()

# -----------------------------
# mean z-uncertainty by particle-to-particle spacing (column)
plt.style.use(['science', 'ieee', 'std-colors'])

# calculate mean rmse_z
dfmbicts_gdpyt = analyze.calculate_bin_local_rmse_z(dfsplicts_gdpyt, column_to_split, splits, min_cm, z_range, round_x_to_decimal, dficts_ground_truth=dfsplicts_gt_gdpyt)
dfmbicts_spc = analyze.calculate_bin_local_rmse_z(dfsplicts_spc, column_to_split, splits, min_cm, z_range, round_x_to_decimal, dficts_ground_truth=dfsplicts_gt_spc)

# stack gdpyt and spc dficts
dfmbicts_compare = modify.stack_dficts([dfmbicts_gdpyt, dfmbicts_spc], [1.0, 11.0])

# stack measurement results by bin and export to excel
dfmbicts_compare = modify.dficts_scale(dfmbicts_compare, columns=['rmse_z'], multipliers=[1/h])
dfmb = modify.stack_dficts_by_key(dfmbicts_compare)
io.export_df_to_excel(dfmb, path_name=join(save_path_name, save_id+'_dx_measurement_results'),
                      include_index=True, index_label='bin_x', filetype='.xlsx', drop_columns=results_drop_columns)

gdpyt_ids = [xk for xk, xv in dfmbicts_gdpyt.items() if xk > filter_keys]
gdpyt_dfs = [xv for xk, xv in dfmbicts_gdpyt.items() if xk > filter_keys]
dfmbicts_gdpyt = {gdpyt_ids[i]: gdpyt_dfs[i] for i in range(len(gdpyt_ids))}
# get dictionary of only gdpt
gdpt_ids = [xk for xk, xv in dfmbicts_spc.items() if xk > filter_keys]
gdpt_dfs = [xv for xk, xv in dfmbicts_spc.items() if xk > filter_keys]
dfmbicts_spc = {gdpt_ids[i]: gdpt_dfs[i] for i in range(len(gdpt_ids))}

# plot global values for static and SPC
"""
# plot global uncertainty - gdpyt
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_gdpyt, parameters='rmse_z', xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_global_rmse_z.png'))
plt.show()

parameters = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_gdpyt, parameters=parameters, xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$\phi\left(z\right)$')
ax2.set_ylim(ylim_percent_true_measured_global)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_global_rmse_z_and_true_percent_meas.png'))
plt.show()

parameters = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_gdpyt, parameters=parameters, xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax2.set_ylim([0, 101])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_global_rmse_z_and_percent_meas.png'))
plt.show()

parameters = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_gdpyt, parameters=parameters, xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim([min_cm, 1.01])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_global_rmse_z_and_cm.png'))
plt.show()

# plot global uncertainty - spc
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_spc, parameters='rmse_z', xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_global_rmse_z.png'))
plt.show()

parameters = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_spc, parameters=parameters, xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$\phi\left(z\right)$')
ax2.set_ylim(ylim_percent_true_measured_global)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_global_rmse_z_and_true_percent_meas.png'))
plt.show()

parameters = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_spc, parameters=parameters, xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax2.set_ylim([0, 101])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_global_rmse_z_and_percent_meas.png'))
plt.show()

parameters = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_global(dfmbicts_spc, parameters=parameters, xlabel=xlabel_for_keys, h=h,
                                            scale=scale_fig_dim)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim([min_cm, 1.01])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_SPC_global_rmse_z_and_cm.png'))
plt.show()
"""

# -----------------------------
# mean z-uncertainty - compare GDPyT and SPC

# plot global uncertainty - gdpyt vs. spc
fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters='rmse_z',
                                                 xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim)
ax.set_ylim(ylim_global)
ax.legend(labels_global, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z.png'))
plt.show()

parameters = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters=parameters,
                                                 xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim,
                                                 ax2_ylim=ylim_percent_true_measured_global)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$\phi$')
ax2.set_ylim(ylim_percent_true_measured_global)
ax.legend(labels_global, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z_and_true_percent_meas.png'))
plt.show()

parameters = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters=parameters,
                                                 xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim,
                                                 ax2_ylim=ylim_percent_measured_global)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$\phi_{ID}$')
ax2.set_ylim(ylim_percent_measured_global)
ax.legend(labels_global, loc=legend_loc)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z_and_percent_meas.png'))
plt.show()

parameters = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_list_global(dfbicts_list=[dfmbicts_gdpyt, dfmbicts_spc], parameters=parameters,
                                                 xlabel=xlabel_for_keys, h=h, scale=scale_fig_dim,
                                                 ax2_ylim=ylim_cm_global)
ax.set_ylim(ylim_global)
ax2.set_ylabel(r'$c_{m}$')
ax2.set_ylim(ylim_cm_global)
ax.legend(labels_global, loc='center right')
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_static_v_SPC_global_rmse_z_and_cm.png'))
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# plot several c_m's per figure
"""
dfbicts2 = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin, bins, min_cm2, z_range, round_to_decimal, dficts_ground_truth)
dfbicts3 = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin, bins, min_cm3, z_range, round_to_decimal, dficts_ground_truth)

dfict_cm_sweep = {'GDPyT cm0.5': dfbicts[1.0], #'GDPT cm0.5': dfbicts[11.0],
                  'GDPyT cm0.9': dfbicts2[1.0], #'GDPT cm0.9': dfbicts2[11.0],
                  'GDPyT cm0.95': dfbicts3[1.0], #'GDPT cm0.95': dfbicts3[11.0]
                  }

labels = [r'GDPyT($c_{m}=0.5$)', #r'GDPT($c_{m}=0.5$)',
          r'GDPyT($c_{m}=0.9$)', #r'GDPT($c_{m}=0.9$)',
          r'GDPyT($c_{m}=0.95$)', #r'GDPT($c_{m}=0.95$)',
          ]
colors = ['tab:blue', #'darkblue',
          'tab:purple', #'purple',
          'tab:pink', #'mediumvioletred']
          ]
linestyles = ['-', '-', '-', #'-', '-', '-',
              ]

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfict_cm_sweep, parameter, h, colors=colors, linestyles=linestyles,
                                           scale=[scale_fig_dim*1.5, scale_fig_dim])
ax.set_ylim([-0.01, 0.15])
ax.legend(labels, fontsize=8, bbox_to_anchor=(1.2, 1), loc='upper left', fancybox=True, shadow=False)
ax2.set_ylabel(r'$\phi\left(z\right)$', fontsize=18)
ax2.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z_sweep_cm.png'))
plt.show()

# plot several c_m's for GDPyT and GDPT per figure
dfict_cm_sweep = {'GDPyT cm0.5': dfbicts[1.0], 'GDPT cm0.5': dfbicts[11.0],
                  'GDPyT cm0.9': dfbicts2[1.0], 'GDPT cm0.9': dfbicts2[11.0],
                  'GDPyT cm0.95': dfbicts3[1.0], 'GDPT cm0.95': dfbicts3[11.0]
                  }

labels = [r'GDPyT($c_{m}=0.5$)', r'GDPT($c_{m}=0.5$)',
          r'GDPyT($c_{m}=0.9$)', r'GDPT($c_{m}=0.9$)',
          r'GDPyT($c_{m}=0.95$)', r'GDPT($c_{m}=0.95$)',
          ]
colors = ['tab:blue', 'darkblue',
          'tab:purple', 'purple',
          'tab:pink', 'mediumvioletred'
          ]
linestyles = ['-', '-', '-', '-', '-', '-',
              ]

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfict_cm_sweep, parameter, h, colors=colors, linestyles=linestyles,
                                           scale=[scale_fig_dim * 1.75, scale_fig_dim * 1.25])
ax.set_ylim([-0.01, 0.15])
ax.legend(labels, fontsize=8, bbox_to_anchor=(1.15, 1), loc='upper left', fancybox=True, shadow=False)
ax2.set_ylabel(r'$\phi\left(z\right)$', fontsize=18)
ax2.set_ylim([0, 110])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z_sweep_cm_compare.png'))
plt.show()
"""