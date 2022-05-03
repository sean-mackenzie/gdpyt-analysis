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

# dx = 5: [93.0, 189.0, 284.0, 380.0, 475.0, 571.0, 666.0, 762.0, 858.0, 930] # for binning
# keys (dx=5): [7.5, 10, 15, 20, 25, 30, 35, 40, 50]  # center-to-center overlap spacing
# dx = 7.5: [79.0, 163.5, 254.0, 348.5, 447.0, 555.5, 665.0, 777.5, 900.0]
# keys (dx=7.5): [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# read files: dx ~ 5
datasets = ['synthetic grid overlap random z nl1']
save_ids = ['grid overlap random z']
subsets = ['dx-5-60-5']

test_id = 0
dataset = datasets[test_id]
save_id = save_ids[test_id]
subset = subsets[test_id]

# read .xlsx result files to dictionary
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/{}'.format(dataset)
read_path_name = join(base_path, 'test_coords', subset)
path_name = join(base_path, 'dx-combined', 'figs')
save_path_name = join(base_path, 'dx-combined', 'results')
settings_sort_strings = ['settings_id', '_coords_']
test_sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'

# split dataframe by parameters/values
column_to_split = 'x'
# splits = np.array([93.0, 189.0, 284.0, 380.0, 475.0, 571.0, 666.0, 762.0, 858.0, 930])  # 10 columns: all data
splits = np.array([93.0, 189.0, 284.0, 380.0, 475.0, 571.0, 666.0, 762.0, 837.0, 930])  # removed double overlap
round_x_to_decimal = 0

# bin data for uncertainty assessment
column_to_bin_and_assess = 'z_true'
bins = 1
round_z_to_decimal = 5

# filters for binning
h = 80
z_range = [-40.001, 40.001]
min_cm = 0.5

# identify data
keys = [60, 5, 10, 15, 20, 25, 30, 35, 40, 50]  # center-to-center overlap spacing

# split dict by key
inspect_gdpyt_by_key = 1.0
inspect_spc_by_key = 11.0
filter_keys = 0  # Note: all values greater than this will be collected

# read files
dficts = io.read_files('df', read_path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])
dfsettings = io.read_files('dict', read_path_name, settings_sort_strings, filetype, startswith=settings_sort_strings[0], columns=['parameter', 'settings'], dtype=str)
dficts_ground_truth = io.read_ground_truth_files(dfsettings)

# filter out particle ID's > 170 (because likely due to image segmentation problems)
dficts = filter.dficts_filter(dfictss=dficts, keys=['id'], values=[170], operations=['lessthan'], copy=True)

# filter out the double overlapped particle at x_true = 875
dficts_dx = filter.dficts_filter(dfictss=dficts, keys=['x_true'], values=[[870.0, 890.0]], operations=['not_between'], copy=True)
dficts_ground_truth_dx = filter.dficts_filter(dfictss=dficts_ground_truth, keys=['x'], values=[[870.0, 890.0]], operations=['not_between'], copy=True)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# read files: dx ~ 7.5
subsets = ['dx-7.5-57.5-5']
subset = subsets[test_id]
read_path_name = join(base_path, 'test_coords', subset)
splits_dxx = np.array([79.0, 163.5, 254.0, 348.5, 447.0, 555.5, 665.0, 777.5, 900.0])
keys_dxx = [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

dficts = io.read_files('df', read_path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])
dfsettings = io.read_files('dict', read_path_name, settings_sort_strings, filetype, startswith=settings_sort_strings[0], columns=['parameter', 'settings'], dtype=str)

dficts_dxx = filter.dficts_filter(dfictss=dficts, keys=['id'], values=[216], operations=['lessthan'], copy=True)
dficts_ground_truth_dxx = io.read_ground_truth_files(dfsettings)


# ----------------------------------------------------------------------------------------------------------------------
split_keys = 10

# sweep c_m
cm_i = 0.5
cm_f = 0.995
cm_steps = 250
xlabel_for_cm_sweep_keys = r'$c_{m}$'
dft_number_ylabel = r'$\frac{\phi_{ID}^2}{\phi}$'
ylim_global_cm_sweep = [-0.0005, 0.1505]
ylim_global_nd_tracking = [-0.0005, 1.5]
smooth_plots = True
show_plots = True

# start analysis

# analyze by cm sweep
analyze_cm_sweep = False

if analyze_cm_sweep:

    min_cm_sweep = np.round(np.linspace(cm_i, cm_f, cm_steps), 6)

    for min_cm in min_cm_sweep:
        save_id = save_ids[test_id] + ' cm={}'.format(min_cm)

        # ---------------------------------------------------------------
        # analyze by method: bin by number of bins

        # calculate local rmse_z
        dfbicts_dx = analyze.calculate_bin_local_rmse_z(dficts_dx, column_to_bin_and_assess, bins, min_cm, z_range,
                                                     round_z_to_decimal, dficts_ground_truth=dficts_ground_truth_dx)
        dfbicts_dxx = analyze.calculate_bin_local_rmse_z(dficts_dxx, column_to_bin_and_assess, bins, min_cm, z_range,
                                                     round_z_to_decimal, dficts_ground_truth=dficts_ground_truth_dxx)


        # get dictionary of only gdpyt
        dfbicts_gdpyt = {1.0: dfbicts_dx[1.0], 2.0: dfbicts_dxx[2.0]}

        # calculate mean measurement results
        dfm_gdpyt = analyze.calculate_bin_measurement_results(dfbicts_gdpyt, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
        dfm_gdpyt['percent_idd'] = dfm_gdpyt['num_bind'] / dfm_gdpyt['true_num_particles'] * 100
        dfm_gdpyt['dft_number'] = dfm_gdpyt['percent_meas'] / dfm_gdpyt['percent_idd']
        dfm_gdpyt['cm_threshold'] = min_cm

        # get dictionary of only spc
        dfbicts_spc = {11.0: dfbicts_dx[11.0], 12.0: dfbicts_dxx[12.0]}

        # calculate mean measurement results
        dfm_spc = analyze.calculate_bin_measurement_results(dfbicts_spc, norm_rmse_z_by_depth=h, norm_num_by_bins=bins)
        dfm_spc['percent_idd'] = dfm_spc['num_bind'] / dfm_spc['true_num_particles'] * 100
        dfm_spc['dft_number'] = dfm_spc['percent_meas'] / dfm_spc['percent_idd']
        dfm_spc['cm_threshold'] = min_cm

        if min_cm == np.min(min_cm_sweep):
            dfm_cm_sweep_gdpyt = dfm_gdpyt.copy()
            dfm_cm_sweep_spc = dfm_spc.copy()
        else:
            dfm_cm_sweep_gdpyt = pd.concat([dfm_cm_sweep_gdpyt, dfm_gdpyt])
            dfm_cm_sweep_spc = pd.concat([dfm_cm_sweep_spc, dfm_spc])
        # ---------------------------------------------------------------

    # plot static - optimal c_m
    fig, ax1, ax2, ax12, ax22 = plotting.plot_optimal_cm_via_percent_change_diff(df=dfm_cm_sweep_gdpyt, true_percent=True,
                                                                                 smooth_plots=False)
    plt.savefig(join(path_name, save_id + '_static_optimal_cm_true_percent.png'))
    if show_plots:
        plt.show()

    fig, ax1, ax2, ax12, ax22 = plotting.plot_optimal_cm_via_percent_change_diff(df=dfm_cm_sweep_gdpyt, true_percent=False,
                                                                                 smooth_plots=False)
    plt.savefig(join(path_name, save_id + '_static_optimal_cm_percent.png'))
    if show_plots:
        plt.show()

    # plot spc - optimal c_m
    fig, ax1, ax2, ax12, ax22 = plotting.plot_optimal_cm_via_percent_change_diff(df=dfm_cm_sweep_spc, true_percent=True,
                                                                                 smooth_plots=False)
    plt.savefig(join(path_name, save_id + '_spc_optimal_cm_true_percent.png'))
    if show_plots:
        plt.show()

    fig, ax1, ax2, ax12, ax22 = plotting.plot_optimal_cm_via_percent_change_diff(df=dfm_cm_sweep_spc, true_percent=False,
                                                                                 smooth_plots=False)
    plt.savefig(join(path_name, save_id + '_spc_optimal_cm_percent.png'))
    if show_plots:
        plt.show()

    fig, ax = plotting.plot_normalized_sigma_by_percent(df=dfm_cm_sweep_gdpyt, smooth_plots=False)
    plt.savefig(join(path_name, save_id + '_static_norm_rmse_z_by_percent.png'))
    if show_plots:
        plt.show()

    fig, ax = plotting.plot_normalized_sigma_by_percent(df=dfm_cm_sweep_spc, smooth_plots=False)
    plt.savefig(join(path_name, save_id + '_spc_norm_rmse_z_by_percent.png'))
    if show_plots:
        plt.show()