# test bin, analyze, and plot functions
from os.path import join
import numpy as np
import pandas as pd

from utils import io, bin, plotting, modify

import matplotlib.pyplot as plt


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
subsets = ['dx-combined']

test_id = 0
dataset = datasets[test_id]
save_id = save_ids[test_id]
subset = subsets[test_id]

# read .xlsx result files to dictionary
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/{}'.format(dataset)
path_name = join(base_path, subset, 'figs')
save_path_name = join(base_path, subset, 'results')

# ------------------------------------------------

# dx = 5: [93.0, 189.0, 284.0, 380.0, 475.0, 571.0, 666.0, 762.0, 858.0, 930] # for binning
# keys (dx=5): [5, 10, 15, 20, 25, 30, 35, 40, 50]  # center-to-center overlap spacing
# dx = 7.5: [79.0, 163.5, 254.0, 348.5, 447.0, 555.5, 665.0, 777.5, 900.0]
# keys (dx=7.5): [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

# split dataframe by parameters/values
dx_keys = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
dxx_keys = [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]
round_x_to_decimal = 0

# filters for binning
h = 80
z_range = [-40.001, 40.001]
min_cm = 0.5
save_id = save_id + '_cm={}'.format(min_cm)


# read dx=5 excel spreadsheet
filepath_dx = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/synthetic grid overlap random z nl1/dx-combined/read/grid overlap random z cm={}_dx-5_measurement_results.xlsx'.format(min_cm)
dfx = io.read_excel(path_name=filepath_dx, filetype='.xlsx')

# read dx=7.5 excel spreadsheet
filepath_dxx = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/synthetic grid overlap random z nl1/dx-combined/read/grid overlap random z cm={}_dx-7.5_measurement_results.xlsx'.format(min_cm)
dfxx = io.read_excel(path_name=filepath_dxx, filetype='.xlsx')


# get dataframe of only gdpyt
dfx_gdpyt = dfx[dfx['filename'] == 1.0].copy()
dfx_gdpyt['dx'] = dx_keys
dfxx_gdpyt = dfxx[dfxx['filename'] == 1.0].copy()
dfxx_gdpyt['dx'] = dxx_keys

# get dataframe of only gdpyt
dfx_spc = dfx[dfx['filename'] == 11.0].copy()
dfx_spc['dx'] = dx_keys
dfxx_spc = dfxx[dfxx['filename'] == 11.0].copy()
dfxx_spc['dx'] = dxx_keys

# merge dataframes
df_gdpyt = pd.concat([dfx_gdpyt, dfxx_gdpyt])
df_spc = pd.concat([dfx_spc, dfxx_spc])

# sort values
df_gdpyt = df_gdpyt.astype(float)
df_gdpyt = df_gdpyt.sort_values(by='dx')
df_gdpyt.to_excel(save_path_name + '/df_idpt.xlsx')
df_gdpyt = df_gdpyt.set_index(keys='dx')

df_spc = df_spc.astype(float)
df_spc = df_spc.sort_values(by='dx')
df_spc.to_excel(save_path_name + '/df_spct.xlsx')
df_spc = df_spc.set_index(keys='dx')

# merge into dictionary
dfbicts = {1.0: df_gdpyt, 11.0: df_spc}

# -----------------------------
# mean z-uncertainty - compare GDPyT and SPC
# formatting figures
save_plots = True
show_plots = True

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
filter_keys = 0
labels_local = [lbl for lbl in dx_keys if lbl > filter_keys]
labels_local.sort()
colors_local = None
linestyles = ['-', '--']
ylim_gdpyt = [-0.0005, 0.1]
ylim_spc = [-0.005, 0.5]
ylim_percent_true_measured_gdpyt = [0, 105]
ylim_percent_true_measured_spc = [0, 105]
ylim_num = 5000

# global
labels_global = ['GDPyT', 'GDPT']
colors_global = None
xlabel_for_keys = r'$\delta x (pix)$'
ylabel_for_sigma = r'$\sigma_{z}\left(z\right) / h$'
ylim_global = [-0.0005, 0.3105]
ylim_percent_true_measured_global = [0, 101]
ylim_percent_measured_global = [0, 101]
ylim_cm_global = [min_cm, 1.01]

# plot global uncertainty - gdpyt vs. spc
if save_plots:
    # plot local - gdpyt
    parameter = 'rmse_z'
    fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h=1, scale=scale_fig_dim, xlabel=xlabel_for_keys,
                                          ylabel=ylabel_for_sigma)
    ax.set_ylim(ylim_global)
    ax.legend(labels_global, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id+'_static_v_spc_global_dx_rmse_z.png'))
    if show_plots:
        plt.show()

    """
    parameter = ['rmse_z', 'true_percent_meas']
    fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h=1, scale=scale_fig_dim_outside_x_legend,
                                               xlabel=xlabel_for_keys, ylabel=ylabel_for_sigma)
    ax.set_ylim(ylim_global)
    ax2.set_ylabel(r'$\phi\left(z\right)$')
    ax2.set_ylim(ylim_percent_true_measured_global)
    ax.legend(labels_global, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id+'_static_v_spc_global_dx_rmse_z_and_true_percent_meas.png'))
    if show_plots:
        plt.show()
    """

    parameter = ['rmse_z', 'percent_meas']
    fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h=1, scale=scale_fig_dim,
                                               xlabel=xlabel_for_keys, ylabel=ylabel_for_sigma)
    ax.set_ylim(ylim_global)
    ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$')
    ax2.set_ylim(ylim_percent_measured_global)
    ax.legend(labels_global, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id+'_static_v_spc_global_dx_rmse_z_and_percent_meas.png'))
    if show_plots:
        plt.show()

    """
    parameter = ['rmse_z', 'num_meas', 'num_bind', 'true_num_particles']
    fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h=1, scale=scale_fig_dim,
                                               xlabel=xlabel_for_keys, ylabel=ylabel_for_sigma)
    ax.set_ylim([x * 2 for x in ylim_global])
    ax2.set_ylabel(r'$\#$')
    ax2.set_ylim([0, ylim_num])
    ax.legend(labels_global, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id+'_static_v_spc_global_dx_rmse_z_and_num_particles.png'))
    if show_plots:
        plt.show()

    parameter = ['rmse_z', 'cm']
    fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h=1, scale=scale_fig_dim,
                                               xlabel=xlabel_for_keys, ylabel=ylabel_for_sigma)
    ax.set_ylim(ylim_global)
    ax2.set_ylabel(r'$c_{m}$')
    ax2.set_ylim([min_cm, 1.01])
    ax.legend(labels_global, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(join(path_name, save_id+'_static_v_spc_global_dx_rmse_z_and_cm.png'))
    if show_plots:
        plt.show()
    """
# ---------------------------------------------------------------



j=1