# test bin, analyze, and plot functions
from os.path import join
import ast
import numpy as np
import pandas as pd
import analyze
from utils import io, bin, plotting, modify
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# formatting
fontP = FontProperties()
fontP.set_size('large')
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'

datasets = ['2.1.1 standard dataset',
            '2.1.2 synthetic grid no overlap dataset/2.1.2.1 uniform z',
            '2.1.2 synthetic grid no overlap dataset/2.1.2.2 random z',
            '2.1.3 synthetic grid overlap dataset/2.1.3.1 uniform z',
            '2.1.3 synthetic grid overlap dataset/2.1.3.2 random z',
            '2.1.4 synthetic density dataset/2.1.4.1 uniform z',
            'S.1.1 baseline image',
            'S.1.2 same ID thresh']

save_ids = ['2.1.1_Dataset_I',
            '2.1.2.1_uniform_z', '2.1.2.2_random_z', '2.1.3.1_uniform_z', '2.1.3.2_random_z',
            '2.1.4.1_uniform_z']

test_id = 0
dataset = datasets[test_id]
save_id = save_ids[test_id]

# analysis
column_to_bin = 'z_true'
bins = 25
min_cm = 0.5
min_cm2 = 0.9
min_cm3 = 0.95
z_range = [-67.001, 18.001]
h = 86
round_to_decimal = 4

# format plots
labels = ['GDPyT', 'GDPT']
colors = ['tab:blue', 'darkblue']
linestyles = None  # ['-', '-']
scale_fig_dim = 1.5

# read .xlsx result files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/Iteration 2 - data by figure/{}'.format(dataset)
settings_sort_strings = ['settings_id', '_coords_']
test_sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'

# end setup
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# start analysis

dficts = io.read_files('df', path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])
dfsettings = io.read_files('dict', path_name, settings_sort_strings, filetype, startswith=settings_sort_strings[0], columns=['parameter', 'settings'], dtype=str)
dficts_ground_truth = io.read_ground_truth_files(dfsettings)

# calculate local rmse_z
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin, bins, min_cm, z_range, round_to_decimal, dficts_ground_truth)

# get dictionary of only gdpyt
gdpyt_ids = [xk for xk, xv in dfbicts.items() if xk < 10]
gdpyt_dfs = [xv for xk, xv in dfbicts.items() if xk < 10]
dfbicts_gdpyt = {gdpyt_ids[i]: gdpyt_dfs[i] for i in range(len(gdpyt_ids))}
# get dictionary of only gdpt
gdpt_ids = [xk for xk, xv in dfbicts.items() if xk > 10]
gdpt_dfs = [xv for xk, xv in dfbicts.items() if xk > 10]
dfbicts_gdpt = {gdpt_ids[i]: gdpt_dfs[i] for i in range(len(gdpt_ids))}

# plot local - gdpyt
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
#ax.legend(labels, prop=fontP, loc='upper left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
ax2.set_ylabel(r'$\phi\left(z\right)$', fontsize=18)
ax2.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z_and_true_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
ax2.set_ylabel(r'$c_{m}$', fontsize=18)
ax2.set_ylim([min_cm, 1.01])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z_and_cm.png'))
plt.show()

parameter = ['true_percent_meas', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$', fontsize=18)
ax2.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_true_percent_meas_and_id_meas.png'))
plt.show()

parameter = ['true_percent_meas', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
ax2.set_ylabel(r'$c_{m}$', fontsize=18)
ax2.set_ylim([min_cm, 1.01])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_true_percent_meas_and_cm.png'))
plt.show()

parameter = 'true_percent_meas'
fig, ax = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
#ax.legend(labels, prop=fontP, loc='lower left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_true_percent_meas.png'))
plt.show()

parameter = 'percent_meas'
fig, ax = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax.set_ylim([0, 105])
#ax.legend(labels, prop=fontP, loc='lower left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_percent_meas.png'))
plt.show()

parameter = 'cm'
fig, ax = plotting.plot_dfbicts_local(dfbicts_gdpyt, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$c_{m}$')
ax.set_ylim([min_cm, 1.05])
#ax.legend(labels, prop=fontP, loc='lower left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_cm.png'))
plt.show()

# plot local - compare gdpyt to gdpt
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
#ax.legend(labels, prop=fontP, loc='upper left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'true_percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
ax2.set_ylabel(r'$\phi\left(z\right)$', fontsize=18)
ax2.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_rmse_z_and_true_percent_meas.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
ax2.set_ylabel(r'$c_{m}$', fontsize=18)
ax2.set_ylim([min_cm, 1.01])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_rmse_z_and_cm.png'))
plt.show()

parameter = ['true_percent_meas', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
ax2.set_ylabel(r'$\phi_{ID}\left(z\right)$', fontsize=18)
ax2.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_true_percent_meas_and_id_meas.png'))
plt.show()

parameter = ['true_percent_meas', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
ax2.set_ylabel(r'$c_{m}$', fontsize=18)
ax2.set_ylim([min_cm, 1.01])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_true_percent_meas_and_cm.png'))
plt.show()

parameter = 'true_percent_meas'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
#ax.legend(labels, prop=fontP, loc='lower left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_true_percent_meas.png'))
plt.show()

parameter = 'percent_meas'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax.set_ylim([0, 105])
#ax.legend(labels, prop=fontP, loc='lower left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_percent_meas.png'))
plt.show()

parameter = 'cm'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$c_{m}$')
ax.set_ylim([min_cm, 1.05])
#ax.legend(labels, prop=fontP, loc='lower left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_local_cm.png'))
plt.show()

# plot global - only gdpyt
parameter = 'rmse_z'
xlabel = 'sigma'
fig, ax = plotting.plot_dfbicts_global(dfbicts_gdpyt, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylim([-0.01, 0.15])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpyt_global_rmse_z.png'))
plt.show()

parameter = 'true_percent_meas'
fig, ax = plotting.plot_dfbicts_global(dfbicts_gdpyt, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpyt_global_true_percent_meas.png'))
plt.show()

parameter = 'percent_meas'
fig, ax = plotting.plot_dfbicts_global(dfbicts_gdpyt, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpyt_global_percent_meas.png'))
plt.show()

# plot global - only gdpt
parameter = 'rmse_z'
xlabel = 'sigma'
fig, ax = plotting.plot_dfbicts_global(dfbicts_gdpt, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylim([-0.01, 0.15])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpt_global_rmse_z.png'))
plt.show()

parameter = 'true_percent_meas'
fig, ax = plotting.plot_dfbicts_global(dfbicts_gdpt, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpt_global_true_percent_meas.png'))
plt.show()

parameter = 'percent_meas'
fig, ax = plotting.plot_dfbicts_global(dfbicts_gdpt, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpt_global_percent_meas.png'))
plt.show()

# plot global - compare gdpyt and gpdt
parameter = 'rmse_z'
xlabel = 'sigma'
fig, ax = plotting.plot_dfbicts_global(dfbicts, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylim([-0.01, 0.15])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_global_rmse_z.png'))
plt.show()

parameter = 'true_percent_meas'
fig, ax = plotting.plot_dfbicts_global(dfbicts, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylabel(r'$\phi\left(z\right)$')
ax.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_global_true_percent_meas.png'))
plt.show()

parameter = 'percent_meas'
fig, ax = plotting.plot_dfbicts_global(dfbicts, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax.set_ylim([0, 105])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_compare_global_percent_meas.png'))
plt.show()


# plot several c_m's per figure
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