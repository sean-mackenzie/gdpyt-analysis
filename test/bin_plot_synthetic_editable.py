# test bin, analyze, and plot functions
from os.path import join
import ast
import numpy as np
import pandas as pd
import analyze
from utils import io, bin, plotting, modify
from tracking import plotting as trackplot

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# formatting
fontP = FontProperties()
fontP.set_size('large')
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'


dataset = 'calib1_testcalib23'
save_id = 'calib1_testcalib23'

# analysis
column_to_bin = 'z_true'
bins = 25
min_cm = 0.5
min_cm2 = 0.9
min_cm3 = 0.95
z_range = [0.9999, 71.0001]
h = 70
round_to_decimal = 4

# format plots
labels = [r'$2_{1img}$', r'$2_{3img}$', r'$3_{1img}$', r'$3_{3img}$']
colors = ['tab:blue', 'darkblue', 'tab:purple', 'purple']
linestyles = ['-', 'dotted', '-', 'dotted']
scale_fig_dim = [1.75, 1.5]

# read .xlsx result files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.02.21-BPE_Pressure_Deflection_20X/analyses/compare_calib_stacks/test_coords'
test_sort_strings = ['test_coords_id', '_calib1_']
filetype = '.xlsx'

# end setup
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# start analysis

dficts = io.read_files('df', path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])

# plot single partilce
fig, ax = trackplot.plot_scatter(dficts, pid=1, xparameter='frame', yparameter='z', min_cm=0.5, z0=0, take_abs=False)
plt.show()

# calculate local rmse_z
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin, bins, min_cm, z_range, round_to_decimal,
                                             dficts_ground_truth=None)

# plot local - gdpyt
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.25])
ax.legend(labels, prop=fontP, loc='upper left',bbox_to_anchor=(1.1, 1.0) , fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z.png'))
plt.show()

parameter = ['rmse_z', 'cm']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.25])
ax2.set_ylabel(r'$c_{m}$', fontsize=18)
ax2.set_ylim([min_cm, 1.01])
ax.legend(labels, prop=fontP, loc='upper left',bbox_to_anchor=(1.1, 1.0) , fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z_and_cm.png'))
plt.show()

parameter = 'percent_meas'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim)
ax.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax.set_ylim([98, 100.5])
ax.legend(labels, prop=fontP, loc='upper left', bbox_to_anchor=(1.1, 1.0) , fancybox=True, shadow=False, title='test calib. stack')
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_percent_meas.png'))
plt.show()

parameter = 'cm'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h, colors=colors, linestyles=linestyles, scale=scale_fig_dim,
                                      scatter_on=False)
ax.set_ylabel(r'$c_{m}$')
ax.set_ylim([0.9, 1.01])
ax.legend(labels, prop=fontP, loc='upper left',bbox_to_anchor=(1.1, 1.0) , fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_cm.png'))
plt.show()

# plot global - only gdpyt
parameter = 'rmse_z'
xlabel = 'sigma'
fig, ax = plotting.plot_dfbicts_global(dfbicts, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylim([-0.01, 0.15])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpyt_global_rmse_z.png'))
plt.show()

parameter = 'percent_meas'
fig, ax = plotting.plot_dfbicts_global(dfbicts, parameter, xlabel, h)
ax.set_xlabel('ID')  # r'$\sigma$'
ax.set_ylabel(r'$\phi_{ID}\left(z\right)$')
ax.set_ylim([90, 100.5])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_gdpyt_global_percent_meas.png'))
plt.show()


# plot several c_m's per figure
dfbicts2 = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin, bins, min_cm2, z_range, round_to_decimal, dficts_ground_truth=None)
dfbicts3 = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin, bins, min_cm3, z_range, round_to_decimal, dficts_ground_truth=None)

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

parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfict_cm_sweep, parameter, h, colors=colors, linestyles=linestyles,
                                           scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
ax.legend(labels, fontsize=8, bbox_to_anchor=(1.2, 1), loc='upper left', fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z_sweep_cm.png'))
plt.show()

# plot several c_m's for GDPyT and GDPT per figure
dfict_cm_sweep = {'GDPyT cm0.5': dfbicts[1.0], 'GDPT cm0.5': dfbicts[2.0],
                  'GDPyT cm0.9': dfbicts2[1.0], 'GDPT cm0.9': dfbicts2[2.0],
                  'GDPyT cm0.95': dfbicts3[1.0], 'GDPT cm0.95': dfbicts3[2.0]
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

parameter = ['rmse_z', 'percent_meas']
fig, ax, ax2 = plotting.plot_dfbicts_local(dfict_cm_sweep, parameter, h, colors=colors, linestyles=linestyles,
                                           scale=scale_fig_dim)
ax.set_ylim([-0.01, 0.15])
ax.legend(labels, fontsize=8, bbox_to_anchor=(1.15, 1), loc='upper left', fancybox=True, shadow=False)
ax2.set_ylabel(r'$\phi\left(z\right)$', fontsize=18)
ax2.set_ylim([0, 110])
plt.tight_layout()
plt.savefig(join(path_name, save_id+'_local_rmse_z_sweep_cm_compare.png'))
plt.show()