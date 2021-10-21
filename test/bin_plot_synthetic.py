# test bin, analyze, and plot functions

import analyze
from utils import io, bin, plotting

import matplotlib.pyplot as plt

# read .xlsx files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/Dataset_I/Random calibration step errors'
sort_strings = ['sigma', '']
filetype = '.xlsx'
dficts = io.read_dataframes(path_name, sort_strings, filetype)


# calculate local rmse_z
column_to_bin='z_true'
bins = 20
min_cm = 0.5
z_range = [-67, 18]
round_to_decimal = 0
true_num_particles = 361
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin, bins, min_cm, z_range, round_to_decimal, true_num_particles)


# plot local
parameter = 'rmse_z'
h = 86
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h)
plt.show()


# plot global
parameter = 'rmse_z'
xlabel = 'sigma'
h = 86
fig, ax = plotting.plot_dfbicts_global(dfbicts, parameter, xlabel, h)
ax.set_xlabel(r'$\sigma$')
plt.show()