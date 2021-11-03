# test bin, analyze, and plot functions

import analyze, filter
from utils import io, plotting, bin, modify, fit

import matplotlib.pyplot as plt

# read .xlsx files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.11.21-BPE_Pressure_Deflection_10X/analyses/Oct 22 - Results - Deflection of over-chip particles/full-30-imgs-range-on-chip'
sort_strings = ['z', 'um']
filetype = '.xlsx'
dficts = io.read_dataframes(path_name, sort_strings, filetype)

# plot all raw data points
xparameter = 'y'
yparameter = 'z'
min_cm = 0.5
z0 = 0
take_abs = False
fig, ax = plotting.plot_scatter(dficts, xparameter, yparameter, min_cm, z0, take_abs)
plt.tight_layout()
plt.show()

# drop NaNs
dficts = filter.dficts_dropna(dficts, columns=['z'])

# filter dataframes
keys = ['z']
values = [116]
operations = ['lessthan']
dficts = filter.dficts_filter(dficts, keys, values, operations)

# plot the absolute value of data points re-centered on estimated focal plane
z0 = 58
mupp = 1.8  # microns per pixel
bins = 50
min_cm = 0.9
take_abs = True
fig, ax = plotting.plot_scatter(dficts, xparameter, yparameter, min_cm, z0, take_abs)
plt.tight_layout()
plt.show()

# modify the dataframe to reflect the coordinates of the device
dficts = modify.dficts_flip(dficts, column='y')
dficts = modify.dficts_shift(dficts, columns=['y'], shifts=[z0])
dficts = modify.dficts_scale(dficts, columns=['x', 'y', 'z'], multipliers=[mupp])

# calculate local rmse_z
column_to_bin='y'
z_range = None
z0 = z0 * mupp
take_abs = True
round_to_decimal = 0
true_num_particles = None
dfbicts = analyze.calculate_bin_local(dficts=dficts, column_to_bin=column_to_bin, bins=bins, min_cm=min_cm,
                                      z_range=z_range, round_to_decimal=round_to_decimal,
                                      true_num_particles=true_num_particles, z0=z0, take_abs=take_abs)

# plot local - scatter
xparameter = 'index'
yparameter = 'z'
z0 = 0
take_abs = False
fig, ax = plotting.plot_scatter(dficts=dfbicts, xparameter=xparameter, yparameter=yparameter, min_cm=min_cm, z0=z0,
                                take_abs=take_abs)
plt.tight_layout()
plt.show()


# plot local - errorbars
yparameter = 'z'
z0 = 0
fig, ax = plotting.plot_errorbars(dfbicts, xparameter='index', yparameter='z', min_cm=min_cm, z0=z0)
plt.tight_layout()
plt.show()

# plot fitted curves on scatter data points
fit_function = fit.parabola
xparameter = 'index'
yparameter = 'z'
z0 = 0
fig, ax = plotting.plot_fit_and_scatter(fit_function=fit_function, dficts=dfbicts, xparameter=xparameter,
                                        yparameter=yparameter, min_cm=min_cm, z0=z0)
plt.tight_layout()
plt.show()

# copy and modify a single dataframe for 2D and 3D plotting
dfabs = dficts[5000.0].copy()

# plot 2D heatmap
fig, ax = plotting.plot_heatmap(dfabs, fig=None, ax=None)
plt.tight_layout()
plt.show()

# plot 3D scatter plot
fig, ax = plotting.plot_scatter_3d(dfabs, fig=None, ax=None, elev=5, azim=-40)
plt.tight_layout()
plt.show()