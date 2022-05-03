# test bin, analyze, and plot functions
from os.path import join
import ast
import numpy as np
import pandas as pd
import analyze
from correction import correct
from utils import io, details, bin, plotting, modify
import filter
from tracking import plotting as trackplot

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------
# Methods to implement:
"""
1. read string to get test details:
    1.1 measurement volume: full or half
    1.2 magnification: 20X or 10X
    1.3 demag: 1X or 0.5X
    1.4 particle diameter: 0.87, 2.15, 5.1, 5.61 um
    1.5 # of images in calibration averaging: 9
2. read test coords to get test details:
    2.1 measurement volume: z min, z max
    2.2 # of particles: p_num
3. compile test details into a single dictionary
    {identifying string: dictionary of results{
                            df: test coords dataframe,
                            h: measurement volume,
                            mag: magnification,
                            demag: demag,
                            pd: particle diameter,
                            p_num: number of particles,
                            calib_img_avg: number of calibration images averaged,
                            }
    }
"""

# ------------------------------------------------
"""
Notes on SciencePlots colors:
    std-colors: 7 colors
            Blue: #0C5DA5
            Green: #00B945
            Orange: #FF9500
            Red: #FF2C00
            Purple: #845B97
            Dark Blue: #474747
            Gray: #9e9e9e
            
    muted: 10 colors
    high-vis: 7 colors + 7 linestyles
    
Other Colors:
            Paler Blue: #0343DF
            Azure: #069AF3
"""

# ------------------------------------------------
# formatting
plt.style.use(['science', 'ieee', 'std-colors'])
scale_fig_dim = [1, 1]
scale_fig_dim_outside_x_legend = [1.5, 1]
legend_loc = 'best'

# ------------------------------------------------
# read files
dataset = 'applicability'
save_id = 'corrected-per-particle_norm-z'

# read .xlsx result files to dictionary
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/{}'.format(dataset)
read_calib_path_name = join(base_path, 'calib_coords')
read_path_name = join(base_path, 'test_coords')
path_figs = join(base_path, 'figs')
path_results = join(base_path, 'results')
calib_sort_strings = ['calib_', '_coords_']
test_sort_strings = ['id_', '_coords_']
settings_sort_strings = ['settings_id', '_coords_']
filetype = '.xlsx'

# ------------------------------------------------


# end setup
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# read files

# calib correction coords
cficts = io.read_files('df', read_calib_path_name, calib_sort_strings, filetype, startswith=calib_sort_strings[0])
# test coords
dficts = io.read_files('df', read_path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])

# read details
dficts_details = details.parse_filename_to_details(read_path_name, test_sort_strings, filetype, startswith=test_sort_strings[0])
dficts_details = details.read_dficts_coords_to_details(dficts, dficts_details)

# ----------------------------------------------------------------------------------------------------------------------

# setup
labels = list(dficts.keys())

# ----------------------------------------------------------------------------------------------------------------------

# start analysis
per_particle_correction = True
show_plots = False

# ---------------------------------------------------------------
# filter particles with errors > h/10 (single rows)
barnkob_error_filter = 0.1   # filter error < h/10 used by Barnkob and Rossi in 'A fast robust algorithm...'
meas_vols = np.array([x[1]['meas_vol'] for x in dficts_details.items()])
meas_vols_inverse = 1 / meas_vols
dficts = modify.dficts_new_column(dficts, new_columns=['percent_error'], columns=['error'], multipliers=meas_vols_inverse)
dficts, dficts_filtered = filter.dficts_filter(dficts, keys=['percent_error'], values=[[-barnkob_error_filter, barnkob_error_filter]],
                              operations=['between'], return_filtered=True)

# filter particles with errors > h/10 (all rows for particles where any row error > h/10)
for name, df in dficts_filtered.items():
    pids_filter = df.id.unique()
    dficts = filter.dficts_filter(dficts, keys=['id'], values=[pids_filter], operations=['notin'], only_keys=[name])

# ---------------------------------------------------------------
# correct particle coordinates

# find sub-image in-focus z-coordinate using interpolated peak intensity
cficts = correct.calc_calib_in_focus_z(cficts, dficts, per_particle_correction=per_particle_correction, round_to_decimals=4)
# correct test coords based on calibration corrected coordinates
dficts = correct.correct_z_by_in_focus_z(cficts, dficts, per_particle_correction=per_particle_correction)

# re-sort dficts
dficts = modify.dficts_sort(dficts)

# filter dficts
# dficts = filter.dficts_filter(dficts, keys=['z_true'], values=[[-65, 65]], operations=['between'])


# ----------------------------------------------------------------------------------------------------------------------
# calculate z-uncertainties: bin by number of bins

# setup
column_to_bin_and_assess = 'z_true'
min_cm = 0.5
round_z_to_decimal = 6

# calculate local rmse_z
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins=dficts_details, min_cm=min_cm,
                                             z_range=None, round_to_decimal=round_z_to_decimal,
                                             dficts_ground_truth=None)

# ---------------------------------------------------------------
# add new columns using normalized measurement volume:

# normalized binned-z: bin_z_norm
meas_vol_zmax = np.array([x[1]['zmax'] - 1 for x in dficts_details.items()])
meas_vols_zmax_inverse = 1 / meas_vol_zmax
dfbicts = modify.dficts_new_column(dfbicts, new_columns=['bin_z_norm'], columns=['index'],
                                   multipliers=meas_vols_zmax_inverse)

# normalized z-uncertainty: rmse_z_norm
dfbicts = modify.dficts_new_column(dfbicts, new_columns=['rmse_z_norm'], columns=['rmse_z'],
                                   multipliers=meas_vols_inverse)

# filter dfbicts
dfbicts = filter.dficts_filter(dfbicts, keys=['bin_z_norm'], values=[[-0.505, 0.505]], operations=['between'])

# set normalized binned-z as index
dfbicts = modify.dficts_set_index(dfbicts, column_to_index='bin_z_norm')


# ---------------------------------------------------------------
# export to Excel - NOTE: the number of bins must be set == 1.
export_results = False
if export_results:
    dfb = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
    dfb = dfb.set_index('filename')
    drop_columns = ['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'bin_z_norm']
    io.export_df_to_excel(dfb,
                          path_name=join(path_results, save_id + '_measurement_results'),
                          include_index=True, index_label='test_id', filetype='.xlsx',
                          drop_columns=drop_columns)

# ---------------------------------------------------------------
# plot binned uncertainties

# setup
xlim_norm = [-0.5125, 0.5125]
xlim_norm_ticks = [-0.5, 0, 0.5]
ylim_microns = [0, 1.125]
ylim_norm = [-0.0005, 0.0205]
ylim_norm_ticks = [0, 0.005, 0.01, 0.015, 0.02]

linestyles = []
for i in list(dfbicts.keys()):
    if i < 10:
        linestyles.append('--')
    else:
        linestyles.append('-')

std_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
shaded_colors = ['mediumblue', 'darkgreen', 'chocolate', 'tab:red', 'indigo', 'midnightblue', 'dimgray']
colors = []
j = 0
for i, k in enumerate(list(dfbicts.keys())):
    if k < 10:
        colors.append(shaded_colors[i])
        j += 1
    else:
        colors.append(std_colors[i - j])

# ---------------------------------------------------------------
# plots

# z-uncertainty (microns)
parameter = 'rmse_z'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h=1, scale=scale_fig_dim,
                                      label_dict=dficts_details, colors=colors, linestyles=linestyles)

ax.set_ylabel(r'$\sigma_{z}\: (\mu m)$')
ax.set_xlabel(r'$z/h$')  # ax.set_xlabel(r'$z \: (\mu m)$')  #
ax.set_ylim(ylim_microns)
ax.set_xlim(xlim_norm)
ax.set_xticks(xlim_norm_ticks)
# ax.legend(loc='upper left', fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0))
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(join(path_figs, save_id+'_z_uncertainty.png'))
plt.show()

# normalized z-uncertainty
parameter = 'rmse_z_norm'
fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h=1, scale=scale_fig_dim,
                                      label_dict=dficts_details, colors=colors, linestyles=linestyles)

ax.set_ylabel(r'$\sigma_{z}/h$')
ax.set_xlabel(r'$z/h$')  # ax.set_xlabel(r'$z \: (\mu m)$')  #
ax.set_ylim(ylim_norm)
ax.set_yticks(ylim_norm_ticks)
ax.set_xlim(xlim_norm)
ax.set_xticks(xlim_norm_ticks)
# ax.legend(loc='upper left', fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0))
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(join(path_figs, save_id+'_norm_z_uncertainty.png'))
plt.show()

"""
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# plot calibration curve: z-measured vs. z-true
plot_calibration_curve = True
if plot_calibration_curve:
    for name, df in dficts.items():
        fig, ax = plt.subplots()
        ax.scatter(df.z_true, df.z, s=1)

        ax.set_xlabel(r'$z_{true}\: (\mu m)$')
        ax.set_ylabel(r'$z_{measured}\: (\mu m)$')
        ax.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(join(path_figs, save_id + '_{}_calibration_curve.png'.format(name)))
        if show_plots:
            plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# plot 3D scatter plot of points in frame closest to in-focus
plot_calibration_scatterplots = True
if plot_calibration_scatterplots:

    for name, df in dficts.items():
        # find in-focus z
        if per_particle_correction:
            dfz_trues = df.groupby('frame').mean()
            dfz_zero = dfz_trues['z_true'].abs().idxmin()
            # get particles at z_zero
            dfz = df[df['frame'] == dfz_zero]
            dfz['dz'] = df['z'] - df['z_true']
        else:
            z_trues = df.z_true.unique()
            z_zero = z_trues[np.argmin(np.abs(z_trues))]
            # get particles at z_zero
            dfz = df[df['z_true'] == z_zero]
            dfz['dz'] = df['z'] - df['z_true']

        # plots

        # format plots
        zmin_lim = -1
        zmax_lim = 1
        if -1 > dfz.z.min() > -1.5:
            zmin_lim = -1.5
        elif -1.5 > dfz.z.min() > -2.5:
            zmin_lim = -2.5
        elif -2.5 > dfz.z.min() > -5:
            zmin_lim = -5
        elif -5 > dfz.z.min():
            zmin_lim = -7.5

        if 1 < dfz.z.max() < 1.5:
            zmax_lim = 1.5
        elif 1.5 < dfz.z.max() < 2.5:
            zmax_lim = 2.5
        elif 2.5 < dfz.z.max() < 5:
            zmax_lim = 5
        elif 5 < dfz.z.max():
            zmax_lim = 7.5

        # plot 1D scatter
        fig, [ax1, ax2] = plt.subplots(nrows=2)
        ax1.scatter(dfz.x, dfz.z, s=5)
        ax1.set_xlabel('x (pixels)')
        ax1.set_ylabel('$z-z_f\: (\mu m)$')
        ax1.set_ylim([zmin_lim, zmax_lim])
        ax1.set_xlim([0, 512])
        ax1.grid(alpha=0.125)
        ax2.scatter(dfz.y, dfz.z, s=5)
        ax2.set_xlabel('y (pixels)')
        ax2.set_ylabel('$z-z_f\: (\mu m)$')
        ax2.set_ylim([zmin_lim, zmax_lim])
        ax2.set_xlim([0, 512])
        ax2.grid(alpha=0.125)
        plt.tight_layout()
        plt.savefig(join(path_figs, save_id + '_{}_calibration_scatter_1d.png'.format(name)))
        if show_plots:
            plt.show()

        # plot 2D scatter
        fig, ax = plt.subplots()
        data = ax.scatter(dfz.x, dfz.y, c=dfz.dz, s=10)
        ax.set_xlim([0, 512])
        ax.set_ylim([0, 512])
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.grid(alpha=0.125)
        # color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.125)
        plt.colorbar(data, cax=cax)
        plt.tight_layout()
        plt.savefig(join(path_figs, save_id + '_{}_calibration_scatter_2d.png'.format(name)))
        if show_plots:
            plt.show()

        # plot 3D scatter
        dfz_list = [dfz.x, dfz.y, dfz.dz]
        fig, ax = plotting.plot_scatter_3d(dfz_list, fig=None, ax=None, elev=5, azim=-40, color=None, alpha=0.75)
        ax.set_xlim([0, 512])
        ax.set_ylim([0, 512])
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_zlabel(r'$z-z_f (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_figs, save_id + '_{}_calibration_scatter_3d.png'.format(name)))
        if show_plots:
            plt.show()
"""
j = 1