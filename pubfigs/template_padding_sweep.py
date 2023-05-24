# test bin, analyze, and plot functions
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import analyze
from utils import io, bin, plotting, modify, functions
from utils.plotting import lighten_color
import filter


# ------------------------------------------------
# formatting
sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF9500'
sciorange = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# -----------------------

# ----------------------------------------------------------------------------------------------------------------------
# 0. SETUP DATASET DETAILS

# spct filepath
fp_spct = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/results/' \
          'spct-no-dz-overlap/results/spct-no-dz-overlap_rmse-z_by_dx_z-true_2d-bin.xlsx'


# setup file paths
dataset = 'grid-overlap-sweep-template-padding'

# sub-dirs
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/{}'.format(dataset)
path_coords_idpt = join(base_path, 'test_coords')
path_coords_spct = join(base_path, 'test_coords/spct')
path_figs = join(base_path, 'figs')
path_results = join(base_path, 'results')
settings_sort_strings = ['settings_id', '_coords_']
test_sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'

# save figures
save_fig_filetype = '.svg'

# template padding --> template size
template_paddings = [0, 3, 5, 8, 11, 15]
template_sizes = [9, 15, 19, 25, 31, 39]

# ----------------------------------------------------------------------------------------------------------------------
# 1. READ TEST COORDS + SETTINGS

method = 'spct'

if method == 'idpt':
    path_coords = path_coords_idpt
else:
    path_coords = path_coords_spct

# ---

# read files
dficts = io.read_files('df',
                       path_coords,
                       test_sort_strings,
                       filetype,
                       startswith=test_sort_strings[0],
                       )
dfsettings = io.read_files('dict',
                           path_coords,
                           settings_sort_strings,
                           filetype,
                           startswith=settings_sort_strings[0],
                           columns=['parameter', 'settings'],
                           dtype=str,
                           )

# read spct
dfs = pd.read_excel(fp_spct)


# ------------------------------------------------------------------------------------------------------------------
# 2. FILTER DATA
apply_filters = True
apply_barnkob_filter = False

# filters
z_range = [-12, 4]  # the calibration range was [-19, 11]
min_cm = 0.0
filter_barnkob = 3  # set this arbitrarily
discard_frames_before = 0.5

# apply filters
if apply_filters:

    if apply_barnkob_filter:
        dficts = filter.dficts_filter(dficts,
                                      keys=['error'],
                                      values=[[-filter_barnkob, filter_barnkob]],
                                      operations=['between'],
                                      copy=False,
                                      only_keys=None,
                                      return_filtered=False)

    dficts = filter.dficts_filter(dficts,
                                  keys=['z_true'],
                                  values=[z_range],
                                  operations=['between'],
                                  copy=False,
                                  only_keys=None,
                                  return_filtered=False)

    dficts = filter.dficts_filter(dficts,
                                  keys=['cm'],
                                  values=[min_cm],
                                  operations=['greaterthan'],
                                  copy=False,
                                  only_keys=None,
                                  return_filtered=False)

    dficts = filter.dficts_filter(dficts,
                                  keys=['frame'],
                                  values=[discard_frames_before],
                                  operations=['greaterthan'],
                                  copy=False,
                                  only_keys=None,
                                  return_filtered=False)

# spct filter
if dfs is not None:
    dfs = dfs[dfs['filename'] == 39]
    dfs = dfs[(dfs['bin'] > z_range[0]) & (dfs['bin'] < z_range[1])]

# ---

# ------------------------------------------------------------------------------------------------------------------
# 3. CALCULATE RMSE-Z FOR EACH TEMPLATE SIZE

export_results = True
save_plots = True
show_plots = True

# calculate mean rmse_z as a function of z_true
plot_mean_uncertainty = True
if plot_mean_uncertainty:

    # calculate mean
    dfbms = analyze.calculate_bin_local_rmse_z(dficts,
                                              column_to_bin='z_true',
                                              bins=1,
                                              min_cm=min_cm,
                                              z_range=z_range,
                                              round_to_decimal=2,
                                              dficts_ground_truth=None,
                                              )
    # ---

    # stack into dataframe
    dfbms = modify.stack_dficts_by_key(dfbms, drop_filename=False)

    # add column for template size
    dfbms['l_t'] = 9 + dfbms['filename'] * 2
    dfbms = dfbms.set_index('filename')

    # export rmse_z(bin-dx)
    if export_results:
        dfbms.to_excel(path_results + '/{}-mean_rmse-z_bin-z-true.xlsx'.format(method), index_label='padding')

    # ---

    # bin-z-true: plot rmse_z by template padding for entire collection
    fig, ax = plt.subplots()

    p1, = ax.plot(dfbms.index, dfbms.rmse_z, '-o')
    ax.set_xlabel(r'$l_{t}^{p} \: (pix.)$')
    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')

    axr = ax.twinx()
    axr.plot(dfbms.index, dfbms.cm, '-d', color=lighten_color(p1.get_color(), 1.15))
    axr.set_ylabel(r'$\diamond \: c_{m}$', color=lighten_color(p1.get_color(), 1.15))
    axr.set_ylim([0.955, 1.01])

    # spct
    if dfs is not None:
        dfsg = dfs.groupby('filename').mean()
        p2, = ax.plot(1, dfsg.rmse_z, 'o', color=scigreen, label='SPCT')
        axr.plot(1, dfsg.cm, 'd', color=lighten_color(p2.get_color(), 1.05))
        ax.legend(loc='lower right',
                  labelspacing=0.25, handlelength=1, handletextpad=0.4, borderaxespad=0.25, columnspacing=1.25)
        ax.set_ylim([0.775, 2.55])

    plt.tight_layout()
    if save_plots:
        plt.savefig(join(path_figs, '{}-all_bin-z-true_plot-rmse-z_by_template-padding'.format(method) + save_fig_filetype))
    if show_plots:
        plt.show()
    plt.close()

    # ---

    # bin-z-true: plot rmse_z by template size for entire collection
    fig, ax = plt.subplots()

    p1, = ax.plot(dfbms.l_t, dfbms.rmse_z, '-o')
    ax.set_xlabel(r'$l_{t} \: (pix.)$')
    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')

    axr = ax.twinx()
    axr.plot(dfbms.l_t, dfbms.cm, '-d', color=lighten_color(p1.get_color(), 1.15))
    axr.set_ylabel(r'$\diamond \: c_{m}$', color=lighten_color(p1.get_color(), 1.15))
    axr.set_ylim([0.955, 1.01])

    # spct
    if dfs is not None:
        p2, = ax.plot(10, dfsg.rmse_z, 'o', color=scigreen, label='SPCT')
        axr.plot(10, dfsg.cm, 'd', color=lighten_color(p2.get_color(), 1.05))
        ax.legend(loc='lower right',
                  labelspacing=0.25, handlelength=1, handletextpad=0.4, borderaxespad=0.25, columnspacing=1.25)
        ax.set_ylim([0.775, 2.5])

    plt.tight_layout()
    if save_plots:
        plt.savefig(join(path_figs, '{}-all_bin-z-true_plot-rmse-z_by_template-size'.format(method) + save_fig_filetype))
    if show_plots:
        plt.show()
    plt.close()

    # ---

    # ---

    # bin by z-true
    dfbs = analyze.calculate_bin_local_rmse_z(dficts,
                                              column_to_bin='z_true',
                                              bins=13,
                                              min_cm=min_cm,
                                              z_range=z_range,
                                              round_to_decimal=2,
                                              dficts_ground_truth=None,
                                              )
    # ---

    # plot rmse_z for each template padding
    ms = 3
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    for name, df in dfbs.items():
        ax1.plot(df.index, df.cm, '-', ms=ms)
        ax2.plot(df.index, df.rmse_z, '-', marker='.', ms=ms, label=name)

    ax1.set_ylabel(r'$c_{m}$')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_z \: (\mu m)$')
    ax2.legend(ncol=2, loc='upper left',
               labelspacing=0.25, handlelength=1, handletextpad=0.4, borderaxespad=0.25, columnspacing=1.25)

    if dfs is not None:
        ax1.plot(dfs.bin, dfs.cm, linestyle='dotted', linewidth=1, color='black', label='SPCT')
        ax2.plot(dfs.bin, dfs.rmse_z, linestyle='dotted', linewidth=1, color='black')
        ax1.legend(loc='lower left',
                   labelspacing=0.25, handlelength=1, handletextpad=0.4, borderaxespad=0.25, columnspacing=1.25)

    plt.tight_layout()
    if save_plots:
        plt.savefig(join(path_figs, '{}-all_bin-z-true_plot-rmse-z-cm_by_z-true_template-padding'.format(method) + save_fig_filetype))
    if show_plots:
        plt.show()
    plt.close()

    # ---

    # plot rmse-z for each template size
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    for name, df in dfbs.items():
        ax1.plot(df.index, df.cm, '-', ms=ms)
        ax2.plot(df.index, df.rmse_z, '-', marker='.', ms=ms, label=name * 2 + 9)

    ax1.set_ylabel(r'$c_{m}$')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_z \: (\mu m)$')
    ax2.legend(ncol=2, loc='upper left',
               labelspacing=0.25, handlelength=1, handletextpad=0.4, borderaxespad=0.25, columnspacing=1.25)

    if dfs is not None:
        ax1.plot(dfs.bin, dfs.cm, linestyle='dotted', linewidth=1, color='black', label='SPCT')
        ax2.plot(dfs.bin, dfs.rmse_z, linestyle='dotted', linewidth=1, color='black')
        ax1.legend(loc='lower left',
                   labelspacing=0.25, handlelength=1, handletextpad=0.4, borderaxespad=0.25, columnspacing=1.25)

    plt.tight_layout()
    if save_plots:
        plt.savefig(join(path_figs, '{}-all_bin-z-true_plot-rmse-z-cm_by_z-true_template-size'.format(method) + save_fig_filetype))
    if show_plots:
        plt.show()
    plt.close()

    # ---

    # export
    dfbs = modify.stack_dficts_by_key(dfbs, drop_filename=False)

    # add column for template size
    dfbs['l_t'] = 9 + dfbs['filename'] * 2

    dfbs = dfbs.set_index('filename')
    if export_results:
        dfbs.to_excel(path_results + '/{}-local_rmse-z_bin-z-true.xlsx'.format(method), index_label='padding')


print("Analysis completed without errors.")