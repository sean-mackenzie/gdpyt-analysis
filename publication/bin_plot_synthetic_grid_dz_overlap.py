# test bin, analyze, and plot functions
from os.path import join
import ast
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
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
sciblue = '#0C5DA5'
scigreen = '#00B945'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

scale_fig_dim = [1, 1]
scale_fig_dim_outside_x_legend = [1.25, 1]
legend_loc = 'best'

# ------------------------------------------------
# read files
dataset = 'synthetic grid dz overlap nl2'
save_id = 'grid dz overlap'

# read .xlsx result files to dictionary
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/{}'.format(dataset)
path_coords = join(base_path, 'test_coords')
path_figs = join(base_path, 'figs')
path_results = join(base_path, 'results')
settings_sort_strings = ['settings_id', '_coords_']
test_sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'

# ------------------------------------------------

"""
IMPORTANT NOTE:
    * Currently, the first column x=110; dx=7.5 is dropped b/c it isn't paired correctly in NearestNeighbors.
    
    splits = np.array([110.0, 245.0, 377.0, 512.0, 644.0, 778.0, 910])
    keys = np.array([1, 2, 3, 4, 5, 6, 7]) * 7.5 

"""
# split dataframe by parameters/values
column_to_split = 'x'
splits = np.array([245.0, 377.0, 512.0, 644.0, 778.0, 910])
round_x_to_decimal = 0

# identify data
keys = np.array([2, 3, 4, 5, 6, 7]) * 7.5  # center-to-center overlap spacing
dict_splits_to_keys = {key: value for (key, value) in zip(splits, keys)}

# split dict by key
inspect_gdpyt_by_key = 1.0
inspect_spc_by_key = 11.0
labels_local = [lbl for lbl in keys]
labels_local.sort()

# -----------------------
# bin data for uncertainty assessment
column_to_bin_and_assess = 'z_true'
bins = 25
round_z_to_decimal = 5

# filters for binning
h = 70
z_range = [-45.001, 25.001]
min_cm = 0.5

# formatting figures
save_plots = True
show_plots = True

# ----------------------------------------------------------------------------------------------------------------------
# read files

dficts = io.read_files('df',
                       path_coords,
                       test_sort_strings,
                       filetype,
                       startswith=test_sort_strings[0])
dfsettings = io.read_files('dict',
                           path_coords,
                           settings_sort_strings,
                           filetype,
                           startswith=settings_sort_strings[0],
                           columns=['parameter', 'settings'],
                           dtype=str)
dficts_ground_truth = io.read_ground_truth_files(dfsettings)

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# add 'z_adj' and 'dz' column to dataframe using NearestNeighbors
if 'dz' not in dficts[inspect_gdpyt_by_key].columns:
    # add 'z_adj' and 'dz' columns
    df_dz = modify.map_adjacent_z_true(dficts[inspect_gdpyt_by_key],
                                       dficts_ground_truth[inspect_gdpyt_by_key],
                                       threshold=46)

    df_dz = df_dz.dropna()

    # export the corrected dataframe
    df_dz.to_excel(path_coords + '/test_id1_coords_z_adj_static_grid-dz-overlap-nl2.xlsx', index=False)

    # update the dictionary
    del dficts
    dficts = {inspect_gdpyt_by_key: df_dz}

# ----------------------------------------------------------------------------------------------------------------------
# start analysis

# filter
apply_filters = True

if apply_filters:
    dficts = filter.dficts_filter(dficts,
                                  keys=['frame'],
                                  values=[0.5],
                                  operations=['greaterthan'],
                                  copy=True,
                                  only_keys=None,
                                  return_filtered=False)

    dficts = filter.dficts_filter(dficts,
                                  keys=['x'],
                                  values=[120],
                                  operations=['greaterthan'],
                                  copy=True,
                                  only_keys=None,
                                  return_filtered=False)

    dficts = filter.dficts_filter(dficts,
                                  keys=['z_true'],
                                  values=[[-45.001, 25.001]],
                                  operations=['between'],
                                  copy=True,
                                  only_keys=None,
                                  return_filtered=False)

    dficts = filter.dficts_dropna(dficts, columns=['dz'])

# 1D binning
dsplicts = modify.split_df_and_merge_dficts(dficts[inspect_gdpyt_by_key],
                                            keys,
                                            column_to_split,
                                            splits,
                                            round_x_to_decimal)

for name, df in dsplicts.items():
    dfb = bin.bin_local(df,
                        column_to_bin='dz',
                        bins=11,
                        min_cm=0.5,
                        z_range=None,
                        round_to_decimal=0,
                        true_num_particles=None,
                        dropna=True)

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.5))

    ax1.plot(dfb.index, -dfb.error, '-o', alpha=0.5, label=r'$-\epsilon \equiv z_{IDPT} < z_{true}$')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel(r'$\epsilon_{z}$')
    ax1.set_ylim([-7.5, 7.5])
    ax1.set_title(r'$\delta x=$' + '{}'.format(name))
    ax1.legend(loc='upper left')

    ax2.plot(dfb.index, dfb.z_std ** 2, '-o', alpha=0.5)
    ax2.set_ylabel(r'$var. \: = \: \sigma^2$')
    ax2.set_ylim([200, 275])

    ax3.errorbar(dfb.index, dfb.z, yerr=dfb.z_std, fmt='o', capsize=2, alpha=0.5,
                 label=r'$\overline{z}_{IDPT} + \sigma$')
    ax3.plot(dfb.index, dfb.z_true, '-o', ms=1, linewidth=0.5, color='black', label=r'$z_{true}$')
    ax3.set_xlabel(r'$\delta z$')
    ax3.set_ylabel(r'$z$')
    ax3.set_ylim([-40, 40])
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path_results + '/error_bias_and_variance_dx{}.png'.format(name))
    plt.show()


j = 1

raise ValueError('ha')

# calculate mean rmse_z
dfbm = bin.bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                            column_to_bin='z_true',
                            bins=1,
                            min_cm=0.5,
                            z_range=None,
                            round_to_decimal=3,
                            df_ground_truth=None,
                            )

dfbm.to_excel(path_results + '/mean_rmse_z.xlsx')

# plot rmse_z for entire collection
dfb = bin.bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                           column_to_bin='z_true',
                           bins=20,
                           min_cm=0.5,
                           z_range=None,
                           round_to_decimal=3,
                           df_ground_truth=None
                           )

fig, ax = plt.subplots()

ax.plot(dfb.index, dfb.rmse_z, '-o', ms=2)

ax.set_xlabel(r'$z_{true} \: (\mu m)$')
ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
plt.tight_layout()
plt.savefig(join(path_figs, save_id + '_all_bin-z-true_plot-rmse-z.png'))
plt.show()

# plot rmse_z for entire collection
dfb = bin.bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                           column_to_bin='dz',
                           bins=20,
                           min_cm=0.5,
                           z_range=None,
                           round_to_decimal=3,
                           df_ground_truth=None
                           )

fig, ax = plt.subplots()

ax.plot(dfb.index, dfb.rmse_z, '-o', ms=2)

ax.set_xlabel(r'$\Delta z \: (\mu m)$')
ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
plt.tight_layout()
plt.savefig(join(path_figs, save_id + '_all_bin-dz_plot-rmse-z.png'))
plt.show()

raise ValueError('ha')

# 1D binning
dsplicts = modify.split_df_and_merge_dficts(dficts[inspect_gdpyt_by_key],
                                            keys,
                                            column_to_split,
                                            splits,
                                            round_x_to_decimal)

# calculate mean rmse-z for each dx
column_to_bin_and_assess = 'z_true'
bins = 1
dfbicts = analyze.calculate_bin_local_rmse_z(dsplicts,
                                             column_to_bin_and_assess,
                                             bins,
                                             min_cm=0.5,
                                             z_range=None,
                                             round_to_decimal=1,
                                             dficts_ground_truth=None)

dfb_stacked = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
dfb_stacked.to_excel(join(path_results, save_id + '_dx_bin-dz-z_plot-rmse-z.xlsx'))

# standard z-uncertainty plots
plot_typical_rmsez = True

if plot_typical_rmsez:
    for name, dfix in dsplicts.items():
        # 2D binning
        columns_to_bin = ['dz', 'z_true']
        bin_dz = 5
        bin_z_true = 20

        dfxbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dfix,
                                                           columns_to_bin=columns_to_bin,
                                                           bins=[bin_dz, bin_z_true],
                                                           round_to_decimals=[2, 2],
                                                           min_cm=0.5,
                                                           equal_bins=[False, False],
                                                           error_column='error',
                                                           )

        fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))
        for name_dz, dfix_rmse in dfxbicts_2d.items():
            ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=name_dz)

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                  title=r'$\Delta z(\delta x=$' + '{}'.format(name) + r'$)$')
        plt.tight_layout()
        plt.savefig(join(path_figs, save_id + '_dx{}_bin-dz-z_plot-rmse-z.png'.format(name)))
        plt.show()

        j = 1

    # 2D binning
    columns_to_bin = ['dz', 'z_true']
    bin_dz = 7
    bin_z_true = 20

    dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[1.0],
                                                      columns_to_bin=columns_to_bin,
                                                      bins=[bin_dz, bin_z_true],
                                                      round_to_decimals=[2, 2],
                                                      min_cm=0.5,
                                                      equal_bins=[False, False],
                                                      error_column='error',
                                                      )

    fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))
    for name_dz, dfix_rmse in dfbicts_2d.items():
        ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=name_dz)

    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\Delta z \: (\mu m)$')
    plt.tight_layout()
    plt.savefig(join(path_figs, save_id + '_all_bin-dz-z_plot-rmse-z.png'))
    plt.show()

    # 2D binning
    columns_to_bin = ['x', 'z_true']
    bin_x = splits
    bin_z_true = 20

    dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[1.0],
                                                      columns_to_bin=columns_to_bin,
                                                      bins=[bin_x, bin_z_true],
                                                      round_to_decimals=[2, 2],
                                                      min_cm=0.5,
                                                      equal_bins=[False, False],
                                                      error_column='error',
                                                      )

    fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))
    for name_dz, dfix_rmse in dfbicts_2d.items():
        ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=dict_splits_to_keys[name_dz])

    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\delta x \: (\mu m)$')
    plt.tight_layout()
    plt.savefig(join(path_figs, save_id + '_all_bin-dx-z_plot-rmse-z.png'))
    plt.show()

# 2D binning
columns_to_bin = ['x', 'dz']
bin_x = splits
bin_dz = 11

dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[1.0],
                                                  columns_to_bin=columns_to_bin,
                                                  bins=[bin_x, bin_dz],
                                                  round_to_decimals=[2, 2],
                                                  min_cm=0.5,
                                                  equal_bins=[False, False],
                                                  error_column='error',
                                                  )

fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))
for name_dz, dfix_rmse in dfbicts_2d.items():
    ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=dict_splits_to_keys[name_dz])

ax.set_xlabel(r'$\Delta z \: (\mu m)$')
ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\delta x \: (pix.)$')
plt.tight_layout()
plt.savefig(join(path_figs, save_id + '_all_bin-dx-dz_plot-rmse-z.png'))
plt.show()

# 2D binning
columns_to_bin = ['x', 'theta']
bin_x = splits
bin_theta = 11

dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[1.0],
                                                  columns_to_bin=columns_to_bin,
                                                  bins=[bin_x, bin_theta],
                                                  round_to_decimals=[2, 2],
                                                  min_cm=0.5,
                                                  equal_bins=[False, False],
                                                  error_column='error',
                                                  )

fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))
for name_dz, dfix_rmse in dfbicts_2d.items():
    ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=dict_splits_to_keys[name_dz])

ax.set_xlabel(r'$\theta \: (deg.)$')
ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\delta x \: (pix.)$')
plt.tight_layout()
plt.savefig(join(path_figs, save_id + '_all_bin-dx-theta_plot-rmse-z.png'))
plt.show()

# 1D binning
column_to_bin = 'theta'
bin_theta = 17

dfb_theta = analyze.bin_local_rmse_z(dficts[1.0],
                                     column_to_bin=column_to_bin,
                                     bins=bin_theta,
                                     min_cm=0.5,
                                     z_range=None,
                                     round_to_decimal=2,
                                     df_ground_truth=None)

fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))

ax.plot(dfb_theta.index, dfb_theta.rmse_z, '-o', ms=2, label='IDPT')

ax.set_xlabel(r'$\theta \: (deg.)$')
ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(join(path_figs, save_id + '_all_bin-theta_plot-rmse-z.png'))
plt.show()

raise ValueError('ha')

fig_5 = True
plot_fit = False

if fig_5:
    columns_to_bin = ['r', 'z_true']
    column_to_count = 'id'
    bins = [bins_r, 27]
    round_to_decimals = [1, 3]
    min_num_bin = 10
    return_groupby = True

    dfm, dfstd = bin.bin_generic_2d(df_error,
                                    columns_to_bin,
                                    column_to_count,
                                    bins,
                                    round_to_decimals,
                                    min_num_bin,
                                    return_groupby
                                    )

    # resolve floating point bin selecting
    dfm = dfm.round({'bin_tl': 0, 'bin_ll': 2})
    dfstd = dfstd.round({'bin_tl': 0, 'bin_ll': 2})

    dfm = dfm.sort_values(['bin_tl', 'bin_ll'])
    dfstd = dfstd.sort_values(['bin_tl', 'bin_ll'])

    # plot
    fig, ax = plt.subplots()

    for i, bin_r in enumerate(dfm.bin_tl.unique()):
        dfbr = dfm[dfm['bin_tl'] == bin_r]
        dfbr_std = dfstd[dfstd['bin_tl'] == bin_r]

        # scatter: mean +/- std
        ax.errorbar(dfbr.bin_ll, dfbr.error, yerr=dfbr_std.error,
                    fmt='-o', ms=2, elinewidth=0.5, capsize=1, label=int(np.round(bin_r, 0)))

        # plot: fit
        if plot_fit:
            fit_bin_ll = np.linspace(dfbr.bin_ll.min(), dfbr.bin_ll.max())
            ax.plot(fit_bin_ll, functions.quadratic(fit_bin_ll, *popts_r[i]),
                    linestyle='--',
                    color=lighten_color(sci_color_cycle[i], amount=1.25),
                    )

    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_xlim(xlim)
    ax.set_xticks(ticks=xyticks, labels=xyticks)
    ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
    ax.set_ylim(yerr_lims)
    ax.set_yticks(ticks=yerr_ticks, labels=yerr_ticks)
    ax.legend(loc='upper right', title=r'$r_{bin}$',
              borderpad=0.2, handletextpad=0.6, borderaxespad=0.25, markerscale=0.75)

    plt.savefig(
        path_figs + '/bin-r-z_normalized-z-errors_by_z_and_fit-quadratic_errlim{}.png'.format(error_threshold))
    plt.tight_layout()
    plt.show()
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTION

# function to add z_true of adjacent particle
map_z_adj = False

if map_z_adj:

    def map_adjacent_z_true(df, threshold=46):
        # initialize list of dataframes
        dfs = []

        # get this frame only
        for fr in df.frame.unique():
            dff = df[df['frame'] == fr]

            # get ID's and coords
            dff = dff.sort_values('id')
            coords_ids = dff['id'].values
            coords_xy = dff[['x', 'y']].values

            # perform NearestNeighbors
            nneigh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords_xy)
            distances, indices = nneigh.kneighbors(np.array(coords_xy))

            # get only 2nd column
            distances = distances[:, 1]
            indices = indices[:, 1]

            # create the mapping of adjacent particle ID's to particle ID's
            mapping_dict = {}
            cids_not_mapped = []

            for distance, idx, cid in zip(distances, indices, coords_ids):
                if distance < threshold:
                    mapping_dict.update({cid: dff.id.values[idx.squeeze()]})
                else:
                    cids_not_mapped.append([cid, distance, idx])

            # create column to map to
            dff['z_adj'] = dff['id']

            # map adjacent particle ID to particle ID
            dff['z_adj'] = dff['z_adj'].map(mapping_dict)

            # map z_true value to adjacent particle ID
            mapping_dict_z_adjacent = {i: z for (i, z) in zip(dff.id.values, dff.z_true.values)}
            dff['z_adj'] = dff['z_adj'].map(mapping_dict_z_adjacent)

            # append to list
            dfs.append(dff)

        # concat list of dataframes
        df_z_adj = pd.concat(dfs)

        return df_z_adj


    # NearestNeighbors to match test coords with true
    df_error = map_adjacent_z_true(df=dficts[1.0], threshold=46)

    # remove frame 0 which is identical to calibration images
    df_error = df_error[df_error['frame'] > 0]
    df_error['dz'] = df_error.z_true - df_error.z_adj
    svp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/synthetic grid dz overlap nl2/test_coords'
    df_error.to_excel(svp + '/test_id1_coords_z-adj_static_grid-dz-overlap-nl2.xlsx', index=False)

print("Analysis completed without errors")