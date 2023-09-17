# imports
from os.path import join
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from utils.plotting import lighten_color
from utils import functions, bin

"""
Blue: #0C5DA5
Green: #00B945
"""

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
sci_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

# --- structure data
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/filtering-effects_z-micrometer-v2'
path_read_similarity = join(base_dir, 'data/similarity')
path_results = join(base_dir, 'results')
filetype = '.xlsx'

# ---

# PLOT PARTICLE TO PARTICLE SIMILARITY

plot_similarity = False
if plot_similarity:
    fp0 = 'collection_similarities_5uM_med-filter-0'
    fp3 = 'collection_similarities_5uM_med-filter-3'
    fp5 = 'collection_similarities_5uM_med-filter-5'

    dfs0 = pd.read_excel(join(path_read_similarity, fp0) + filetype)
    dfs3 = pd.read_excel(join(path_read_similarity, fp3) + filetype)
    dfs5 = pd.read_excel(join(path_read_similarity, fp5) + filetype)

    dfsg0 = dfs0.groupby('z').mean().reset_index()
    dfsg0_std = dfs0.groupby('z').std().reset_index()
    dfsg3 = dfs3.groupby('z').mean().reset_index()
    dfsg3_std = dfs3.groupby('z').std().reset_index()
    dfsg5 = dfs5.groupby('z').mean().reset_index()
    dfsg5_std = dfs5.groupby('z').std().reset_index()

    # ---

    # setup
    param_z = 'z'
    z_offset = -50

    # ---

    # line plot
    ms = 1
    fig, ax = plt.subplots()

    ax.plot(dfsg0[param_z] + z_offset, dfsg0.cm, '-o', ms=ms, label='0')
    ax.plot(dfsg3[param_z] + z_offset, dfsg3.cm, '-o', ms=ms, label='3')
    ax.plot(dfsg5[param_z] + z_offset, dfsg5.cm, '-o', ms=ms, label='5')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
    ax.legend(title=r'$d_{med}$')
    plt.tight_layout()
    plt.savefig(path_results + '/average-particle-image-similarity_by_{}_plot.svg'.format(param_z))
    plt.show()
    plt.close()

    # ---

    # error bars
    capsize = 1
    elinewidth = 0.25
    fig, ax = plt.subplots()

    ax.errorbar(dfsg0[param_z] + z_offset, dfsg0.cm, yerr=dfsg0_std.cm,
                fmt='o', ms=ms, capsize=capsize, elinewidth=elinewidth, label='0')
    ax.errorbar(dfsg3[param_z] + z_offset, dfsg3.cm, yerr=dfsg3_std.cm,
                fmt='o', ms=ms, capsize=capsize, elinewidth=elinewidth, label='3')
    ax.errorbar(dfsg5[param_z] + z_offset, dfsg5.cm, yerr=dfsg5_std.cm,
                fmt='o', ms=ms, capsize=capsize, elinewidth=elinewidth, label='5')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
    ax.legend(title=r'$d_{med}$')
    plt.tight_layout()
    plt.savefig(path_results + '/average-particle-image-similarity_by_{}_errorbars.svg'.format(param_z))
    plt.show()
    plt.close()

# ---

# ---

# PLOT FORWARD SELF SIMILARITY

plot_self_similarity = False
if plot_self_similarity:
    median0_stack_id = 46
    median3_stack_id = 46
    median5_stack_id = 44

    fp0 = 'calib_stacks_forward_self-similarity_5uM_med-filter-0'
    fp3 = 'calib_stacks_forward_self-similarity_5uM_med-filter-3'
    fp5 = 'calib_stacks_forward_self-similarity_5uM_med-filter-5'

    dfs0 = pd.read_excel(join(path_read_similarity, fp0) + filetype)
    dfs3 = pd.read_excel(join(path_read_similarity, fp3) + filetype)
    dfs5 = pd.read_excel(join(path_read_similarity, fp5) + filetype)

    dfs0 = dfs0[dfs0['id'] == median0_stack_id]
    dfs3 = dfs3[dfs3['id'] == median3_stack_id]
    dfs5 = dfs5[dfs5['id'] == median5_stack_id]

    # ---

    # setup
    param_z = 'z'
    z_offset = -50

    # ---

    # line plot
    ms = 1
    fig, ax = plt.subplots()

    ax.plot(dfs0[param_z] + z_offset, dfs0.cm, '-o', ms=ms, label='0')
    ax.plot(dfs3[param_z] + z_offset, dfs3.cm, '-o', ms=ms, label='3')
    ax.plot(dfs5[param_z] + z_offset, dfs5.cm, '-o', ms=ms, label='5')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$S(z_{i}, z)$')
    ax.legend(title=r'$d_{med}$')
    plt.tight_layout()
    plt.savefig(path_results + '/forward-self-similarity_by_{}_plot.svg'.format(param_z))
    plt.show()
    plt.close()

    # ---

# ---

# --- COMPARE COLLECTION SIMILARITY TO SELF-SIMILARITY

# --- compare collection similarity to self similarity
compare_collection_and_self_similarity = False

if compare_collection_and_self_similarity:

    path_results_compare = join(path_results, 'compare-col-and-self-sim')
    if not os.path.exists(path_results_compare):
        os.makedirs(path_results_compare)

    # --- read + processing

    # setup
    median0_stack_id = 46

    # collection similarity
    fp0 = 'collection_similarities_5uM_med-filter-0'
    dfs0 = pd.read_excel(join(path_read_similarity, fp0) + filetype)
    dfsg0 = dfs0.groupby('z').mean().reset_index()
    dfsg0_std = dfs0.groupby('z').std().reset_index()

    # self-similarity
    fp0 = 'calib_stacks_forward_self-similarity_5uM_med-filter-0'
    dfs0 = pd.read_excel(join(path_read_similarity, fp0) + filetype)
    dfs0 = dfs0[dfs0['id'] == median0_stack_id]

    # ---

    # --- plotting

    # setup
    param_z = 'z'
    z_offset = -50
    ms = 3
    capsize = 1.5
    elinewidth = 0.5
    errorevery = 5

    # plot

    # figure: error bars collection similarity + plot single particle calibration used for testing
    fig, ax = plt.subplots()

    ax.errorbar(dfsg0[param_z] + z_offset, dfsg0.cm, yerr=dfsg0_std.cm,
                fmt='o', ms=ms, capsize=capsize, elinewidth=elinewidth, errorevery=errorevery, color=sciblue,
                label=r'$\overline{S}(p_{i}, p_{N})$')

    ax.plot(dfs0[param_z] + z_offset, dfs0.cm,
            '-o', ms=ms, color=scired,
            label=r'${S} (z_{i}, z_{i+1})$')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_ylabel(r'$S$')
    ax.set_ylim(bottom=0.525)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path_results_compare + '/spct_compare-col-and-self-sim-spc_by_z_error-every-{}.svg'.format(errorevery))
    plt.show()
    plt.close()

    # ---

    # figure: (same as above but with title showing the mean + std across the full axial range)

    # processing
    p2p_mean_sim = dfsg0.cm.mean()
    p2p_std_sim = dfsg0.cm.std()
    p2p_mean_std = dfsg0_std.cm.mean()

    ss_mean = dfs0.cm.mean()
    ss_std = dfs0.cm.std()

    # plot

    fig, ax = plt.subplots()

    ax.errorbar(dfsg0[param_z] + z_offset, dfsg0.cm, yerr=dfsg0_std.cm,
                fmt='o', ms=ms, capsize=capsize, elinewidth=elinewidth, errorevery=errorevery, color=sciblue,
                label=r'$\overline{S}(p_{i}, p_{N})$')

    ax.plot(dfs0[param_z] + z_offset, dfs0.cm,
            '-o', ms=ms, color=scired,
            label=r'${S} (z_{i}, z_{i+1})$')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_ylabel(r'$S$')
    ax.set_ylim(bottom=0.525)
    ax.legend()

    ax.set_title(r'$\overline{S} (z_{i}, z_{i+1}) \pm \sigma =$' +
                 ' {} '.format(np.round(ss_mean, 2)) +
                 r'$\pm$' +
                 ' {}'.format(np.round(ss_std, 2)))

    plt.suptitle(r'$| \overline{S}(p_{i}, p_{N}) | \pm \sigma \pm \overline{\sigma_{cm}} =$' +
                 ' {} '.format(np.round(p2p_mean_sim, 2)) +
                 r'$\pm$' +
                 ' {} '.format(np.round(p2p_std_sim, 2)) +
                 r'$\pm$' +
                 ' {}'.format(np.round(p2p_mean_std, 2)))

    plt.tight_layout()
    plt.savefig(path_results_compare + '/spct_compare-col-and-self-sim-spc_by_z_titled.svg')
    plt.show()
    plt.close()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------

# ---

# READ AND CORRECT RAW TEST COORDS

correct_raw_test_coords = False

if correct_raw_test_coords:

    test_type = 'vary-test-template-size'

    path_test_coords = join(base_dir, 'data/test-coords/{}/raw'.format(test_type))
    path_test_corr_coords = join(base_dir, 'data/test-coords/{}/custom'.format(test_type))

    if not os.path.exists(path_test_corr_coords):
        os.makedirs(path_test_corr_coords)

    files = [f for f in os.listdir(path_test_coords) if f.endswith('.xlsx')]

    for f in files:
        df = pd.read_excel(join(path_test_coords, f))

        # adjust z_true
        df['z_true_test'] = (df['z_true'] - df['z_true'] % 3) / 3 * 5

        # adjust z_true_corr
        df['z_true_corr'] = df['z_true_test'] - 68.6519

        # adjust z_corr
        df['z_corr'] = df['z'] - 49.9176 - 5

        # adjust error_corr
        df['error_corr'] = df['z_corr'] - df['z_true_corr']

        # make the "corrected" columns the actual columns
        df['z_true'] = df['z_true_corr']
        df['z'] = df['z_corr']
        df['error'] = df['error_corr']

        # drop the "corrected" columns
        df = df.drop(columns=['z_true_test', 'z_true_corr', 'z_corr', 'error_corr'])

        # export
        df.to_excel(join(path_test_corr_coords, f), index=False)

# ---

# ANALYZE TEST COORDS

# experimental
mag_eff = 10.0
microns_per_pixel = 1.6
area_pixels = 512 ** 2
area_microns = (512 * microns_per_pixel) ** 2

# processing
z_range = [-55, 55]
measurement_depth = z_range[1] - z_range[0]
filter_barnkob = measurement_depth / 10
true_num_particles_per_frame = 92
num_frames_per_step = 3
num_z_steps = 22
true_total_num_particles = true_num_particles_per_frame * num_frames_per_step * num_z_steps
min_cm = 0.5

# ---

analyze_test_coords = False
if analyze_test_coords:

    test_type = 'vary-both-template-size'

    # file paths
    path_test_coords = join(base_dir, 'data/test-coords/{}/custom'.format(test_type))
    path_results = join(base_dir, 'results', test_type, 'test')

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # read files
    files = [f for f in os.listdir(path_test_coords) if f.endswith('.xlsx')]

    # processing
    min_percent_layers = 0.5
    remove_ids = None

    for f in files:

        # modifiers
        if 'idpt' in f:
            method = 'idpt'
            padding = 5
            img_xc, img_yc = 256 + padding, 256 + padding
        else:
            method = 'spct'
            padding = 0
            img_xc, img_yc = 256, 256

        dft = pd.read_excel(join(path_test_coords, f))
        dft['z_corr'] = dft['z']

        # compute the radial distance
        dft['r'] = np.sqrt((img_xc - dft.x) ** 2 + (img_yc - dft.y) ** 2)

        # ------------------------------------------------------------------------------------------------------------------
        # 4. STORE RAW STATS (AFTER FILTERING RANGE)

        # filter range so test matches calibration range
        dft = dft[(dft['z_true'] > z_range[0]) & (dft['z_true'] < z_range[1])]

        i_num_rows = len(dft)
        i_num_pids = len(dft.id.unique())

        # ------------------------------------------------------------------------------------------------------------------
        # 5. SPLIT DATAFRAME INTO BINS

        true_zs = dft.z_true.unique()
        dft['bin'] = np.round(dft['z_true'], 0).astype(int)
        dzs = dft.bin.unique()
        num_dz_steps = len(dzs)

        dfs = []
        names = []
        initial_stats = []

        for i, dz in enumerate(dzs):
            dftz = dft[dft['bin'] == dz]
            dfs.append(dftz)
            names.append(i)
            initial_stats.append([i, dz, len(dftz), len(dftz.id.unique())])

        # ----------------------------------------------------------------------------------------------------------------------
        # 8. EVALUATE RMSE Z
        analyze_rmse = True

        if analyze_rmse:

            export_results = True
            save_plots = True
            show_plots = False

            # create file paths
            if test_type == 'vary-median-filter':
                string_idx = -8
            else:
                string_idx = -10
            path_results_per = join(path_results, method + '-' + f[string_idx:-5])
            path_figs = join(path_results, method + '-' + f[string_idx:-5], 'figs')

            if not os.path.exists(path_results_per):
                os.makedirs(path_results_per)
            if not os.path.exists(path_figs):
                os.makedirs(path_figs)

            # filter out bad particle ids and barnkob filter
            dffs_errors = []
            dffs = []
            dfilt = []
            for df, name, dz in zip(dfs, names, dzs):

                # initial rows and pids
                i_pid_num_rows = true_num_particles_per_frame * num_frames_per_step
                i_pid_num_pids = true_num_particles_per_frame

                # filter
                if remove_ids:
                    exclude_ids = remove_ids[name][2]
                    df = df[~df.id.isin(exclude_ids)]

                # final rows and pids
                f_pid_num_rows = len(df)
                f_pid_num_pids = len(df.id.unique())

                # barnkob filter
                df_error = df[df['error'].abs() > filter_barnkob]
                df = df[df['error'].abs() < filter_barnkob]
                f_barnkob_num_rows = len(df)
                f_barnkob_num_pids = len(df.id.unique())

                # calculate stats
                dz_percent_pids = f_barnkob_num_pids / i_pid_num_pids * 100
                dz_percent_rows = f_barnkob_num_rows / i_pid_num_rows * 100

                dffs_errors.append(df_error)
                dffs.append(df)
                dfilt.append([name, dz, dz_percent_pids, i_pid_num_pids, f_pid_num_pids, f_barnkob_num_pids,
                              dz_percent_rows, i_pid_num_rows, f_pid_num_rows, f_barnkob_num_rows])

            # focal plane bias errors
            dffp_errors = pd.concat(dffs_errors, ignore_index=True)
            dffp_errors.to_excel(path_results_per + '/test_coords_focal-plane-bias-errors.xlsx', index=False)

            # filtered and corrected test coords
            dffp = pd.concat(dffs, ignore_index=True)
            dfilt = pd.DataFrame(np.array(dfilt),
                                 columns=['name', 'dz', 'dz_percent_pids', 'i_pid_num_pids', 'f_pid_num_pids',
                                          'f_barnkob_num_pids', 'dz_percent_rows', 'i_pid_num_rows',
                                          'f_pid_num_rows', 'f_barnkob_num_rows'])
            dfilt = dfilt.set_index('dz')

            # store stats
            f_num_rows = len(dffp)
            f_num_pids = len(dffp.id.unique())

            # rmse
            dffp = dffp.rename(columns={'bin': 'binn'})

            # export final + filtered test coords
            if export_results:
                dffp.to_excel(path_results_per + '/test_coords_spct-corrected-and-filtered.xlsx', index=False)

            # bin generic
            dfrmsem, dfrmsestd = bin.bin_generic(dffp,
                                                 column_to_bin='binn',
                                                 column_to_count='id',
                                                 bins=dzs,
                                                 round_to_decimal=0,
                                                 return_groupby=True)

            dfrmse_mean = bin.bin_local_rmse_z(dffp, column_to_bin='binn', bins=1, min_cm=min_cm,
                                               z_range=None, round_to_decimal=0, df_ground_truth=None, dropna=True,
                                               error_column='error')

            dfrmse_bin = bin.bin_local_rmse_z(dffp, column_to_bin='binn', bins=dzs, min_cm=min_cm,
                                              z_range=None, round_to_decimal=0, df_ground_truth=None, dropna=True,
                                              error_column='error')

            # join rmse and filter dataframes
            dfrmse = pd.concat([dfrmse_bin, dfilt], axis=1, join='inner')

            # export
            if export_results:
                dfrmse.to_excel(path_results_per + '/rmse-z_binned.xlsx')
                dfrmse_mean.to_excel(path_results_per + '/rmse-z_mean.xlsx')

            # --- plot
            if save_plots or show_plots:

                # fit line
                popt, pcov = curve_fit(functions.line, dfrmse.z_true, dfrmse.z)
                z_fit = np.linspace(dfrmse.z_true.min(), dfrmse.z_true.max())

                rmse_fit_line = np.sqrt(np.sum((functions.line(dfrmse.z_true, *popt) - dfrmse.z) ** 2) / len(dfrmse.z))
                print("{} = r.m.s. error of the mean z per z-step from fitted line".format(rmse_fit_line))

                xylim = 62.5
                xyticks = [-50, -25, 0, 25, 50]

                # close all figs
                plt.close('all')

                # binned calibration curve with std-z errorbars (microns) + fit line
                if len(dfrmse) == len(dfrmsestd):
                    fig, ax = plt.subplots()
                    ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmsestd.z, fmt='o', ms=3, elinewidth=0.5, capsize=1,
                                color=sciblue,
                                label=r'$\overline{z} \pm \sigma$')  #
                    ax.plot(z_fit, functions.line(z_fit, *popt), linestyle='--', linewidth=1.5, color='black',
                            alpha=0.25,
                            label=r'$dz/dz_{true} = $' + ' {}'.format(np.round(popt[0], 3)))
                    ax.plot(dffp_errors.z_true, dffp_errors.z, 'x', ms=3, color='tab:red',
                            label=r'$\epsilon_{z} > h/10$')  #
                    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                    ax.set_xlim([-xylim, xylim])
                    ax.set_xticks(ticks=xyticks, labels=xyticks)
                    ax.set_ylabel(r'$z \: (\mu m)$')
                    ax.set_ylim([-xylim, xylim])
                    ax.set_yticks(ticks=xyticks, labels=xyticks)
                    ax.legend(loc='lower right', handletextpad=0.25, borderaxespad=0.3)
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_figs +
                                    '/calibration_curve_z+std-errobars_fit_line_a{}_b{}_slope-label-blk.png'.format(
                                        np.round(popt[0],
                                                 3),
                                        np.round(popt[1],
                                                 3))
                                    )
                    if show_plots:
                        plt.show()
                    plt.close()

                # rmse-z: microns
                fig, ax = plt.subplots()
                ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/rmse-z_microns.png')
                if show_plots:
                    plt.show()
                plt.close()

                # rmse-z: normalized
                fig, ax = plt.subplots()
                ax.plot(dfrmse.index, dfrmse.rmse_z / measurement_depth, '-o')
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$\sigma_{z}/h$')
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/rmse-z_normalized.png')
                if show_plots:
                    plt.show()
                plt.close()

                # rmse-z + percent rows
                fig, ax = plt.subplots()
                ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')

                axr = ax.twinx()
                axr.plot(dfrmse.index, dfrmse.dz_percent_rows, '-s', ms=2, color='gray', alpha=0.5)
                axr.set_ylabel(r'$\phi_{ID} \: (\%)$', color='gray')
                axr.set_ylim([-5, 105])
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/rmse-z_and_percent_measure.png')
                if show_plots:
                    plt.show()
                plt.close()

                # binned calibration curve with rmse-z errorbars (microns)
                fig, ax = plt.subplots()
                ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmse.rmse_z / 2, fmt='o', ms=1, elinewidth=0.5, capsize=1,
                            label=r'$\overline{z} \pm \sigma_{z}/2$')
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.set_ylim([-xylim, xylim])
                ax.set_yticks(ticks=xyticks, labels=xyticks)
                ax.legend()
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/rmse-z_microns_errorbars-are-rmse.png')
                if show_plots:
                    plt.show()
                plt.close()

                # binned calibration curve with rmse-z errorbars (microns) + fit line
                fig, ax = plt.subplots()
                ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmse.rmse_z / 2, fmt='o', ms=1.5, elinewidth=0.75,
                            capsize=1,
                            label=r'$\overline{z} \pm \sigma_{z}/2$')
                ax.plot(z_fit, functions.line(z_fit, *popt), linestyle='--', linewidth=1, color='black', alpha=0.25,
                        label=r'$dz/dz_{true} = $' + ' {}'.format(np.round(popt[0], 3)))
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.set_ylim([-xylim, xylim])
                ax.set_yticks(ticks=xyticks, labels=xyticks)
                ax.legend()
                plt.tight_layout()
                if save_plots:
                    plt.savefig(
                        path_figs + '/rmse-z_microns_errorbars-are-rmse_fit_line_a{}_b{}.png'.format(
                            np.round(popt[0], 3),
                            np.round(popt[1], 3)))
                if show_plots:
                    plt.show()
                plt.close()

                # all points calibration curve (filtered)
                fig, ax = plt.subplots()
                ax.scatter(dffp.z_true, dffp.z, s=3)
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.set_ylim([-xylim, xylim])
                ax.set_yticks(ticks=xyticks, labels=xyticks)
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/calibration_curve_filtered.png')
                if show_plots:
                    plt.show()
                plt.close()

                # all points calibration curve (raw)
                fig, ax = plt.subplots()
                ax.scatter(dft.z_true, dft.z, s=3)
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.set_ylim([-xylim, xylim])
                ax.set_yticks(ticks=xyticks, labels=xyticks)
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/calibration_curve_raw.png')
                if show_plots:
                    plt.show()
                plt.close()

                # all points calibration curve (filtered + errors)
                fig, ax = plt.subplots()
                ax.plot(dffp.z_true, dffp.z, linestyle='', marker="o", ms=3,
                        label=r'Valid%'.format(np.round(len(dffp) / (len(dffp) + len(dffp_errors)) * 100, 1)))
                ax.plot(dffp_errors.z_true, dffp_errors.z, linestyle='', marker=r'$\times$', ms=3, mfc='white',
                        fillstyle='none', color='tab:red', label=r'$\epsilon_{z} > h/10$')
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$z \: (\mu m)$')
                ax.set_ylim([-xylim, xylim])
                ax.set_yticks(ticks=xyticks, labels=xyticks)
                # ax.legend()
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/calibration_curve_filtered_and_outliers_no-label.png')
                if show_plots:
                    plt.show()
                plt.close()

        # ---

# ---

# ANALYZE TEST COORDS

analyze_test_outputs = True
if analyze_test_outputs:

    test_type = 'vary-test-template-size'

    # file paths
    path_results = join(base_dir, 'results', test_type, 'test')

    if test_type == 'vary-median-filter':
        save_id = 'median-filters'

        dirs = ['idpt-0-0', 'idpt-0-3', 'idpt-0-5', 'idpt-3-3', 'idpt-5-5',
                'spct-0-0', 'spct-0-3', 'spct-0-5', 'spct-3-3', 'spct-5-5']
        lss = ['none', 'none', 'none', 'none', 'none',
               'none', 'none', 'none', 'none', 'none']
        mrks = ['o', '^', 'p', 'h', '*',
                'o', '^', 'p', 'h', '*']
        clrs = [sciblue, sciblue, sciblue, sciblue, sciblue,
                scigreen, scigreen, scigreen, scigreen, scigreen]
        clr_mods = [1, 0.9, 0.8, 1.1, 1.2,
                    1, 0.9, 0.8, 1.1, 1.2]

    elif test_type == 'vary-test-template-size':
        save_id = 'test-templates'

        dirs = ['idpt-13-19', 'idpt-14-19', 'idpt-15-19', 'idpt-16-19', 'idpt-17-19', 'idpt-18-19', 'idpt-19-19']
        lss = ['none', 'none', 'none', 'none', 'none', 'none', 'none']
        mrks = ['o', '^', 'p', 'h', '*', 's', 'd']
        clrs = sci_colors
        clr_mods = [1, 1, 1, 1, 1, 1, 1]

    elif test_type == 'vary-both-template-size':
        save_id = 'both-templates'

        dirs = ['idpt-13-16', 'idpt-14-17', 'idpt-15-18', 'idpt-16-19', 'idpt-17-20', 'idpt-18-21']
        lss = ['none', 'none', 'none', 'none', 'none', 'none']
        mrks = ['o', '^', 'p', 'h', '*', 's']
        clrs = sci_colors
        clr_mods = [1, 1, 1, 1, 1, 1]

    else:
        raise ValueError("test type not understood.")

    # --- PLOT LOCAL AXIAL UNCERTAINTY

    # read
    dfbs = []
    for read_dir in dirs:
        dfb = pd.read_excel(join(path_results, read_dir, 'rmse-z_binned.xlsx'), index_col=0)
        dfbs.append(dfb)

    # setup

    ms = 3

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
    for dfb, read_dir, ls, mrk, clr, clr_mod in zip(dfbs, dirs, lss, mrks, clrs, clr_mods):
        ax.plot(dfb.index, dfb.rmse_z, linestyle=ls, marker=mrk, ms=ms, color=lighten_color(clr, clr_mod),
                label=read_dir)
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(join(path_results, 'compare-all_{}_local-rmse-z.svg'.format(save_id)))
    plt.show()
    plt.close()

    # plot: rmse_z and true percent measure
    ms = 1
    fig, (axr, ax) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.125))
    for dfb, read_dir, ls, mrk, clr, clr_mod in zip(dfbs, dirs, lss, mrks, clrs, clr_mods):
        axr.plot(dfb.index, dfb.dz_percent_rows, linestyle='-', marker=mrk, ms=ms, color=lighten_color(clr, clr_mod))
        ax.plot(dfb.index, dfb.rmse_z, linestyle='-', marker=mrk, ms=ms, color=lighten_color(clr, clr_mod),
                label=read_dir)
    axr.set_ylabel(r'$\phi \: (\%)$')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(join(path_results, 'compare-all_{}_local-rmse-z-and-true-percent-meas.svg'.format(save_id)))
    plt.show()
    plt.close()

    # plot: rmse_z and true percent measure
    fig, (axr, ax) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.125))
    for dfb, read_dir, ls, mrk, clr, clr_mod in zip(dfbs, dirs, lss, mrks, clrs, clr_mods):
        axr.plot(dfb.index, dfb.cm, linestyle='-', marker=mrk, ms=ms, color=lighten_color(clr, clr_mod))
        ax.plot(dfb.index, dfb.rmse_z, linestyle='-', marker=mrk, ms=ms, color=lighten_color(clr, clr_mod),
                label=read_dir)
    axr.set_ylabel(r'$c_{m}$')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(join(path_results, 'compare-all_{}_local-rmse-z-and-cm.svg'.format(save_id)))
    plt.show()
    plt.close()

    # ---

    # ---

    # compare IDPT to SPCT
    compare_idpt_to_spct = False

    if compare_idpt_to_spct:

        # setup
        i_idx = [0, 3]
        s_idx = [5, 8]
        ms = 2.5

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches))
        for dfb, read_dir, ls, mrk, clr, clr_mod in zip(dfbs[i_idx[0]:i_idx[1]],
                                                        dirs[i_idx[0]:i_idx[1]],
                                                        lss[i_idx[0]:i_idx[1]],
                                                        mrks[i_idx[0]:i_idx[1]],
                                                        clrs[i_idx[0]:i_idx[1]],
                                                        clr_mods[i_idx[0]:i_idx[1]]):
            ax.plot(dfb.index, dfb.rmse_z, linestyle=ls, marker=mrk, ms=ms, color=lighten_color(clr, clr_mod),
                    label=read_dir)

        for dfb, read_dir, ls, mrk, clr, clr_mod in zip(dfbs[s_idx[0]:s_idx[1]],
                                                        dirs[s_idx[0]:s_idx[1]],
                                                        lss[s_idx[0]:s_idx[1]],
                                                        mrks[s_idx[0]:s_idx[1]],
                                                        clrs[s_idx[0]:s_idx[1]],
                                                        clr_mods[s_idx[0]:s_idx[1]]):
            ax.plot(dfb.index, dfb.rmse_z, linestyle=ls, marker=mrk, ms=ms, color=lighten_color(clr, clr_mod),
                    label=read_dir)

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(join(path_results, 'compare-test_median-filters_local-rmse-z.svg'))
        plt.show()
        plt.close()

        # ---

        # setup
        ms = 3.5
        dfbs_i = [dfbs[0], dfbs[3], dfbs[4]]
        dirs_i = [dirs[0], dirs[3], dirs[4]]
        lss_i = [lss[0], lss[3], lss[4]]
        mrks_i = [mrks[0], mrks[3], mrks[4]]
        clrs_i = [clrs[0], clrs[3], clrs[4]]
        clr_mods_i = [clr_mods[0], clr_mods[3], clr_mods[4]]

        dfbs_s = [dfbs[5], dfbs[8], dfbs[9]]
        dirs_s = [dirs[5], dirs[8], dirs[9]]
        lss_s = [lss[5], lss[8], lss[9]]
        mrks_s = [mrks[5], mrks[8], mrks[9]]
        clrs_s = [clrs[5], clrs[8], clrs[9]]
        clr_mods_s = [clr_mods[5], clr_mods[8], clr_mods[9]]

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches))
        for dfb, read_dir, ls, mrk, clr, clr_mod in zip(dfbs_i, dirs_i, lss_i, mrks_i, clrs_i, clr_mods_i):
            ax.plot(dfb.index, dfb.rmse_z, linestyle=ls, marker=mrk, ms=ms, color=lighten_color(clr, clr_mod), label=read_dir)

        for dfb, read_dir, ls, mrk, clr, clr_mod in zip(dfbs_s, dirs_s, lss_s, mrks_s, clrs_s, clr_mods_s):
            ax.plot(dfb.index, dfb.rmse_z, linestyle=ls, marker=mrk, ms=ms, color=lighten_color(clr, clr_mod), label=read_dir)

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(join(path_results, 'compare-cal-and-test_median-filters_local-rmse-z.svg'))
        plt.show()
        plt.close()

        # ---

    # ---

    # ---

    # --- PLOT GLOBAL AXIAL UNCERTAINTY

    # ---
    plot_global = True

    if plot_global:

        # median filtering

        if test_type == 'vary-median-filter':

            """path_results = join(base_dir, 'results', 'test')
            dirs = ['idpt-0-0', 'idpt-0-3', 'idpt-0-5', 'idpt-3-3', 'idpt-5-5',
                    'spct-0-0', 'spct-0-3', 'spct-0-5', 'spct-3-3', 'spct-5-5']"""

            # --- CALIBRATION MEDIAN FILTERING = 0

            # read
            dfbs = []
            for read_dir in dirs[0:3]:
                dfb = pd.read_excel(join(path_results, read_dir, 'rmse-z_mean.xlsx'), index_col=0)
                dfbs.append(dfb)
            dfbs_i_c0 = pd.concat(dfbs)

            dfbs = []
            for read_dir in dirs[5:8]:
                dfb = pd.read_excel(join(path_results, read_dir, 'rmse-z_mean.xlsx'), index_col=0)
                dfbs.append(dfb)
            dfbs_s_c0 = pd.concat(dfbs)

            # setup
            x = np.array([0, 3, 5])
            ms = 5

            fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))

            ax.plot(x, dfbs_i_c0.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
            ax.plot(x, dfbs_s_c0.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')

            ax.set_xlabel(r'$d_{med}^{t}$')
            ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(join(path_results, 'compare-test_median-filters_global-rmse-z.svg'))
            plt.show()
            plt.close()

            # ---

            # --- MEDIAN FILTERING BOTH CALIBRATION AND TEST

            # read
            dfbs = []
            for read_dir in [dirs[0], dirs[3], dirs[4]]:
                dfb = pd.read_excel(join(path_results, read_dir, 'rmse-z_mean.xlsx'), index_col=0)
                dfbs.append(dfb)
            dfbs_i_c0 = pd.concat(dfbs)

            dfbs = []
            for read_dir in [dirs[5], dirs[8], dirs[9]]:
                dfb = pd.read_excel(join(path_results, read_dir, 'rmse-z_mean.xlsx'), index_col=0)
                dfbs.append(dfb)
            dfbs_s_c0 = pd.concat(dfbs)

            # setup
            x = np.array([0, 3, 5])
            ms = 5

            # plot
            fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))

            ax.plot(x, dfbs_i_c0.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
            ax.plot(x, dfbs_s_c0.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')

            ax.set_xlabel(r'$d_{med}$')
            ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(join(path_results, 'compare-cal-and-test_median-filters_global-rmse-z.svg'))
            plt.show()
            plt.close()

            # ---

            # plot rmse_z and true percent measure
            fig, (axr, ax) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))

            axr.plot(x, dfbs_i_c0.num_meas / true_total_num_particles * 100, '-o', ms=ms, color=sciblue, label='IDPT')
            axr.plot(x, dfbs_s_c0.num_meas / true_total_num_particles * 100, '-o', ms=ms, color=scigreen, label='SPCT')
            ax.plot(x, dfbs_i_c0.rmse_z, '-o', ms=ms, color=sciblue)
            ax.plot(x, dfbs_s_c0.rmse_z, '-o', ms=ms, color=scigreen)

            axr.set_ylabel(r'$\phi \: (\%)$')
            axr.set_ylim([41, 101])
            axr.legend()
            ax.set_xlabel(r'$d_{med}$')
            ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_results, 'compare-cal-and-test_median-filters_global-rmse-z-and-true-percent-meas.svg'))
            plt.show()
            plt.close()

            # ---

            # plot rmse_z and cm
            fig, (axr, ax) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))

            axr.plot(x, dfbs_i_c0.cm, '-o', ms=ms, color=sciblue, label='IDPT')
            axr.plot(x, dfbs_s_c0.cm, '-o', ms=ms, color=scigreen, label='SPCT')
            ax.plot(x, dfbs_i_c0.rmse_z, '-o', ms=ms, color=sciblue)
            ax.plot(x, dfbs_s_c0.rmse_z, '-o', ms=ms, color=scigreen)

            axr.set_ylabel(r'$c_{m}$')
            axr.set_ylim([0.875, 1.00925])
            axr.legend()
            ax.set_xlabel(r'$d_{med}$')
            ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_results, 'compare-cal-and-test_median-filters_global-rmse-z-and-cm.svg'))
            plt.show()
            plt.close()

            raise ValueError("Done ! ")

            # ---

            # ---

        # modifiers
        elif test_type == 'vary-test-template-size':
            # dirs = ['idpt-13-19', 'idpt-14-19', 'idpt-15-19', 'idpt-16-19', 'idpt-17-19', 'idpt-18-19', 'idpt-19-19']
            x = np.arange(13, 20)
            xlbl = r'$l_{t}, \: l_{c}=19  \: (pix.)$'
        elif test_type == 'vary-both-template-size':
            # dirs = ['idpt-13-16', 'idpt-14-17', 'idpt-15-18', 'idpt-16-19', 'idpt-17-20', 'idpt-18-21']
            x = np.arange(13, 19)
            xlbl = r'$l_{t}, \: l_{c}=l_{t}+3 \: (pix.)$'

        # --- MEDIAN FILTERING BOTH CALIBRATION AND TEST

        # read
        dfbs = []
        for read_dir in dirs:
            dfb = pd.read_excel(join(path_results, read_dir, 'rmse-z_mean.xlsx'), index_col=0)
            dfbs.append(dfb)
        dfbs_i_c0 = pd.concat(dfbs)

        # ---

        # export
        dfbs_i_c0.to_excel(join(path_results, 'combined_{}_global-rmse-z.xlsx'.format(save_id)))
        raise ValueError()

        # ---

        # setup
        ms = 5

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))

        ax.plot(x, dfbs_i_c0.rmse_z, '-o', ms=ms, color=sciblue)

        ax.set_xlabel(xlbl)
        ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_results, 'compare-cal-and-test_{}_global-rmse-z.svg'.format(save_id)))
        plt.show()
        plt.close()

        # ---

        # plot rmse_z and true percent measure
        fig, (axr, ax) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))

        axr.plot(x, dfbs_i_c0.num_meas / true_total_num_particles * 100, '-o', ms=ms, color=sciblue)
        ax.plot(x, dfbs_i_c0.rmse_z, '-o', ms=ms, color=sciblue)

        axr.set_ylabel(r'$\phi \: (\%)$')
        axr.set_ylim([41, 101])
        ax.set_xlabel(xlbl)
        ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_results, 'compare-{}_global-rmse-z-and-true-percent-meas.svg'.format(save_id)))
        plt.show()
        plt.close()

        # ---

        # plot rmse_z and cm
        fig, (axr, ax) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))

        axr.plot(x, dfbs_i_c0.cm, '-o', ms=ms, color=sciblue, label='IDPT')
        ax.plot(x, dfbs_i_c0.rmse_z, '-o', ms=ms, color=sciblue)

        axr.set_ylabel(r'$c_{m}$')
        axr.set_ylim([0.875, 1.00925])
        ax.set_xlabel(xlbl)
        ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(join(path_results, 'compare-{}_global-rmse-z-and-cm.svg'.format(save_id)))
        plt.show()
        plt.close()

    # ---

# ---

print("Analysis completed without errors.")