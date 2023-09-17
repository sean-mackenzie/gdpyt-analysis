# test bin, analyze, and plot functions
import os
from os.path import join
import ast
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy.optimize import curve_fit

import analyze
from utils import io, bin, plotting, modify, functions, plot_collections
from utils.plotting import lighten_color
import filter
from datasets.grid_overlap import DatasetUnpacker

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# ------------------------------------------------
# formatting
sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF9500'
sciorange = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# -----------------------

# ----------------------------------------------------------------------------------------------------------------------
# 0. SETUP DATASET DETAILS

# read files
dataset = 'grid-dz-overlap'

# read .xlsx result files to dictionary
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/{}'.format(dataset)
path_coords = join(base_path, 'test_coords/z_adj')
path_figs = join(base_path, 'figs')
path_results = join(base_path, 'results')
settings_sort_strings = ['settings_id', '_coords_']
test_sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'

# save figures
save_fig_filetype = '.svg'

# ----------------------------------------------------------------------------------------------------------------------
# A. ANALYZE TEST COORDS
analyze_test_coords = True

# run
if analyze_test_coords:

    # define
    def analyze_grid_dz_overlap(inspect_key, path_results):
        """
        Comprehensive script to process raw data, compute tracking performance, and plot results.

        Key Processes:
            (1) Read 'raw' test_coords and compute x_adj, z_adj, dz_adj using Nearest Neighbors.
            (2) Compute tracking performance for all relevant binning schemes.
                (a) export results to Excel.
                (b) plot results.

        :param inspect_key:
        :param path_results:
        :return:
        """
        # setup
        inspect_gdpyt_by_key = inspect_key
        print("TEST ID {} IS BEING EVALUATED.".format(inspect_key))

        # read dataset details
        details = DatasetUnpacker(dataset='grid-overlap', key=inspect_key).unpack()
        inspect_key = details['key']
        splits = details['splits']
        keys = details['keys']
        dict_splits_to_keys = details['dict_splits_to_keys']
        intercolumn_spacing_threshold = details['intercolumn_spacing_threshold']
        x_filter = details['x_filter']
        x_filter_operation = details['x_filter_operation']
        min_length_per_split = details['min_length_per_split']
        single_column_dx = details['single_column_dx']
        single_column_x = details['single_column_x']
        save_id = details['save_id']
        template_size = details['template_size']
        max_diameter = details['max_diameter']

        # make dataset-specific folders to save figs/results
        path_results = join(path_results, save_id)
        if not os.path.exists(path_results):
            os.makedirs(path_results)

        path_figs = join(path_results, 'figs')
        if not os.path.exists(path_figs):
            os.makedirs(path_figs)

        path_results = join(path_results, 'results')
        if not os.path.exists(path_results):
            os.makedirs(path_results)

        # data slicing
        column_to_split = 'x'
        round_x_to_decimal = 0

        # filters for removing artificial baseline
        discard_frames_before = 0.5

        # read files
        dficts = io.read_files('df',
                               path_coords,
                               test_sort_strings,
                               filetype,
                               startswith=test_sort_strings[0],
                               subset=inspect_key,
                               )
        dfsettings = io.read_files('dict',
                                   path_coords,
                                   settings_sort_strings,
                                   filetype,
                                   startswith=settings_sort_strings[0],
                                   columns=['parameter', 'settings'],
                                   dtype=str,
                                   subset=inspect_key,
                                   )
        dficts_ground_truth = io.read_ground_truth_files(dfsettings)

        """
        Packaging true coords for GDPTlab evaluation:
        
        dftrue = modify.stack_dficts_by_key(dficts_ground_truth, drop_filename=True)
        dftrue = dftrue[dftrue['filename'] > 1700]
        dftrue = dftrue.drop(columns=['p_d'])
        arr_true = dftrue.to_numpy()

        sp = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level2/grid-dz/misc/gdpt'
        np.savetxt(sp + '/true_coordinates.txt', arr_true, fmt='%.4f', delimiter=',', header='fr,X,Y,Z')
        """

        # add 'z_adj' and 'dz' column to dataframe using NearestNeighbors
        if 'dz' not in dficts[inspect_gdpyt_by_key].columns:
            # add 'z_adj' and 'dz' columns
            df_dz = modify.map_adjacent_z_true(dficts[inspect_gdpyt_by_key],
                                               dficts_ground_truth[inspect_gdpyt_by_key],
                                               dfsettings[inspect_gdpyt_by_key],
                                               threshold=intercolumn_spacing_threshold,
                                               single_column_x=single_column_x,
                                               single_column_dx=single_column_dx,
                                               )
            print('{} rows before drop NaNs'.format(len(df_dz)))
            df_dz = df_dz.dropna()
            print('{} rows after drop NaNs'.format(len(df_dz)))

            # export the corrected dataframe
            df_dz.to_excel(path_coords + '/test_id{}_coords_z_adj_{}.xlsx'.format(inspect_gdpyt_by_key, save_id),
                           index=False)

            # update the dictionary
            del dficts
            dficts = {inspect_gdpyt_by_key: df_dz}

            # ---

        # ---

        # ------------------------------------------------------------------------------------------------------------------
        # 0.5 READ PERTINENT DATA FROM SETTINGS
        nominal_zf = float(dfsettings[inspect_gdpyt_by_key]['calib_baseline_image'].split(
            dfsettings[inspect_gdpyt_by_key]['calib_base_string'])[-1].split('.tif')[0])

        # ------------------------------------------------------------------------------------------------------------------
        # 1. FILTER DATA
        apply_filters = True
        apply_barnkob_filter = True

        # filters
        z_range = [-50, 15]
        theta_range = [-5, 5]
        min_cm = 0.5
        h = z_range[1] - z_range[0]
        filter_barnkob = h / 10

        if inspect_key in [1, 11, 12, 21]:
            theta_range = [-1, 1]
            print("Adjusting theta range to (-1, 1) because no-dz-overlap dataset.")

        # apply filters
        if apply_filters:

            dficts = filter.dficts_filter(dficts,
                                          keys=['frame'],
                                          values=[discard_frames_before],
                                          operations=['greaterthan'],
                                          copy=False,
                                          only_keys=None,
                                          return_filtered=False)

            if x_filter is not None:
                dficts = filter.dficts_filter(dficts,
                                              keys=['x'],
                                              values=[x_filter],
                                              operations=[x_filter_operation],
                                              copy=False,
                                              only_keys=None,
                                              return_filtered=False)

                dficts_ground_truth = filter.dficts_filter(dficts_ground_truth,
                                                           keys=['x'],
                                                           values=[x_filter],
                                                           operations=[x_filter_operation],
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
                                          keys=['theta'],
                                          values=[theta_range],
                                          operations=['between'],
                                          copy=False,
                                          only_keys=None,
                                          return_filtered=False)

            dficts = filter.dficts_dropna(dficts, columns=['dz'])

            dficts = filter.dficts_filter(dficts,
                                          keys=['cm'],
                                          values=[min_cm],
                                          operations=['greaterthan'],
                                          copy=False,
                                          only_keys=None,
                                          return_filtered=False)

            if apply_barnkob_filter:
                dficts = filter.dficts_filter(dficts,
                                              keys=['error'],
                                              values=[[-filter_barnkob, filter_barnkob]],
                                              operations=['between'],
                                              copy=False,
                                              only_keys=None,
                                              return_filtered=False)

            # --- filter ground truth

            dficts_ground_truth = filter.dficts_filter(dficts_ground_truth,
                                                       keys=['z'],
                                                       values=[z_range],
                                                       operations=['between'],
                                                       copy=False,
                                                       only_keys=None,
                                                       return_filtered=False)

            dficts_ground_truth = filter.dficts_filter(dficts_ground_truth,
                                                       keys=['filename'],
                                                       values=[discard_frames_before],
                                                       operations=['greaterthan'],
                                                       copy=False,
                                                       only_keys=None,
                                                       return_filtered=False)

            # ---

        # ---

        # --------------------------------------------------------------------------------------------------------------
        # 1.5. CORRECT X-Y ERRORS

        print("WARNING! Correcting x-y errors somewhat arbitrarily!")

        # fix x value to reflect xm template matching (same for y)
        dficts[inspect_gdpyt_by_key]['x'] = dficts[inspect_gdpyt_by_key]['x'] + 1
        dficts[inspect_gdpyt_by_key]['y'] = dficts[inspect_gdpyt_by_key]['y'] + 1

        if inspect_key in [1, 2]:
            dficts[inspect_gdpyt_by_key]['x'] = dficts[inspect_gdpyt_by_key]['x'] + dficts[inspect_gdpyt_by_key]['xm'] - 3
            dficts[inspect_gdpyt_by_key]['y'] = dficts[inspect_gdpyt_by_key]['y'] + dficts[inspect_gdpyt_by_key]['ym'] - 3

        # ---

        # --------------------------------------------------------------------------------------------------------------
        # 2. SPLIT DATA BY X-COLUMN (EQUIV. TO DX OR PARTICLE-TO-PARTICLE SPACING)

        dsplicts = modify.split_df_and_merge_dficts(dficts[inspect_gdpyt_by_key],
                                                    keys,
                                                    column_to_split,
                                                    splits,
                                                    round_x_to_decimal,
                                                    min_length=min_length_per_split,
                                                    )

        # ground truth
        dsplicts_ground_truth = modify.split_df_and_merge_dficts(dficts_ground_truth[inspect_gdpyt_by_key],
                                                                 keys,
                                                                 column_to_split,
                                                                 splits,
                                                                 round_x_to_decimal,
                                                                 min_length=min_length_per_split,
                                                                 )

        # ------------------------------------------------------------------------------------------------------------------
        # 3. PLOTTING

        # setup filepaths
        path_figs_zero_dz = join(path_figs, 'zero-dz')  # not created
        path_figs_sub_onehalf_dz = join(path_figs, 'sub-onehalf-dz')
        path_figs_averages = join(path_figs, 'averages')
        path_figs_per_dx = join(path_figs, 'per-dx')
        path_figs_bin_2d = join(path_figs, 'bin-2d')
        path_figs_surface = join(path_figs, 'uncertainty_surface')
        path_figs_pair_error = join(path_figs, 'mean-pair-error')
        path_figs_bias_errors = join(path_figs, 'bias_errors')

        # common directories
        for fp in [path_figs_averages, path_figs_bin_2d]:
            if not os.path.exists(fp):
                os.makedirs(fp)

        # directories for dz-overlap
        if save_id.find('no') == -1:
            for fp in [path_figs_per_dx, path_figs_sub_onehalf_dz, path_figs_pair_error, path_figs_bias_errors,
                       path_figs_surface]:
                if not os.path.exists(fp):
                    os.makedirs(fp)

        # ---

        # formatting figures
        save_plots = True
        show_plots = False

        # ---

        # calculate mean rmse_z as a function of (dx, z_true, dz)
        plot_mean_uncertainty = True
        if plot_mean_uncertainty:

            # export mean results: bin = 1
            dfbm = bin.bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                        column_to_bin='z_true',
                                        bins=[nominal_zf],
                                        min_cm=min_cm,
                                        z_range=z_range,
                                        round_to_decimal=1,
                                        df_ground_truth=dficts_ground_truth[inspect_gdpyt_by_key],
                                        include_xy=True,
                                        )
            dfbm.to_excel(path_results + '/mean_rmse-z.xlsx')

            # ---

            # bin-dx: entire collection
            dfbx = {}
            for name, dfix in dsplicts.items():
                dfb = bin.bin_local_rmse_z(dfix,
                                           column_to_bin='dx',
                                           bins=1,
                                           min_cm=min_cm,
                                           z_range=z_range,
                                           round_to_decimal=0,
                                           df_ground_truth=None,
                                           include_xy=True,
                                           )
                dfbx.update({name: dfb})
            dfbx = modify.stack_dficts_by_key(dfbx, drop_filename=False)
            dfbx = dfbx.set_index('filename')

            # export rmse_z(bin-dx)
            dfbx.to_excel(path_results + '/mean_rmse-z_bin-dx.xlsx', index_label='dx')

            # bin-dx: plot rmse_z for entire collection
            fig, ax = plt.subplots()
            ax.plot(dfbx.index, dfbx.rmse_z, '-o')
            ax.set_xlabel(r'$\delta x \: (pixels)$')
            ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_figs_averages, save_id + '_all_bin-dx_plot-rmse-z.png'))
            if show_plots:
                plt.show()
            plt.close()

            # bin-dx: plot rmse_z, rmse_x, rmse_y for entire collection
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))
            ax1.plot(dfbx.index, dfbx.rmse_y, '-o')
            ax1.set_ylabel(r'$\sigma_y \: (\mu m)$')
            ax2.plot(dfbx.index, dfbx.rmse_x, '-o')
            ax2.set_ylabel(r'$\sigma_x \: (\mu m)$')
            ax3.plot(dfbx.index, dfbx.rmse_z, '-o')
            ax3.set_ylabel(r'$\sigma_z \: (\mu m)$')
            ax3.set_xlabel(r'$\delta x \: (pixels)$')

            plt.tight_layout()
            plt.savefig(join(path_figs_averages, save_id + '_all_bin-dx_plot-rmse-x-y-z.png'))
            if show_plots:
                plt.show()
            plt.close()

            # bin-dx: plot rmse_z, c_m for entire collection
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))
            ax1.plot(dfbx.index, dfbx.cm, '-o')
            ax1.set_ylabel(r'$c_{m}$')
            ax2.plot(dfbx.index, dfbx.rmse_z, '-o')
            ax2.set_ylabel(r'$\sigma_z \: (\mu m)$')
            ax2.set_xlabel(r'$\delta x \: (pixels)$')

            plt.tight_layout()
            plt.savefig(join(path_figs_averages, save_id + '_all_bin-dx_plot-rmse-z_and_cm.svg'))
            if show_plots:
                plt.show()
            plt.close()

            # ---

            # bin-z_true: entire collection
            bins_z = np.linspace(z_range[0], z_range[1], 21)
            dfb = bin.bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                       column_to_bin='z_true',
                                       bins=bins_z,
                                       min_cm=min_cm,
                                       z_range=z_range,
                                       round_to_decimal=3,
                                       df_ground_truth=dficts_ground_truth[inspect_gdpyt_by_key],
                                       include_xy=True,
                                       )

            # export mean bin-dx results
            dfb.to_excel(path_results + '/mean_rmse-z_bin_z-true.xlsx')

            # bin-z_true: plot rmse_z for entire collection
            fig, ax = plt.subplots()
            ax.plot(dfb.index, dfb.rmse_z, '-o', color=sciblue)
            ax.set_xlabel(r'$z_{true} \: (\mu m)$')
            ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
            axr = ax.twinx()
            axr.plot(dfb.index, dfb.true_percent_meas, '--', color=sciblue)
            axr.set_ylim(bottom=-2.5, top=102.5)
            axr.set_ylabel(r'$\phi \: (\%)$')
            plt.tight_layout()
            plt.savefig(join(path_figs_averages, save_id + '_all_bin-z-true_plot-rmse-z.png'))
            if show_plots:
                plt.show()
            plt.close()

            # bin-z_true: plot rmse_xyz for entire collection
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))
            ax1.plot(dfb.index, dfb.rmse_y, '-o')
            ax1.set_ylabel(r'$\sigma_y \: (\mu m)$')
            ax2.plot(dfb.index, dfb.rmse_x, '-o')
            ax2.set_ylabel(r'$\sigma_x \: (\mu m)$')
            ax3.plot(dfb.index, dfb.rmse_z, '-o')
            ax3.set_ylabel(r'$\sigma_z \: (\mu m)$')
            ax3.set_xlabel(r'$z_{true} \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_figs_averages, save_id + '_all_bin-z-true_plot-rmse-x-y-z.png'))
            if show_plots:
                plt.show()
            plt.close()

            # ---

            # bin-dz: entire collection
            if len(dficts[inspect_gdpyt_by_key]['dz'].unique()) > 1:

                bins_dz = 20
                dfb = bin.bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                           column_to_bin='dz',
                                           bins=bins_dz,
                                           min_cm=min_cm,
                                           z_range=None,
                                           round_to_decimal=3,
                                           df_ground_truth=None,
                                           include_xy=True,
                                           )

                # export mean bin-dz results
                dfb.to_excel(path_results + '/mean_rmse-z_bin_dz.xlsx')

                # bin-dz: plot rmse_z for entire collection
                fig, ax = plt.subplots()
                ax.plot(dfb.index, dfb.rmse_z, '-o')
                ax.set_xlabel(r'$\Delta z \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(join(path_figs_averages, save_id + '_all_bin-dz_plot-rmse-z.png'))
                if show_plots:
                    plt.show()
                plt.close()

                # bin-dz: plot rmse_z for entire collection
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))
                ax1.plot(dfb.index, dfb.rmse_y, '-o')
                ax1.set_ylabel(r'$\sigma_y \: (\mu m)$')
                ax2.plot(dfb.index, dfb.rmse_x, '-o')
                ax2.set_ylabel(r'$\sigma_x \: (\mu m)$')
                ax3.plot(dfb.index, dfb.rmse_z, '-o')
                ax3.set_ylabel(r'$\sigma_z \: (\mu m)$')
                ax3.set_xlabel(r'$\Delta z \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(join(path_figs_averages, save_id + '_all_bin-dz_plot-rmse-x-y-z.png'))
                if show_plots:
                    plt.show()
                plt.close()

                # ---

                # bin-theta: entire collection
                column_to_bin = 'theta'
                bin_theta = [-25, -20, -15, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 20, 25]

                dfb_theta = analyze.bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                                     column_to_bin=column_to_bin,
                                                     bins=bin_theta,
                                                     min_cm=min_cm,
                                                     z_range=z_range,
                                                     round_to_decimal=2,
                                                     df_ground_truth=None,
                                                     include_xy=True,
                                                     )

                # export mean bin-theta results
                dfb_theta.to_excel(path_results + '/mean_rmse-z_bin_theta.xlsx')

                # plots: (1) calculated results, (2) fitted parabola
                for alpha, lbl in zip([1, 0.0625], ['', '_fit']):

                    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
                    ax.plot(dfb_theta.index, dfb_theta.rmse_z, '-o', ms=2, alpha=alpha, label='IDPT')

                    if lbl == '_fit':
                        popt, pcov = curve_fit(functions.quadratic_slide, dfb_theta.index, dfb_theta.rmse_z)
                        fit_x = np.linspace(dfb_theta.index.min(), dfb_theta.index.max())
                        ax.plot(fit_x, functions.quadratic_slide(fit_x, *popt), '--', color=sciblue,
                                label=r'$f(x_{0}=$' + '\n' +
                                      '{}'.format(np.round(popt[1], 2)) + r'$)=$'
                                                                          '{}'.format(
                                    np.round(popt[0], 4)) + r'$(x+x_{0})^2+$' +
                                      '{}'.format(np.round(popt[1], 3)) + r'$(x+x_{0})+$' +
                                      '{}'.format(np.round(popt[2], 3)))

                    ax.set_xlabel(r'$\theta \: (deg.)$')
                    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                    # ax.set_ylim(bottom=0.85, top=1.65)
                    # ax.set_yticks([1, 1.25, 1.5])
                    ax.legend(loc='upper left')
                    plt.tight_layout()
                    plt.savefig(join(path_figs_averages, save_id + '_all_bin-theta_plot-rmse-z{}.png'.format(lbl)))
                    if show_plots:
                        plt.show()
                    plt.close()

            # ---

        # ---

        # --- calculate/plot percent diameter overlap
        export_percent_diameter_overlap = False
        plot_percent_diameter_overlap = False
        calculate_percent_overlap = False

        if plot_percent_diameter_overlap or export_percent_diameter_overlap:

            # --- setup
            path_calib_spct_stats = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/' \
                                    'grid-dz-overlap/extra-analyses/results-06.21.22_spct-no-dz-overlap-1701-2000/' \
                                    'calibration-spct-no-dz-no-dz/calib_spct_stats_grid-dz_calib_nll2_spct_no-dz.xlsx'

            # --- read each percent diameter overlap dataframe (if available)
            param_diameter = 'contour_diameter'
            max_n_neighbors = 4

            # create directories for files
            path_overlap = path_results + '/percent-overlap'
            if not os.path.exists(path_overlap):
                os.makedirs(path_overlap)

            if calculate_percent_overlap:

                if param_diameter == 'contour_diameter':
                    popt_contour = analyze.fit_contour_diameter(path_calib_spct_stats, fit_z_dist=50, show_plot=False)
                else:
                    raise ValueError('Parameter for percent diameter overlap is not understood.')

                # calculate overlap
                dfo = analyze.calculate_particle_to_particle_spacing(
                    test_coords_path=dficts[inspect_gdpyt_by_key],
                    theoretical_diameter_params_path=None,
                    mag_eff=None,
                    z_param='z',
                    zf_at_zero=-4,
                    zf_param=None,
                    max_n_neighbors=max_n_neighbors,
                    true_coords_path=dficts_ground_truth[inspect_gdpyt_by_key],
                    maximum_allowable_diameter=None,
                    popt_contour=popt_contour,
                    param_percent_diameter_overlap=param_diameter
                )

                # save to excel
                if export_percent_diameter_overlap:
                    dfo.to_excel(path_overlap + '/{}_percent_overlap_{}.xlsx'.format(save_id, param_diameter),
                                 index=False)

            else:
                dfo = pd.read_excel(path_overlap + '/{}_percent_overlap_{}.xlsx'.format(save_id, param_diameter))

            # ---

            # --- --- EVALUATE RMSE Z

            # --- setup
            method = 'idpt'
            plot_all_errors = True
            plot_spacing_dependent_rmse = True
            export_min_dx = True
            export_precision = True
            plot_min_dx = True
            save_plots = True
            show_plots = False

            # binning
            bin_z = np.arange(-47.5, 15, 12)  # [-35, -25, -12.5, 12.5, 25, 35]
            bin_pdo = np.linspace(0.125, 5.125, 25)  # [-2.5, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8]
            bin_min_dx = splits
            num_bins = 8
            min_num_per_bin = 50
            round_z = 3
            round_pdo = 2
            round_min_dx = 1
            ylim_rmse = [-0.0625, 10.25]

            # filters
            error_limits = [None]  # filter_barnkob [None, 5, None, 5]  # 5
            depth_of_focuss = [None]  # [None, None, 7.5, 7.5]  # 7.5
            max_overlap = 5.25

            for error_limit, depth_of_focus in zip(error_limits, depth_of_focuss):

                # apply filters
                if error_limit is not None:
                    dfo = dfo[dfo['error'].abs() < error_limit]
                if depth_of_focus is not None:
                    dfo = dfo[(dfo['z_true'] < -depth_of_focus) | (dfo['z_true'] > depth_of_focus)]
                if max_overlap is not None:
                    dfo = dfo[dfo['percent_dx_diameter'] < max_overlap]
                    # dfo['percent_dx_diameter'] = dfo['percent_dx_diameter'].where(dfo['percent_dx_diameter'] < max_overlap, max_overlap)

                # create directories for files
                path_pdo = path_overlap + \
                           '/percent-overlap_{}/max-pdo-{}_error-limit-{}_exclude-dof-{}_min-num-{}'.format(
                               param_diameter,
                               max_overlap,
                               error_limit,
                               depth_of_focus,
                               min_num_per_bin,
                               )
                if not os.path.exists(path_pdo):
                    os.makedirs(path_pdo)

                # --- --- EVALUATE DATA

                # evaluate precision by min_dx and pdo
                plot_precision = True
                if plot_precision:

                    # precision columns
                    if 'xm' in dfo.columns:
                        dfo['rm'] = np.sqrt(dfo.xm ** 2 + dfo.ym ** 2)
                        precision_columns = ['xm', 'ym', 'rm']
                    else:
                        if 'r' not in dfo.columns:
                            dfo['r'] = np.sqrt(dfo.x ** 2 + dfo.y ** 2)
                        precision_columns = ['x', 'y', 'r']

                    # min dx
                    column_to_bin = 'min_dx'
                    plot_collections.bin_plot_spct_stats_2d_static_precision_mindx_id(
                        dfo, column_to_bin, precision_columns, bin_min_dx, round_min_dx, min_num_per_bin,
                        export_results=export_precision, path_results=path_pdo,
                        save_figs=save_plots, path_figs=path_pdo, show_figs=show_plots,
                    )

                    # percent diameter overlap
                    column_to_bin = 'percent_dx_diameter'
                    pdo_threshold = None
                    plot_collections.bin_plot_spct_stats_2d_static_precision_pdo_id(
                        dfo, column_to_bin, precision_columns, bin_pdo, round_pdo, pdo_threshold, min_num_per_bin,
                        export_results=export_precision, path_results=path_pdo,
                        save_figs=save_plots, path_figs=path_pdo, show_figs=show_plots,
                    )

                # scatter plots of errors
                if plot_all_errors:

                    # similarity
                    fig, ax = plt.subplots()
                    sc = ax.scatter(dfo.cm, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
                    cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
                    cbar.minorticks_on()
                    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                    ax.set_xlabel(r'$c_{m}$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_error_by_cm.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # percent diameter
                    fig, ax = plt.subplots()
                    sc = ax.scatter(dfo.percent_dx_diameter, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
                    cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
                    cbar.minorticks_on()
                    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                    ax.set_xlabel(r'$\varphi \: $(\%)')
                    ax.set_xscale('log')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_error_by_percent_dx_diameter.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # min dx
                    fig, ax = plt.subplots()
                    sc = ax.scatter(dfo.min_dx, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
                    cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
                    cbar.minorticks_on()
                    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                    ax.set_xlabel(r'$\delta x_{min} \: $ (pixels)')
                    ax.set_xscale('log')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_error_by_min_dx.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # mean dx
                    fig, ax = plt.subplots()
                    sc = ax.scatter(dfo.mean_dx, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
                    cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
                    cbar.minorticks_on()
                    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                    ax.set_xlabel(r'$\overline {\delta x}_{n=5} \: (pix.)$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_error_by_mean_dx.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # mean dxo
                    fig, ax = plt.subplots()
                    sc = ax.scatter(dfo.mean_dxo, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
                    cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
                    cbar.minorticks_on()
                    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                    ax.set_xlabel(r'$\overline {\delta x_{o}} \: (pix.)$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_error_by_mean_dxo.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # num dxo
                    fig, ax = plt.subplots()
                    sc = ax.scatter(dfo.num_dxo, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
                    cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
                    cbar.minorticks_on()
                    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                    ax.set_xlabel(r'$N_{o} \: (\#)$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_error_by_num_dxo.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # group by - num dxo
                    dfogm = dfo.groupby('num_dxo').mean().reset_index()
                    dfogstd = dfo.groupby('num_dxo').std().reset_index()

                    fig, ax = plt.subplots()
                    ax.errorbar(dfogm.num_dxo, dfogm.error.abs(), yerr=dfogstd.error, fmt='o', ms=2, elinewidth=1,
                                capsize=2)
                    ax.plot(dfogm.num_dxo, dfogm.error.abs())

                    axr = ax.twinx()
                    axr.errorbar(dfogm.num_dxo, dfogm.cm, yerr=dfogstd.cm, fmt='o', ms=1, elinewidth=0.75,
                                 capsize=1.5,
                                 color='gray', alpha=0.25)
                    axr.plot(dfogm.num_dxo, dfogm.cm, color='gray', alpha=0.125)

                    ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                    ax.set_xlabel(r'$N_{o} \: (\#)$')
                    axr.set_ylabel(r'$c_{m}$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_error-cm_grouped-by_num_dxo.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                # spacing dependent uncertainty
                if plot_spacing_dependent_rmse:

                    columns_to_bin = ['mean_dx', 'min_dx', 'mean_dxo', 'num_dxo', 'percent_dx_diameter']
                    column_labels = [r'$\overline {\delta x}_{n=5} \: (pix.)$',
                                     r'$\delta x_{min} \: (pix.)$',
                                     r'$\overline {\delta x_{o}} \: (pix.)$',
                                     r'$N_{o} \: (\#)$',
                                     r'$\tilde{\varphi} \: (\%)$',
                                     ]

                    for col, col_lbl in zip(columns_to_bin, column_labels):

                        if col == 'num_dxo':
                            num_bins = np.arange(max_n_neighbors + 1)
                        elif col == 'percent_dx_diameter':
                            num_bins = bin_pdo

                        dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin=col, bins=num_bins, min_cm=min_cm,
                                                    z_range=None, round_to_decimal=3, df_ground_truth=None)

                        fig, ax = plt.subplots()
                        ax.plot(dfob.index, dfob.rmse_z, '-o')
                        ax.set_xlabel(col_lbl)
                        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                        ax.set_ylim(ylim_rmse)
                        axr = ax.twinx()
                        axr.plot(dfob.index, dfob.num_bind, '-d', ms=3, color='gray', alpha=0.125)
                        axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
                        axr.set_ylim([0, dfob.num_bind.max() * 1.1])
                        plt.tight_layout()
                        if save_plots:
                            plt.savefig(path_pdo + '/{}_binned_rmsez_by_{}.png'.format(method, col))
                        if show_plots:
                            plt.show()
                        plt.close()

                # --- --- EVALUATE RMSE Z

                # binning
                columns_to_bin = ['z_true', 'percent_dx_diameter']

                dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfo,
                                                               columns_to_bin=columns_to_bin,
                                                               bins=[bin_z, bin_pdo],
                                                               round_to_decimals=[round_z, round_pdo],
                                                               min_cm=min_cm,
                                                               equal_bins=[False, False])

                # --- --- PLOT RMSE Z

                if plot_percent_diameter_overlap:

                    # Plot rmse z + number of particles binned as a function of percent diameter overlap for z bins

                    # linear y-scale
                    fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True,
                                                  figsize=(size_x_inches * 1.35, size_y_inches * 1.5))
                    for name, df in dfbicts.items():
                        df_nums = df[df['num_bind'] > min_num_per_bin]
                        ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                        ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

                    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                    ax.grid(alpha=0.25)
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                    ax2.set_xlabel(r'$\tilde{\varphi} \: (\%)$')
                    ax2.set_ylabel(r'$N_{p}$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_rmsez_num-binned_pdo.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # log y-scale
                    fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True,
                                                  figsize=(size_x_inches * 1.35, size_y_inches * 1.5))
                    for name, df in dfbicts.items():
                        df_nums = df[df['num_bind'] > min_num_per_bin]
                        ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                        ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

                    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                    ax.set_yscale('log')
                    ax.grid(alpha=0.25)
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                    ax2.set_xlabel(r'$\tilde{\varphi} \: (\%)$')
                    ax2.set_ylabel(r'$N_{p}$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_rmsez_num-binned_pdo_log-y.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                # --- --- EXPORT RMSE Z TO EXCEL
                if export_percent_diameter_overlap:
                    dfstack = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
                    dfstack.to_excel(path_pdo + '/{}_binned_rmsez_by_z_pdo.xlsx'.format(method), index=False)

                # --- --- PLOT OTHER METRICS

                # --- calculate the local rmse_z uncertainty

                # bin by percent diameter overlap
                if plot_percent_diameter_overlap:
                    dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin='percent_dx_diameter', bins=bin_pdo,
                                                min_cm=min_cm,
                                                z_range=None, round_to_decimal=round_z, df_ground_truth=None)

                    fig, ax = plt.subplots()
                    ax.plot(dfob.index, dfob.rmse_z, '-o')
                    ax.set_xlabel(r'$\varphi \: (\%)$')
                    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                    ax.set_ylim(ylim_rmse)
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_binned_rmsez_by_pdo.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # bin by z
                    dfobz = bin.bin_local_rmse_z(df=dfo, column_to_bin='z_true', bins=25, min_cm=min_cm,
                                                 z_range=None,
                                                 round_to_decimal=round_z, df_ground_truth=None)

                    fig, ax = plt.subplots()
                    ax.plot(dfobz.index, dfobz.rmse_z, '-o')
                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_pdo + '/{}_binned_rmsez_by_z.png'.format(method))
                    if show_plots:
                        plt.show()
                    plt.close()

                # ------------------------------------------------------------------------------------------------------------------
                # --- --- MINIMUM PARTICLE TO PARTICLE SPACING (MIN_DX)

                if export_min_dx or plot_min_dx:

                    # read percent overlap dataframe
                    dfo = pd.read_excel(path_overlap + '/{}_percent_overlap_{}.xlsx'.format(save_id, param_diameter))

                    # create directories for files
                    path_min_dx = path_results + '/min_dx/error-limit-{}_exclude-dof-{}_min-num-{}'.format(
                        error_limit,
                        depth_of_focus,
                        min_num_per_bin)
                    if not os.path.exists(path_min_dx):
                        os.makedirs(path_min_dx)

                    # binning
                    columns_to_bin = ['z_true', 'min_dx']

                    # apply filters
                    if error_limit is not None:
                        dfo = dfo[dfo['error'].abs() < error_limit]
                    if depth_of_focus is not None:
                        dfo = dfo[(dfo['z_true'] < -depth_of_focus) | (dfo['z_true'] > depth_of_focus)]

                    # compute rmse-z

                    dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfo,
                                                                   columns_to_bin=columns_to_bin,
                                                                   bins=[bin_z, bin_min_dx],
                                                                   round_to_decimals=[round_z, round_min_dx],
                                                                   min_cm=min_cm,
                                                                   equal_bins=[False, False])

                    # --- --- PLOT RMSE Z

                    if plot_min_dx:

                        # Plot rmse z + number of particles binned as a function of percent diameter overlap for different z bins

                        # linear y-scale
                        fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True,
                                                      figsize=(size_x_inches * 1.35, size_y_inches * 1.5))

                        for name, df in dfbicts.items():
                            df_nums = df[df['num_bind'] > min_num_per_bin]

                            ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                            ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

                        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                        ax.grid(alpha=0.25)
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                        ax2.set_xlabel(r'$\delta x_{min} \: (pix.)$')
                        ax2.set_ylabel(r'$N_{p}$')
                        plt.tight_layout()
                        plt.savefig(path_min_dx + '/{}_rmsez_num-binned_min_dx.png'.format(method))
                        if show_plots:
                            plt.show()
                        plt.close()

                        # log y-scale
                        fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True,
                                                      figsize=(size_x_inches * 1.35, size_y_inches * 1.5))

                        for name, df in dfbicts.items():
                            df_nums = df[df['num_bind'] > min_num_per_bin]

                            ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                            ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

                        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                        ax.set_yscale('log')
                        ax.grid(alpha=0.25)
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                        ax2.set_xlabel(r'$\delta x_{min} \: (pix.)$')
                        ax2.set_ylabel(r'$N_{p}$')
                        plt.tight_layout()
                        plt.savefig(path_min_dx + '/{}_rmsez_num-binned_min_dx_log-y.png'.format(method))
                        if show_plots:
                            plt.show()
                        plt.close()

                    # --- --- EXPORT RMSE Z TO EXCEL
                    if export_min_dx:
                        dfstack = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
                        dfstack.to_excel(path_min_dx + '/{}_binned_rmsez_by_z_min_dx.xlsx'.format(method),
                                         index=False)

                    # --- --- PLOT OTHER METRICS

                    # --- calculate the local rmse_z uncertainty

                    if plot_min_dx:

                        # bin by min dx
                        dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin='min_dx', bins=bin_min_dx, min_cm=min_cm,
                                                    z_range=None, round_to_decimal=round_z, df_ground_truth=None)

                        fig, ax = plt.subplots()
                        ax.plot(dfob.index, dfob.rmse_z, '-o')
                        ax.set_xlabel(r'$\delta x_{min} \: (pix.)$')
                        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                        plt.tight_layout()
                        plt.savefig(path_min_dx + '/{}_binned_rmsez_by_min_dx.png'.format(method))
                        if show_plots:
                            plt.show()
                        plt.close()

                        # bin by z
                        dfobz = bin.bin_local_rmse_z(df=dfo, column_to_bin='z_true', bins=40, min_cm=min_cm,
                                                     z_range=None,
                                                     round_to_decimal=round_z, df_ground_truth=None)

                        fig, ax = plt.subplots()
                        ax.plot(dfobz.index, dfobz.rmse_z, '-o')
                        ax.set_xlabel(r'$z \: (\mu m)$')
                        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                        plt.tight_layout()
                        plt.savefig(path_min_dx + '/{}_binned_rmsez_by_z.png'.format(method))
                        if show_plots:
                            plt.show()
                        plt.close()

        # ---

        # plot per-dx binned 2D: dz and z_true,
        plot_per_dx = False
        if plot_per_dx:

            if len(dficts[inspect_gdpyt_by_key]['dz'].unique()) > 1:

                for name, dfix in dsplicts.items():
                    # 2D binning
                    columns_to_bin = ['dz', 'z_true']
                    bin_dz = 5
                    bin_z_true = 20

                    dfxbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dfix,
                                                                       columns_to_bin=columns_to_bin,
                                                                       bins=[bin_dz, bin_z_true],
                                                                       round_to_decimals=[2, 2],
                                                                       min_cm=min_cm,
                                                                       equal_bins=[False, False],
                                                                       error_column='error',
                                                                       include_xy=True,
                                                                       )

                    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))
                    for name_dz, dfix_rmse in dfxbicts_2d.items():
                        ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=name_dz)

                    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                              title=r'$\Delta z(\delta x=$' + '{}'.format(name) + r'$)$')
                    plt.tight_layout()
                    plt.savefig(join(path_figs_per_dx, save_id + '_dx{}_bin-dz-z_plot-rmse-z.png'.format(name)))
                    if show_plots:
                        plt.show()
                    plt.close()

            # ---

        # ---

        # bin 2D
        plot_2d_binning = True

        if plot_2d_binning:

            # filter
            min_num_per_bin = 50

            # 2D binning: rmse_z (dx, z_true)
            columns_to_bin = ['x', 'z_true']
            bin_x = splits
            bin_z_true = 30
            clrs = iter(cm.magma(np.linspace(0.05, 0.95, len(splits))))

            dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                                              columns_to_bin=columns_to_bin,
                                                              bins=[bin_x, bin_z_true],
                                                              round_to_decimals=[2, 2],
                                                              min_cm=min_cm,
                                                              equal_bins=[False, False],
                                                              error_column='error',
                                                              include_xy=True,
                                                              )

            # export results
            dfbicts_2d_stacked = modify.stack_dficts_by_key(dfbicts_2d, drop_filename=False)
            dfbicts_2d_stacked.to_excel(join(path_results, save_id + '_rmse-z_by_dx_z-true_2d-bin.xlsx'))

            # plot
            fig, ax = plt.subplots(figsize=(size_x_inches * 1.3, size_y_inches * 1.25))
            for name_dz, dfix_rmse in dfbicts_2d.items():
                dfix_rmse = dfix_rmse[
                    dfix_rmse['num_meas'] > min_num_per_bin]  # NOTE: this changes the values in the df.
                ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, color=next(clrs),
                        label=dict_splits_to_keys[name_dz])

            ax.set_xlabel(r'$z_{true} \: (\mu m)$')
            ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
            ax.legend(loc='upper left', ncol=5, title=r'$\delta x \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_figs_bin_2d, save_id + '_all_2d-bin-dx-z_plot-rmse-z.png'))
            if show_plots:
                plt.show()
            plt.close()

            # plot dx values in groups
            dx_groups = [[38]]  # , [5, 6, 7], [8, 9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20, 21, 22]]
            for dxg in dx_groups:
                fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
                axr = ax.twinx()

                for name_dz, dfix_rmse in dfbicts_2d.items():
                    if dict_splits_to_keys[name_dz] in dxg:
                        dfix_rmse = dfix_rmse[
                            dfix_rmse['num_meas'] > min_num_per_bin]  # NOTE: this changes values in df.
                        ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=dict_splits_to_keys[name_dz])
                        axr.plot(dfix_rmse.bin, dfix_rmse.cm, '--', linewidth=0.5, alpha=0.5)

                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                axr.set_ylabel(r'$c_{m}$')
                ax.legend(title=r'$\delta x \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(join(path_figs_bin_2d, save_id + '_2d-bin-dx-z_plot-rmse-z_dx={}.png'.format(dxg)))
                if show_plots:
                    plt.show()
                plt.close()

            # ---

            # 2D binning: rmse-z (dz, z_true)
            if len(dficts[inspect_gdpyt_by_key]['dz'].unique()) > 1:

                columns_to_bin = ['dz', 'z_true']
                bin_dz = 13  # [-15, -10, -5, 0, 5, 10, 15]
                bin_z_true = 20

                dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                                                  columns_to_bin=columns_to_bin,
                                                                  bins=[bin_dz, bin_z_true],
                                                                  round_to_decimals=[2, 2],
                                                                  min_cm=min_cm,
                                                                  equal_bins=[False, False],
                                                                  error_column='error',
                                                                  include_xy=True,
                                                                  )

                # export results
                dfbicts_2d_stacked = modify.stack_dficts_by_key(dfbicts_2d, drop_filename=False)
                dfbicts_2d_stacked.to_excel(join(path_results, save_id + '_rmse-z_by_dz_z-true_2d-bin.xlsx'))

                # plot
                fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))
                for name_dz, dfix_rmse in dfbicts_2d.items():
                    ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, label=name_dz)

                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                ax.legend(loc='upper left', title=r'$\Delta z \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(join(path_figs_bin_2d, save_id + '_all_bin-dz-z_plot-rmse-z.png'))
                if show_plots:
                    plt.show()
                plt.close()

                # ---

                # 2D binning: rmse_z (dx, dz)
                columns_to_bin = ['x', 'dz']
                bin_x = splits
                bin_dz = 7
                clrs = iter(cm.plasma(np.linspace(0.05, 0.95, len(splits))))

                dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                                                  columns_to_bin=columns_to_bin,
                                                                  bins=[bin_x, bin_dz],
                                                                  round_to_decimals=[2, 2],
                                                                  min_cm=min_cm,
                                                                  equal_bins=[False, False],
                                                                  error_column='error',
                                                                  include_xy=True,
                                                                  )

                # export results
                dfbicts_2d_stacked = modify.stack_dficts_by_key(dfbicts_2d, drop_filename=False)
                dfbicts_2d_stacked.to_excel(join(path_results, save_id + '_rmse-z_by_dx_dz_2d-bin.xlsx'))

                # plot
                fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches * 1.5))
                for name_dz, dfix_rmse in dfbicts_2d.items():
                    ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, color=next(clrs),
                            label=dict_splits_to_keys[name_dz])

                ax.set_xlabel(r'$\Delta z \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                ax.legend(loc='upper left', ncol=5, title=r'$\delta x \: (pix.)$')
                plt.tight_layout()
                plt.savefig(join(path_figs_bin_2d, save_id + '_all_bin-dx-dz_plot-rmse-z.png'))
                if show_plots:
                    plt.show()
                plt.close()

                # ---

                # 2D binning: rmse-z (dx, theta)
                columns_to_bin = ['x', 'theta']
                bin_x = splits
                bin_theta = 13  # [-25, -20, -15, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 20, 25]

                dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dficts[inspect_gdpyt_by_key],
                                                                  columns_to_bin=columns_to_bin,
                                                                  bins=[bin_x, bin_theta],
                                                                  round_to_decimals=[2, 2],
                                                                  min_cm=min_cm,
                                                                  equal_bins=[False, False],
                                                                  error_column='error',
                                                                  include_xy=True,
                                                                  )

                # export results
                dfbicts_2d_stacked = modify.stack_dficts_by_key(dfbicts_2d, drop_filename=False)
                dfbicts_2d_stacked.to_excel(join(path_results, save_id + '_rmse-z_by_dx_theta_2d-bin.xlsx'))

                # plots: (1) calculated results, (2) fitted parabola
                for lbl in ['', '_fit']:
                    clrs = iter(cm.coolwarm(np.linspace(0.05, 0.95, len(splits))))

                    fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches))
                    for name_dz, dfix_rmse in dfbicts_2d.items():

                        if lbl == '_fit':
                            if len(dfix_rmse.rmse_z) < 5:
                                continue
                            popt, pcov = curve_fit(functions.quadratic, dfix_rmse.bin, dfix_rmse.rmse_z)
                            fit_x = np.linspace(dfix_rmse.bin.min(), dfix_rmse.bin.max())
                            ax.plot(fit_x, functions.quadratic(fit_x, *popt), '-', color=next(clrs),
                                    label=dict_splits_to_keys[name_dz])
                        else:
                            ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=2, color=next(clrs),
                                    label=dict_splits_to_keys[name_dz])

                    ax.set_xlabel(r'$\theta \: (deg.)$')
                    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2, title=r'$\delta x \: (pix.)$')
                    plt.tight_layout()
                    plt.savefig(join(path_figs_bin_2d, save_id + '_all_bin-dx-theta_plot-rmse-z{}.png'.format(lbl)))
                    if show_plots:
                        plt.show()
                    plt.close()

        # ---
        raise ValueError()
        # ------------------------------------------------------------------------------------------------------------------
        # SPECIALIZED PLOTS OR... PLOTS OF 'NOT-FULLY-UNDERSTOOD VALUE'

        # calculate mean rmse_z as a function of (dx, z_true) where dz = 0
        plot_zero_dz = False
        if plot_zero_dz and os.path.exists(path_figs_zero_dz):

            # bin-dx: where dz = 0
            dfdzero_bx = {}
            for name, dfix in dsplicts.items():
                dfix_dzero = dfix[dfix['dz'] == 0]

                dfb = bin.bin_local_rmse_z(dfix_dzero,
                                           column_to_bin='dx',
                                           bins=1,
                                           min_cm=min_cm,
                                           z_range=z_range,
                                           round_to_decimal=0,
                                           df_ground_truth=None,
                                           )
                dfdzero_bx.update({name: dfb})
            dfbx = modify.stack_dficts_by_key(dfdzero_bx, drop_filename=False)
            dfbx = dfbx.set_index('filename')

            # export rmse_z(bin-dx)
            dfbx.to_excel(path_results + '/mean_rmse-z_bin-dx_dz=zero.xlsx', index_label='dx')

            # bin-dx: plot rmse_z for entire collection
            fig, ax = plt.subplots()
            ax.plot(dfbx.index, dfbx.rmse_z, '-o')
            ax.set_xlabel(r'$\delta x \: (pixels)$')
            ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(join(path_figs_zero_dz, save_id + '_all_bin-dx_plot-rmse-z_dz=zero.png'))
            if show_plots:
                plt.show()
            plt.close()

        # ---

        # bin-dx: where dz < 0.5
        plot_onehalf_dz = False
        if plot_onehalf_dz:

            if len(dficts[inspect_gdpyt_by_key]['dz'].unique()) > 1:

                dfdz_sub_onehalf_bx = {}
                for name, dfix in dsplicts.items():
                    dfix_dzero = dfix[dfix['dz'].abs() < 0.5]

                    dfb = bin.bin_local_rmse_z(dfix_dzero,
                                               column_to_bin='dx',
                                               bins=1,
                                               min_cm=min_cm,
                                               z_range=z_range,
                                               round_to_decimal=0,
                                               df_ground_truth=None,
                                               )
                    dfdz_sub_onehalf_bx.update({name: dfb})
                dfbx = modify.stack_dficts_by_key(dfdz_sub_onehalf_bx, drop_filename=False)
                dfbx = dfbx.set_index('filename')

                # export rmse_z(bin-dx)
                dfbx.to_excel(path_figs_sub_onehalf_dz + '/mean_rmse-z_bin-dx_dz=sub-one-half.xlsx', index_label='dx')

                # bin-dx: plot rmse_z for dz < 0.5
                fig, ax = plt.subplots()
                ax.plot(dfbx.index, dfbx.rmse_z, '-o')
                ax.set_xlabel(r'$\delta x \: (pixels)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(join(path_figs_sub_onehalf_dz, save_id + '_all_bin-dx_plot-rmse-z_dz=sub-one-half.png'))
                if show_plots:
                    plt.show()
                plt.close()

            # ---

        # ---

        if len(dficts[inspect_gdpyt_by_key]['dz'].unique()) > 1:

            # ---

            # calculate z-uncertainty on a 2D grid (dx by dz) and plot 3D surface
            plot_rmse_z_by_dx_dz = True
            if plot_rmse_z_by_dx_dz:

                # 2D binning
                dfxz = dficts[inspect_gdpyt_by_key].copy()
                dfxz = dfxz[(dfxz['dz'] > -11) & (dfxz['dz'] < 11)]
                column_to_bin = 'dz'
                bin_dz = 13  # [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

                dfxzbicts = analyze.calculate_bin_local_rmse_z(dsplicts,
                                                               column_to_bin,
                                                               bin_dz,
                                                               min_cm=min_cm,
                                                               z_range=None,
                                                               round_to_decimal=2,
                                                               dficts_ground_truth=None)

                dfbxz = modify.stack_dficts_by_key(dfxzbicts, drop_filename=False)
                print("Initial length {}".format(len(dfbxz)))
                dfbxz = dfbxz[dfbxz['num_meas'] > 200]
                print("Final length {}".format(len(dfbxz)))

                # get data arrays
                x = dfbxz.index.to_numpy()
                y = dfbxz.filename.to_numpy()
                z = dfbxz.rmse_z.to_numpy()

                for num in [150]:
                    # get the range of points for the 2D surface space
                    xr = (x.min(), x.max())
                    yr = (y.min(), y.max())
                    X = np.linspace(min(x), max(x), num)
                    Y = np.linspace(min(y), max(y), num)
                    X, Y = np.meshgrid(X, Y)
                    interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
                    Z = interp(X, Y)

                    plt.pcolormesh(X, Y, Z, shading='auto', cmap=cm.coolwarm)
                    plt.title(num)
                    plt.colorbar()
                    plt.axis("equal")
                    if save_plots:
                        plt.savefig(
                            join(path_figs_surface, save_id + '_bin-dx-dz_plot-rmse-z-2Dsurf_num{}.png'.format(num)))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # plot
                    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                    ax.set_xlabel(r'$\Delta z \: (\mu m)$')
                    ax.set_ylabel(r'$\delta x \: (pixels)$')
                    ax.set_zlabel(r'$\sigma_z \: (\mu m)$')
                    ax.view_init(30, 160)
                    if save_plots:
                        plt.savefig(
                            join(path_figs_surface, save_id + '_bin-dx-dz_plot-rmse-z-3Dsurf_num{}.png'.format(num)))
                    if show_plots:
                        plt.show()
                    plt.close()

            # ---

            # bias error + variance
            plot_bias_error = True
            if plot_bias_error:

                for name, df in dsplicts.items():
                    dfb = bin.bin_local(df,
                                        column_to_bin='dz',
                                        bins=11,
                                        min_cm=min_cm,
                                        z_range=z_range,
                                        round_to_decimal=0,
                                        true_num_particles=None,
                                        dropna=True)

                    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True,
                                                        figsize=(size_x_inches, size_y_inches * 1.5))

                    ax1.plot(dfb.index, dfb.error, '-o', alpha=0.5, label=r'$-\epsilon \equiv z_{IDPT} < z_{true}$')
                    ax1.axhline(y=0, color='black', linewidth=0.5)
                    ax1.set_ylabel(r'$\epsilon_{z}$')
                    ax1.set_ylim([-10, 10])  # [-7.5, 5]
                    ax1.set_yticks([-7.5, -5, -2.5, 0, 2.5, 5])
                    ax1.set_title(r'$\delta x=$' + '{}'.format(name))
                    ax1.legend(loc='lower left')

                    ax2.plot(dfb.index, dfb.z_std ** 2, '-o', alpha=0.5)
                    ax2.set_ylabel(r'$var. \: = \: \sigma^2$')
                    ax2.set_ylim([0, 750])  # [120, 660]
                    ax2.set_yticks([150, 300, 450, 600])

                    ax3.errorbar(dfb.index, dfb.z, yerr=dfb.z_std, fmt='o', capsize=2, alpha=0.5,
                                 label=r'$\overline{z}_{IDPT} + \sigma$')
                    ax3.plot(dfb.index, dfb.z_true, '-o', ms=1, linewidth=0.5, color='black', label=r'$z_{true}$')
                    ax3.set_xlabel(r'$\delta z$')
                    ax3.set_ylabel(r'$z$')
                    ax3.set_ylim([-44, 24])
                    ax3.set_yticks([-40, -20, 0, 20])
                    ax3.legend(loc='upper left', ncol=2)

                    plt.tight_layout()
                    plt.savefig(path_figs_bias_errors + '/error_bias_and_variance_dx{}.png'.format(name))
                    # plt.show()
                    plt.close()

            # ---

            # calculate mean error of the particle pair and plot
            plot_mean_error_of_pair = True
            if plot_mean_error_of_pair:

                dz_limit = 5

                # create sub-directory
                path_figs_pair_error_sub = join(path_figs_pair_error, 'per-dx')
                if not os.path.exists(path_figs_pair_error_sub):
                    os.makedirs(path_figs_pair_error_sub)

                # create pair error split dictionary
                dsplicts_pair = {}
                for name, df in dsplicts.items():
                    dfpair = df.groupby(['frame', 'y']).mean().reset_index()
                    dfpair = dfpair[(dfpair['dz'] < dz_limit) & (dfpair['dz'] > -dz_limit)]
                    dsplicts_pair.update({name: dfpair})

                # calculate mean rmse-z for each dx
                column_to_bin_and_assess = 'z_true'
                bins = 1
                dfbicts = analyze.calculate_bin_local_rmse_z(dsplicts_pair,
                                                             column_to_bin_and_assess,
                                                             bins,
                                                             min_cm=min_cm,
                                                             z_range=None,
                                                             round_to_decimal=1,
                                                             dficts_ground_truth=None)

                dfb_stacked = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
                dfb_stacked.to_excel(join(path_figs_pair_error, save_id +
                                          '_dx-pair-error_bin-dz-z_plot-rmse-z_dz-limit{}.xlsx'.format(dz_limit)))

                # plot mean rmse-z pair error for all dx on the same figure
                fig, ax = plt.subplots()
                ax.plot(dfb_stacked.filename, dfb_stacked.rmse_z, '-o')
                ax.set_xlabel(r'$\delta x \: (pixels)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(
                    join(path_figs_pair_error,
                         save_id + '_dx-pair-error_dx_plot-rmse-z_dz-limit{}.png'.format(dz_limit)))
                if show_plots:
                    plt.show()
                plt.close()

                # standard z-uncertainty plots (for mean pair error) (one figure per dx)
                plot_each_pair_error = True
                if plot_each_pair_error:
                    for name, dfix in dsplicts_pair.items():
                        # 2D binning
                        columns_to_bin = ['dz', 'z_true']
                        bin_dz = [0.0]
                        bin_z_true = 20

                        dfxbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dfix,
                                                                           columns_to_bin=columns_to_bin,
                                                                           bins=[bin_dz, bin_z_true],
                                                                           round_to_decimals=[2, 2],
                                                                           min_cm=min_cm,
                                                                           equal_bins=[False, False],
                                                                           error_column='error',
                                                                           )

                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))
                        for name_dz, dfix_rmse in dfxbicts_2d.items():
                            ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', label=name_dz)

                        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                        ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                                  title=r'$\Delta z(\delta x=$' + '{}'.format(name) + r'$)$')
                        plt.tight_layout()
                        plt.savefig(join(path_figs_pair_error_sub,
                                         save_id + '_dx{}-pair-error_bin-dz-z_plot-rmse-z.png'.format(name)))
                        if show_plots:
                            plt.show()
                        plt.close()

                    # standard z-uncertainty plots (for mean pair error) (all plots on the same figure)
                    fig, ax = plt.subplots(figsize=(size_x_inches * 1.15, size_y_inches * 1.15))
                    clrs = iter(plt.cm.viridis(np.linspace(0.01, 0.99, len(dsplicts_pair))))

                    for name, dfix in dsplicts_pair.items():
                        # 2D binning
                        columns_to_bin = ['dz', 'z_true']
                        bin_dz = [0.0]
                        bin_z_true = 20

                        dfxbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(dfix,
                                                                           columns_to_bin=columns_to_bin,
                                                                           bins=[bin_dz, bin_z_true],
                                                                           round_to_decimals=[2, 2],
                                                                           min_cm=min_cm,
                                                                           equal_bins=[False, False],
                                                                           error_column='error',
                                                                           )

                        for name_dz, dfix_rmse in dfxbicts_2d.items():
                            ax.plot(dfix_rmse.bin, dfix_rmse.rmse_z, '-o', ms=1, color=next(clrs), label=name)

                    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                    ax.legend(ncol=2, loc='upper left')  # , title=r'$\delta x$'
                    plt.tight_layout()
                    plt.savefig(join(path_figs_pair_error, save_id + '_dx-pair-error-all_bin-dz-z_plot-rmse-z.png'))
                    if show_plots:
                        plt.show()
                    plt.close()

            # ---

        # ---


    inspect_keys = [2]
    for inspect_key in inspect_keys:
        analyze_grid_dz_overlap(inspect_key, path_results)
        raise ValueError()
    # ---

# ----------------------------------------------------------------------------------------------------------------------
# B. ANALYZE TEST OUTPUTS
analyze_test_outputs = False

if analyze_test_outputs:

    # compare no-dz overlap
    compare_no_dz = False
    if compare_no_dz:

        # setup
        path_figs = base_path + '/compare/no-dz'
        if not os.path.exists(path_figs):
            os.makedirs(path_figs)

        # modifiers
        save_plots = True
        show_plots = False

        # ---

        # plot mean rmse-z by z_true (single, isolated particle; dx = 38)
        plot_rmse_by_ztrue_per_dx_for_single_particle = False
        if plot_rmse_by_ztrue_per_dx_for_single_particle:
            
            # , 'barnkob-filter'

            # read
            dfi = pd.read_excel(join(path_results, 'idpt-no-dz-overlap', 'results',
                                     'idpt-no-dz-overlap_rmse-z_by_dx_z-true_2d-bin.xlsx'), index_col=0)
            dfs = pd.read_excel(join(path_results, 'spct-no-dz-overlap', 'results',
                                     'spct-no-dz-overlap_rmse-z_by_dx_z-true_2d-bin.xlsx'), index_col=0)

            # setup
            px = 'bin'
            plot_columns = ['', 'true_percent_meas', 'percent_meas', 'num_meas', 'cm']
            plot_column_labels = ['', r'$\phi^{\delta} \: (\%)$', r'$\phi_{ID}^{\delta} \: (\%)$',
                                  r'$N_{p}^{\delta} \: (\#)$', r'$c_{m}^{\delta}$']
            ms = 4

            # data selection
            bin_tl_list = [39]
            for bin_tl in bin_tl_list:
                dfii = dfi[dfi['bin_tl'] == bin_tl]
                dfsi = dfs[dfs['bin_tl'] == bin_tl]

                # processing
                if bin_tl == 39:
                    dfii['true_percent_meas'] = dfii['num_meas'] / dfii['num_bind']
                    dfsi['true_percent_meas'] = dfsi['num_meas'] / dfii['num_bind']

                # ---

                # plot rmse_x-y-z
                ms = 3
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))
                ax1.plot(dfii[px], dfii.rmse_y, '-o', ms=ms, color=sciblue, label='IDPT')
                ax1.plot(dfsi[px], dfsi.rmse_y, '-o', ms=ms, color=scigreen, label='SPCT')
                ax1.set_ylabel(r'$\sigma_{y}^{\delta} \: (\mu m)$')
                ax1.legend()

                ax2.plot(dfii[px], dfii.rmse_x, '-o', ms=ms, color=sciblue)
                ax2.plot(dfsi[px], dfsi.rmse_x, '-o', ms=ms, color=scigreen)
                ax2.set_ylabel(r'$\sigma_{x}^{\delta} \: (\mu m)$')

                ax3.plot(dfii[px], dfii.rmse_z, '-o', ms=ms, color=sciblue)
                ax3.plot(dfsi[px], dfsi.rmse_z, '-o', ms=ms, color=scigreen)
                ax3.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                ax3.set_xlabel(r'$z \: (\mu m)$')

                # ax.set_ylim(top=3.75)
                plt.tight_layout()
                fig.subplots_adjust(hspace=0.075)  # adjust space between axes
                if save_plots:
                    plt.savefig(path_figs + '/compare_rmse-x-y-z_by_z-true_for_dx={}{}'.format(bin_tl, save_fig_filetype))
                if show_plots:
                    plt.show()
                plt.close()

                # plot other columns
                for pc, pl in zip(plot_columns, plot_column_labels):
                    if 'true_percent_meas' not in dfii.columns:
                        continue

                    fig, (axr, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

                    ax.plot(dfii[px], dfii.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
                    ax.plot(dfsi[px], dfsi.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')
                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                    ax.set_ylim(top=3.75)
                    ax.legend(loc='upper left')

                    if len(pc) > 0:
                        axr.plot(dfii[px], dfii[pc], '--', color=sciblue)
                        axr.plot(dfsi[px], dfsi[pc], '--', color=scigreen)
                        axr.set_ylabel(pl)
                        axr.set_ylim(bottom=0.2, top=1.075)
                        axr.set_yticks([0.25, 0.5, 0.75, 1.0])

                    plt.tight_layout()
                    fig.subplots_adjust(hspace=0.075)  # adjust space between axes
                    if save_plots:
                        plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_z-true_for_dx={}{}'.format(pc, bin_tl, save_fig_filetype))
                    if show_plots:
                        plt.show()
                    plt.close()

            # ---

        # ---

        # plot mean rmse-z by dx
        plot_rmse_by_dx = False
        if plot_rmse_by_dx:

            # , 'barnkob-filter'

            # read
            dfi = pd.read_excel(join(path_results, 'idpt-no-dz-overlap', 'results', 'mean_rmse-z_bin-dx.xlsx'))
            dfs = pd.read_excel(join(path_results, 'spct-no-dz-overlap', 'results', 'mean_rmse-z_bin-dx.xlsx'))

            # processing
            dfi['true_percent_meas'] = dfi['num_meas'] / dfi['num_bind']
            dfs['true_percent_meas'] = dfs['num_meas'] / dfi['num_bind']

            # setup
            plot_columns = ['num_meas', 'true_percent_meas', 'cm']
            plot_column_labels = [r'$N_{p} \: (\#)$', r'$\phi \: (\%)$', r'$c_{m}$']  #
            ms = 3

            # ---

            # plot rmse_x-y-z
            ms = 3
            px = 'dx'

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(size_x_inches, size_y_inches * 1.5))
            ax1.plot(dfi[px], dfi.rmse_y, '-o', ms=ms, color=sciblue, label='IDPT')
            ax1.plot(dfs[px], dfs.rmse_y, '-o', ms=ms, color=scigreen, label='SPCT')
            ax1.set_ylabel(r'$\sigma_{y} \: (\mu m)$')
            ax1.set_ylim([-0.05, 0.55])
            ax1.set_yticks([0, 0.25, 0.5])
            ax1.legend()

            ax2.plot(dfi[px], dfi.rmse_x, '-o', ms=ms, color=sciblue)
            ax2.plot(dfs[px], dfs.rmse_x, '-o', ms=ms, color=scigreen)
            ax2.set_ylabel(r'$\sigma_{x} \: (\mu m)$')

            ax3.plot(dfi[px], dfi.rmse_z, '-o', ms=ms, color=sciblue)
            ax3.plot(dfs[px], dfs.rmse_z, '-o', ms=ms, color=scigreen)
            ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')

            ax4.plot(dfi[px], dfi.true_percent_meas, '-o', ms=ms, color=sciblue)
            ax4.plot(dfs[px], dfs.true_percent_meas, '-o', ms=ms, color=scigreen)
            ax4.set_ylabel(r'$\phi \: (\%)$')
            ax4.set_xlabel(r'$\delta x \: (pix.)$')

            # ax.set_ylim(top=3.75)
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.075)  # adjust space between axes
            if save_plots:
                plt.savefig(path_figs + '/compare_rmse-x-y-z_by_dx{}'.format(save_fig_filetype))
            if show_plots:
                plt.show()
            plt.close()

            #

            # plot other columns
            """for pc, pl in zip(plot_columns, plot_column_labels):
                fig, (axr, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

                ax.plot(dfi.dx, dfi.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
                ax.plot(dfs.dx, dfs.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')
                ax.set_xlabel(r'$\delta x \: (pix.)$')
                ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                # ax.set_ylim(bottom=0.245, top=2.45)

                # axr = ax.twinx()
                axr.plot(dfi.dx, dfi[pc], '-s', ms=ms, color=sciblue)
                axr.plot(dfs.dx, dfs[pc], '-s', ms=ms, color=scigreen)
                axr.set_ylabel(pl)
                axr.set_ylim(bottom=0)

                plt.tight_layout()
                fig.subplots_adjust(hspace=0.075)  # adjust space between axes
                if save_plots:
                    plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_dx.png'.format(pc))
                if show_plots:
                    plt.show()
                plt.close()"""

        # ---

        # plot mean rmse-z by dx (broken x-axis)
        plot_by_dx_broken_axis = False
        if plot_by_dx_broken_axis:

            # read
            dfi = pd.read_excel(join(path_results, 'idpt-no-dz-overlap', 'results', 'mean_rmse-z_bin-dx.xlsx'))
            dfs = pd.read_excel(join(path_results, 'spct-no-dz-overlap', 'results', 'mean_rmse-z_bin-dx.xlsx'))

            # processing
            dfi['true_percent_meas'] = dfi['num_meas'] / dfi['num_bind'] * 100
            dfs['true_percent_meas'] = dfs['num_meas'] / dfi['num_bind'] * 100

            # setup
            plot_columns = ['true_percent_meas', 'percent_meas', 'cm']
            plot_column_labels = [r'$\phi^{\delta} \: (\%)$', r'$\phi_{ID}^{\delta} \: (\%)$', r'$c_{m}^{\delta}$']
            ms = 4

            # y-axes limits
            ylim = [0.4, 5.6]
            yylim_s = [[-5, 105], [0.45, 1.05]]
            yticks_s = [[0, 25, 50, 75, 100], [0.5, 0.75, 1.0]]

            # plot
            for pc, pl, yylim, yticks in zip(plot_columns, plot_column_labels, yylim_s, yticks_s):
                fig, (ax, axr) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]})
                fig.subplots_adjust(wspace=0.075, right=0.875, bottom=0.15)  # adjust space between axes

                # plot the same data on both axes
                # rmse-z
                ax.plot(dfi.dx, dfi.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
                ax.plot(dfs.dx, dfs.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')
                axr.plot(dfi.dx, dfi.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
                axr.plot(dfs.dx, dfs.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')

                # scale 2nd plot to fit on 1st plot y-axis
                yyi = (dfi[pc] - yylim[0]) / (yylim[1] - yylim[0]) * (ylim[1] - ylim[0]) + ylim[0]
                yys = (dfs[pc] - yylim[0]) / (yylim[1] - yylim[0]) * (ylim[1] - ylim[0]) + ylim[0]

                ax.plot(dfi.dx, yyi, '--', ms=ms/1.25, color=lighten_color(sciblue, 1.1))
                ax.plot(dfs.dx, yys, '--', ms=ms/1.25, color=lighten_color(scigreen, 1.1))
                axr.plot(dfi.dx, yyi, '--', ms=ms/1.25, color=lighten_color(sciblue, 1.1))
                axr.plot(dfs.dx, yys, '--', ms=ms/1.25, color=lighten_color(scigreen, 1.1))

                # zoom-in / limit the view to different portions of the data

                # regularly spaced data
                ax.set_xlim(3.5, 23.5)
                ax.set_xticks([5, 10, 15, 20])
                ax.set_ylim(ylim)
                ax.set_yticks([1, 2, 3, 4, 5])
                ax.set_xlabel(r'$\delta x \: (pix.)$', x=0.6125, horizontalalignment='center')
                ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')

                # outliers
                axr.set_xlim(36.5, 39.5)
                axr.set_xticks([38])

                # hide the spines between ax and ax2
                ax.spines.right.set_visible(False)
                ax.yaxis.tick_left()
                axr.spines.left.set_visible(False)

                axrr = axr.twinx()
                axr.tick_params(axis='y', which='both',
                                left=False, right=False,
                                labelleft=False, labelright=False,
                                )  # don't put tick labels at the left
                axrr.set_ylim(yylim)
                axrr.set_yticks(yticks)
                axrr.set_ylabel(pl, labelpad=2)
                axrr.spines.left.set_visible(False)
                axrr.tick_params(axis='y', which='both', left=False, labelleft=False)

                # axes breaks
                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=8,
                              linestyle="none", color='k', mec='k', mew=0.5, clip_on=False)
                ax.plot([1, 1], [1, 0], transform=ax.transAxes, **kwargs)
                axr.plot([0, 0], [0, 1], transform=axr.transAxes, **kwargs)

                if save_plots:
                    plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_dx_broken-axis{}'.format(pc, save_fig_filetype))
                if show_plots:
                    plt.show()
                plt.close()

        # ---

        # plot mean rmse-z by z_true
        plot_rmse_by_ztrue = False
        if plot_rmse_by_ztrue:
            # read
            dfi = pd.read_excel(join(path_results, 'idpt-no-dz-overlap', 'results', 'mean_rmse-z_bin_z-true.xlsx'))
            dfs = pd.read_excel(join(path_results, 'spct-no-dz-overlap', 'results', 'mean_rmse-z_bin_z-true.xlsx'))

            # setup
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(size_x_inches, size_y_inches * 1.5))
            ax1.plot(dfi.bin, dfi.rmse_y, '-o', ms=ms, color=sciblue, label='IDPT')
            ax1.plot(dfs.bin, dfs.rmse_y, '-o', ms=ms, color=scigreen, label='SPCT')
            ax1.set_ylabel(r'$\sigma_{y} \: (\mu m)$')
            ax1.legend()

            ax2.plot(dfi.bin, dfi.rmse_x, '-o', ms=ms, color=sciblue)
            ax2.plot(dfs.bin, dfs.rmse_x, '-o', ms=ms, color=scigreen)
            ax2.set_ylabel(r'$\sigma_{x} \: (\mu m)$')

            ax3.plot(dfi.bin, dfi.rmse_z, '-o', ms=ms, color=sciblue)
            ax3.plot(dfs.bin, dfs.rmse_z, '-o', ms=ms, color=scigreen)
            ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')

            ax4.plot(dfi.bin, dfi.true_percent_meas, '-o', ms=ms, color=sciblue)
            ax4.plot(dfs.bin, dfs.true_percent_meas, '-o', ms=ms, color=scigreen)
            ax4.set_ylabel(r'$\phi \: (\%)$')
            ax4.set_xlabel(r'$z \: (\mu m)$')

            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs + '/compare_rmse-x-y-z_percent-true_by_z-true{}'.format(save_fig_filetype))
            if show_plots:
                plt.show()
            plt.close()

            """plot_columns = ['true_percent_meas', 'cm']
            plot_column_labels = [r'$\phi \: (\%)$', r'$c_{m}$']
            ms = 4

            # plot
            for pc, pl in zip(plot_columns, plot_column_labels):
                fig, ax = plt.subplots()
                ax.plot(dfi.index, dfi.rmse_z, 'o', ms=ms, color=sciblue, label='IDPT')
                ax.plot(dfs.index, dfs.rmse_z, 'o', ms=ms, color=scigreen, label='SPCT')
                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                ax.set_ylim(bottom=-0.2, top=5.2)

                axr = ax.twinx()
                axr.plot(dfi.index, dfi[pc], 'd', ms=ms/1.25, color=lighten_color(sciblue, 1.1))
                axr.plot(dfs.index, dfs[pc], 'd', ms=ms/1.25, color=lighten_color(scigreen, 1.1))
                axr.set_ylabel(pl)
                axr.set_ylim(bottom=0)

                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_z-true{}'.format(pc, save_fig_filetype))
                if show_plots:
                    plt.show()
                plt.close()"""

        # ---

    # ---

    # compare dz overlap
    compare_dz = False
    if compare_dz:
        # setup
        path_figs = base_path + '/compare/dz'
        dir_idpt = 'idpt-dz-overlap'
        dir_spct = 'spct-dz-overlap'
        save_plots = True
        show_plots = True

        # plot mean rmse-z by z_true (single, isolated particle; dx = 38)
        plot_rmse_by_ztrue_per_dx_for_single_particle = False
        if plot_rmse_by_ztrue_per_dx_for_single_particle:
            # read

            dfi = pd.read_excel(join(path_results, 'idpt-dz-overlap', 'results',
                                     'idpt-dz-overlap_rmse-z_by_dx_z-true_2d-bin.xlsx'), index_col=0)
            dfs = pd.read_excel(join(path_results, 'spct-dz-overlap', 'results',
                                     'spct-dz-overlap_rmse-z_by_dx_z-true_2d-bin.xlsx'), index_col=0)

            # setup
            px = 'bin'
            plot_columns = ['num_meas', 'cm']
            plot_column_labels = [r'$N_{p} \: (\#)$', r'$c_{m}$']
            ms = 4

            # data selection
            bin_tl_list = [39]
            for bin_tl in bin_tl_list:
                dfii = dfi[dfi['bin_tl'] == bin_tl]
                dfsi = dfs[dfs['bin_tl'] == bin_tl]

                # plot
                for pc, pl in zip(plot_columns, plot_column_labels):
                    fig, ax = plt.subplots()
                    ax.plot(dfii[px], dfii.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
                    ax.plot(dfsi[px], dfsi.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')
                    ax.set_xlabel(r'$\delta x \: (pix.)$')
                    ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                    # ax.set_ylim(top=5.05)

                    axr = ax.twinx()
                    axr.plot(dfii[px], dfii[pc], '--', color=sciblue)
                    axr.plot(dfsi[px], dfsi[pc], '--', color=scigreen)
                    axr.set_ylabel(pl)
                    axr.set_ylim(bottom=0)

                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_z-true_for_dx={}.png'.format(pc, bin_tl))
                    if show_plots:
                        plt.show()
                    plt.close()

            # ---

        # ---

        # plot mean rmse-z by dx
        plot_rmse_by_dx = False
        if plot_rmse_by_dx:

            # read
            dfi = pd.read_excel(join(path_results, dir_idpt, 'results', 'mean_rmse-z_bin-dx.xlsx'))
            dfs = pd.read_excel(join(path_results, dir_spct, 'results', 'mean_rmse-z_bin-dx.xlsx'))

            # setup

            """plot_columns = ['num_meas', 'cm']
            plot_column_labels = [r'$N_{p} \: (\#)$', r'$c_{m}$']
            ms = 4

            # plot
            for pc, pl in zip(plot_columns, plot_column_labels):
                fig, ax = plt.subplots()
                ax.plot(dfi.index, dfi.rmse_z, '-o', color=sciblue, label='IDPT')
                ax.plot(dfs.index, dfs.rmse_z, '-o', color=scigreen, label='SPCT')
                ax.set_xlabel(r'$\delta x \: (pix.)$')
                ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                # ax.set_ylim(bottom=0.245, top=2.45)

                axr = ax.twinx()
                axr.plot(dfi.index, dfi[pc], '--', color=sciblue)
                axr.plot(dfs.index, dfs[pc], '--', color=scigreen)
                axr.set_ylabel(pl)
                axr.set_ylim(bottom=0)

                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_dx.png'.format(pc))
                if show_plots:
                    plt.show()
                plt.close()"""

        # ---

        # plot mean rmse-z by theta
        plot_rmse_by_theta = False
        if plot_rmse_by_theta:

            # read
            dfi = pd.read_excel(join(path_results, dir_idpt, 'results', 'mean_rmse-z_bin_theta.xlsx'))
            dfs = pd.read_excel(join(path_results, dir_spct, 'results', 'mean_rmse-z_bin_theta.xlsx'))

            # setup
            px = 'bin'
            plot_columns = ['num_meas', 'cm']
            plot_column_labels = [r'$N_{p} \: (\#)$', r'$c_{m}$']
            ms = 4

            # plot
            for pc, pl in zip(plot_columns, plot_column_labels):
                fig, ax = plt.subplots()
                ax.plot(dfi[px], dfi.rmse_z, '-o', color=sciblue, label='IDPT')
                ax.plot(dfs[px], dfs.rmse_z, '-o', color=scigreen, label='SPCT')
                ax.set_xlabel(r'$\theta \: (\circ)$')
                ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                # ax.set_ylim(bottom=0.245, top=2.45)

                axr = ax.twinx()
                axr.plot(dfi[px], dfi[pc], '--', color=sciblue)
                axr.plot(dfs[px], dfs[pc], '--', color=scigreen)
                axr.set_ylabel(pl)
                axr.set_ylim(bottom=0)

                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_theta.png'.format(pc))
                if show_plots:
                    plt.show()
                plt.close()

        # ---

        # plot mean rmse-z by z_true
        plot_rmse_by_z_true = False
        if plot_rmse_by_z_true:

            # read
            dfi = pd.read_excel(join(path_results, dir_idpt, 'results', 'mean_rmse-z_bin_z-true.xlsx'))
            dfs = pd.read_excel(join(path_results, dir_spct, 'results', 'mean_rmse-z_bin_z-true.xlsx'))

            # setup
            px = 'bin'
            plot_columns = ['true_percent_meas', 'cm']
            plot_column_labels = [r'$\phi \: (\%)$', r'$c_{m}$']
            ms = 4

            # plot
            for pc, pl in zip(plot_columns, plot_column_labels):
                fig, ax = plt.subplots()
                ax.plot(dfi[px], dfi.rmse_z, '-o', color=sciblue, label='IDPT')
                ax.plot(dfs[px], dfs.rmse_z, '-o', color=scigreen, label='SPCT')
                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                # ax.set_ylim(bottom=0.245, top=2.45)

                axr = ax.twinx()
                axr.plot(dfi[px], dfi[pc], '--', color=sciblue)
                axr.plot(dfs[px], dfs[pc], '--', color=scigreen)
                axr.set_ylabel(pl)
                axr.set_ylim(bottom=0)

                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/compare_rmse-z_and_{}_by_z-true.png'.format(pc))
                if show_plots:
                    plt.show()
                plt.close()

        # ---

    # ---

    # specifics: plot dz overlap
    plot_dz = True
    if plot_dz:
        # setup
        path_read = join(path_results, 'idpt-dz-overlap', 'results')
        path_figs = join(path_read, 'post-figs')

        if not os.path.exists(path_figs):
            os.makedirs(path_figs)

        save_plots = True
        show_plots = False

        # plot standard rmse z
        plot_standard = False
        if plot_standard:
            dfi = pd.read_excel(join(path_read, 'mean_rmse-z_bin-dx.xlsx'))
            dfi_no_bnkb = pd.read_excel('/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/'
                                        'results/idpt-dz-overlap/no-barnkob-filter/results/'
                                        'mean_rmse-z_bin-dx.xlsx')

            # add true number of particles
            dfi['true_num'] = dfi_no_bnkb.num_bind
            dfi['true_percent_meas'] = dfi['num_meas'] / dfi['true_num']

            # setup
            ms = 4

            # plot: rmse_z and true percent measure
            fig, (axr, ax) = plt.subplots(2, 1, sharex=True, figsize=(size_x_inches, size_y_inches),
                                          gridspec_kw={'height_ratios': [1, 3]})

            p1, = ax.plot(dfi.dx, dfi.rmse_z, '-o', ms=ms)
            axr.plot(dfi.dx, dfi.true_percent_meas, '-o', ms=ms / 1.25, color=p1.get_color())
            ax.set_xlabel(r'$\delta x \: (pix.)$')
            ax.set_ylabel(r'$\sigma_z^{\delta} \: (\mu m)$')
            axr.set_ylabel(r'$\phi^{\delta} \: (\%)$')
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.075)
            if save_plots:
                plt.savefig(join(path_figs, 'all_bin-dx_plot-rmse-z-true-percent{}'.format(save_fig_filetype)))
            if show_plots:
                plt.show()
            plt.close()

            # ---

            # plot rmse xyz and true percent measure
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(size_x_inches, size_y_inches * 1.5))
            ax1.plot(dfi.dx, dfi.rmse_y, '-o', ms=ms, color=sciblue, label='IDPT')
            ax1.set_ylabel(r'$\sigma_{y} \: (\mu m)$')
            ax1.legend()

            ax2.plot(dfi.dx, dfi.rmse_x, '-o', ms=ms, color=sciblue)
            ax2.set_ylabel(r'$\sigma_{x} \: (\mu m)$')

            ax3.plot(dfi.dx, dfi.rmse_z, '-o', ms=ms, color=sciblue)
            ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')

            ax4.plot(dfi.dx, dfi.true_percent_meas, '-o', ms=ms, color=sciblue)
            ax4.set_ylabel(r'$\phi \: (\%)$')
            ax4.set_xlabel(r'$\delta x \: (pix.)$')
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.075)
            if save_plots:
                plt.savefig(join(path_figs, 'all_bin-dx_plot-rmse-xyz-true-percent{}'.format(save_fig_filetype)))
            if show_plots:
                plt.show()
            plt.close()

        # ---

        # plot mean rmse-z by 2d
        plot_2d_bins = False
        if plot_2d_bins:
            dfi = pd.read_excel(join(path_read, 'idpt-dz-overlap_rmse-z_by_dx_theta_2d-bin.xlsx'), index_col=0)
            dfi_no_bnkb = pd.read_excel('/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/'
                                        'results/idpt-dz-overlap/no-barnkob-filter/results/'
                                        'idpt-dz-overlap_rmse-z_by_dx_theta_2d-bin.xlsx', index_col=0)

            # processing
            details = DatasetUnpacker(dataset='grid-overlap', key=1).unpack()
            inspect_key = details['key']
            splits = details['splits']
            keys = details['keys']
            dict_splits_to_keys = details['dict_splits_to_keys']
            intercolumn_spacing_threshold = details['intercolumn_spacing_threshold']
            template_size = details['template_size']
            max_diameter = details['max_diameter']

            # map dx to filename
            dfi['dx_id'] = dfi['filename']
            mapping_dict = {splits[i]: keys[i] for i in range(len(splits))}
            dfi.loc[:, 'dx_id'] = dfi.loc[:, 'dx_id'].map(mapping_dict)

            # add true number of particles
            dfi['true_num'] = dfi_no_bnkb.num_bind
            dfi['true_percent_meas'] = dfi['num_meas'] / dfi['true_num']

            # average +/- theta values
            dfi['abs_theta'] = dfi['bin'].abs()
            dfig = dfi.groupby(by=['filename', 'abs_theta']).mean().reset_index()

            # sort zipped(splits, keys)
            keys, splits = keys[1:], splits[1:]

            # setup
            ms = 2
            clrs = iter(cm.plasma(np.linspace(0.0, 0.95, len(keys))))

            # plot abs(theta)
            plot_abs_theta = False
            if plot_abs_theta:
                fig, (axy, axx, ax, axr) = plt.subplots(4, 1, sharex=True,
                                                        figsize=(size_x_inches * 1.0625, size_y_inches * 1.75)
                                                        )  # gridspec_kw={'height_ratios': [1, 2]}

                for xc, dx in zip(splits, keys):
                    dft = dfig[dfig['filename'] == xc]
                    p1, = ax.plot(dft.abs_theta, dft.rmse_z, '-', marker='.', ms=ms, color=next(clrs), label=dx)
                    axy.plot(dft.abs_theta, dft.rmse_y, '-', marker='.', ms=ms, color=p1.get_color())
                    axx.plot(dft.abs_theta, dft.rmse_x, '-', marker='.', ms=ms, color=p1.get_color())
                    axr.plot(dft.abs_theta, dft.true_percent_meas, '-', marker='.', ms=ms, color=p1.get_color())

                axy.set_ylabel(r'$\sigma_y \: (\mu m)$')
                axx.set_ylabel(r'$\sigma_x \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1.625), ncol=1, title=r'$\delta x$',
                          labelspacing=0.1, handletextpad=0.4, columnspacing=1)  #

                axr.set_xlabel(r'$\theta \: (deg.)$')
                axr.set_ylabel(r'$\phi \: (\%)$')
                axr.set_ylim(bottom=0.55)

                plt.tight_layout()
                fig.subplots_adjust(hspace=0.075)
                if save_plots:
                    plt.savefig(join(path_figs, 'all_bin-dx-abs-theta_plot-rmse-xyz{}'.format(save_fig_filetype)))
                if show_plots:
                    plt.show()
                plt.close()

            # ---

            # plot +/- theta
            plot_pm_theta = False
            if plot_pm_theta:
                fig, ax = plt.subplots(figsize=(size_x_inches * 1.0625, size_y_inches * 1.0625))
                clrs = iter(cm.plasma(np.linspace(0.0, 0.95, len(keys))))

                for xc, dx in zip(splits, keys):
                    dft = dfi[dfi['filename'] == xc]
                    ax.plot(dft.bin, dft.rmse_z, '-o', ms=2, color=next(clrs), label=dx)

                ax.set_xlabel(r'$\theta \: (deg.)$')
                ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
                # ax.set_ylim([0.5, 15.5])
                # ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1.075), ncol=1, title=r'$\delta x$',
                          labelspacing=0.1, handletextpad=0.4, columnspacing=1)
                plt.tight_layout()
                if save_plots:
                    plt.savefig(join(path_figs, 'all_bin-dx-theta_plot-rmse-z{}'.format(save_fig_filetype)))
                if show_plots:
                    plt.show()
                plt.close()

            # ---

            # plot uncertainty surface: rmse-z(dx, theta)
            # calculate z-uncertainty on a 2D grid (dx by dz) and plot 3D surface
            plot_rmse_z_by_dx_dz = False

            if plot_rmse_z_by_dx_dz:

                dfig_surf = dfig[dfig['dx_id'] < 30]

                # get data arrays
                x = dfig_surf.dx_id.to_numpy()
                y = dfig_surf.abs_theta.to_numpy()
                z = dfig_surf.rmse_z.to_numpy()

                for num in [250]:
                    # get the range of points for the 2D surface space
                    xr = (x.min(), x.max())
                    yr = (y.min(), y.max())
                    X = np.linspace(min(x), max(x), num)
                    Y = np.linspace(min(y), max(y), num)
                    X, Y = np.meshgrid(X, Y)
                    interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
                    Z = interp(X, Y)

                    plt.pcolormesh(X, Y, Z, shading='auto', cmap=cm.coolwarm)
                    plt.colorbar(label=r'$\sigma_z^{\delta} \: (\mu m)$')
                    plt.xlabel(r'$\delta x \: (pix.)$')
                    plt.ylabel(r'$\theta \: (^{\circ})$')
                    plt.tight_layout()

                    if save_plots:
                        plt.savefig(join(path_figs, 'bin-dx-abs-theta_2d-surf-plot-rmse-z_units-pixels.png'))
                    if show_plots:
                        plt.show()
                    plt.close()

                    # plot
                    plot_surface_3d = False
                    if plot_surface_3d:
                        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                        ax.set_xlabel(r'$\delta x \: (pix.)$')
                        ax.set_ylabel(r'$\theta \: (^{\circ})$')
                        ax.set_zlabel(r'$\sigma_z \: (\mu m)$')
                        ax.view_init(25, -70)
                        plt.tight_layout()
                        if save_plots:
                            plt.savefig(
                                join(join(path_figs, 'bin-dx-abs-theta_3d-surf-plot-rmse-z.png')))
                        if show_plots:
                            plt.show()
                        plt.close()

        # ---

        # compare rmse-z by dx to mean-pair-error by dx
        compare_error_to_mean_pair_error = False
        if compare_error_to_mean_pair_error:
            dfi = pd.read_excel(join(path_read, 'mean_rmse-z_bin-dx.xlsx'))
            dfip = pd.read_excel(join(path_results, 'idpt-dz-overlap', 'figs', 'mean-pair-error',
                                      'idpt-dz-overlap_dx-pair-error_bin-dz-z_plot-rmse-z_dz-limit25.xlsx'))

            # drop the single particle
            dfi = dfi[dfi['dx'] < 30]
            dfip = dfip[dfip['filename'] < 30]

            # setup
            ms = 4

            # plot: rmse_z and true percent measure
            fig, ax = plt.subplots()

            ax.plot(dfi.dx, dfi.rmse_z, '-o', ms=ms, label=r'$\epsilon_{i}$')
            ax.plot(dfip.filename, dfip.rmse_z, '-o', ms=ms, color=sciorange, label=r'$\overline{\epsilon}_{i, i+1}$')
            ax.set_xlabel(r'$\delta x \: (pix.)$')
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax.legend()
            plt.tight_layout()
            if save_plots:
                plt.savefig(join(path_figs, 'compare_bin-dx_rmse-z-to-mean-pair-error{}'.format(save_fig_filetype)))
            #if show_plots:
            plt.show()
            plt.close()

        # ---

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# C. ANALYZE SPCT STATS
analyze_spct_stats = False

if analyze_spct_stats:
    path_calib_coords = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level2/grid-dz/results/iter1/calibration-spct-get-spct-cal'
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/synthetic grid dz overlap nl25/results/idpt-dz-overlap/results/mean_rmse-z_bin-dx.xlsx'

    # read calib coords
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='spct')

    # read test coords
    df = pd.read_excel(fp)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# B. ANALYZE TEST OUTPUTS
analyze_single_particle = False

if analyze_single_particle:

    path_single = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/results/' \
                  'idpt-no-dz-overlap/single-particle'
    path_figs = path_single + '/figs'

    fpi = path_single + '/results/idpt-no-dz-overlap_rmse-z_by_dx_z-true_2d-bin.xlsx'
    dfi = pd.read_excel(fpi, index_col=0)

    # ---

    # processing
    dfi['dx'] = dfi['dx'].abs()

    # ---

    # plot cm by dx and z

    # setup
    dxs = dfi.dx.unique()
    dxs = dxs[dxs < 30]
    dxs = np.append(dxs, 38)
    clrs = iter(cm.magma(np.linspace(0.25, 0.95, len(dxs))))

    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 1.125))

    for dx in dxs:
        dfidx = dfi[dfi['dx'] == dx]

        if dx == 38:
            ax.plot(dfidx.z_true, dfidx.cm, '-', color='black', label=dx)
        else:
            ax.plot(dfidx.z_true, dfidx.cm, '-', color=next(clrs), label=dx)

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$c_{m}^{\delta}$')
    ax.legend(title=r'$\delta x$', loc='upper left',
              bbox_to_anchor=(1, 1.05), labelspacing=0.1, handletextpad=0.4, columnspacing=1)
    plt.tight_layout()
    plt.savefig(path_figs + '/single-particle_cm_by_dx_and_by_z.svg')
    plt.show()
    plt.close()

    # ---

    # plot mean cm by dx
    dfig = dfi.groupby('dx').mean()

    fig, ax = plt.subplots()
    ax.plot(dfig.index, dfig.cm, '-o')
    ax.set_xlabel(r'$\delta x \: (pix.)$')
    ax.set_ylabel(r'$c_{m}$')
    plt.tight_layout()
    plt.savefig(path_figs + '/single-particle_mean-cm_by_dx.svg')
    plt.show()
    plt.close()



    j = 1


# ---

# ----------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS


# ---

print("Analysis completed without errors")