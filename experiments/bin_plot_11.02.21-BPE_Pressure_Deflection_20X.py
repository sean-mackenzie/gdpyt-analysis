# test bin, analyze, and plot functions

# imports
import os
from os.path import join
from os import listdir

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import filter
import analyze
from correction import correct
from utils import fit, functions, bin, io, plotting, modify, plot_collections
from utils.plotting import lighten_color

# A note on SciencePlots colors
"""
Blue: #0C5DA5
Green: #00B945
Red: #FF9500
Orange: #FF2C00

Other Colors:
Light Blue: #7BC8F6
Paler Blue: #0343DF
Azure: #069AF3
Dark Green: #054907
"""

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF9500'
sciorange = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# 1. SETUP - BASE DIRECTORY

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.02.21-BPE_Pressure_Deflection_20X/analyses/'

# ----------------------------------------------------------------------------------------------------------------------
# 2. SETUP - IDPT

path_idpt = join(base_dir, 'results-04.26.22_idpt')
path_test_coords = join(path_idpt, 'coords/test-coords')
path_calib_coords = join(path_idpt, 'coords/calib-coords')
path_similarity = join(path_idpt, 'similarity')
path_results = join(path_idpt, 'results')
path_figs = join(path_idpt, 'figs')

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3. ANALYSIS - READ FILES

method = 'idpt'
microns_per_pixel = 0.8

# ----- 4.1 CORRECT TEST COORDS
correct_test_coords = False

if correct_test_coords:

    use_idpt_zf = False
    use_spct_zf = False

    # ------------------------------------------------------------------------------------------------------------------

    if use_idpt_zf:
        """
        NOTE: This correction scheme fits a 2D spline to the in-focus particle positions and uses this to set their 
        z_f = 0 position.        
        """

        param_zf = 'zf_from_peak_int'
        plot_calib_plane = False
        plot_calib_spline = False
        kx, ky = 2, 2

        # step 1. read calibration coords
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)

        # step 2. remove outliers

        # 2.1 get z_in-focus mean + standard deviation
        zf_c_mean = dfcpid[param_zf].mean()
        zf_c_std = dfcpid[param_zf].std()

        # 2.2 filter calibration coords
        dfcpid = dfcpid[(dfcpid[param_zf] > zf_c_mean - zf_c_std) & (dfcpid[param_zf] < zf_c_mean + zf_c_std)]

        # step 3. fit plane
        dictc_fit_plane = correct.fit_in_focus_plane(df=dfcpid, param_zf=param_zf, microns_per_pixel=microns_per_pixel)
        popt_c = dictc_fit_plane['popt_pixels']

        if plot_calib_plane:
            fig = plotting.plot_fitted_plane_and_points(df=dfcpid, dict_fit_plane=dictc_fit_plane)
            plt.savefig(path_figs + '/idpt-calib-coords_fit-plane_raw.png')
            plt.close()

            dfict_fit_plane = pd.DataFrame.from_dict(dictc_fit_plane, orient='index', columns=['value'])
            dfict_fit_plane.to_excel(path_figs + '/idpt-calib-coords_fit-plane_raw.xlsx')

        # step 4. FIT SMOOTH 2D SPLINE AND PLOT RAW POINTS + FITTED SURFACE (NO CORRECTION)
        bispl_c, rmse_c = fit.fit_3d_spline(x=dfcpid.x,
                                            y=dfcpid.y,
                                            z=dfcpid[param_zf],
                                            kx=kx,
                                            ky=ky)

        if plot_calib_spline:
            fig, ax = plotting.scatter_3d_and_spline(dfcpid.x, dfcpid.y, dfcpid[param_zf],
                                                     bispl_c,
                                                     cmap='RdBu',
                                                     grid_resolution=30,
                                                     view='multi')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_zlabel(r'$z_{f} \: (\mu m)$')
            plt.suptitle('fit RMSE = {}'.format(np.round(rmse_c, 3)))
            plt.savefig(path_figs + '/idpt-calib-coords_fit-spline_kx{}_ky{}.png'.format(kx, ky))
            plt.close()

        # step 5. read test_coords
        dft = io.read_test_coords(path_test_coords)

        # step 6. drop unnecessary columns in dft
        dft = dft[['frame', 'id', 'z', 'z_true', 'x', 'y', 'cm', 'error']]

        # step 7. create a z_corr column by using fitted spline to correct z
        dft = correct.correct_z_by_spline(dft, bispl=bispl_c, param_z='z')
        dft['z_true_corr'] = dft['z_true'] - dft['z_cal_surf']

        # step 8. export corrected test_coords
        dft.to_excel(path_results + '/test_coords_corrected_t-calib2_c-calib1.xlsx', index=False)

    elif use_spct_zf:
        """
        NOTE: No correction is currently performed. The z-coords are well aligned enough in both calibration image sets
        to just ignore. This is not necessarily surprising because the calibration images were acquired with the intention
        of making the z-coords identical for all calibration image sets (by using the same beginning and ending tick mark
        on the fine adjustment knob during image acquisition).
        """

        # --------------------------------------------------------------------------------------------------------------
        # SETUP - SPCT CALIBRATION IN-FOCUS COORDS

        # SPCT analysis of images used for IDPT calibration
        path_spct_calib_coords = join(base_dir, 'results-04.26.22_spct_calib1_test-2-3/coords/calib-coords')
        path_calib_pid_defocus = join(path_spct_calib_coords, 'calib_spct_pid_defocus_stats_c-calib1_t-calib2.xlsx')
        path_calib_spct_stats = join(path_spct_calib_coords, 'calib_spct_stats_c-calib1_t-calib2.xlsx')
        path_calib_spct_pop = join(path_spct_calib_coords, 'calib_spct_pop_defocus_stats_c-calib1_t-calib2.xlsx')

        # SPCT analysis of images used for IDPT test
        path_spct_test_coords = join(base_dir, 'results-04.28.22_spct-calib2_test3/coords/calib-coords')
        path_test_pid_defocus = join(path_spct_test_coords, 'calib_spct_pid_defocus_stats_c-calib2_t-calib3.xlsx')
        path_test_spct_stats = join(path_spct_test_coords, 'calib_spct_stats_c-calib2_t-calib3.xlsx')
        path_test_spct_pop = join(path_spct_test_coords, 'calib_spct_pop_defocus_stats_c-calib2_t-calib3.xlsx')

        # --------------------------------------------------------------------------------------------------------------

        # --- PART A. READ COORDS USED FOR IDPT CALIBRATION (i.e. 'calib1')

        merge_spct_stats = True
        param_zf = 'zf_from_peak_int'
        plot_calib_plane = True
        plot_test_plane = True
        kx, ky = 2, 2

        # step 1. merge [['x', 'y']] into spct pid defocus stats.
        if merge_spct_stats:
            # read SPCT calibration coords and merge ['x', 'y'] into pid_defocus_stats
            dfcpid = pd.read_excel(path_calib_pid_defocus)
            dfcstats = pd.read_excel(path_calib_spct_stats)
            dfcpid = modify.merge_calib_pid_defocus_and_correction_coords(path_calib_coords, method, dfs=[dfcstats,
                                                                                                          dfcpid])
        else:
            # read SPCT pid defocus stats that have already been merged
            path_calib_pid_defocus = join(path_calib_coords, 'calib_spct_pid_defocus_stats_calib1_xy.xlsx')
            dfcpid = pd.read_excel(path_calib_pid_defocus)

        # step 2. remove outliers

        # 2.1 get z_in-focus mean + standard deviation
        zf_c_mean = dfcpid[param_zf].mean()
        zf_c_std = dfcpid[param_zf].std()

        # 2.2 filter calibration coords
        dfcpid = dfcpid[(dfcpid[param_zf] > 34) & (dfcpid[param_zf] < zf_c_mean + zf_c_std / 2)]
        dfcpid = dfcpid[dfcpid['x'] > 120]

        # step 3. fit plane
        dictc_fit_plane = correct.fit_in_focus_plane(df=dfcpid, param_zf=param_zf, microns_per_pixel=microns_per_pixel)
        popt_c = dictc_fit_plane['popt_pixels']

        if plot_calib_plane:
            fig = plotting.plot_fitted_plane_and_points(df=dfcpid, dict_fit_plane=dictc_fit_plane)
            plt.savefig(path_figs + '/calibration-coords_fit-plane_raw.png')
            plt.close()

            dfict_fit_plane = pd.DataFrame.from_dict(dictc_fit_plane, orient='index', columns=['value'])
            dfict_fit_plane.to_excel(path_figs + '/calibration-coords_fit-plane_raw.xlsx')

            # FIT SMOOTH 2D SPLINE AND PLOT RAW POINTS + FITTED SURFACE (NO CORRECTION)
            bispl_c, rmse_c = fit.fit_3d_spline(x=dfcpid.x,
                                                y=dfcpid.y,
                                                z=dfcpid[param_zf],
                                                kx=kx,
                                                ky=ky)

            fig, ax = plotting.scatter_3d_and_spline(dfcpid.x, dfcpid.y, dfcpid[param_zf],
                                                     bispl_c,
                                                     cmap='RdBu',
                                                     grid_resolution=30,
                                                     view='multi')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_zlabel(r'$z_{f} \: (\mu m)$')
            plt.suptitle('fit RMSE = {}'.format(np.round(rmse_c, 3)))
            plt.savefig(path_figs + '/calibration-coords_fit-spline_kx{}_ky{}.png'.format(kx, ky))
            plt.close()

        # ---

        # --- PART B. READ COORDS USED FOR IDPT TEST (i.e. 'calib2')

        # step 1. merge [['x', 'y']] into spct pid defocus stats.
        if merge_spct_stats:
            # read SPCT calibration coords and merge ['x', 'y'] into pid_defocus_stats
            dfcpid = pd.read_excel(path_test_pid_defocus)
            dfcstats = pd.read_excel(path_test_spct_stats)
            dfcpid = modify.merge_calib_pid_defocus_and_correction_coords(path_calib_coords, method, dfs=[dfcstats,
                                                                                                          dfcpid])
        else:
            # read SPCT pid defocus stats that have already been merged
            path_calib_pid_defocus = join(path_calib_coords, 'calib_spct_pid_defocus_stats_calib2_xy.xlsx')
            dfcpid = pd.read_excel(path_calib_pid_defocus)

        # step 2. remove outliers

        # 2.1 get z_in-focus mean + standard deviation
        zf_c_mean = dfcpid[param_zf].mean()
        zf_c_std = dfcpid[param_zf].std()

        # 2.2 filter calibration coords
        dfcpid = dfcpid[(dfcpid[param_zf] > zf_c_mean - zf_c_std / 2) & (dfcpid[param_zf] < zf_c_mean + zf_c_std / 2)]

        # step 3. fit plane
        dictc_fit_plane = correct.fit_in_focus_plane(df=dfcpid, param_zf=param_zf, microns_per_pixel=microns_per_pixel)
        popt_c = dictc_fit_plane['popt_pixels']

        if plot_test_plane:
            fig = plotting.plot_fitted_plane_and_points(df=dfcpid, dict_fit_plane=dictc_fit_plane)
            plt.savefig(path_figs + '/test-coords_fit-plane_raw.png')
            plt.close()

            dfict_fit_plane = pd.DataFrame.from_dict(dictc_fit_plane, orient='index', columns=['value'])
            dfict_fit_plane.to_excel(path_figs + '/test-coords_fit-plane_raw.xlsx')

            # FIT SMOOTH 2D SPLINE AND PLOT RAW POINTS + FITTED SURFACE (NO CORRECTION)
            bispl_c, rmse_c = fit.fit_3d_spline(x=dfcpid.x,
                                                y=dfcpid.y,
                                                z=dfcpid[param_zf],
                                                kx=kx,
                                                ky=ky)

            fig, ax = plotting.scatter_3d_and_spline(dfcpid.x, dfcpid.y, dfcpid[param_zf],
                                                     bispl_c,
                                                     cmap='RdBu',
                                                     grid_resolution=30,
                                                     view='multi')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_zlabel(r'$z_{f} \: (\mu m)$')
            plt.suptitle('fit RMSE = {}'.format(np.round(rmse_c, 3)))
            plt.savefig(path_figs + '/test-coords_fit-spline_kx{}_ky{}.png'.format(kx, ky))
            plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# 4. PLOT TEST COORDS RMSE-Z

analyze_test_coords = False
save_plots = False
show_plots = False

if analyze_test_coords:
    # read test coords
    dft = io.read_test_coords(path_test_coords)

    # test coords stats
    mag_eff = 20.0
    area_pixels = 512 ** 2
    area_microns = (512 * microns_per_pixel) ** 2
    i_num_rows = len(dft)
    i_num_pids = len(dft.id.unique())

    # ---

    # --- STEP 0. drop and rename columns for simplicity
    dft = dft.drop(columns=['z', 'z_true'])
    dft = dft.rename(columns={'z_corr': 'z', 'z_true_corr': 'z_true'})

    # ---

    rmse_all_particles = False
    rmse_on_off_bpe = False
    rmse_compare = False

    # format plots
    xylim = 37.25
    xyticks = [-30, -15, 0, 15, 30]
    lbls = ['On', 'Border', 'Off']
    markers = ['s', 'd', 'o']

    if rmse_all_particles:

        # --- STEP 1. CALCULATE RMSE-Z FOR ALL PARTICLES
        column_to_bin = 'z_true'
        bins_z = 20
        round_z_to_decimal = 3
        min_cm = 0.5

        # 1.1 mean rmse-z
        dfrmse_mean = bin.bin_local_rmse_z(dft,
                                           column_to_bin=column_to_bin,
                                           bins=1,
                                           min_cm=min_cm,
                                           z_range=None,
                                           round_to_decimal=round_z_to_decimal,
                                           df_ground_truth=None,
                                           dropna=True,
                                           error_column='error',
                                           )
        dfrmse_mean.to_excel(path_results + '/mean-rmse-z_bin=1_no-filters.xlsx')

        # 1.2 binned rmse-z
        dfrmse = bin.bin_local_rmse_z(dft,
                                      column_to_bin=column_to_bin,
                                      bins=bins_z,
                                      min_cm=min_cm,
                                      z_range=None,
                                      round_to_decimal=round_z_to_decimal,
                                      df_ground_truth=None,
                                      dropna=True,
                                      error_column='error',
                                      )

        dfrmse.to_excel(path_results + '/binned-rmse-z_bins={}_no-filters.xlsx'.format(bins_z))

        # 1.3 groupby 'bin' rmse-z mean + std
        dfrmsem, dfrmsestd = bin.bin_generic(dft,
                                             column_to_bin='bin',
                                             column_to_count='id',
                                             bins=bins_z,
                                             round_to_decimal=round_z_to_decimal,
                                             return_groupby=True)

        # 1.3 plot binned rmse-z
        if save_plots or show_plots:

            # close all figs
            plt.close('all')

            # ----------------------- BASIC RMSE-Z PLOTS

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

            # ----------------------- Z-MEAN +/- Z-STD PLOTS

            # fit line
            popt, pcov = curve_fit(functions.line, dfrmse.z_true, dfrmse.z)
            z_fit = np.linspace(dfrmse.z_true.min(), dfrmse.z_true.max())

            rmse_fit_line = np.sqrt(np.sum((functions.line(dfrmse.z_true, *popt) - dfrmse.z)**2) / len(dfrmse.z))
            print(rmse_fit_line)

            # binned calibration curve with std-z errorbars (microns) + fit line
            fig, ax = plt.subplots()

            ax.errorbar(dfrmsem.z_true, dfrmsem.z, yerr=dfrmsestd.z, fmt='o', ms=3, elinewidth=0.5, capsize=1, color=sciblue,
                        label=r'$\overline{z} \pm \sigma$')  #
            ax.plot(z_fit, functions.line(z_fit, *popt), linestyle='--', linewidth=1.5, color='black', alpha=0.25,
                    label=r'$dz/dz_{true} = $' + ' {}'.format(np.round(popt[0], 3)))

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

    if rmse_on_off_bpe:

        # --- STEP 0. SPLIT DATAFRAME INTO (1) OFF BPE and (2) OFF BPE.
        column_to_bin = 'x'
        bins_x = [145, 175, 205]
        round_x_to_decimal = 0

        dfbx = bin.bin_by_list(dft,
                               column_to_bin=column_to_bin,
                               bins=bins_x,
                               round_to_decimal=round_x_to_decimal,
                               )

        df_on = dfbx[dfbx['bin'] == bins_x[0]]
        df_edge = dfbx[dfbx['bin'] == bins_x[1]]
        df_off = dfbx[dfbx['bin'] == bins_x[2]]

        # --- plotting

        # --- STEP 1. PLOT CALIBRATION CURVE (Z VS. Z_TRUE) FOR EACH DATAFRAME (ON, EDGE, OFF)
        ss = 1

        fig, ax = plt.subplots()

        ax.scatter(df_off.z_true, df_off.z, s=ss, marker=markers[2], color=sciblue, label=lbls[2])
        ax.scatter(df_on.z_true, df_on.z, s=ss, marker=markers[0], color=sciorange, label=lbls[0])
        ax.scatter(df_edge.z_true, df_edge.z, s=ss, marker=markers[1], color=scired, label=lbls[1])

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        ax.legend(loc='lower right', markerscale=2.5)
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/on-edge-off-bpe_calibration_curve.png')
        if show_plots:
            plt.show()
        plt.close()

        # --- STEP 2. FOR EACH DATAFRAME (ON, EDGE, OFF), COMPUTE RMSE-Z AND PLOT

        for lbl, dft in zip(lbls, [df_on, df_edge, df_off]):

            # --- STEP 1. CALCULATE RMSE-Z FOR ALL PARTICLES
            column_to_bin = 'z_true'
            bins_z = 20
            round_z_to_decimal = 3
            min_cm = 0.5

            # 1.1 mean rmse-z
            dfrmse_mean = bin.bin_local_rmse_z(dft,
                                               column_to_bin=column_to_bin,
                                               bins=1,
                                               min_cm=min_cm,
                                               z_range=None,
                                               round_to_decimal=round_z_to_decimal,
                                               df_ground_truth=None,
                                               dropna=True,
                                               error_column='error',
                                               )
            dfrmse_mean.to_excel(path_results + '/{}_mean-rmse-z_bin=1_no-filters.xlsx'.format(lbl))

            # 1.2 binned rmse-z
            dfrmse = bin.bin_local_rmse_z(dft,
                                          column_to_bin=column_to_bin,
                                          bins=bins_z,
                                          min_cm=min_cm,
                                          z_range=None,
                                          round_to_decimal=round_z_to_decimal,
                                          df_ground_truth=None,
                                          dropna=True,
                                          error_column='error',
                                          )

            dfrmse.to_excel(path_results + '/{}_binned-rmse-z_bins={}_no-filters.xlsx'.format(lbl, bins_z))

            # 1.3 groupby 'bin' rmse-z mean + std
            dfrmsem, dfrmsestd = bin.bin_generic(dft,
                                                 column_to_bin='bin',
                                                 column_to_count='id',
                                                 bins=bins_z,
                                                 round_to_decimal=round_z_to_decimal,
                                                 return_groupby=True)

            # 1.3 plot binned rmse-z
            if save_plots or show_plots:

                # close all figs
                plt.close('all')

                # ----------------------- BASIC RMSE-Z PLOTS

                # rmse-z: microns
                fig, ax = plt.subplots()
                ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_xlim([-xylim, xylim])
                ax.set_xticks(ticks=xyticks, labels=xyticks)
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_figs + '/{}_rmse-z_microns.png'.format(lbl))
                if show_plots:
                    plt.show()
                plt.close()

                # ----------------------- Z-MEAN +/- Z-STD PLOTS

                # fit line
                popt, pcov = curve_fit(functions.line, dfrmse.z_true, dfrmse.z)
                z_fit = np.linspace(dfrmse.z_true.min(), dfrmse.z_true.max())

                rmse_fit_line = np.sqrt(np.sum((functions.line(dfrmse.z_true, *popt) - dfrmse.z) ** 2) / len(dfrmse.z))
                print(rmse_fit_line)

                # binned calibration curve with std-z errorbars (microns) + fit line
                fig, ax = plt.subplots()

                ax.errorbar(dfrmsem.z_true, dfrmsem.z, yerr=dfrmsestd.z, fmt='o', ms=3, elinewidth=0.5, capsize=1,
                            color=sciblue,
                            label=r'$\overline{z} \pm \sigma$')  #
                ax.plot(z_fit, functions.line(z_fit, *popt), linestyle='--', linewidth=1.5, color='black', alpha=0.25,
                        label=r'$dz/dz_{true} = $' + ' {}'.format(np.round(popt[0], 3)))

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
                                '/{}_calibration_curve_z+std-errobars_fit_line_a{}_b{}_slope-label-blk.png'.format(
                                    lbl,
                                    np.round(popt[0],
                                             3),
                                    np.round(popt[1],
                                             3))
                                )
                if show_plots:
                    plt.show()
                plt.close()

    if rmse_compare:

        # 1. read binned rmse-z dataframes from Excel
        path_rmse_compare = join(path_results, 'on-edge-off-bpe')

        df1 = pd.read_excel(join(path_rmse_compare, '{}_binned-rmse-z_bins=20_no-filters.xlsx'.format(lbls[0])))
        df2 = pd.read_excel(join(path_rmse_compare, '{}_binned-rmse-z_bins=20_no-filters.xlsx'.format(lbls[1])))
        df3 = pd.read_excel(join(path_rmse_compare, '{}_binned-rmse-z_bins=20_no-filters.xlsx'.format(lbls[2])))

        # 1.3 plot binned rmse-z
        if save_plots or show_plots:

            ms = 4

            # ----------------------- BASIC RMSE-Z PLOTS

            # rmse-z: microns
            fig, ax = plt.subplots()

            ax.plot(df3.bin, df3.rmse_z, '-o', ms=ms, label=lbls[2], color=sciblue)
            ax.plot(df2.bin, df2.rmse_z, '-o', ms=ms, label=lbls[1], color=scired)
            ax.plot(df1.bin, df1.rmse_z, '-o', ms=ms, label=lbls[0], color=sciorange)

            ax.set_xlabel(r'$z_{true} \: (\mu m)$')
            ax.set_xlim([-xylim, xylim])
            ax.set_xticks(ticks=xyticks, labels=xyticks)
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax.legend()

            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs + '/compare-on-edge-off-bpe_rmse-z_microns.png')
            if show_plots:
                plt.show()
            plt.close()

            # rmse-z (microns) + c_m
            darken_clr = 1.0
            alpha_clr = 1.0

            fig, [axr, ax] = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]})

            axr.plot(df3.bin, df3.cm, '-', ms=ms-2, marker=markers[2], color=sciblue)
            axr.plot(df2.bin, df2.cm, '-', ms=ms-2, marker=markers[1], color=scired)
            axr.plot(df1.bin, df1.cm, '-', ms=ms-2, marker=markers[0], color=sciorange)
            axr.set_ylabel(r'$c_{m}$')

            ax.plot(df3.bin, df3.rmse_z, '-', ms=ms-0.75, marker=markers[2], color=sciblue, label=lbls[2])
            ax.plot(df2.bin, df2.rmse_z, '-', ms=ms-0.75, marker=markers[1], color=scired, label=lbls[1])
            ax.plot(df1.bin, df1.rmse_z, '-', ms=ms-0.75, marker=markers[0], color=sciorange, label=lbls[0])
            ax.set_xlabel(r'$z_{true} \: (\mu m)$')
            ax.set_xlim([-xylim, xylim])
            ax.set_xticks(ticks=xyticks, labels=xyticks)
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax.legend()

            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs + '/compare-on-edge-off-bpe_rmse-z_microns_cm.png')
            if show_plots:
                plt.show()
            plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# 5. IDPT VS. SPCT - COMPARE NUMBER OF PARTICLES PER Z

compare_idpt_spct = False
save_plots = False
show_plots = False

if compare_idpt_spct:

    # --- 1. IDPT

    # read IDPT test coords
    dft = io.read_test_coords(path_test_coords)

    # test coords stats
    mag_eff = 20.0
    area_pixels = 512 ** 2
    area_microns = (512 * microns_per_pixel) ** 2
    i_num_rows = len(dft)
    i_num_pids = len(dft.id.unique())
    dft = dft.drop(columns=['z', 'z_true'])
    dft = dft.rename(columns={'z_corr': 'z', 'z_true_corr': 'z_true'})

    # --- 2. SPCT

    # 2.1 read SPCT off-bpe test coords
    dfs_off = pd.read_excel('/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.02.21-BPE_Pressure_Deflection_20X/analyses/results-04.26.22_spct_calib1_test-2-3/coords/test-coords/test_coords_t-calib2_c-calib1.xlsx')
    dfs_on = pd.read_excel('/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.02.21-BPE_Pressure_Deflection_20X/analyses/results-04.26.22_spct_stack-id-on-bpe/testcalib2_calcalib1/test_coords_t_20X_ccalib1_tcalib2_c_20X_tcalib2_ccalib1_2022-04-26 20:45:34.334931.xlsx')

    # 2.2 correct z by mean z_f from peak_intensity
    z_f_mean = 35.1

    dfs_off['z'] = dfs_off['z'] - z_f_mean
    dfs_off['z_true'] = dfs_off['z_true'] - z_f_mean
    dfs_on['z'] = dfs_on['z'] - z_f_mean
    dfs_on['z_true'] = dfs_on['z_true'] - z_f_mean

    # --- 3. GROUPBY Z_TRUE
    dftg = dft.copy()
    dftg = dftg.round({'z_true': 0})
    dftc = dftg.groupby('z_true').count().reset_index()

    dfs_offc = dfs_off.groupby('z_true').count().reset_index()
    dfs_onc = dfs_on.groupby('z_true').count().reset_index()

    # filter z_true for pretty plotting
    zlim = 35
    dftc = dftc[(dftc['z_true'] > -zlim) & (dftc['z_true'] < zlim)]
    dfs_offc = dfs_offc[(dfs_offc['z_true'] > -zlim) & (dfs_offc['z_true'] < zlim)]
    dfs_onc = dfs_onc[(dfs_onc['z_true'] > -zlim) & (dfs_onc['z_true'] < zlim)]

    # ---

    # --- plotting

    # format plots
    xylim = 37.25
    xyticks = [-30, -15, 0, 15, 30]
    ms = 3

    # FIGURE 1. PLOT NUMBER OF PARTICLES PER Z_TRUE
    fig, ax = plt.subplots()

    ax.plot(dftc.z_true, dftc.z, '-o', ms=ms, color=sciblue, label=r'$IDPT$')
    ax.plot(dfs_offc.z_true, dfs_offc.z, '-o', ms=ms, color=lighten_color(scigreen, 1.0), label=r'$SPCT_{Low}$')
    ax.plot(dfs_onc.z_true, dfs_onc.z, '-o', ms=ms, color=lighten_color(scigreen, 1.2), label=r'$SPCT_{High}$')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xlim([-xylim, xylim])
    ax.set_xticks(xyticks)
    ax.set_ylabel(r'$N_{p} \: (\#)$')
    ax.set_ylim([0, 200])
    ax.legend()

    plt.tight_layout()
    if save_plots:
        plt.savefig(path_figs + '/compare-idpt-spct_num-particles.png')
    if show_plots:
        plt.show()
    plt.close()

    # ---

    # FIGURE 2. PLOT NUMBER OF PARTICLES PER Z_TRUE AND CM

    dftm = dftg.groupby('z_true').mean().reset_index()

    dfs_offm = dfs_off.groupby('z_true').mean().reset_index()
    dfs_onm = dfs_on.groupby('z_true').mean().reset_index()

    # filter z_true for pretty plotting
    dftm = dftm[(dftm['z_true'] > -zlim) & (dftm['z_true'] < zlim)]
    dfs_offm = dfs_offm[(dfs_offm['z_true'] > -zlim) & (dfs_offm['z_true'] < zlim)]
    dfs_onm = dfs_onm[(dfs_onm['z_true'] > -zlim) & (dfs_onm['z_true'] < zlim)]

    # plot
    fig, [axr, ax] = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    axr.plot(dftm.z_true, dftm.cm, '-o', ms=ms - 1, color=sciblue)
    axr.plot(dfs_offm.z_true, dfs_offm.cm, '-o', ms=ms - 1, color=lighten_color(scigreen, 1.0))
    axr.plot(dfs_onm.z_true, dfs_onm.cm, '-o', ms=ms - 1, color=lighten_color(scigreen, 1.2))
    axr.set_ylabel(r'$c_{m}$')
    axr.set_ylim([0.790, 1.01])
    axr.set_yticks([0.8, 0.9, 1.0])

    ax.plot(dftc.z_true, dftc.z, '-o', ms=ms, color=sciblue, label=r'$IDPT$')
    ax.plot(dfs_offc.z_true, dfs_offc.z, '-o', ms=ms, color=lighten_color(scigreen, 1.0), label=r'$SPCT_{Low}$')
    ax.plot(dfs_onc.z_true, dfs_onc.z, '-o', ms=ms, color=lighten_color(scigreen, 1.2), label=r'$SPCT_{High}$')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xlim([-xylim, xylim])
    ax.set_xticks(xyticks)
    ax.set_ylabel(r'$N_{p} \: (\#)$')
    ax.set_ylim([0, 185])
    ax.set_yticks([0, 50, 100, 150])
    ax.legend()

    plt.tight_layout()
    if save_plots:
        plt.savefig(path_figs + '/compare-idpt-spct_num-particles_and_cm.png')
    if show_plots:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# 6. AVERAGE PARTICLE-TO-PARTICLE SIMILARITY PER-FRAME

plot_average_particle_similarity = False

if plot_average_particle_similarity:

    # setup
    save_plots = True
    xylim = 37.25
    xyticks = [-30, -15, 0, 15, 30]
    ms = 3

    # read dataframe
    fp = join(base_dir, 'average-particle-similarity/'
                        'average_similarity_SPCT_11.02.21-BPE_Pressure_Deflection_20X_c-calib1_t-calib2.xlsx')
    dfsim = pd.read_excel(fp)

    # plot
    fig, ax = plt.subplots()

    ax.plot(dfsim.z_corr, dfsim.sim, '-o', ms=ms)

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xlim([-xylim, xylim])
    ax.set_xticks(xyticks)
    ax.set_ylabel(r'$S (p_{i}, p_{N})$')
    ax.set_ylim([0.49, 1.01])

    plt.tight_layout()
    if save_plots:
        plt.savefig(path_figs + '/average-particle-to-particle-similarity.png')
    plt.show()
    plt.close()


j = 1

print("Analysis completed without errors.")