# imports
from os.path import join
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate

import analyze
from correction.correct import correct_z_by_xy_surface, correct_z_by_spline
from utils import plot_collections, bin, modify, plotting, fit, functions

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

"""
Blue: #0C5DA5
Green: #00B945
"""

plt.style.use(['science', 'ieee', 'std-colors'])
sci_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'

# --- file paths
base_dir = '/Users/mackenzie/Desktop/idpt_experiments/10.07.21-BPE_Pressure_Deflection_20X/analyses/results-04.06.22-min-temp-pos-and-neg/'
svp = base_dir + 'figs/'
rvp = base_dir + 'results/'

# --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  ---
# READ FILES


# --- ---  --- ---  --- ---  --- --- READ CALIBRATION DIAMETER PARAMETERS
theory_diam_path = base_dir + 'spct-calib-coords/calib_spct_pop_defocus_stats.xlsx'

# --- ---  --- ---  --- ---  --- --- READ CALIBRATION IN-FOCUS COORDS

fpcal = base_dir + 'spct-calib-coords/calib_idpt_pid_defocus_stats_xy.xlsx'


# --- ---  --- ---  --- ---  --- --- READ POSITIVE TEST COORDS
fdir = base_dir + 'test-coords/combined'

files = [f for f in os.listdir(fdir) if f.endswith('.xlsx')]
pids = [float(xf.split('z')[-1].split('um.xlsx')[0]) for xf in files]

pids = sorted(pids)
files = sorted(files, key=lambda x: float(x.split('z')[-1].split('um.xlsx')[0]))


# --- ---  --- ---  --- ---  --- --- READ NEGATIVE TEST COORDS


# --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  ---
# DEFINE FUNCTIONS


def flip_y(df, y0):
    df['y'] = (df['y'] - 512) * -1 + y0
    return df


def center_on_origin(df, z0=40, y0=25):
    df['z'] = df['z'] - z0
    df = flip_y(df, y0)
    return df


# --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  ---


# --- ---  --- ---  --- ---  --- --- ANALYZE TEST COORDINATES
analyze_test = False

if analyze_test:

    show_figs = False
    save_figs = False
    export_results = True

    analyze_all_rows = True
    plot_raw = False
    plot_zcorr = False
    analyze_per_particle_precision = True
    save_plots, show_plots = False, False

    fit_general_surface = True
    save_fitted_surface = False

    analyze_all_pids = False
    analyze_rmsez_by_overlap = True
    fit_pid_general_surface = True
    show_percent_figs = False
    save_percent_figs = False
    fit_beam_theory = False
    plot_fit_surface = False
    filter_z_std = 2
    filter_precision = 2

    analyze_percent_measure_by_precision = False
    export_precision_sweep_results = False
    save_precision_sweep_figs = False
    save_precision_figs = False
    show_precision_figs = False
    export_precision_results = False

    analyze_combined_tests = True

    # --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- --- -
    # SETUP PROCESS MODIFIERS

    # experimental
    mag_eff = 20
    microns_per_pixel = 0.8
    meas_depth = 100

    # fitting
    kx = 1
    ky = 2

    # origin centering
    z0 = 38  # ~z @ zero deflection (microns)
    z0c = 35
    y0 = 22.5  # offset from wall boundary (pixels)

    # filtering
    filter_z = 41
    cm_min = 0.5
    z_corr_min = 0
    min_num_bind = 15
    min_num_frames = 5

    # --- define beam equations
    E = 6e6  # elastic modulus of SILPURAN (Pa)
    t = 20e-6  # thickness of SILPURAN sheet (m)
    L = 2.5e-3  # width of channel (m)

    instRectPlate = functions.fRectangularUniformLoad(plate_width=L, youngs_modulus=E, plate_thickness=t)
    f_ssRectPlate = instRectPlate.rectangular_uniformly_loaded_simply_supported_plate

    # --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- --- -
    # ANALYZE IN-FOCUS CALIBRATION COORDINATES FROM SPCT

    # read calibration file
    dfc = pd.read_excel(fpcal)

    # re-center origin
    dfc['zf_from_nsv'] = dfc['zf_from_nsv'] - z0c
    dfc = flip_y(dfc, y0)

    # fit smooth 3d spline
    bispl_c, rmse_c = fit.fit_3d_spline(x=dfc.x, y=dfc.y, z=dfc.zf_from_nsv, kx=kx, ky=ky)

    # plot scatter points + fitted surface
    plot_calib_surface = False

    if plot_calib_surface:

        plot_raw_calib = False
        if plot_raw_calib:
            fig, ax = plt.subplots()

            ax.scatter(dfc.y, dfc.zf_from_nsv, c=dfc.id, s=1)
            ax.set_xlabel('y (pixels)')
            ax.set_xlim([0, dfc.y.max() * 1.05])
            ax.set_ylabel(r'$z_{f} \: (\mu m)$')
            ax.set_ylim([np.min([0, dfc.zf_from_nsv.min()]), dfc.zf_from_nsv.max() * 1.05])

            plt.tight_layout()
            plt.savefig(svp + 'calibration_raw_scatter_z_by_y_pixels.png')
            # plt.show()
            plt.close()

        fig, ax = plotting.scatter_3d_and_spline(dfc.x, dfc.y, dfc.zf_from_nsv,
                                                 bispl_c,
                                                 cmap='RdBu',
                                                 grid_resolution=30,
                                                 view='multi')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_zlabel(r'$z_{f} \: (\mu m)$')
        plt.suptitle('fit RMSE = {}'.format(np.round(rmse_c, 3)))
        plt.savefig(svp + 'calibration_fit-spline_kx{}_ky{}.png'.format(kx, ky))
        # plt.show()
        plt.close()


    # --- --- READ TESTS
    dfps = []
    for f in files:
        df = pd.read_excel(join(fdir, f))
        df = df[df['frame'] > -1]
        dfps.append(df)

    # --- --- PRECISION SWEEP

    if analyze_percent_measure_by_precision:

        data_percent_measure_by_precision = []

        for pid, df in zip(pids, dfps):

            # precision @ z

            xparam = 'id'
            pos = ['z']
            particle_ids = df[xparam].unique()
            count_column = 'counts'

            dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                                column_to_bin=xparam,
                                                                precision_columns=pos,
                                                                bins=particle_ids,
                                                                round_to_decimal=0)

            # filter out particles with precision > 2 microns
            filter_precisions = [5, 2.5, 1.25, 1, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125]

            for filt_p in filter_precisions:
                i_num_pids = len(dfp_id.id.unique())
                remove_ids = dfp_id[dfp_id[pos[0]] > filt_p].id.unique()

                dffp = df.copy()
                dffp = dffp[~dffp.id.isin(remove_ids)]

                # sampling frequency results
                num_pids = len(dffp.id.unique())
                percent_pids = num_pids / i_num_pids
                particle_density = num_pids / 512**2
                emitter_density_microns = num_pids / (512 * microns_per_pixel) ** 2
                data_percent_measure_by_precision.append([pid, filt_p, i_num_pids, num_pids, percent_pids,
                                                          particle_density, emitter_density_microns])

        # evaluate
        dfpmp = pd.DataFrame(np.array(data_percent_measure_by_precision), columns=['name',
                                                                                   'fprecision',
                                                                                   'i_num_pids',
                                                                                   'f_num_pids',
                                                                                   'percent_pids',
                                                                                   'p_density',
                                                                                   'p_density_microns',
                                                                                   ])

        # export results
        if export_precision_sweep_results:
            dfpmp.to_excel(rvp + '/percent_measure_by_precision.xlsx', index=False)

        # plot
        if save_precision_sweep_figs:
            dfpmpg = dfpmp.groupby('fprecision').mean().reset_index()

            for pc in ['percent_pids', 'p_density', 'p_density_microns']:
                fig, ax = plt.subplots()
                ax.plot(dfpmpg.fprecision, dfpmpg[pc], '-o')
                ax.set_xlabel(r'z-precision $(\mu m)$')
                ax.set_ylabel(pc)
                plt.tight_layout()
                plt.savefig(svp + '/meas-{}_by_precision_filter.png'.format(pc))
                plt.show()

    # --- --- ANALYZE ALL ROWS

    dfpsf = []
    dfms = []
    fit_parameters = []

    for pid, df in zip(pids, dfps):

        df_original = df.copy()

        # count # of measurements and particle ID's
        initial_rows, initial_pids = len(df), len(df.id.unique())

        # --- ---  --- ---  --- ---  --- --- ANALYZE ALL ROWS
        # compute r.m.s. error on fitted surface on all particle measurements
        if analyze_all_rows:

            # filter
            df = df[df['z'] > filter_z]
            df = df[df['cm'] > cm_min]

            df['z_mean_pid'] = 0
            df['num_frames_pid'] = 0
            for particle_id in df.id.unique():
                z_mean_pid = df[df['id'] == particle_id].z.mean()
                df.loc[df['id'] == particle_id, 'z_mean_pid'] = z_mean_pid

                num_frames_pid = len(df[df['id'] == particle_id].z)
                df.loc[df['id'] == particle_id, 'num_frames_pid'] = num_frames_pid

            df['z_mean_error'] = df['z_mean_pid'] - df['z']
            df = df[df['z_mean_error'].abs() < meas_depth/10]
            df = df[df['num_frames_pid'] > min_num_frames]

            # plot raw measurements (after minor filtering)
            if plot_raw:
                fig, ax = plt.subplots()
                ax.scatter(df.y, df.z, c=df.id, s=1)
                ax.set_xlabel('y (pixels)')
                ax.set_xlim([0, df.y.max() * 1.05])
                ax.set_ylabel(r'$z_{raw} \: (\mu m)$')
                plt.tight_layout()
                if save_figs is True:
                    plt.savefig(svp + 'z{}um_raw_scatter_z_by_y_microns.png'.format(pid))
                if show_figs:
                    plt.show()
                plt.close()

            # restructure
            df = center_on_origin(df, z0=z0, y0=y0)

            df = correct_z_by_spline(df, bispl_c, param_z='z')

            # filter out particles with z_corr < 0
            df = df[df['z_corr'] > z_corr_min]

            # calculate per-particle precision (ID)
            if analyze_per_particle_precision:
                xparam = 'id'
                pos = ['z_corr', 'x', 'y']
                num_bins = len(df[xparam].unique())
                count_column = 'counts'

                dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                                    column_to_bin=xparam,
                                                                    precision_columns=pos,
                                                                    bins=num_bins,
                                                                    round_to_decimal=0)

                # filter precision
                remove_ids = dfp_id[dfp_id['z_corr'] > filter_precision].id.unique()
                dfp_id = dfp_id[dfp_id['z_corr'] <= filter_precision]

                mean_precision = dfp_id['z_corr'].mean()

                # export results
                if export_precision_results:
                    dfp_id.to_excel(rvp + '/fprecision{}_per-particle_precision_initial.xlsx'.format(filter_precision),
                                    index=False)

                # plot bin(id)
                if save_precision_figs or show_precision_figs:

                    # plot z-precision by y
                    fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches))

                    ax.errorbar(dfp_id['y_m'], dfp_id['z_corr_m'], yerr=dfp_id['z_corr'],
                                ms=0.5, fmt='o', elinewidth=0.5, capsize=1, ecolor='gray')

                    ax.set_xlabel('y (pixels)')
                    ax.set_ylabel('z-precision')
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(svp + 'z{}um_fprecision{}_z-precision-by-id_deflection-errorbars.png'.format(pid, filter_precision))
                    if show_figs:
                        plt.show()
                    plt.close()

                    # plot z-precision by ID
                    for pc in pos[:1]:
                        fig, ax = plt.subplots(figsize=(size_x_inches * 2, size_y_inches))

                        ax.scatter(dfp_id[xparam], dfp_id[pc], c=dfp_id['id'], s=0.5)

                        ax.set_xlabel(xparam)
                        ax.set_ylabel('{} precision'.format(pc))

                        axr = ax.twinx()
                        axr.plot(dfp_id[xparam], dfp_id[count_column], '-s', markersize=2, alpha=0.25)
                        axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
                        axr.set_ylim([0, int(np.round(dfp_id[count_column].max() + 6, -1))])
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(svp + 'z{}um_fprecision{}_z-precision-by-id.png'.format(pid, filter_precision))
                        if show_figs:
                            plt.show()
                        plt.close()

            # --- FILTER BY PRECISION

            df = df[~df.id.isin(remove_ids)]

            # count # of final measurements and particle ID's
            final_rows, final_pids = len(df), len(df.id.unique())
            percent_meas = np.round(final_rows / initial_rows * 100, 1)
            percent_pids = np.round(final_pids / initial_pids * 100, 1)

            # plot z_corr measurements (after filtering)

            if plot_zcorr:
                fig, ax = plt.subplots()

                ax.scatter(df.y * microns_per_pixel, df.z_corr, c=df.id, s=1)
                ax.set_xlabel(r'$y \: (\mu m)$')
                ax.set_xlim([0, df.y.max() * microns_per_pixel * 1.05])
                ax.set_ylabel(r'$z_{corr} \: (\mu m)$')

                plt.suptitle(r'{} measurements ({}\%); {} particles ({}\%)'.format(final_rows,
                                                                                   percent_meas,
                                                                                   final_pids,
                                                                                   percent_pids))
                plt.tight_layout()
                if save_figs is True:
                    plt.savefig(svp + 'z{}um_all_rows_z_corr_scatter_z_by_y_microns.png'.format(pid))
                if show_figs:
                    plt.show()
                plt.close()

            # --- Fit generalized 3D surface
            if fit_general_surface:
                df = df.drop(columns=['frame', 'stack_id', 'z_true', 'max_sim'])

                # fit smooth 3d spline
                bispl_t, rmse_t = fit.fit_3d_spline(x=df.x, y=df.y, z=df.z_corr, kx=kx, ky=ky)

                if save_fitted_surface:
                    fig, ax = plotting.scatter_3d_and_spline(df.x, df.y, df.z_corr,
                                                             bispl_t,
                                                             cmap='RdBu',
                                                             view='multi')
                    ax.set_xlabel('x (pixels)')
                    ax.set_ylabel('y (pixels)')
                    ax.set_zlabel(r'$z_{corr} \: (\mu m)$')
                    plt.suptitle('fit RMSE = {}'.format(np.round(rmse_t, 3)))
                    plt.savefig(svp + 'z{}um_multi-surface-view.png'.format(pid))
                    plt.close()

                df['fit_z'] = bispl_t.ev(df.x, df.y)
                df['fit_error'] = df.fit_z - df.z_corr

                df['test_id'] = pid
                dfms.append(df)
                fit_parameters.append(bispl_t)

                df = df.drop(columns=['id'])

                # --- Calculate r.m.s. error on fitted surface
                dfrmse_m = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=1, min_cm=0.5, z_range=None,
                                              round_to_decimal=5, dropna=True, error_column='fit_error')
                mean_rmse_z_calc = dfrmse_m.rmse_z.values[0]

                dfrmse = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=20, min_cm=0.5, z_range=None,
                                              round_to_decimal=5, dropna=True, error_column='fit_error')

                dfrmse_min_binned = dfrmse[dfrmse['num_bind'] > min_num_bind]
                fig, ax = plt.subplots()
                ax.plot(dfrmse_min_binned.index, dfrmse_min_binned.rmse_z, '-o')
                ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(svp + 'z{}um_all_rows_rmse_z_bin_z-corr_fit-spline.png'.format(pid))
                if show_figs:
                    plt.show()
                plt.close()

                # save to excel
                if export_results:
                    dfrmse.to_excel(rvp + 'z{}um_all_rows_rmse_z_bin_z-corr_fit-spline.xlsx'.format(pid), index=True)
                    dict_results_all_rows = {'test_id': pid,
                                             'mag_eff': mag_eff,
                                             'microns_per_pixel': microns_per_pixel,
                                             'filter_z': filter_z,
                                             'cm_threshold': cm_min,
                                             'precision_filter': filter_precision,
                                             'cal_z0': z0c,
                                             'test_z0': z0,
                                             'boundary_offset': y0,
                                             'cal_surf_rmse': rmse_c,
                                             'initial_rows': initial_rows,
                                             'final_rows': final_rows,
                                             'percent_meas_rows': percent_meas,
                                             'initial_p_ids': initial_pids,
                                             'final_p_ids': final_pids,
                                             'percent_meas_p_ids': percent_pids,
                                             'precision': mean_precision,
                                             'test_surf_rmse': rmse_t,
                                             'test_calc_rmse': mean_rmse_z_calc,
                                             'test_surf_mean_error': df['fit_error'].mean(),
                                             }
                    dfict_results_all_rows = pd.DataFrame.from_dict(dict_results_all_rows, orient='index')
                    dfict_results_all_rows.to_excel(rvp + 'z{}um_results_all-rows.xlsx'.format(pid))

        # --- ---  --- ---  --- ---  --- --- ANALYZE ALL PARTICLE ID'S

        # compute r.m.s. error on fitted surface on all particle measurements
        if analyze_all_pids:

            df = df_original.copy()

            # filter and restructure
            df = df[df['z'] > filter_z]
            df = df[df['cm'] > cm_min]

            # recenter data on estimated origin
            dfm = modify.groupby_stats(df, group_by='id',
                                       drop_columns=['frame', 'stack_id', 'z_true', 'max_sim', 'error'])
            dfm = dfm.reset_index()
            dfm['frame'] = 1
            dfm = center_on_origin(dfm, z0=z0, y0=y0)
            dfm = correct_z_by_xy_surface(dfm, functions.smooth_surface, cal_fit_surface_params, fit_var='z')

            # filters
            dfm = dfm[dfm['z_counts'] > min_num_frames]
            dfm = dfm[dfm['z_corr'] > z_corr_min]
            dfm = dfm[dfm['z_std'] < filter_z_std]

            # count # of final measurements and particle ID's
            final_rows, final_pids = dfm.z_counts.sum(), len(dfm)
            percent_meas = np.round(final_rows / initial_rows * 100, 1)
            percent_pids = np.round(final_pids / initial_pids * 100, 1)

            # scatter plot of z_corr(x, y)
            if show_percent_figs or save_percent_figs:
                fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 2, size_y_inches))

                ax1.scatter(dfm.x, dfm.z_corr, c=dfm.id, s=1)
                ax1.set_xlabel('x (pixels)')
                ax1.set_ylabel(r'$z_{corr} \: (\mu m)$')
                ax1.set_ylim([dfm.z_corr.min(), dfm.z_corr.max()*1.05])
                ax1.set_xlim([0, dfm.x.max() * 1.05])

                ax2.scatter(dfm.y, dfm.z_corr, c=dfm.id, s=1)
                ax2.set_xlabel('y (pixels)')
                ax2.set_xlim([0, dfm.y.max()*1.05])

                plt.suptitle(r'{} measurements ({}\%); {} particles ({}\%)'.format(final_rows,
                                                                                  percent_meas,
                                                                                  final_pids,
                                                                                  percent_pids))
                plt.tight_layout()
                if save_figs is True:
                    plt.savefig(svp + 'z{}um_scatter_x_y.png'.format(pid))
                if show_figs:
                    plt.show()
                plt.close()

            # --- calculate percent overlap
            dfm = analyze.calculate_particle_to_particle_spacing(test_coords_path=dfm,
                                                                 theoretical_diameter_params_path=theory_diam_path,
                                                                 mag_eff=mag_eff,
                                                                 z_param='z_corr',
                                                                 zf_at_zero=True,
                                                                 max_n_neighbors=5,
                                                                 true_coords_path=None,
                                                                 maximum_allowable_diameter=None)

            # save to excel
            if export_results:
                dfm.to_excel(rvp + 'z{}um_percent_overlap.xlsx'.format(pid), index=False)

            # --- Fit uniformly loaded thin plate with simply boundary conditions
            if fit_beam_theory:

                # create a 'x' and 'y' column in units of microns
                dfm['x_um'] = dfm.x * microns_per_pixel
                dfm['y_um'] = dfm.y * microns_per_pixel

                dfm = dfm.sort_values('y_um')
                x_fit = dfm.x_um.to_numpy()
                y_fit = dfm.y_um.to_numpy()
                z_fit = dfm.z_corr.to_numpy()

                popt, pcov = curve_fit(f_ssRectPlate, y_fit, z_fit)

                dfm['fit_z'] = f_ssRectPlate(y_fit, *popt)
                dfm['fit_error'] = dfm.fit_z - dfm.z_corr

                # organize fit params for congruency
                rmse = None
                rr = None
                fit_surface_params = popt

                # plot scatter points + fitted surface
                fig, ax = plt.subplots()
                ax.scatter(dfm.y_um, dfm.z_corr, s=1)
                ax.plot(y_fit, f_ssRectPlate(y_fit, *popt))
                ax.set_ylabel(r'$z_{corr} \: (\mu m)$')
                ax.set_xlabel(r'$y \: (\mu m)$')
                ax.set_xlim([0, dfm.y_um.max() * 1.05])
                plt.tight_layout()
                if save_figs is True:
                    plt.savefig(svp + 'z{}um_scatter_y_and_fit_plate_ss.png'.format(pid))
                if show_figs:
                    plt.show()
                plt.close()

                if plot_fit_surface:
                    fig, ax = plotting.scatter_3d_and_surface(x_fit, y_fit, z_fit,
                                                              func=f_ssRectPlate,
                                                              func_params='y',
                                                              fit_params=popt,
                                                              cmap='RdBu',
                                                              grid_resolution=30)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.view_init(10, -10)
                    if save_figs is True:
                        plt.savefig(svp + 'z{}um_fit_rectangular_plate_ss.png'.format(pid))
                    if show_figs:
                        plt.show()
                    plt.close()

            elif fit_pid_general_surface:

                # --- Fit generalized 3D surface
                rmse, rr, fit_surface_params = fit.fit_smooth_surface(dfm, z_param='z_corr')

                x_fit = dfm.x.to_numpy()
                y_fit = dfm.y.to_numpy()
                z_fit = dfm.z_corr.to_numpy()
                data = [x_fit, y_fit]

                dfm['fit_z'] = functions.smooth_surface(data, *fit_surface_params)
                dfm['fit_error'] = dfm.fit_z - dfm.z_corr

                # plot multi view scatter + fitted surface
                if save_fitted_surface:
                    fig, ax = plotting.scatter_3d_and_surface(x=x_fit, y=y_fit, z=z_fit,
                                                              func=functions.smooth_surface,
                                                              func_params=['x', 'y'],
                                                              fit_params=fit_surface_params,
                                                              cmap='RdBu',
                                                              grid_resolution=30,
                                                              view='multi',
                                                              )
                    plt.savefig(svp + 'z{}um_multi-surface-view-all-pids.png'.format(pid))
                    plt.close()

                # plot scatter points + fitted surface
                if plot_fit_surface:
                    fig, ax = plt.subplots()
                    ax.scatter(y_fit, z_fit, s=0.125, alpha=0.75, color='gray', label='Data')
                    y_plot_surface = np.linspace(1, y_fit.max() * 1.05, 100)
                    x_plot_surface = np.ones_like(y_plot_surface) * np.mean(x_fit)
                    ax.plot(y_plot_surface,
                            functions.smooth_surface([x_plot_surface, y_plot_surface], *fit_surface_params),
                            color='black', linestyle='--', label='Fit')
                    ax.set_ylabel(r'$z_{corr} \: (\mu m)$')
                    ax.set_xlabel('y (pixels)')
                    ax.set_xlim([0, y_plot_surface.max() * 1.025])
                    ax.legend()
                    plt.tight_layout()
                    if save_figs is True:
                        plt.savefig(svp + 'z{}um_scatter_y_and_fit_plate_ss.png'.format(pid))
                    if show_figs:
                        plt.show()
                    plt.close()

            dfm['test_id'] = pid
            dfms.append(dfm)
            fit_parameters.append(fit_surface_params)

            if fit_beam_theory or fit_pid_general_surface:
                # --- Calculate r.m.s. error on fitted surface
                dfrmse = bin.bin_local_rmse_z(dfm, column_to_bin='z_corr', bins=20, min_cm=0.5, z_range=None,
                                              round_to_decimal=5, dropna=True, error_column='fit_error')

                dfrmse_min_binned = dfrmse[dfrmse['num_bind'] > min_num_bind]
                fig, ax = plt.subplots()
                ax.plot(dfrmse_min_binned.index, dfrmse_min_binned.rmse_z, '-o')
                ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(svp + 'z{}um_all_pids_rmse_z_bin_z-corr.png'.format(pid))
                if show_figs:
                    plt.show()
                plt.close()

                # save to excel
                if export_results:
                    dfrmse.to_excel(rvp + 'z{}um_all_pids_rmse_z_bin_z-corr_generalized-3d-surface.xlsx'.format(pid),
                                    index=True)
                    dict_results_all_pids = {'test_id': pid,
                                             'mag_eff': mag_eff,
                                             'microns_per_pixel': microns_per_pixel,
                                             'filter_z': filter_z,
                                             'cm_threshold': cm_min,
                                             'std_filter': filter_z_std,
                                             'cal_z0': z0c,
                                             'test_z0': z0,
                                             'boundary_offset': y0,
                                             'cal_surf_rmse': rmse_c,
                                             'initial_rows': initial_rows,
                                             'final_rows': final_rows,
                                             'percent_meas_rows': percent_meas,
                                             'initial_p_ids': initial_pids,
                                             'final_p_ids': final_pids,
                                             'percent_meas_p_ids': percent_pids,
                                             'test_surf_rmse': rmse_t,
                                             'test_surf_mean_error': dfm['fit_error'].mean(),
                                             }
                    dfict_results_all_pids = pd.DataFrame.from_dict(dict_results_all_pids, orient='index')
                    dfict_results_all_pids.to_excel(rvp + 'z{}um_results_all-pids.xlsx'.format(pid))

                # --- analyze rmse z by percent overlap
                if analyze_rmsez_by_overlap:

                    # limit percent diameter overlap to -25% (b/c all negative numbers are not overlapping here)
                    # dfm['percent_dx_diameter'] = dfm['percent_dx_diameter'].where(dfm['percent_dx_diameter'] > -0.5, -0.5)
                    dfm = dfm[dfm['percent_dx_diameter'] > -2]

                    # binning
                    columns_to_bin = ['z_corr', 'percent_dx_diameter']
                    bin_z = 3
                    bin_pdo = [-2, -1, -0.5, -0.25, 0, 0.33, 0.66]

                    dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfm,
                                                                   columns_to_bin=columns_to_bin,
                                                                   bins=[bin_z, bin_pdo],
                                                                   round_to_decimals=[4, 4],
                                                                   min_cm=0.5,
                                                                   equal_bins=[True, False],
                                                                   error_column='fit_error')

                    # Plot rmse z + number of particles binned
                    fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True,
                                                  figsize=(size_x_inches * 1.35, size_y_inches * 1.5))
                    for name, dfb in dfbicts.items():

                        dfb = dfb[dfb['num_bind'] > min_num_bind]

                        ax.plot(dfb.bin, dfb.rmse_z, '-o', label=np.round(name, 1))
                        ax2.plot(dfb.bin, dfb.num_bind, '-o')

                    ax.set_ylabel(r'$\sigma_{z} \: (\mu m$)')
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                    ax2.set_xlabel(r'$\gamma \: $(\%)')
                    ax2.set_ylabel(r'$N_{p}$')
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(svp + 'z{}um_all_pids_rmsez_num-binned_pdo.png'.format(pid))
                    if show_figs:
                        plt.show()
                    plt.close()

                    # save to excel
                    if export_results:
                        dfb_rmse_stack = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
                        dfb_rmse_stack.to_excel(rvp + 'z{}um_pids_rmse_z_bin_z-corr_and_pdo.xlsx'.format(pid),
                                        index=False)

    # export the dataframes
    # dfmstack = pd.concat(dfms, ignore_index=True)
    # dfmstack.to_excel(svp + 'all-tests_dfmstack.xlsx', index=False)

    if analyze_combined_tests:

        plot_combined_scatter = False
        if plot_combined_scatter:
            fig, ax = plt.subplots()
            for pid, dfm in zip(pids, dfms):
                ax.scatter(dfm.y, dfm.z_corr, s=0.125, label=pid)

            ax.set_xlabel('y (pixels)')
            ax.set_xlim([0, 550])
            ax.set_ylabel(r'$z_{corr} \: (\mu m)$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(svp + 'combined_scatter_plot_all-rows.png')
            plt.show()

            fig, ax = plt.subplots()
            for pid, dfm in zip(pids, dfms):
                dfmg = dfm.groupby('id').mean()
                dfmgstd = dfm.groupby('id').std()

                ax.errorbar(dfmg.y, dfmg.z_corr, yerr=dfmgstd.z_corr,
                            ms=0.25, fmt='.', elinewidth=0.25, capsize=0.5, alpha=0.75, label=pid)

            ax.set_xlabel('y (pixels)')
            ax.set_xlim([0, 550])
            ax.set_ylabel(r'$z_{corr} \: (\mu m)$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(svp + 'combined_scatter_plot_all-rows-groupby-id.png')
            plt.show()

        # plot cross-section deflections
        xloc = 225 * microns_per_pixel
        dx = 25 * microns_per_pixel
        yloc = 390 * microns_per_pixel
        dy = 20 * microns_per_pixel

        fig, [axx, axy] = plt.subplots(ncols=2, figsize=(size_x_inches * 2, size_y_inches))

        xy_slopes = []
        i = 0
        for pid, dfm in zip(pids, dfms):

            dfm['x'] = dfm.x * microns_per_pixel
            dfm['y'] = dfm.y * microns_per_pixel

            # fit smooth 3d spline
            bispl_t, rmse_t = fit.fit_3d_spline(x=dfm.x, y=dfm.y, z=dfm.z_corr)

            # --- z(x)

            # and evaluate spline at data points
            dfm = dfm.sort_values('x')
            x_fit = np.linspace(dfm.x.min(), dfm.x.max())
            y_fit = np.ones_like(x_fit) * yloc
            z_fit = bispl_t.ev(x_fit, y_fit)

            # calculate slope
            x_slope = (z_fit.max() - z_fit.min()) / (x_fit.max() - x_fit.min())
            x_angle = np.rad2deg(np.arctan((z_fit.max() - z_fit.min()) / (x_fit.max() - x_fit.min())))

            # plot fitted surface
            axx.plot(x_fit, z_fit, color='gray', alpha=0.25, linewidth=0.5)

            # scatter points
            dfmx = dfm[(dfm['y'] > yloc - dy) & (dfm['y'] < yloc + dy)]
            dfmxstd = dfmx.groupby('id').std().reset_index()
            dfmxm = dfmx.groupby('id').mean().reset_index()
            axx.scatter(dfmxm.x, dfmxm.z_corr, s=1, color=sci_colors[i])
            axx.errorbar(dfmxm.x, dfmxm.z_corr, yerr=dfmxstd.z_corr,
                        color=sci_colors[i], ms=0.5, fmt='o', elinewidth=0.5, capsize=1, alpha=0.5)

            # --- z(y)

            # and evaluate spline at data points
            dfm = dfm.sort_values('y')
            y_fit = np.linspace(40, 410)
            x_fit = np.ones_like(y_fit) * xloc
            z_fit = bispl_t.ev(x_fit, y_fit)

            # calculate slope
            y_slope = (z_fit[7] - z_fit[0]) / (y_fit[7] - y_fit[0])
            y_angle = np.rad2deg(np.arctan((z_fit[7] - z_fit[0]) / (y_fit[7] - y_fit[0])))
            xy_slopes.append([pid, x_slope, x_angle, y_slope, y_angle])

            # plot fitted surface
            axy.plot(y_fit, z_fit, color='gray', alpha=0.25, linewidth=0.5)

            dfmy = dfm[(dfm['x'] > xloc - dx) & (dfm['x'] < xloc + dx)]
            dfmystd = dfmy.groupby('id').std().reset_index()
            dfmym = dfmy.groupby('id').mean().reset_index()
            axy.scatter(dfmym.y, dfmym.z_corr, s=1, color=sci_colors[i], label=pid * 1e-3)
            axy.errorbar(dfmym.y, dfmym.z_corr, yerr=dfmystd.z_corr,
                         color=sci_colors[i], ms=0.5, fmt='o', elinewidth=0.5, capsize=1, alpha=0.5)

            i += 1

        axx.set_xlabel(r'$l \: (\mu m)$')
        # axx.set_xlim([10, 425])
        axx.set_ylabel(r'$\Delta z|_{w_0} \: (\mu m)$')

        axy.set_xlabel(r'$w \: (\mu m)$')
        axy.set_xlim([30, 420])
        axy.set_ylabel(r'$z|_{l_0} \: (\mu m)$')
        axy.legend(title=r'$\Delta h \: (mm)$', labelspacing=0.3, handletextpad=0.4)

        plt.tight_layout()
        plt.savefig(svp + 'combined_cross-section-deflection_plots_and_errorbars.png')
        plt.show()

        # export slopes
        dfslope = pd.DataFrame(xy_slopes, columns=['test_id', 'x_slope', 'x_angle_deg', 'y_slope', 'y_angle_deg'])
        dfslope.to_excel(svp + 'per-test_xy-slopes-of-fitted-spline_units-microns.xlsx', index=False)

        # plot only z(y)

        fig, axy = plt.subplots()
        i = 0

        for pid, dfm in zip(pids, dfms):

            # fit smooth 3d spline
            bispl_t, rmse_t = fit.fit_3d_spline(x=dfm.x, y=dfm.y, z=dfm.z_corr)

            dfm = dfm.sort_values('y')

            # and evaluate spline at data points
            y_fit = np.linspace(40, 410)
            x_fit = np.ones_like(y_fit) * xloc
            z_fit = bispl_t.ev(x_fit, y_fit)

            # plot fitted surface
            axy.plot(y_fit, z_fit, color='gray', alpha=0.25, linewidth=0.5)

            dfmy = dfm[(dfm['x'] > xloc - dx) & (dfm['x'] < xloc + dx)]
            dfmystd = dfmy.groupby('id').std().reset_index()
            dfmym = dfmy.groupby('id').mean().reset_index()

            axy.scatter(dfmym.y, dfmym.z_corr, s=1, color=sci_colors[i], label=pid * 1e-3)
            axy.errorbar(dfmym.y, dfmym.z_corr, yerr=dfmystd.z_corr,
                         color=sci_colors[i], ms=0.5, fmt='o', elinewidth=0.5, capsize=1, alpha=0.5)

            i += 1

        axy.set_xlabel(r'$w \: (\mu m)$')
        axy.set_xlim([30, 420])
        axy.set_ylabel(r'$z(x=l_0) \: (\mu m)$')
        axy.legend(title=r'$\Delta h \: (mm)$', labelspacing=0.3, handletextpad=0.4)

        plt.tight_layout()
        plt.savefig(svp + 'combined_deflection_plots_and_errorbars.png')
        plt.show()


# --- ---  --- ---  --- ---  --- --- ANALYZE COMBINED TEST COORDINATES

analyze_bin_z_all_tests = False

if analyze_bin_z_all_tests:

    save_figs, show_figs = False, False
    export_results = True

    # results of tests
    fp = base_dir + 'results/all-tests_dfmstack.xlsx'
    df = pd.read_excel(fp)
    df['z_corr'] = df.z_corr - 1.2

    # export mean rmse-z
    if export_results:
        dfrmse_mean = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=1, min_cm=0.5, z_range=None,
                                      round_to_decimal=4, dropna=True, error_column='fit_error')
        dfrmse_mean.to_excel(rvp + 'all-tests_dfmstack_bin-rmsez_mean-1bin.xlsx')


    bins = [2.5, 5.0, 7.5, 10.0, 12.5, 15, 20, 25]
    dfrmse = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=bins, min_cm=0.5, z_range=None,
                                  round_to_decimal=4, dropna=True, error_column='fit_error')

    fig, ax = plt.subplots()

    ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
    ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
    ax.set_xlim([0, 27.5])
    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')

    axr = ax.twinx()
    axr.plot(dfrmse.index, dfrmse.num_bind, '-d', color='gray', ms=3, alpha=0.25)
    axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')

    plt.tight_layout()
    if save_figs:
        plt.savefig(svp + 'all-tests_all-rows_rmse_z_bin_z-corr.png')
    if show_figs:
        plt.show()
    plt.close()
    if export_results:
        dfrmse.to_excel(rvp + 'all-tests_dfmstack_bin-rmsez.xlsx')

    # results of meta assessment
    fpmeta = '/Users/mackenzie/Desktop/idpt_experiments/10.07.21-BPE_Pressure_Deflection_20X/analyses/' \
             'results-04.11.22-idpt-meta-assessment/coords/test-coords/' \
             'test_coords_t_Deflection_idpt-cal_c10.07.21_BPE_sim-sym_2022-04-11 18:04:22.334060.xlsx'
    dfm = pd.read_excel(fpmeta)
    dfm['z_true'] = dfm.z_true - 41

    # export mean rmse-z
    if export_results:
        dfrmse_meta_mean = bin.bin_local_rmse_z(dfm, column_to_bin='z_true', bins=1, min_cm=0.5, z_range=None,
                                  round_to_decimal=4, dropna=True, error_column='error')
        dfrmse_meta_mean.to_excel(rvp + 'meta_bin-rmsez_mean-1bin.xlsx')

    # local bins
    dfrmse_meta = bin.bin_local_rmse_z(dfm, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None,
                                  round_to_decimal=4, dropna=True, error_column='error')

    fig, ax = plt.subplots()

    ax.plot(dfrmse.index, dfrmse.rmse_z, '-o', ms=4, label='Test')
    ax.plot(dfrmse_meta.index, dfrmse_meta.rmse_z, '-s', ms=4, label='Self-Assessment')
    ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax.legend(loc='upper left')

    axr = ax.twinx()
    axr.plot(dfrmse.index, dfrmse.num_bind, '-o', color='midnightblue', ms=2, alpha=0.35)
    axr.plot(dfrmse_meta.index, dfrmse_meta.num_bind, '-s', color='darkgreen', ms=2, alpha=0.35)
    axr.set_ylabel(r'$N_{p} \: (\#)$')

    plt.tight_layout()
    if save_figs:
        plt.savefig(svp + 'all-tests_all-rows+meta_rmse_z_bin_z-corr.png')
    if show_figs:
        plt.show()
    plt.close()
    if export_results:
        dfrmse_meta.to_excel(rvp + 'meta_bin-rmsez.xlsx')


# --- ---  --- ---  --- ---  --- --- ANALYZE META ASSESSMENT
analyze_meta = False

if analyze_meta:
    # --- --- SETUP
    meta_dir = base_dir + 'meta_coords/'
    mfp = 'meta_coords'

    # --- --- READ FILES

    # --- read meta assessment
    dfmeta = pd.read_excel(meta_dir + mfp + '.xlsx')

    # --- add column for in-focus z
    dfmeta['z_f'] = dfmeta['id']
    dfmeta = dfmeta.replace({'z_f': mapping_dict['zf_from_peak_int']})

    # --- correct z_true and z by in-focus z
    dfmeta['z_true_cf'] = dfmeta.z_true - dfmeta.z_f
    dfmeta['z_cf'] = dfmeta.z - dfmeta.z_f

    # --- filter z_true
    z_range = [-45, 50]
    dfmeta = dfmeta[(dfmeta['z_true_cf'] > z_range[0]) & (dfmeta['z_true_cf'] < z_range[1])]

    # --- plotting
    ms = 4

    # plot calibration curve
    fig, ax = plt.subplots()

    ax.scatter(dfmeta.z_true_cf, dfmeta.z_cf, s=ms / 2)

    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    plt.tight_layout()
    plt.savefig(svp + 'meta-assessment_calibration_curve.png')
    plt.show()

    # --- bin by z_true corrected by z_f
    column_to_bin = 'z_true_cf'
    bins = 25
    min_cm = 0.9
    round_to_decimal = 4

    dfrmse = bin.bin_local_rmse_z(dfmeta, column_to_bin, bins, min_cm, z_range, round_to_decimal, dropna=True)
    dfrmse = dfrmse.reset_index()

    # --- plot rmse_z of meta assessment
    h = 80

    # plot in microns
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(dfrmse.bin, dfrmse.rmse_z, '-o', markersize=ms, label=r'$\sigma_z$')
    ax2.plot(dfrmse.bin, dfrmse.percent_meas, '--d', markersize=ms, label=r'$\phi_{ID}$')

    ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylim([-0.0005, 0.3])

    ax2.set_ylabel(r'$\phi_{ID}$')
    ax2.set_ylim([50, 102])

    plt.tight_layout()
    plt.savefig(svp + 'meta-assessment_rmsez.png')
    plt.show()

    # ---

    # plot normalized by h
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(dfrmse.bin, dfrmse.rmse_z / h, '-o', markersize=ms, label=r'$\sigma_z$')
    ax2.plot(dfrmse.bin, dfrmse.percent_meas, '--d', markersize=ms, label=r'$\phi_{ID}$')

    ax.set_ylabel(r'$\sigma_z/h$')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylim([-0.000025, 0.003])

    ax2.set_ylabel(r'$\phi_{ID}$')
    ax2.set_ylim([50, 102])

    plt.tight_layout()
    plt.savefig(svp + 'meta-assessment_norm_rmsez.png')
    plt.show()

# --- ---  --- ---  --- ---  --- --- ANALYZE BY PERCENT OVERLAP
analyze_percent_overlap = False

if analyze_percent_overlap:

    # --- --- SETUP
    meta_dir = base_dir + 'meta_coords/'
    mfp = 'meta_coords'
    theory_diam_path = base_dir + 'idpt_outputs/SPCT-full-image/calibration/calib_spct_pop_defocus_stats_10.07.21-BPE_Pressure_Deflection.xlsx'
    mag_eff = 20

    # --- --- READ FILES

    # --- read meta assessment
    dfmeta = pd.read_excel(meta_dir + mfp + '.xlsx')

    # --- add column for in-focus z
    dfmeta['z_f'] = dfmeta['id']
    dfmeta = dfmeta.replace({'z_f': mapping_dict['zf_from_peak_int']})

    # --- correct z_true and z by in-focus z
    dfmeta['z_true_cf'] = dfmeta.z_true - dfmeta.z_f
    dfmeta['z_cf'] = dfmeta.z - dfmeta.z_f

    # --- filter z_true
    z_range = [-45, 50]
    dfmeta = dfmeta[(dfmeta['z_true_cf'] > z_range[0]) & (dfmeta['z_true_cf'] < z_range[1])]

    dfo = analyze.calculate_particle_to_particle_spacing(test_coords_path=dfmeta,
                                                         theoretical_diameter_params_path=theory_diam_path,
                                                         mag_eff=mag_eff,
                                                         z_param='z_cf',
                                                         zf_at_zero=True,
                                                         max_n_neighbors=5,
                                                         true_coords_path=None,
                                                         maximum_allowable_diameter=None)

    # save to excel
    # dfo.to_excel(svp + 'meta-assessment_percent_overlap.xlsx', index=False)

    # limit percent diameter overlap to -25% (not overlapping here)
    # dfo['percent_dx_diameter'] = dfo['percent_dx_diameter'].where(dfo['percent_dx_diameter'] > 0, 0)

    # --- plotting
    ms = 4

    # binning
    columns_to_bin = ['z_true_cf', 'percent_dx_diameter']
    bin_z = [-30, -15, 0, 15, 30]
    bin_pdo = 4  # [0.0, 0.25, 0.5, 0.75, 0.85, 0.95]

    dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfo,
                                                   columns_to_bin=columns_to_bin,
                                                   bins=[bin_z, bin_pdo],
                                                   round_to_decimals=[3, 4],
                                                   min_cm=0.5,
                                                   equal_bins=[False, True])

    # --- --- PLOT RMSE Z
    plot_percent_diameter_overlap = True
    markers = ['o', 'd', 's', '*', '^']

    if plot_percent_diameter_overlap:
        # Plot rmse z + number of particles binned as a function of percent diameter overlap for different z bins
        fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.35, size_y_inches * 1.5))
        i = 0
        for name, df in dfbicts.items():
            ax.plot(df.bin, df.rmse_z, '-', marker=markers[i], label=name)
            ax2.plot(df.bin, df.num_bind, '-', marker=markers[i])
            i += 1

        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        ax.set_yscale('log')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
        ax2.set_xlabel(r'$\gamma \: $(\%)')
        ax2.set_ylabel(r'$N_{p}$')
        plt.tight_layout()
        plt.savefig(svp + 'meta-assessment_rmse_z_equi_bin_percent_overlap_markers.png')
        plt.show()


# --- ---  --- ---  --- ---  --- --- ANALYZE BY SPCT STATS
analyze_spct_stats = True

if analyze_spct_stats:

    # filpaths
    base_dir = '/Users/mackenzie/Desktop/idpt_experiments/10.07.21-BPE_Pressure_Deflection_20X/analyses/' \
               'results-04.12.22_spct-meta-assessment-later'

    # read
    plot_collections.plot_spct_stats(base_dir)


# --- ---  --- ---  --- ---  --- --- ANALYZE BY SPCT META ASSESSMENT
analyze_spct_meta = False

if analyze_spct_meta:
    # filpaths
    base_dir = '/Users/mackenzie/Desktop/idpt_experiments/10.07.21-BPE_Pressure_Deflection_20X/analyses/' \
               'results-04.12.22_spct-meta-assessment-later'

    # read
    plot_collections.plot_meta_assessment(base_dir,
                                          method='spct',
                                          min_cm=0.5,
                                          min_percent_layers=0.5,
                                          microns_per_pixel=0.8,
                                          path_calib_spct_pop=None
                                          )