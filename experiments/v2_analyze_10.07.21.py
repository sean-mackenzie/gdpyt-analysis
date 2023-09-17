# imports
from os.path import join
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity

import filter, analyze
from correction import correct
from datasets import field_curvatures
from utils import io, plot_collections, bin, modify, plotting, fit, functions
from utils.plotting import lighten_color

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

"""
Blue: #0C5DA5
Green: #00B945
"""

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
sci_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

# --- experimental
microns_per_pixel = 1.6

# ---

# ----------------------------------------------------------------------------------------------------------------------
# A. SETUP FILE PATHS

dataset = 'idpt-gaussian'

if dataset == 'min-temp':
    # min-temp-pos-and-neg
    fpc = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/' \
          'analyses/results-06.13.22_spct-meta-last/coords/calib-coords/calib_spct_pid_defocus_stats_xy.xlsx'
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/' \
               'analyses/results-04.06.22-min-temp-pos-and-neg/'
    fdir = base_dir + 'coords/test-coords/combined'

elif dataset == 'idpt-gaussian':
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/' \
               'analyses/results-06.30.22-idpt-gaussian-iter2'
    fpc = base_dir + '/coords/calib-coords/calib_idpt_pid_defocus_stats_gen-cal.xlsx'
    fdir = base_dir + '/coords/test-coords'

elif dataset == 'spct':
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/' \
               'analyses/results-06.30.22-spct'
    fpc = None  # base_dir + '/coords/calib-coords/calib_idpt_pid_defocus_stats_gen-cal.xlsx'
    fdir = base_dir + '/coords/test-coords'

else:
    raise ValueError()

path_calib_coords = join(base_dir, 'coords/calib-coords')
path_test_coords = join(base_dir, 'coords/test-coords')
path_results = join(base_dir, 'results')
path_figs = join(path_results, 'figs')
path_calibration_surface = join(path_figs, 'calibration-surface')

if not os.path.exists(path_calibration_surface):
    os.makedirs(path_calibration_surface)

if not os.path.exists(path_figs):
    os.makedirs(path_figs)

# ----------------------------------------------------------------------------------------------------------------------
# B. (OPTIONAL) ASSESS QUALITY OF CORRECTION
assess_correction = False
if assess_correction:

    # --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---
    # READ FILES

    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/' \
               'analyses/results-06.13.22_spct-meta-last'
    path_calib_coords = join(base_dir, 'coords/calib-coords')
    path_test_coords = join(base_dir, 'coords/test-coords')
    path_results = join(base_dir, 'results')
    path_figs = join(base_dir, 'figs')

    # ---

    # read test coords
    dft, dfti = io.read_test_coords(path_test_coords)

    # read calib coords
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='spct')

    # ---

    # read from spct analysis
    mag_eff, zf, c1, c2 = io.read_pop_gauss_diameter_properties(dfcpop)
    if mag_eff == 10:
        microns_per_pixel = 1.6
    elif mag_eff == 20:
        microns_per_pixel = 0.8
    elif mag_eff == 5:
        microns_per_pixel = 3.2
    else:
        raise ValueError('Effective magnification is not equal to 5, 10 or 20?')

    # spct stats
    image_dimensions = (512, 512)
    area = (image_dimensions[0] * microns_per_pixel) * (image_dimensions[1] * microns_per_pixel)
    particle_ids = dfcstats.id.unique()
    num_pids = len(dfcstats.id.unique())
    num_frames = len(dfcstats.frame.unique())
    measurement_depth = dfcstats.z_corr.max() - dfcstats.z_corr.min()

    dict_penetrance = filter.get_pid_penetrance(dfti)
    dfp = dict_penetrance['dfp']
    penetrance_pids = dict_penetrance['penetrance_pids']
    penetrance_num_pids = dict_penetrance['penetrance_num_pids']

    num_lines = 20
    num_plots = int(np.floor(penetrance_num_pids / num_lines))
    zs_of_peak_int = []

    for n in range(num_plots):
        pids_this_plot = penetrance_pids[n + n * num_lines:(n + 1) * num_lines]
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.5, size_y_inches * 1.5))
        for pid in pids_this_plot:
            dfpid = dfp[dfp['id'] == pid].reset_index()

            ax1.plot(dfpid.z_true, dfpid.gauss_A, '-o', ms=1, label=pid)
            ax1.set_ylabel('A')
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
            ax2.plot(dfpid.z_true, dfpid.contour_area, '-o', ms=1)
            ax2.set_ylabel('Area')
            ax2.set_xlabel(r'$z_{true}$')

            zs_of_peak_int.append([pid, dfpid.iloc[dfpid.gauss_A.idxmax()].z_true])

        plt.tight_layout()
        plt.savefig(path_results + '/gauss_A_by_z_true_{}.svg'.format(n))
        plt.close()

    zs_of_peak_int = np.array(zs_of_peak_int)
    dfzps = pd.DataFrame(zs_of_peak_int, columns=['id', 'z_p'])
    dfzps.to_excel(path_results + '/df_z_of_peak_int_raw.xlsx')
    print(dfzps.z_p.mean())

    # ------------------------------------------- CORRECT BEFORE ANALYSIS ----------------------------------------------

    param_z = 'z'
    param_z_true = 'z_true'
    param_zf = 'zf_from_peak_int'
    kx_c = 2
    ky_c = 2
    img_xc, img_yc = 256, 256

    zf_nearest_calib_mean = dfcpid.zf_nearest_calib.mean()
    dfcpid_f = dfcpid[(dfcpid['zf_nearest_calib'] > zf_nearest_calib_mean - 5) &
                      (dfcpid['zf_nearest_calib'] < zf_nearest_calib_mean + 5)]

    dfti_corr = correct.correct_test_by_calibration_fitting(dfti,
                                                            param_z,
                                                            param_z_true,
                                                            dfcpid_f,
                                                            param_zf,
                                                            microns_per_pixel,
                                                            img_xc, img_yc,
                                                            kx_c, ky_c,
                                                            path_figs,
                                                            )

    dict_penetrance_corr = filter.get_pid_penetrance(dfti_corr)
    dfpc = dict_penetrance_corr['dfp']
    penetrance_pidsc = dict_penetrance_corr['penetrance_pids']
    penetrance_num_pidsc = dict_penetrance_corr['penetrance_num_pids']

    num_plots = int(np.floor(len(penetrance_pidsc) / num_lines))
    zs_of_peak_int = []

    for n in range(num_plots):
        pids_this_plot = penetrance_pidsc[n + n * num_lines:(n + 1) * num_lines]
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.5, size_y_inches * 1.5))
        for pid in pids_this_plot:
            dfpid = dfpc[dfpc['id'] == pid].reset_index()

            ax1.plot(dfpid.z_true_corr, dfpid.gauss_A, '-o', ms=1, label=pid)
            ax1.set_ylabel('A')
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
            ax2.plot(dfpid.z_true_corr, dfpid.contour_area, '-o', ms=1)
            ax2.set_ylabel('Area')
            ax2.set_xlabel(r'$z_{true corr}$')

            zs_of_peak_int.append([pid, dfpid.iloc[dfpid.gauss_A.idxmax()].z_true_corr])

        plt.tight_layout()
        plt.savefig(path_results + '/gauss_A_by_z_true_corr_{}.svg'.format(n))
        plt.close()

    zs_of_peak_int = np.array(zs_of_peak_int)
    dfzps = pd.DataFrame(zs_of_peak_int, columns=['id', 'z_corr_p'])
    dfzps.to_excel(path_results + '/df_z_of_peak_int_corr.xlsx')
    print(dfzps.z_corr_p.mean())

# ---

# ----------------------------------------------------------------------------------------------------------------------
# C. ANALYZE TEST COORDS
analyze_test = False
if analyze_test:

    correct_field_curvature = False
    save_figs = True
    show_figs = False

    # ------------------------------------------------------------------------------------------------------------------
    # 1. FIT CORRECTION

    if fpc is not None:
        # read spct stats
        dfc = pd.read_excel(fpc)

        # setup
        param_zf = 'zf_from_nsv'
        kx_c = 2
        ky_c = 2

        if show_figs or save_figs:
            fig, ax = plt.subplots()
            ax.scatter(dfc.x, dfc.y, c=dfc[param_zf])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.tight_layout()
            if save_figs:
                plt.savefig(path_calibration_surface + '/raw_calibration-in-focus_{}.png'.format(param_zf))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # step 1. correct field curvature
        if correct_field_curvature:
            bispl_fc = field_curvatures.get_field_curvature(dataset='20X-0.5X', path_results=path_results)
            dfc = correct.correct_z_by_spline(dfc, bispl_fc, param_z=param_zf)

        # ---

        # step 2. fit bivariate spline to in-focus points
        bispl_c, rmse_c = fit.fit_3d_spline(x=dfc.x,
                                            y=dfc.y,
                                            z=dfc[param_zf],
                                            kx=kx_c,
                                            ky=ky_c,
                                            )

        # plot fit + points
        if save_figs or show_figs:
            fig, ax = plotting.scatter_3d_and_spline(dfc.x, dfc.y, dfc[param_zf],
                                                     bispl_c,
                                                     cmap='RdBu',
                                                     view='multi')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_zlabel(r'$z_{f} \: (\mu m)$')
            plt.suptitle('Calibration fit RMSE = {}'.format(np.round(rmse_c, 3)))
            if save_figs:
                plt.savefig(
                    path_calibration_surface + '/calibration_multi-surface-view_kx{}ky{}_{}.png'.format(kx_c, ky_c,
                                                                                                        param_zf))
            if show_figs:
                plt.show()
            plt.close()

    else:
        bispl_c = None

    # ---

    # -------------------------------------------------------------------------------------------------------------------
    # 2. ANALYZE TEST COORDS

    save_figs = True
    show_figs = False
    export_results = True
    calculate_precision = True
    calculate_rmse = True

    # --- ---  --- ---  --- ---  --- --- SETUP

    files = [f for f in os.listdir(fdir) if f.endswith('.xlsx')]
    test_ids = [float(xf.split('z')[-1].split('um.xlsx')[0]) for xf in files]

    test_ids = sorted(test_ids)
    files = sorted(files, key=lambda x: float(x.split('z')[-1].split('um.xlsx')[0]))

    # ---

    # --- ---  --- ---  --- ---  --- --- READ POSITIVE TEST COORDS
    df_errors = []
    df_precisions = []

    for tid, f in zip(test_ids, files):

        df = pd.read_excel(join(fdir, f))
        df = df[df['frame'] > -1]

        # initial stats
        i_num = len(df)

        # ---

        # FILTER #1: Z > 41 MICRONS --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  ---
        df = df[df['z'] > 40]

        f_num = len(df)
        percent_removed = 1 - f_num / i_num
        print("{} rows ({} \%) out of {} removed from {}.".format(i_num - f_num, percent_removed, i_num, tid))

        # FILTER #2: cm >
        # df = df[df['cm'] > 0.925]
        # --- ---  --- ---  --- ---  --- --- --- ---  --- ---  --- ---  --- --- --- ---  --- --- --- --- ---  --- ------

        # PROCESSING #1 AND 2: FLIP Y DIRECTION + RAISE UP 3 MICRONS --  --- ---  --- ---  --- --- --- ---  --- ---  ---

        # ---

        # plot raw test coords
        if save_figs or show_figs:
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 2, size_y_inches))
            ax1.scatter(df.y * microns_per_pixel, df.z, c=df.x, s=1)
            ax1.set_xlabel(r'$y \: (\mu m)$')
            ax1.set_ylabel(r'$z_{raw} \: (\mu m)$')
            ax2.scatter(df.x * microns_per_pixel, df.z, c=df.y, s=1)
            ax2.set_xlabel(r'$x \: (\mu m)$')
            plt.tight_layout()
            if save_figs:
                plt.savefig(path_figs + '/H={}_raw_scatter.svg'.format(tid))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # PROCESSING # 3 - CORRECT TEST COORDS FROM CALIBRATION IN-FOCUS COORDS ---------- --- --- -- --- --- - ---  ---

        # step 2. correct from bispl
        if bispl_c is not None:
            if correct_field_curvature:
                df_fc_corrected = correct.correct_z_by_spline(df, bispl_fc, param_z='z')
                df_bspl_corrected = correct.correct_z_by_spline(df_fc_corrected, bispl_c, param_z='z_corr')
            else:
                df_bspl_corrected = correct.correct_z_by_spline(df, bispl_c, param_z='z')

            # plot spline corrected test coords
            if save_figs or show_figs:
                fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches, size_y_inches))
                ax1.scatter(df_bspl_corrected.y * microns_per_pixel, df_bspl_corrected.z_corr, c=df_bspl_corrected.id, s=1)
                ax1.set_xlabel(r'$y \: (\mu m)$')
                ax1.set_ylabel(r'$z_{corr} \: (\mu m)$')
                ax2.scatter(df_bspl_corrected.x * microns_per_pixel, df_bspl_corrected.z_corr, c=df_bspl_corrected.id, s=1)
                ax2.set_xlabel(r'$x \: (\mu m)$')
                plt.suptitle('H = {}, calib bspl r.m.s.e = {}'.format(tid, np.round(rmse_c, 3)))
                plt.tight_layout()
                if save_figs:
                    plt.savefig(
                        path_figs + '/H={}_bspl-kx{}ky{}-{}-corrected_scatter.svg'.format(tid, kx_c, ky_c, param_zf))
                if show_figs:
                    plt.show()
                plt.close()

            # overwrite raw coords for simplicity
            df = df_bspl_corrected

        # --- ---  --- ---  --- ---  --- --- --- - ---  --- --- ---  --- ---  --- ---  --- --- --- - ---  --- ----------

        # ---

        # PROCESSING # 4 - FIT 2D SPLINE TO CALIBRATION-CORRECTED TEST COORDS ---------- --- ---  --- ---  -- ---  -----

        # --- Fit generalized 3D surface
        if 'z_corr' not in df.columns:
            df['z_corr'] = df['z']

        # fit smooth 3d spline
        kx, ky = 2, 2
        bispl_t, rmse_t = fit.fit_3d_spline(x=df.x, y=df.y, z=df.z_corr, kx=kx, ky=ky)

        # plot fit + points
        if save_figs or show_figs:
            fig, ax = plotting.scatter_3d_and_spline(df.x, df.y, df.z_corr,
                                                     bispl_t,
                                                     cmap='RdBu',
                                                     view='multi')
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_zlabel(r'$z_{corr} \: (\mu m)$')
            plt.suptitle('fit RMSE = {}'.format(np.round(rmse_t, 3)))
            if save_figs:
                plt.savefig(path_figs + '/z{}um_multi-surface-view_kx{}-ky{}.png'.format(tid, kx, ky))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # --- Calculate error
        df['fit_z'] = bispl_t.ev(df.x, df.y)
        df['fit_error'] = df.fit_z - df.z_corr
        df['test_id'] = tid

        # --- store results
        df_errors.append(df)

        # export test coords with 'error' column
        if export_results:
            df.to_excel(join(path_results, f), index=False)

        # ---

        # --- Calculate precision
        if calculate_precision:

            precision_data = []
            for pid in df.id.unique():
                # this pid
                dfpid = df[df['id'] == pid]
                # mean
                mx = dfpid.xm.mean()
                my = dfpid.ym.mean()
                mz = dfpid.z_corr.mean()
                # precision
                px = functions.calculate_precision(dfpid.xm.to_numpy())
                py = functions.calculate_precision(dfpid.ym.to_numpy())
                pz = functions.calculate_precision(dfpid.z_corr.to_numpy())
                # store
                precision_data.append([tid, pid, mx, my, mz, px, py, pz])

            dfprecision = pd.DataFrame(np.array(precision_data),
                                       columns=['test_id', 'id', 'mx', 'my', 'mz', 'px', 'py', 'pz'])
            df_precisions.append(dfprecision)

        # ---

        # --- Calculate r.m.s. error on fitted surface
        if calculate_rmse:
            dfrmse_m = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=1, min_cm=0.5, z_range=None,
                                            round_to_decimal=5, dropna=True, error_column='fit_error')
            dfrmse_m['true_percent_meas'] = dfrmse_m['num_meas'] / i_num

            dfrmse = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=20, min_cm=0.5, z_range=None,
                                          round_to_decimal=4, dropna=True, error_column='fit_error')

            dfrmse_m.to_excel(path_results + '/mean_rmse_H={}.xlsx'.format(tid))
            dfrmse.to_excel(path_results + '/local_rmse_H={}.xlsx'.format(tid))

        # ---

    # ---

    # after analyzing all tests, export the results
    if len(df_errors) > 0:
        df_errors = pd.concat(df_errors)
        if 'z_cal_surf' in df_errors.columns:
            df_errors = df_errors.drop(
                columns=['stack_id', 'z_true', 'max_sim', 'error', 'z_cal_surf'])  # 'temp_z_corr', 'bin'
        else:
            df_errors = df_errors.drop(columns=['stack_id', 'z_true', 'max_sim', 'error'])
        df_errors.to_excel(path_results + '/df_errors_combined.xlsx', index=False)

    if len(df_precisions) > 0:
        df_precisions = pd.concat(df_precisions)

        # compute lateral precision in pixels and microns
        df_precisions['pxy_pixels'] = np.sqrt(df_precisions['px'] ** 2 + df_precisions['py'] ** 2)
        df_precisions['pxy_microns'] = df_precisions['pxy_pixels'] * microns_per_pixel

        # export
        df_precisions.to_excel(path_results + '/df_precisions_combined.xlsx', index=False)

        # compute mean precision
        df_precisions['bin'] = 1
        df_precision_mean = df_precisions.groupby('bin').mean()
        df_precision_mean.to_excel(path_results + '/df_precisions_mean.xlsx', index=False)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# D. EVALUATE TEST COORD ANALYSIS
analyze_results = False
if analyze_results:

    df = pd.read_excel(path_results + '/df_errors_combined.xlsx')

    # --- Calculate r.m.s. error on fitted surface
    total_rmse = True
    if total_rmse:
        dfrmse_m = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=1, min_cm=0.5, z_range=None,
                                        round_to_decimal=5, dropna=True, error_column='fit_error')

        bins_z = np.linspace(4, 29.5, 18)
        dfrmse = bin.bin_local_rmse_z(df, column_to_bin='z_corr', bins=bins_z, min_cm=0.5, z_range=None,
                                      round_to_decimal=4, dropna=True, error_column='fit_error')

        dfrmse_m.to_excel(path_results + '/total_mean_rmse.xlsx')
        dfrmse.to_excel(path_results + '/total_local_rmse.xlsx')

        # plot rmse-z
        fig, ax = plt.subplots()
        ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(path_results + '/total_local_rmse_z_by_z.svg')
        plt.show()

    # ---

    # Calculate total z-precision.
    total_precision = True
    if total_precision:
        dfp = pd.read_excel(path_results + '/df_precisions_combined.xlsx')
        x_precision_mean = dfp.px.mean()
        y_precision_mean = dfp.py.mean()
        z_precision_mean = dfp.pz.mean()
        print("mean precision (x, y, z) = ({}, {}, {}) pixels".format(np.round(x_precision_mean, 4),
                                                                       np.round(y_precision_mean, 4),
                                                                       np.round(z_precision_mean, 4)))

    # ---

    # plot rmse_z and precision as 2D scatter
    plot_2d_surface = True
    if plot_2d_surface:
        # --- plot 2D uncertainty scatter

        # duplicate for plotting purposes
        df_surface = df.copy()

        # calculate r.m.s.e. z
        df_surface['sqerr'] = df_surface['fit_error'] ** 2
        df_surface = df_surface.groupby('id').mean()
        df_surface['rmse_z'] = np.sqrt(df_surface['sqerr'])

        # get data arrays
        x = df_surface.x.to_numpy() * microns_per_pixel
        y = df_surface.y.to_numpy() * microns_per_pixel
        z = df_surface.rmse_z.to_numpy()

        # get the range of points for the 2D surface space
        plt.scatter(x, y, c=z, s=2, cmap='coolwarm')
        plt.colorbar(label=r'$\sigma_{z} \: (\mu m)$')
        plt.xlabel(r'$x \: (\mu m)$')
        plt.ylabel(r'$y \: (\mu m)$')
        plt.savefig(path_results + '/rmse_z_by_xy_2d-surface.svg')
        plt.tight_layout()
        plt.show()
        plt.close()

        # ---

        # --- precision
        dfp = pd.read_excel(path_results + '/df_precisions_combined.xlsx')

        # --- plot 2D uncertainty surface

        # get data arrays
        x = dfp.mx.to_numpy() * microns_per_pixel
        y = dfp.my.to_numpy() * microns_per_pixel
        z = dfp.pz.to_numpy()

        # get the range of points for the 2D surface space
        plt.scatter(x, y, c=z, s=2, cmap='coolwarm')
        plt.colorbar(label=r'$\nu_{z} \: (\mu m)$')
        plt.xlabel(r'$x \: (\mu m)$')
        plt.ylabel(r'$y \: (\mu m)$')
        plt.savefig(path_results + '/precision-z_by_xy_2d-surface.svg')
        plt.tight_layout()
        plt.close()

    # ---

    # plot kernel density estimation
    plot_kde = True
    if plot_kde:
        # df = df[df['fit_error'].abs() < 5]

        x = df.z_corr.to_numpy()
        y = df.fit_error.to_numpy()
        fig = fig = plt.figure()
        color = None
        colormap = 'coolwarm'
        scatter_size = 0.5
        kde = True

        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.075, hspace=0.075)

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        if color is not None:
            ax.scatter(x, y, c=color, cmap=colormap, s=scatter_size)
        else:
            ax.scatter(x, y, s=scatter_size, color='black')

        # vertical and horizontal lines denote the mean value
        # ax.axvline(np.mean(x), ymin=0, ymax=0.5, color='black', linestyle='--', linewidth=0.25, alpha=0.25)
        # ax.axhline(np.mean(y), xmin=0, xmax=0.5, color='black', linestyle='--', linewidth=0.25, alpha=0.25)

        # x
        binwidth_x = 2
        xlim_low = (int(np.min(x) / binwidth_x) - 1) * binwidth_x  # + binwidth_x
        xlim_high = (int(np.max(x) / binwidth_x) + 1) * binwidth_x
        xbins = np.arange(xlim_low, xlim_high + binwidth_x, binwidth_x)

        # y
        binwidth_y = 0.5
        ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
        ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
        ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)

        nx, binsx, patchesx = ax_histx.hist(x, bins=xbins, zorder=2.5, color='gray')
        ny, binsy, patchesy = ax_histy.hist(y, bins=ybins, orientation='horizontal', color='gray', zorder=2.5)

        # kernel density estimation
        if kde:
            distance_from_mean_x, distance_from_mean_y = 15, 2
            # x_plot = np.linspace(np.mean(x) - distance_from_mean_x, np.mean(x) + distance_from_mean_x, 1000)
            # y_plot = np.linspace(np.mean(y) - distance_from_mean_x, np.mean(y) + distance_from_mean_x, 1000)
            x_plot = np.linspace(np.min(x), np.max(x), 500)
            y_plot = np.linspace(np.min(y), np.max(y), 500)

            x = x[:, np.newaxis]
            x_plot = x_plot[:, np.newaxis]
            kde_x = KernelDensity(kernel="gaussian", bandwidth=2.5).fit(x)
            log_dens_x = kde_x.score_samples(x_plot)
            scale_to_max = np.max(nx) / np.max(np.exp(log_dens_x))
            # ax_histx.fill(x_plot[:, 0], np.exp(log_dens_x) * scale_to_max, fc='lightsteelblue', zorder=2)
            p1 = ax_histx.fill_between(x_plot[:, 0], 0, np.exp(log_dens_x) * scale_to_max,
                                       fc="None", ec=scired, zorder=2.5)
            p1.set_linewidth(0.5)
            # ax_histx.plot(x_plot[:, 0], np.exp(log_dens_x) * scale_to_max, linestyle='-', color=scired, zorder=3.5)
            ax_histx.set_ylabel('counts')

            y = y[:, np.newaxis]
            y_plot = y_plot[:, np.newaxis]
            kde_y = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(y)
            log_dens_y = kde_y.score_samples(y_plot)
            scale_to_max = np.max(ny) / np.max(np.exp(log_dens_y))
            p2 = ax_histy.fill_betweenx(y_plot[:, 0], 0, np.exp(log_dens_y) * scale_to_max,
                                        fc="None", ec=scired, zorder=2.5)
            p2.set_linewidth(0.5)
            # ax_histy.plot(y_plot[:, 0], np.exp(log_dens_y) * scale_to_max, linestyle='-', color=scired)
            ax_histy.set_xlabel('counts')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        fig.subplots_adjust(bottom=0.1, left=0.1)  # adjust space between axes
        plt.savefig(path_results + '/kde_error-z_by_z.svg')
        plt.show()

    # ---

    # plot scatter z(y)
    plot_multi_scatter = True
    if plot_multi_scatter:

        def flip_y(df, y0):
            df['y'] = (df['y'] - 512) * -1 + y0
            return df


        fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))
        test_ids = sorted(df['test_id'].unique())
        for tid, cmap_id in zip(test_ids, ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']):
            dft = df[df['test_id'] == tid]
            dft = flip_y(dft, 0)

            c = dft.x.to_numpy()
            c = np.flip(c)
            print(np.min(c))

            ax.scatter(dft.y * microns_per_pixel, dft.z_corr, c=c, s=10, cmap=cmap_id, label=np.round(tid / 1000, 2))
        ax.set_xlabel(r'$y \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\Delta H \: (mm)$', markerscale=1.5)
        plt.tight_layout()
        plt.savefig(path_results + '/combined_z_by_y.svg')
        plt.show()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# F. PLOT MISC. SPCT STATS
plot_misc_spct = False
if plot_misc_spct:
    path_results = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/' \
                   'analyses/results-04.06.22-min-temp-pos-and-neg/results'
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/analyses/' \
         'results-04.12.22_spct-meta-assessment-later/coords/calib-coords/' \
         'calib_spct_stats_10.07.21-BPE_Pressure_Deflection_spct-cal.xlsx'

    df = pd.read_excel(fp)
    dfg = df.groupby('z_true').mean().reset_index()

    zf = 40.5
    microns_per_pixel = 1.6
    dfg['z_corr'] = dfg['z_true'] - zf

    fig, ax1 = plt.subplots()
    ax1.scatter(dfg.z_corr, dfg.diameter_contour * microns_per_pixel, s=2, label=r'$d_{e, contour}$')
    ax1.scatter(dfg.z_corr, dfg.min_dx * microns_per_pixel, s=2, marker='s', color=sciorange,
                label=r'$\overline{\delta x}_{min}$')
    ax1.set_ylabel(r'$\mu m$')
    ax1.set_ylim([2, 78])
    ax1.set_xlabel(r'$z \: (\mu m)$')
    ax1.legend()
    plt.tight_layout()
    plt.savefig(path_results + '/contour-dia_and_dx_by_z.svg')
    plt.show()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# G. COMPARE DEFLECTION SLICE TO COMSOL
analyze_deflection_slice = False
if analyze_deflection_slice:

    # read COMSOL coordinates
    cmsl_y1 = pd.read_excel(path_results + '/comsol/y-slice-small-def.xlsx')
    cmsl_y2 = pd.read_excel(path_results + '/comsol/y-slice-medium-def.xlsx')
    cmsl_y3 = pd.read_excel(path_results + '/comsol/y-slice-large-def.xlsx')

    # read IDPT coordinates
    df_orig = pd.read_excel(path_results + '/df_errors_combined.xlsx')

    # setup
    y0 = 100  # y offset
    z0 = 2.875  # z offset
    dx, dy = 25, 25

    # plot cross-section deflections
    xlocs = np.linspace(0, 512, 9)  # 225
    ylocs = np.linspace(0, 512, 9)  # 390

    for xloc in xlocs:
        df = df_orig.copy()

        # processing
        df.loc[:, 'y'] = (df.loc[:, 'y'] - 512) * -1
        df.loc[:, 'y'] = df.loc[:, 'y'] * microns_per_pixel
        df.loc[:, 'y'] = df.loc[:, 'y'] + y0
        df = df.sort_values('y')

        df.loc[:, 'fit_z'] = df.loc[:, 'fit_z'] - z0
        df.loc[:, 'z_corr'] = df.loc[:, 'z_corr'] - z0

        # ---

        # plot only z(y)

        fig, axy = plt.subplots()

        # plot COMSOL slice
        axy.plot(cmsl_y1.y, cmsl_y1.z, color='black', alpha=0.5, label='COMSOL')
        axy.plot(cmsl_y2.y, cmsl_y2.z, color='black', alpha=0.5)
        axy.plot(cmsl_y3.y, cmsl_y3.z, color='black', alpha=0.5)

        i = 0

        for tid in df.test_id.unique():
            # get dataframe for this test
            dft = df[df['test_id'] == tid]

            # get particles in this slice
            dfty = dft[(dft['x'] > xloc - dx) & (dft['x'] < xloc + dx)]

            # ---

            # plot fitted surface
            axy.plot(dfty.y, dfty.fit_z, color='gray', alpha=0.5, linewidth=0.75)

            dfty_m = dfty.groupby('id').mean().reset_index()
            dfty_std = dfty.groupby('id').std().reset_index()

            axy.scatter(dfty_m.y, dfty_m.z_corr,
                        s=1, color=sci_colors[i], label=tid * 1e-3)
            axy.errorbar(dfty_m.y, dfty_m.z_corr, yerr=dfty_std.z_corr,
                         color=sci_colors[i], ms=0.5, fmt='o', elinewidth=0.5, capsize=1, alpha=0.5)

            i += 1

        axy.set_xlabel(r'$w \: (\mu m)$')
        # axy.set_xlim([30, 420])
        axy.set_ylabel(r'$z(x=l_0) \: (\mu m)$')
        axy.legend(title=r'$\Delta h \: (mm)$', bbox_to_anchor=(1, 1), labelspacing=0.3, handletextpad=0.4)

        plt.tight_layout()
        plt.savefig(path_results + '/comsol/y-slice-def_errorbars+COMSOL_xc{}.svg'.format(xloc))
        plt.show()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# H. COMPARE DEFLECTION SLICE TO COMSOL
compare_comsol_surface = True
if compare_comsol_surface:

    save_figs = True
    show_figs = False

    # COMSOL path
    path_results_cmsl_surface = path_results + '/comsol/surfaces'

    # read IDPT coordinates
    df_orig = pd.read_excel(path_results + '/df_errors_combined.xlsx')
    df_orig['x'] = df_orig.x * microns_per_pixel
    df_orig['y'] = df_orig.y * microns_per_pixel

    # mirror across x-y plane
    df_orig['xc'] = df_orig['y']
    df_orig['yc'] = df_orig['x']

    # mirror across x plane
    df_orig['yc'] = df_orig['yc'] * -1 + 512 * microns_per_pixel

    # translate
    df_orig['xc'] = df_orig['xc'] + 381
    df_orig['yc'] = df_orig['yc'] - 1329.2

    # ---

    # compute z_offset minimum
    compute_z_offset = False
    if compute_z_offset:

        # setup
        test_ids = df_orig.test_id.unique()
        cmsl_Hs = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5,
                   10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14]

        # stack dataframes into a list
        df_tids = []
        for tid in test_ids:
            df_tids.append(df_orig[df_orig['test_id'] == tid])
        df_cmsls = []
        for cmsl_H in cmsl_Hs:
            cmsl = pd.read_excel(path_results_cmsl_surface + '/xlsx/surf_H{}umDef.xlsx'.format(cmsl_H))
            cmsl = cmsl * 1e6
            df_cmsls.append(cmsl)

        # ---

        def fit_3d_surfaces_by_z_offset(df, bspl, guess_z_offset=2.5):
            """ popt, img_norm = fit_2d_gaussian_on_image(img, normalize, bkg_mean) """

            XYZ = df[['xc', 'yc', 'z_corr']].to_numpy()

            def bspl_offset(xy, z_offset):
                return bspl.ev(xy[:, 0], xy[:, 1]) + z_offset

            # fit surface
            try:
                popt, pcov = curve_fit(bspl_offset, XYZ[:, :2], XYZ[:, 2], guess_z_offset)
            except RuntimeError:
                popt = None

            # calculate fit error
            if popt is not None:
                res = popt[0]
                fit_results = bspl_offset(xy=XYZ[:, :2], z_offset=res)
                rmse, r_squared = fit.calculate_fit_error(fit_results, data_fit_to=XYZ[:, 2])
            else:
                res, rmse, r_squared = None, None, None

            return res, rmse, r_squared


        def compute_rmse_z_by_z_offset(df, bispl, z_i, z_f, num_z, show_fig=True):
            """ min_rmse_z_offset, min_rmse_z = compute_rmse_z_by_z_offset(df, bispl, z_i, z_f, num_z=500, show_fig=True)"""

            z_offsets = np.linspace(z_i, z_f, num_z)
            rmse_zs = []
            for z_offset in z_offsets:
                df_off = df.copy()
                df_off['z_corr'] = df_off['z_corr']
                df_off['fit_z'] = bispl.ev(df_off.xc, df_off.yc) + z_offset
                df_off['fit_error'] = df_off.fit_z - df_off.z_corr
                df_off['sqerr'] = df_off['fit_error'] ** 2
                df_off = df_off.groupby('id').mean()
                df_off['rmse_z'] = np.sqrt(df_off['sqerr'])
                mean_rmse_z = df_off.rmse_z.mean()
                rmse_zs.append(mean_rmse_z)

            min_rmse_z = np.min(rmse_zs)
            min_rmse_z_offset = z_offsets[np.argmin(rmse_zs)]

            if show_fig:
                fig, ax = plt.subplots()
                ax.plot(z_offsets, rmse_zs)
                ax.set_xlabel('z offset')
                ax.set_ylabel('rmse z')
                ax.set_title('Min rmse z = {} at {}'.format(np.round(min_rmse_z, 4),
                                                            np.round(min_rmse_z_offset, 4))
                             )
                plt.show()

            return min_rmse_z_offset, min_rmse_z


        # ---
        data = []
        for tid, dft in zip(test_ids, df_tids):
            print("Analyzing test id {}.".format(tid))

            for cmsl_H, cmsl in zip(cmsl_Hs, df_cmsls):
                print("Analyzing H = {}.".format(cmsl_H))

                # fit smooth 3d spline
                kx, ky = 2, 2
                bispl_cmsl, rmse_cmsl = fit.fit_3d_spline(x=cmsl.x, y=cmsl.y, z=cmsl.dz, kx=kx, ky=ky)

                # use scipy.optimize.curve_fit
                z_offset_min, rmse_min, r_squared_min = fit_3d_surfaces_by_z_offset(dft, bispl_cmsl, guess_z_offset=2.5)

                # compute via for loop
                z_offset_comp, rmse_comp = compute_rmse_z_by_z_offset(dft, bispl_cmsl, z_i=-15, z_f=25, num_z=1000,
                                                                           show_fig=False)

                datum = [tid, cmsl_H, rmse_cmsl, z_offset_min, z_offset_comp, rmse_min, rmse_comp, r_squared_min]
                data.append(datum)

        df_res = pd.DataFrame(np.array(data),
                              columns=['tid', 'H', 'bspl_H_rmse',
                                       'z_o_min', 'z_o_comp', 'rmse_min', 'rmse_comp', 'rr_min'])
        df_res.to_excel(path_results_cmsl_surface + '/minimized-z-offset.xlsx')

    # ---

    # evaluate z_offset
    evaluate_z_offset = False
    if evaluate_z_offset:
        df_res = pd.read_excel(path_results_cmsl_surface + '/minimized-z-offset.xlsx')

        for tid in df_res.tid.unique():
            dfrt = df_res[df_res['tid'] == tid].reset_index()

            ms = 1
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            ax1.plot(dfrt.H, dfrt.rmse_min, 'o', ms=ms, label='Min.')
            ax1.plot(dfrt.H, dfrt.rmse_comp, 's', ms=ms, label='Comp.')
            ax1.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            ax1.set_title('Min r.m.s.e. (Min., Comp.) = ({}, {}) at ({}, {})'.format(
                np.round(dfrt.rmse_min.min(), 3),
                np.round(dfrt.rmse_comp.min(), 3),
                np.round(dfrt.iloc[dfrt.rmse_min.idxmin()].z_o_min, 3),
                np.round(dfrt.iloc[dfrt.rmse_comp.idxmin()].z_o_comp, 3),
            )
            )

            ax2.plot(dfrt.H, dfrt.z_o_min, 'o', ms=ms, label='Min.')
            ax2.plot(dfrt.H, dfrt.z_o_comp, 's', ms=ms, label='Comp.')
            ax2.set_ylabel(r'$z_{offset} \: (\mu m)$')
            ax2.set_xlabel(r'$H$')

            plt.tight_layout()
            plt.savefig(path_results_cmsl_surface + '/tid{}_rmse_z_offset_by_H.svg'.format(tid))
            plt.show()

    # ---

    # plot x-y scatter to confirm coordinate matching
    plot_sidebyside_scatter = False
    if plot_sidebyside_scatter:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(size_x_inches * 2, size_y_inches))
        ax1.scatter(cmsl.x, cmsl.y, c=cmsl.dz, cmap='viridis', vmin=0, vmax=35)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('COMSOL')
        ax2.scatter(df.xc, df.yc, c=df.z_corr, cmap='viridis', vmin=0, vmax=35)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Experiment')
        plt.tight_layout()
        plt.savefig(path_results_cmsl_surface + '/side-by-side.svg')
        plt.show()
        plt.close()

    # ---

    # plot IDPT test coords and fitted surface for each test
    plot_optimal_surface = False
    if plot_optimal_surface:

        # setup
        kx, ky = 2, 2

        # read z_offset optimization data
        df_res = pd.read_excel(path_results_cmsl_surface + '/minimized-z-offset.xlsx')

        # step through each test id
        test_ids = sorted(df_orig.test_id.unique())
        z_offsets = []
        rmses = []
        kde_xs_wrt_cmsl = []
        kde_xs_wrt_calib = []
        kde_ys = []
        dfs = []

        for tid in test_ids:

            # get results from z_offset minimization
            dfrt = df_res[df_res['tid'] == tid].reset_index()
            cmsl_H = dfrt.iloc[dfrt.rmse_min.idxmin()].H
            z_offset = dfrt.iloc[dfrt.rmse_min.idxmin()].z_o_min
            rmse_pred = dfrt.rmse_min.min()
            r_squared_pred = dfrt.iloc[dfrt.rmse_min.idxmin()].rr_min

            # ---

            # get test surface data and perform offset correction
            df = df_orig[df_orig['test_id'] == tid].reset_index()

            # correct IDPT data using optimal z_offset
            df['z_corr_offset'] = df['z_corr'] - z_offset

            # ---

            # get COMSOL surface data
            if cmsl_H % 1 == 0:
                read_cmsl_H = int(cmsl_H)
            else:
                read_cmsl_H = cmsl_H

            cmsl = pd.read_excel(path_results_cmsl_surface + '/xlsx/surf_H{}umDef.xlsx'.format(read_cmsl_H))
            cmsl = cmsl * 1e6

            # fit spline to COMSOL
            bispl_cmsl, rmse_cmsl = fit.fit_3d_spline(x=cmsl.x, y=cmsl.y, z=cmsl.dz, kx=kx, ky=ky)

            # ---

            # plot fit
            plot_fit = False
            if plot_fit:
                fig, ax = plotting.scatter_3d_and_spline(cmsl.x, cmsl.y, cmsl.dz,
                                                         bispl_cmsl,
                                                         cmap='RdBu',
                                                         view='multi',
                                                         units=r'$(\mu m)$')

                plt.suptitle('fit RMSE = {}'.format(np.round(rmse_cmsl, 4)))
                if save_figs:
                    plt.savefig(path_results_cmsl_surface + '/fit-bspl_surf_H{}umDef_multi-surface-view.png'.format(cmsl_H))
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # compute error
            df['fit_z'] = bispl_cmsl.ev(df.xc, df.yc)
            df['fit_error'] = df.z_corr_offset - df.fit_z
            df['sqerr'] = df['fit_error'] ** 2
            dfg = df.groupby('id').mean()
            dfg['rmse_z'] = np.sqrt(dfg['sqerr'])
            mean_rmse_z = dfg.rmse_z.mean()

            # store results
            z_offsets.append(z_offset)
            rmses.append(mean_rmse_z)
            kde_xs_wrt_cmsl.append(df.z_corr_offset.to_numpy())
            kde_xs_wrt_calib.append(df.z_corr.to_numpy())
            kde_ys.append(df.fit_error.to_numpy())
            dfs.append(df)

            # ---

            # plot fit + points
            plot_fit_and_data = False
            if plot_fit_and_data:
                fig, ax = plotting.scatter_3d_and_spline(df.xc, df.yc, df.z_corr_offset,
                                                         bispl_cmsl,
                                                         cmap='RdBu',
                                                         view='multi',
                                                         units=r'$(\mu m)$')

                plt.suptitle('COMSOL, Experiment RMSE = {}, {}'.format(np.round(rmse_cmsl, 4), np.round(mean_rmse_z, 4)))
                if save_figs:
                    plt.savefig(
                        path_results_cmsl_surface +
                        '/compare_tid{}_surf_H{}umDef_zOffset{}_3d-surface-multi.png'.format(tid,
                                                                                             cmsl_H,
                                                                                             np.round(z_offset, 1),
                                                                                             )
                    )
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # per-test plot kernel density estimation
            plot_kde_on_cmsl = False
            if plot_kde_on_cmsl:

                # data to plot
                x = df.z_corr_offset.to_numpy()
                y = df.fit_error.to_numpy()

                # setup
                kde = True
                binwidth_x, bandwidth_x = 2.5, 2.5
                binwidth_y, bandwidth_y = 1, 0.5
                color = None
                colormap = 'coolwarm'
                scatter_size = 0.5
                fig = plt.figure()

                # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
                # the size of the marginal axes and the main axes in both directions.
                # Also adjust the subplot parameters for a square plot.
                gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.075, hspace=0.075)

                ax = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                # no labels
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)

                # the scatter plot:
                if color is not None:
                    ax.scatter(x, y, c=color, cmap=colormap, s=scatter_size)
                else:
                    ax.scatter(x, y, s=scatter_size, color='black')

                # now determine nice limits by hand:

                # x
                xlim_low = (int(np.min(x) / binwidth_x) - 1) * binwidth_x # - binwidth_x * 2
                xlim_high = (int(np.max(x) / binwidth_x) + 1) * binwidth_x + binwidth_x
                xbins = np.arange(xlim_low, xlim_high + binwidth_x, binwidth_x)

                # y
                ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y - binwidth_y
                ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y + binwidth_y
                ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)

                nx, binsx, patchesx = ax_histx.hist(x, bins=xbins, zorder=2.5, color='gray')
                ny, binsy, patchesy = ax_histy.hist(y, bins=ybins, orientation='horizontal', color='gray',
                                                    zorder=2.5)

                # kernel density estimation
                if kde:
                    x_plot = np.linspace(np.min(x) - binwidth_x, np.max(x) + binwidth_x * 2, 500)
                    y_plot = np.linspace(np.min(y) - binwidth_y, np.max(y) + binwidth_y, 500)

                    x = x[:, np.newaxis]
                    x_plot = x_plot[:, np.newaxis]
                    kde_x = KernelDensity(kernel="gaussian", bandwidth=bandwidth_x).fit(x)
                    log_dens_x = kde_x.score_samples(x_plot)
                    scale_to_max = np.max(nx) / np.max(np.exp(log_dens_x))
                    p1 = ax_histx.fill_between(x_plot[:, 0], 0, np.exp(log_dens_x) * scale_to_max,
                                               fc="None", ec=scired, zorder=2.5)
                    p1.set_linewidth(0.5)
                    ax_histx.set_ylabel('counts')

                    y = y[:, np.newaxis]
                    y_plot = y_plot[:, np.newaxis]
                    kde_y = KernelDensity(kernel="gaussian", bandwidth=bandwidth_y).fit(y)
                    log_dens_y = kde_y.score_samples(y_plot)
                    scale_to_max = np.max(ny) / np.max(np.exp(log_dens_y))
                    p2 = ax_histy.fill_betweenx(y_plot[:, 0], 0, np.exp(log_dens_y) * scale_to_max,
                                                fc="None", ec=scired, zorder=2.5)
                    p2.set_linewidth(0.5)
                    ax_histy.set_xlabel('counts')

                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                fig.subplots_adjust(bottom=0.1, left=0.1)  # adjust space between axes
                plt.savefig(path_results_cmsl_surface + '/tid{}_surf_H{}umDef_kde_error-z_by_z.svg'.format(tid, cmsl_H))
                plt.show()

            # ---

        # ---

        # export combined dataframe w/ errors with respect to COMSOL surface
        dfs = pd.concat(dfs)
        dfs.to_excel(path_results_cmsl_surface + '/df_errors_combined_wrt_cmsl_coords.xlsx')

        # ---

        # plot z_offsets and rmse_zs by test_id
        plot_rmse_comparison = False
        if plot_rmse_comparison:
            test_ids = np.array(test_ids) * 1e-3

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

            ax1.scatter(test_ids, z_offsets)
            ax1.set_ylabel(r'$z_{offset} \: (\mu m)$')
            ax1.set_ylim([5.25, 7.75])
            ax1.set_yticks([5.5, 6.5, 7.5])
            ax1.set_title(r'$z_{offset} = $' +
                         ' {}'.format(np.round(np.mean(z_offsets), 2)) +
                         r'$\pm$' +
                         ' {}'.format(np.round(np.std(z_offsets), 2))
                         )

            ax2.scatter(test_ids, rmses)
            ax2.set_xlabel(r'$\Delta H \: (mm)$')
            ax2.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            ax2.set_ylim([0.55, 0.95])
            ax2.set_yticks([0.6, 0.7, 0.8, 0.9])
            ax2.set_title(r'$| \overline{\sigma_{z}} | = $' +
                         ' {}'.format(np.round(np.mean(rmses), 2)) +
                         r'$\pm$' +
                         ' {}'.format(np.round(np.std(rmses), 2))
                         )

            plt.tight_layout()
            plt.savefig(path_results_cmsl_surface + '/compare_z_offset_and_rmse_z_by_test_id.svg')
            plt.show()
            plt.close()

        # ---

    # ---

    # evaluate IDPT error on COMSOL fitted surface
    evaluate_optimal_surface_results = False
    if evaluate_optimal_surface_results:

        dfs = pd.read_excel(path_results_cmsl_surface + '/df_errors_combined_wrt_cmsl_coords.xlsx')

        # plot local rmse_z for combined tests (includes z_offset so rmse_z wrt COMSOL coordinates)
        plot_combined_local_rmse_wrt_cmsl = False
        if plot_combined_local_rmse_wrt_cmsl:

            print("{} rows analyzed for global and local rmse_z".format(len(dfs)))

            dfrmse_m = bin.bin_local_rmse_z(dfs, column_to_bin='z_corr_offset', bins=1, min_cm=0.5, z_range=None,
                                            round_to_decimal=5, dropna=True, error_column='fit_error')

            bins_z = 12  # np.linspace(4, 29.5, 18)
            dfrmse = bin.bin_local_rmse_z(dfs, column_to_bin='z_corr_offset', bins=bins_z, min_cm=0.5, z_range=None,
                                          round_to_decimal=4, dropna=True, error_column='fit_error')

            dfrmse_m.to_excel(path_results_cmsl_surface + '/total_mean_rmse_wrt_cmsl_coords.xlsx')
            dfrmse.to_excel(path_results_cmsl_surface + '/total_local_rmse_wrt_cmsl_coords.xlsx')

            # plot rmse-z
            fig, ax = plt.subplots()
            ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(path_results_cmsl_surface + '/total_local_rmse_z_by_z_wrt_cmsl_coords_bins{}.svg'.format(bins_z))
            plt.show()

        # ---

        # plot local rmse_z for combined tests (z_corr wrt calibration in-focus coords)
        plot_combined_local_rmse_wrt_calib_coords = True
        if plot_combined_local_rmse_wrt_calib_coords:

            print("{} rows analyzed for global and local rmse_z".format(len(dfs)))

            dfrmse_m = bin.bin_local_rmse_z(dfs, column_to_bin='z_corr', bins=1, min_cm=0.5, z_range=None,
                                            round_to_decimal=5, dropna=True, error_column='fit_error')

            bins_z = np.linspace(4, 29, 13)
            dfrmse = bin.bin_local_rmse_z(dfs, column_to_bin='z_corr', bins=bins_z, min_cm=0.5, z_range=None,
                                          round_to_decimal=4, dropna=True, error_column='fit_error')

            dfrmse_m.to_excel(path_results_cmsl_surface + '/total_mean_rmse_wrt_idpt-in-focus-coords.xlsx')
            dfrmse.to_excel(path_results_cmsl_surface + '/total_local_rmse_wrt_idpt-in-focus-coords.xlsx')

            # plot rmse-z
            fig, ax = plt.subplots()
            ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(path_results_cmsl_surface +
                        '/total_local_rmse_z_by_z_wrt_idpt-in-focus-coords_bins{}.svg'.format(len(bins_z)))
            plt.show()

        # ---

        # compare all: kernel density estimation
        plot_all_kde_on_cmsl = True
        if plot_all_kde_on_cmsl:

            x_wrt_cmsl = dfs.z_corr_offset.to_numpy()
            x_wrt_calib = dfs.z_corr.to_numpy()
            y_errs = dfs.fit_error.to_numpy()

            # setup
            kde = True
            binwidth_x, bandwidth_x = 5, 5
            binwidth_y, bandwidth_y = 1, 0.5
            color = None
            colormap = 'coolwarm'
            scatter_size = 0.5

            # plot wrt to each: COMSOL and calib coordinates
            for x, y, save_id in zip([x_wrt_cmsl, x_wrt_calib], [y_errs, y_errs], ['wrt_cmsl', 'wrt_calib']):

                fig = plt.figure()

                # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
                # the size of the marginal axes and the main axes in both directions.
                # Also adjust the subplot parameters for a square plot.
                gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.075, hspace=0.075)

                ax = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                # no labels
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)

                # the scatter plot:
                if color is not None:
                    ax.scatter(x, y, c=color, cmap=colormap, s=scatter_size)
                else:
                    ax.scatter(x, y, s=scatter_size, color='black')

                # now determine nice limits by hand:

                # x
                xlim_low = (int(np.min(x) / binwidth_x) - 1) * binwidth_x + binwidth_x * 1
                xlim_high = (int(np.max(x) / binwidth_x) + 1) * binwidth_x  # + binwidth_x
                xbins = np.arange(xlim_low, xlim_high + binwidth_x, binwidth_x)

                # y
                ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # - binwidth_y
                ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y + binwidth_y
                ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)

                # histogram
                nx, binsx, patchesx = ax_histx.hist(x, bins=xbins, zorder=2.5, color='gray')
                ny, binsy, patchesy = ax_histy.hist(y, bins=ybins, orientation='horizontal', color='gray',
                                                    zorder=2.5)

                # kernel density estimation
                if kde:
                    x_plot = np.linspace(np.min(x) - binwidth_x * 1.25, np.max(x) + binwidth_x * 1, 500)
                    y_plot = np.linspace(np.min(y) - binwidth_y * 1, np.max(y) + binwidth_y * 1, 500)

                    x = x[:, np.newaxis]
                    x_plot = x_plot[:, np.newaxis]
                    kde_x = KernelDensity(kernel="gaussian", bandwidth=bandwidth_x).fit(x)
                    log_dens_x = kde_x.score_samples(x_plot)
                    scale_to_max = np.max(nx) / np.max(np.exp(log_dens_x))
                    p1 = ax_histx.fill_between(x_plot[:, 0], 0, np.exp(log_dens_x) * scale_to_max,
                                               fc="None", ec=scired, zorder=2.5)
                    p1.set_linewidth(0.5)
                    ax_histx.set_ylabel('counts')

                    y = y[:, np.newaxis]
                    y_plot = y_plot[:, np.newaxis]
                    kde_y = KernelDensity(kernel="gaussian", bandwidth=bandwidth_y).fit(y)
                    log_dens_y = kde_y.score_samples(y_plot)
                    scale_to_max = np.max(ny) / np.max(np.exp(log_dens_y))
                    p2 = ax_histy.fill_betweenx(y_plot[:, 0], 0, np.exp(log_dens_y) * scale_to_max,
                                                fc="None", ec=scired, zorder=2.5)
                    p2.set_linewidth(0.5)
                    ax_histy.set_xlabel('counts')

                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                fig.subplots_adjust(bottom=0.1, left=0.1)  # adjust space between axes
                plt.savefig(path_results_cmsl_surface + '/plot_all_kde_error-z_by_z_{}.svg'.format(save_id))
                plt.show()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# I. COMPARE LOCAL RMSE-Z EXPERIMENTAL TO SYNTHETIC DZ-OVERLAP
compare_experiment_to_synthetic = False
if compare_experiment_to_synthetic:

    # setup
    theta_limit = 5

    # read paths

    # experimental local rmse-z
    path_results_cmsl_surface = path_results + '/comsol/surfaces'
    dfe = pd.read_excel(path_results_cmsl_surface + '/total_local_rmse_wrt_idpt-in-focus-coords.xlsx')

    # synthetic local rmse-z
    fp_synthetic = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/results/' \
                   'final/idpt-dz-overlap-theta-less-than-{}/results/mean_rmse-z_bin_z-true.xlsx'.format(theta_limit)
    dfs = pd.read_excel(fp_synthetic)

    # save paths
    path_results_cmsl_compare = path_results + '/comsol/surfaces/compare-to-synthetic'
    if not os.path.exists(path_results_cmsl_compare):
        os.makedirs(path_results_cmsl_compare)

    # ---

    # setup
    x = 'bin'
    y = 'rmse_z'
    y2 = 'cm'
    clr_exp = lighten_color(sciblue, 0.90)
    clr_syn = lighten_color(sciblue, 1.25)
    ms = 4

    # ---

    # processing

    # 1. flip z-coordinates
    dfs[x] = dfs[x] * -1 - 4

    # 2. remove extraneous data points
    z_min, z_max = -4, 30
    dfe = dfe[(dfe[x] > z_min) & (dfe[x] < z_max)]
    dfs = dfs[(dfs[x] > z_min) & (dfs[x] < z_max)]

    # ---

    # plot rmse-z
    fig, ax = plt.subplots()
    ax.plot(dfe[x], dfe[y], '-o', color=clr_exp, label='Experiment', zorder=3.2)
    ax.plot(dfs[x], dfs[y], '-o', color=clr_syn, label='Synthetic', zorder=3.1)
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_results_cmsl_compare +
                '/compare_local_rmse_z_by_z_wrt_idpt-in-focus-coords_syn-theta-lim{}.svg'.format(theta_limit))
    plt.show()

    # ---

    # plot rmse-z + c_m
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(dfe[x], dfe[y2], '-o', color=clr_exp, zorder=3.2)
    ax1.plot(dfs[x], dfs[y2], '-o', color=clr_syn, zorder=3.1)
    ax1.set_ylabel(r'$c_{m}^{\delta}$')

    ax2.plot(dfe[x], dfe[y], '-o', color=clr_exp, label='Experiment', zorder=3.2)
    ax2.plot(dfs[x], dfs[y], '-o', color=clr_syn, label='Synthetic', zorder=3.1)
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path_results_cmsl_compare +
                '/compare_local_rmse_z_and_cm_by_z_wrt_idpt-in-focus-coords_syn-theta-lim{}.svg'.format(theta_limit))
    plt.show()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# J. ANALYZE PARTICLE IMAGE OVERLAP
analyze_overlap = False
if analyze_overlap:

    # --- file paths

    # calibration coords with measured contour diameter
    path_calib_spct_stats = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/10.07.21-BPE_Pressure_Deflection_20X/analyses/' \
                            'results-04.12.22_spct-meta-assessment-later/coords/calib-coords/calib_spct_stats_10.07.21-BPE_Pressure_Deflection_spct-cal.xlsx'

    # calibration coords with maximum number of particles identified for ground_truth overlapping calculation
    path_ground_truth_positions = path_calib_coords + '/calib_idpt_pid_defocus_stats_gen-cal.xlsx'

    # tests
    fp_combined_tests_with_errors = base_dir + '/results/comsol/surfaces/df_errors_combined_wrt_cmsl_coords.xlsx'

    # save paths
    path_overlap = join(path_results, 'overlap')
    if not os.path.exists(path_overlap):
        os.makedirs(path_overlap)

    # ---

    # --- READ TEST COORDS
    # NOTE: x-y units are in microns
    df = pd.read_excel(fp_combined_tests_with_errors)

    # ---

    # --- COMPUTE OVERLAP
    compute_overlap = True
    show_fit_plot = True

    # setup
    param_diameter = 'contour_diameter'
    max_n_neighbors = 5
    xy_units = 'pixels'
    xy_unit_label = r' $(pix.)$'

    path_overlap = join(path_overlap, 'units_in_' + xy_units)
    if not os.path.exists(path_overlap):
        os.makedirs(path_overlap)

    # compute overlap
    if compute_overlap:

        # setup
        fit_z_dist = 40

        if xy_units == 'microns':
            scale_diameter_by = microns_per_pixel
        else:
            scale_diameter_by = None

        # fit contour diameter
        if show_fit_plot:
            popt_contour, fig, ax = analyze.fit_contour_diameter(path_calib_spct_stats,
                                                                 fit_z_dist,
                                                                 show_plot=True,
                                                                 scale_diameter_by=scale_diameter_by,
                                                                 )
            ax.set_xlim([-45, 45])
            ax.set_xticks([-40, -20, 0, 20, 40])
            ax.set_ylabel('Contour diameter' + ' ' + xy_unit_label)
            plt.tight_layout()
            plt.savefig(path_overlap + '/fit-spct-contour-diameter_units={}.png'.format(xy_units))
            plt.show()
        else:
            popt_contour = analyze.fit_contour_diameter(path_calib_spct_stats,
                                                        fit_z_dist,
                                                        show_plot=False,
                                                        scale_diameter_by=scale_diameter_by,
                                                        )

        # ---

        # calculate overlap for each test individually (b/c overlap calc. is per frame)
        dfois = []
        for tid in df.test_id.unique():
            dft = df[df['test_id'] == tid]

            # calculate overlap
            dfo = analyze.calculate_particle_to_particle_spacing(
                test_coords_path=dft,
                theoretical_diameter_params_path=None,
                mag_eff=10.0,
                z_param='z_corr',
                zf_at_zero=True,
                zf_param=None,
                max_n_neighbors=max_n_neighbors,
                true_coords_path=path_ground_truth_positions,
                maximum_allowable_diameter=None,
                popt_contour=popt_contour,
                param_percent_diameter_overlap=param_diameter,
                microns_per_pixels=['invert', microns_per_pixel],
                path_save_true_coords_overlap=path_overlap,
                id_save_true_coords_overlap=tid,
            )

            dfois.append(dfo)

        # ---

        dfois = pd.concat(dfois)

        # export combined overlap coords
        dfois.to_excel(path_overlap + '/df-combined_overlap-wrt-calib_errors-wrt-cmsl.xlsx', index=False)

    else:
        dfois = pd.read_excel(path_overlap + '/df-combined_overlap-wrt-calib_errors-wrt-cmsl.xlsx')

    # ---

    # add z_true just to go along with rmse-computing-functions formatting
    if 'z_true' not in dfois.columns:
        dfois['z_true'] = dfois['z_corr']

    # ---

    compute_aggregated = True
    if compute_aggregated:

        df = dfois
        tid = 'combined'

        # error column - NOTE: THIS NEEDS TO BE CHANGED. IT'S STUPID.
        if 'error' in df.columns:
            df = df.drop(columns=['error'])
        df['error'] = df['fit_error']

        # plot setup
        param_z = 'z_corr'
        z_range = None
        min_cm = 0.0

        # binning
        bin_z = np.linspace(df[param_z].min(), df[param_z].max(), 10)[0:-1]
        round_z = 3
        bin_min_dx = np.arange(5, 60, 5)  # units = microns: np.arange(2.5, 38, 2.5)
        min_num_per_bin = 50

        save_plots = True
        show_plots = True

        path_results_id = join(path_overlap, str(tid))
        if not os.path.exists(path_results_id):
            os.makedirs(path_results_id)

        # ---

        # plot mean rmse-z by z_true
        plot_rmse_by_z_true = True
        if plot_rmse_by_z_true:

            # export mean rmse: bin = 1
            dfbm = bin.bin_local_rmse_z(df,
                                        column_to_bin='z_corr',
                                        bins=1,
                                        min_cm=min_cm,
                                        z_range=z_range,
                                        round_to_decimal=3,
                                        df_ground_truth=None,
                                        include_xy=False,
                                        )
            dfbm.to_excel(path_results_id + '/mean_rmse-z.xlsx')

            # export local rmse results:
            dfi = bin.bin_local_rmse_z(df,
                                       column_to_bin='z_corr',
                                       bins=bin_z,
                                       min_cm=min_cm,
                                       z_range=z_range,
                                       round_to_decimal=3,
                                       df_ground_truth=None,
                                       include_xy=False,
                                       )
            dfi = dfi.reset_index()
            dfi = dfi.rename(columns={'index': 'bin'})
            dfi = dfi.round({'bin': 2})
            dfi.to_excel(path_results_id + '/total_local_rmse-z.xlsx')

            # setup
            px = 'bin'
            plot_columns = ['percent_meas', 'cm', 'num_meas']
            plot_column_labels = [r'$\phi_{ID} \: (\%)$', r'$c_{m}$', r'$N_{p} \: (\#)$']
            ms = 4

            # plot
            for pc, pl in zip(plot_columns, plot_column_labels):
                fig, ax = plt.subplots()
                ax.plot(dfi[px], dfi.rmse_z, '-o', color=sciblue, label='IDPT')
                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                # ax.set_ylim(bottom=0.245, top=2.45)

                axr = ax.twinx()
                axr.plot(dfi[px], dfi[pc], '--', color=sciblue)
                axr.set_ylabel(pl)
                axr.set_ylim(bottom=0)

                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_results_id + '/compare_rmse-z_and_{}_by_z-corr.png'.format(pc))
                if show_plots:
                    plt.show()
                plt.close()

            # ---

        # ---

        plot_2d_binning = True

        if plot_2d_binning:

            # filter
            bin_z_wide = np.array([1, 2, 3, 4, 5, 6, 7]) * 5
            bin_min_dx_wide = np.array([1, 2, 3, 4, 5, 6, 7]) * 5
            min_num_per_bin = 50

            # 2D binning: rmse_z (dx, z_true)
            columns_to_bin = ['min_dx', 'z_corr']
            clrs = iter(cm.magma(np.linspace(0.05, 0.95, len(bin_min_dx))))

            dfbicts_2d = analyze.evaluate_2d_bin_local_rmse_z(df,
                                                              columns_to_bin=columns_to_bin,
                                                              bins=[bin_min_dx_wide, bin_z_wide],
                                                              round_to_decimals=[0, 2],
                                                              min_cm=min_cm,
                                                              equal_bins=[False, False],
                                                              error_column='error',
                                                              include_xy=False,
                                                              )

            # export results
            dfbicts_2d_stacked = modify.stack_dficts_by_key(dfbicts_2d, drop_filename=False)
            dfbicts_2d_stacked.to_excel(join(path_results_id, 'rmse-z_by_z-corr_min-dx_2d-bin.xlsx'))

            # plot
            fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
            for name_dz, dfix_rmse in dfbicts_2d.items():
                dfix_rmse_plot = dfix_rmse[dfix_rmse['num_meas'] > min_num_per_bin]  # NOTE: this MIGHT change the df.
                ax.plot(dfix_rmse_plot.bin, dfix_rmse_plot.rmse_z, '-o', ms=2, color=next(clrs),
                        label=np.round(name_dz, 1))

            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\sigma_z \: (\mu m)$')
            ax.legend(loc='upper left', ncol=2, title=r'$\delta x_{min} \: $' + xy_unit_label)
            plt.tight_layout()
            plt.savefig(join(path_results_id, 'all_2d-bin-z-min-dx_plot-rmse-z.png'))
            if show_plots:
                plt.show()
            plt.close()

        # ---

        # Figure: multi plot - percent overlap, error, diameter
        plot_overlap_multi = True
        if plot_overlap_multi:
            ms, ms2 = 4, 2
            sciblue_mod, scigreen_mod = 1, 1
            plot_columns = [['cm'],
                            ['contour_diameter', 'mean_dx'],
                            ['contour_diameter', 'min_dx'],
                            ['num_meas'],
                            ['percent_meas'],
                            ]
            plot_labels = [[r'$c_{m}$'],
                           [r'$\bullet \: d_{e} \: $' + xy_unit_label, r'$\diamond \: \overline{\delta x} \: $' + xy_unit_label],
                           [r'$\bullet \: d_{e} \: $' + xy_unit_label, r'$\diamond \: \delta x_{min} \: $' + xy_unit_label],
                           [r'$N_{p} \: (\#)$'],
                           [r'$\phi_{ID}$'],
                           ]

            # binning
            bin_pdo = np.round(np.linspace(0.125, 2.125, 12), 3)
            num_bins = 6
            round_pdo = 2

            # filters
            error_limits = [None]  # 5
            depth_of_focuss = [None]  # 7.5
            max_overlap = None  # bin_pdo[-1]  # 1.125

            for error_limit, depth_of_focus in zip(error_limits, depth_of_focuss):

                # apply filters
                if error_limit is not None:
                    df = df[df['error'].abs() < error_limit]
                if depth_of_focus is not None:
                    raise ValueError()  # dfoi = dfoi[(dfoi['z_true'] < -depth_of_focus) | (dfoi['z_true'] > depth_of_focus)]
                if max_overlap is not None:
                    df = df[df['percent_dx_diameter'] < max_overlap]

                # compute rmse-z; bin by min dx
                dfoib = bin.bin_local_rmse_z(df=df, column_to_bin='min_dx', bins=bin_min_dx, min_cm=min_cm,
                                             z_range=z_range, round_to_decimal=round_z)

                # remove bins with < min_num_per_bin measurements
                dfoib = dfoib[dfoib['num_meas'] > min_num_per_bin]

                # ---

                # plot: cm, percent measure, rmse-z
                ms = 3
                fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True,
                                                    figsize=(size_x_inches * 0.6, size_y_inches * 1.265))
                ax1.plot(dfoib.index, dfoib.cm, '-o', ms=ms, label='IDPT')
                ax1.set_ylabel(r'$c_{m}$')
                # ax1.set_ylim([0.89, 1.01])
                # ax1.set_yticks([0.9, 1.0])
                # ax1.set_xticks([-50, 0, 50], [])
                ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
                ax1.tick_params(axis='y', which='minor', left=False, right=False)
                ax1.legend(markerscale=0.75, handlelength=1, borderpad=0.2, labelspacing=0.25,
                           handletextpad=0.4, borderaxespad=0.25)

                ax2.plot(dfoib.index, dfoib.percent_meas, '-o', ms=ms, zorder=2)
                ax2.set_ylabel(r'$\phi_{ID} \: (\%)$')
                ax2.set_ylim([-5, 105])
                ax2.set_yticks([0, 50, 100])
                # ax1.set_xticks([-50, 0, 50], [])
                ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
                ax2.tick_params(axis='y', which='minor', left=False, right=False)

                ax3.plot(dfoib.index, dfoib.rmse_z, '-o', ms=ms, zorder=2)
                ax3.set_xlabel(r'$\delta x_{min} \: $' + xy_unit_label)
                ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                # ax3.set_ylim([0.875, 4.125])
                # ax3.set_yticks([1, 2, 3, 4])
                # ax3.set_xlim([-5, 68])
                # ax3.set_xticks([0, 20, 40, 60])
                ax3.tick_params(axis='x', which='minor', bottom=False, top=False)
                ax3.tick_params(axis='y', which='minor', left=False, right=False)

                plt.tight_layout()
                if save_plots:
                    plt.savefig(
                        path_results_id + '/idpt_rmsez_by_min_dx_erlim{}dof{}_small.svg'.format(error_limit,
                                                                                                depth_of_focus,
                                                                                                )
                    )
                if show_plots:
                    plt.show()
                plt.close('all')

                # plot - multi
                for pc, pl in zip(plot_columns, plot_labels):
                    fig, [axr, ax] = plt.subplots(nrows=2, sharex=True)
                    ax.plot(dfoib.index, dfoib.rmse_z, '-o', ms=ms, label='IDPT')
                    ax.set_xlabel(r'$\delta x_{min} \: $' + xy_unit_label)
                    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                    ax.legend()

                    pi, = axr.plot(dfoib.index, dfoib[pc[0]], '-o', ms=ms, zorder=2)
                    axr.set_ylabel(pl[0])

                    if len(pc) == 2:
                        axrr = axr.twinx()
                        axrr.plot(dfoib.index, dfoib[pc[1]], '-D',
                                  color=lighten_color(pi.get_color(), sciblue_mod),
                                  mfc='white',
                                  mec=lighten_color(pi.get_color(), sciblue_mod),
                                  ms=ms2,
                                  lw=1,
                                  zorder=1.5,
                                  )
                        axrr.set_ylabel(pl[1])

                        # make y-limits consistent
                        axr.set_ylim(bottom=2.5)
                        axrr.set_ylim(bottom=2.5)

                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(
                            path_results_id + '/idpt-spct_rmsez_by_min_dx_erlim{}dof{}_{}.png'.format(error_limit,
                                                                                                      depth_of_focus,
                                                                                                      pc,
                                                                                                      )
                        )
                    if show_plots:
                        plt.show()
                    plt.close('all')

                # bin by percent diameter overlap
                plot_overlap = True
                if plot_overlap:
                    dfoib = bin.bin_local_rmse_z(df=df, column_to_bin='percent_dx_diameter', bins=bin_pdo,
                                                 min_cm=min_cm,
                                                 z_range=z_range, round_to_decimal=round_z, df_ground_truth=None)

                    # remove bins with < min_num_per_bin measurements
                    dfoib = dfoib[dfoib['num_meas'] > min_num_per_bin]

                    # plot
                    for pc, pl in zip(plot_columns, plot_labels):
                        fig, [axr, ax] = plt.subplots(nrows=2, sharex=True)
                        ax.plot(dfoib.index, dfoib.rmse_z, '-o', ms=ms, label='IDPT')
                        ax.set_xlabel(r'$\overline{\varphi} \: (\%)$')
                        ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                        # ax.set_ylim(ylim_rmse)
                        ax.legend()

                        pi, = axr.plot(dfoib.index, dfoib[pc[0]], 'o', ms=ms2, zorder=2)
                        axr.set_ylabel(pl[0])

                        if len(pc) == 2:
                            axrr = axr.twinx()
                            axrr.plot(dfoib.index, dfoib[pc[1]], 'D',
                                      color=lighten_color(pi.get_color(), sciblue_mod),
                                      mfc='white',
                                      mec=lighten_color(pi.get_color(), sciblue_mod),
                                      ms=ms2,
                                      lw=0.5,
                                      zorder=1.5,
                                      )
                            axrr.set_ylabel(pl[1])

                            # make y-limits consistent
                            axr.set_ylim(bottom=2.5, top=45)
                            axr.set_yticks([10, 20, 30, 40])
                            axrr.set_ylim(bottom=2.5, top=45)
                            axrr.set_yticks([10, 20, 30, 40])

                        plt.tight_layout()
                        if save_plots:
                            plt.savefig(
                                path_results_id + '/idpt-spct_rmsez_by_pdo_erlim{}dof{}_{}.png'.format(error_limit,
                                                                                                       depth_of_focus,
                                                                                                       pc,
                                                                                                       )
                            )
                        if show_plots:
                            plt.show()
                        plt.close('all')
        # ---

# ---

print("analysis completed without errors.")