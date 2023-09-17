# meta-assessment - 11.06.21_z-micrometer-v2
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

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
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# TEST COORDS
"""
IDPT:
base_dir = ''

SPCT:
base_dir = ''
"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
           'results-08.23.22_meta-spct/calib_1umSteps_test_5umSteps'

method = 'spct'

test_name = 'test_coords_particle_image_stats_spct'
padding = 5
calib_id_from_calibset = 42
calib_id_from_testset = 42
z_inspect = [10, 35]  # note, this will include both: (-35, -10) and (10, 35)

path_test_coords = join(base_dir, 'coords/test-coords')
path_test_coords_post_processed = path_test_coords + '/post-processed'

path_calib_coords = join(base_dir, 'coords/calib-coords')
path_similarity = join(base_dir, 'similarity')
path_results = join(base_dir, 'results')
path_results_combined = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# ----------------------------------------------------------------------------------------------------------------------
# 0. SETUP PROCESS CONTROLS

# subject to change
true_num_particles_per_frame = 88
baseline_frame = 50
z_zero_from_calibration = 50.0

# experimental
mag_eff = 10.01
NA_eff = 0.45
microns_per_pixel = 1.6
size_pixels = 16  # microns
depth_of_field = functions.depth_of_field(mag_eff, NA_eff, 600e-9, 1.0, size_pixels * 1e-6) * 1e6
num_pixels = 512
area_pixels = num_pixels ** 2
img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding
area_microns = (num_pixels * microns_per_pixel) ** 2

# processing
z_range = [-50, 50]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
num_frames_per_step = 3
filter_barnkob = measurement_depth / 10
min_cm = 0.5
min_percent_layers = 0.5
remove_ids = None

# ---

# ----------------------------------------------------------------------------------------------------------------------
# A. EVALUATE STAGE TILT AND FIELD CURVATURE ON CALIBRATION COORDS

# read test coords (where z is based on 1-micron steps and z_true is 5-micron steps every 3 frames)
evaluate_initial_surface = True
correct_stage_tilt = True
correct_field_curvature = True


# ---

# iterator
param_zfs = ['zf_from_peak_int', 'zf_from_nsv_signal', 'zf_from_nsv', 'zf_from_dia']
flips = [True, False]

mean_error_results = []
std_error_results = []

# iterate
for param_zf in param_zfs:

    # fit plane

    def fit_plane_and_bispl(path_figs):
        # read coords
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='spct')
        dfc = dfcpid

        # fit spline to 'raw' data
        bispl_raw, rmse = fit.fit_3d_spline(x=dfc.x,
                                            y=dfc.y,
                                            z=dfc[param_zf],
                                            kx=2,
                                            ky=2)

        # return 3 fits to actual data
        dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_and_tilt_corrected, bispl = \
            correct.fit_plane_correct_plane_fit_spline(dfc,
                                                       param_zf,
                                                       microns_per_pixel,
                                                       img_xc,
                                                       img_yc,
                                                       kx=2,
                                                       ky=2,
                                                       path_figs=None)

        # return faux flat plane
        faux_zf = 'faux_zf_flat'
        dfc[faux_zf] = dfc['zf_from_nsv'].mean()
        dict_flat_plane = correct.fit_in_focus_plane(dfc, faux_zf, microns_per_pixel, img_xc, img_yc)

        return dict_fit_plane, dict_fit_plane_bspl_corrected, dict_flat_plane, bispl, bispl_raw


    # ---

    path_calib_surface = path_results + '/calibration-surface'
    if not os.path.exists(path_calib_surface):
        os.makedirs(path_calib_surface)

    # fit
    dict_fit_plane, dict_fit_plane_bspl_corrected, dict_flat_plane, bispl, bispl_raw = fit_plane_and_bispl(
        path_calib_surface)

    # ---

    for flip_correction in flips:

        # ----------------------------------------------------------------------------------------------------------------------
        # B. ANALYZE TEST COORDS AFTER CORRECTION

        analyze_test = True
        if analyze_test:

            path_results_corr = path_results + '/{}_flip-{}'.format(param_zf, flip_correction)
            if not os.path.exists(path_results_corr):
                os.makedirs(path_results_corr)

            # 1. read test coords
            dft = io.read_test_coords(path_test_coords)

            #   1.a - drop uneccessary columns
            dft = dft.drop(columns=['xm', 'ym', 'xg', 'yg', 'contour_area', 'contour_diameter',
                                    'gauss_dia_x', 'gauss_dia_y', 'gauss_A', 'gauss_sigma_x', 'gauss_sigma_y'])

            #   1.b - make 'frame' and 'z_true' the same for readability
            dft['frame'] = dft['z_true']

            #       1.b.i - make 'stack_id' for groupby and referncing
            dft['stack_id'] = calib_id_from_calibset

            #   1.c - center z on focal plane
            dft['z_true_corr'] = dft['z_true'] - z_zero_from_calibration
            dft['z_corr'] = dft['z'] - z_zero_from_calibration
            dft['error_corr'] = dft['z_corr'] - dft['z_true_corr']

            #   1.d - apply filters

            #       1.d.i - z_range filter
            dft = dft[(dft['z_true_corr'] > z_range[0]) & (dft['z_true_corr'] < z_range[1])]

            #       1.d.ii - min_cm filter
            dft = dft[dft['cm'] > min_cm]

            #       1.d.iii - Barnkob filter
            dft = dft[dft['error_corr'].abs() < filter_barnkob]

            # ---

            # ---

            # 2. corrections

            #   2.a - calibration particle is "zero" position
            dft_baseline = dft[dft['frame'] == baseline_frame]
            cx = dft_baseline[dft_baseline['id'] == calib_id_from_testset].x.values[0]
            cy = dft_baseline[dft_baseline['id'] == calib_id_from_testset].y.values[0]
            cz = dft_baseline[dft_baseline['id'] == calib_id_from_testset].z.values[0]

            #   2.b - stage tilt
            if correct_stage_tilt:
                # tilt
                dft = correct.correct_z_by_plane_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                          df=dft,
                                                                          dict_fit_plane=dict_fit_plane_bspl_corrected,
                                                                          param_z='z_corr',
                                                                          param_z_corr='z_corr_tilt',
                                                                          param_z_surface='z_plane',
                                                                          flip_correction=flip_correction,
                                                                          )

            # ---

            #   2.c - field curvature
            if correct_field_curvature:
                #   2.c.i - correct 'raw' z for field curvature
                dft = correct.correct_z_by_spline_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                           df=dft,
                                                                           bispl=bispl,
                                                                           param_z='z_corr',
                                                                           param_z_corr='z_corr_fc',
                                                                           param_z_surface='z_fc',
                                                                           flip_correction=flip_correction,
                                                                           )

                #   2.c.ii - correct 'corr_tilt' for field curvature
                dft = correct.correct_z_by_spline_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                           df=dft,
                                                                           bispl=bispl,
                                                                           param_z='z_corr_tilt',
                                                                           param_z_corr='z_corr_tilt_fc',
                                                                           param_z_surface='z_plane_fc',
                                                                           flip_correction=flip_correction,
                                                                           )

            # ---

            #   2.d - correct field-curvature on 'raw' and flat plane

            # 2D spline on 'raw'
            dft = correct.correct_z_by_spline_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                       df=dft,
                                                                       bispl=bispl_raw,
                                                                       param_z='z_corr',
                                                                       param_z_corr='z_corr_bspl',
                                                                       param_z_surface='z_bspl',
                                                                       flip_correction=flip_correction,
                                                                       )

            # flat plane
            dft = correct.correct_z_by_plane_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                      df=dft,
                                                                      dict_fit_plane=dict_flat_plane,
                                                                      param_z='z_corr',
                                                                      param_z_corr='z_corr_flat',
                                                                      param_z_surface='z_flat',
                                                                      flip_correction=flip_correction,
                                                                      )

            # ---

            #   2.d - add columns for 'error' relative to corrections
            dft['error_corr_tilt'] = dft['z_corr_tilt'] - dft['z_true_corr']
            dft['error_corr_fc'] = dft['z_corr_fc'] - dft['z_true_corr']
            dft['error_corr_tilt_fc'] = dft['z_corr_tilt_fc'] - dft['z_true_corr']
            dft['error_corr_bspl'] = dft['z_corr_bspl'] - dft['z_true_corr']
            dft['error_corr_flat'] = dft['z_corr_flat'] - dft['z_true_corr']

            # ---

            # ---

            # 3. evaluate

            # setup
            lbl_columns = [r'$z$', r'$z_{tilt}$', r'$z^{f.c.}$', r'$z_{tilt}^{f.c.}$', r'$z^{bspl}$', r'$z_{flat}$']
            z_columns = ['z_corr', 'z_corr_tilt', 'z_corr_fc', 'z_corr_tilt_fc', 'z_corr_bspl', 'z_corr_flat']
            error_columns = ['error_corr', 'error_corr_tilt', 'error_corr_fc', 'error_corr_tilt_fc', 'error_corr_bspl', 'error_corr_flat']

            # setup
            num = len(z_columns)

            # groupby collection
            dfm = dft.groupby('stack_id').mean()
            dfstd = dft.groupby('stack_id').std()

            # store results
            mean_error_results.append(dfm)
            std_error_results.append(dfstd)

            # export
            dfm.to_excel(path_results_corr + '/df_mean.xlsx')
            dfstd.to_excel(path_results_corr + '/df_std.xlsx')

            # print mean
            print("MEAN of ERROR Columns")
            print(dfm[error_columns])

            print("STDEV. of ERROR Columns")
            print(dfstd[error_columns])

            # data
            data_z = dft[z_columns].values
            data_error = dft[error_columns].values

            mean_z = dfm[z_columns].values
            mean_error = dfm[error_columns].values

            std_z = dfstd[z_columns].values
            std_error = dfstd[z_columns].values

            # ---

            # ---

            # 4. plot collection

            # modifiers
            save_figs = True
            show_figs = False
            plot_basics = True
            plot_z_by_z_true = True

            # box and whiskers, violin,
            if plot_basics:
                # box and whisker
                fig, ax = plt.subplots()
                ax.boxplot(data_error, showfliers=False)
                ax.set_xticks(ticks=np.arange(1, len(error_columns) + 1), labels=lbl_columns)
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_corr + '/box_and_whisker.png')
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # violin
                fig, ax = plt.subplots()
                ax.violinplot(data_error, showmeans=True, points=1000)
                ax.set_xticks(ticks=np.arange(1, len(error_columns) + 1), labels=lbl_columns)
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_corr + '/violin.png')
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # z by z_true
            if plot_z_by_z_true:

                # setup
                x = 'z_true_corr'

                # all corrections on same figure
                fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
                for zc, lbl in zip(z_columns, lbl_columns):
                    ax.plot(dft[x], dft[zc], '.', ms=0.5, alpha=0.35, label=lbl)
                ax.set_xlabel(r'$z_{true}$')
                ax.set_ylabel(r'$z$')
                ax.legend(loc='upper left')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_corr + '/z_by_z-true_all.png')
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # individually plot corrections
                for zc, lbl in zip(z_columns, lbl_columns):
                    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
                    ax.plot(dft[x], dft[zc], '.', ms=0.5, label=lbl)
                    ax.set_xlabel(r'$z_{true}$')
                    ax.set_ylabel(r'$z$')
                    ax.legend(loc='upper left')
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_results_corr + '/z_by_z-true_{}.png'.format(zc))
                    if show_figs:
                        plt.show()
                    plt.close()

                # ---

                # fit line to z by z_true and report the r.m.s.e. and quality of fit
                for zc, lbl in zip(z_columns, lbl_columns):

                    # data
                    x_data = dft[x].to_numpy()
                    y_data = dft[zc].to_numpy()

                    # fit
                    popt, pcov = curve_fit(functions.line, x_data, y_data)

                    # sample fitted function
                    y_fit = functions.line(x_data, *popt)

                    # r.m.s.e. and r_squared
                    rmse, r_squared = fit.calculate_fit_error(y_fit, y_data)

                    # ---

                    # plot
                    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
                    ax.plot(x_data, y_data, '.', ms=0.5, label=lbl)
                    ax.plot(x_data, y_fit, ls='--', lw=0.5, color='black', alpha=0.75,
                            label='({}, {})'.format(np.round(rmse, 3), np.round(r_squared, 3)))
                    ax.set_xlabel(r'$z_{true}$')
                    ax.set_ylabel(r'$z$')
                    ax.legend(loc='upper left', title=r'$(\sigma_{z}, \: R^2)$')
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_results_corr + '/z_by_z-true_fit-line_{}.png'.format(zc))
                    if show_figs:
                        plt.show()
                    plt.close()


            # ---

            # ---

            # 4. plot by z_true
            plot_groupby_z_true = True

            # plot stdev at each z-position
            if plot_groupby_z_true:

                # setup
                x = 'z_true_corr'

                # groupby z_true
                dfzm = dft.groupby(x).mean().reset_index()
                dfzstd = dft.groupby(x).std().reset_index()
                dfzcounts = dft.groupby(x).count().reset_index()

                # ERRORBAR - all corrections on same figure
                fig, ax = plt.subplots(figsize=(size_x_inches * 1.1, size_y_inches))
                for zc, ec, lbl in zip(z_columns, error_columns, lbl_columns):
                    ax.errorbar(dfzm[x], dfzm[ec], yerr=dfzstd[ec],
                                fmt='o', ms=1, capsize=1, elinewidth=0.25, alpha=0.5, label=lbl)
                ax.set_xlabel(r'$z_{true}$')
                ax.set_ylabel(r'$\epsilon_{z} + \sigma_{z}$')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_corr + '/errorbars_z-error_all-same.png')
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # ERRORBAR - all corrections on individual figure
                fig, ax = plt.subplots(nrows=num, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.5))
                ii = np.arange(num)
                for zc, ec, lbl, i in zip(z_columns, error_columns, lbl_columns, ii):
                    ax[i].errorbar(dfzm[x], dfzm[ec], yerr=dfzstd[ec],
                                   fmt='o', ms=1, capsize=1, elinewidth=0.25, alpha=0.5, label=lbl)
                    ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

                ax[ii[-1]].set_xlabel(r'$z_{true}$')
                ax[ii[-1]].set_ylabel(r'$\epsilon_{z} + \sigma_{z}$')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_corr + '/errorbars_z-error_all.png')
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # PLOT MEAN - all corrections on individual figure
                fig, ax = plt.subplots(nrows=num, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.5))
                ii = np.arange(num)
                for zc, ec, lbl, i in zip(z_columns, error_columns, lbl_columns, ii):
                    ax[i].plot(dfzm[x], dfzm[ec], '-o', ms=1, label=lbl)
                    ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

                ax[ii[-1]].set_xlabel(r'$z_{true}$')
                ax[ii[-1]].set_ylabel(r'$\epsilon_{z}$')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_corr + '/mean_z-error_all.png')
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # COUNTS
                fig, ax = plt.subplots(nrows=num, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.5))
                ii = np.arange(num)
                for zc, ec, lbl, i in zip(z_columns, error_columns, lbl_columns, ii):
                    ax[i].plot(dfzm[x], dfzcounts[ec], '.', ms=1, label=lbl)
                    ax[i].legend(loc='upper left')

                ax[ii[-1]].set_xlabel(r'$z_{true}$')
                ax[ii[-1]].set_ylabel(r'$N_{p}$')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_corr + '/counts_all.png')
                if show_figs:
                    plt.show()
                plt.close()

            # plot z positions on plane

            # setup
            df = dft
            column_to_bin = 'z_true_corr'
            column_to_fit = 'z_corr'
            column_to_color = 'id'
            xy_cols = ['x', 'y']
            dict_plane = dict_fit_plane
            scatter_size = 5
            plane_alpha = 0.2

            # df_inspect = dft[(dft['z_true_corr'] > z_inspect[0]) & (dft['z_true_corr'] < z_inspect[1])]
            df_inspect = dft[(dft['z_true_corr'].abs() > z_inspect[0]) & (dft['z_true_corr'].abs() < z_inspect[1])]

            for zc in z_columns:
                analyze.scatter_xy_on_plane_by_bin(df_inspect,
                                                   column_to_bin,
                                                   column_to_fit,
                                                   column_to_color,
                                                   xy_cols,
                                                   dict_plane,
                                                   path_results_corr,
                                                   save_id=zc,
                                                   scatter_size=scatter_size,
                                                   plane_alpha=plane_alpha,
                                                   relative=True,
                                                   cx=cx,
                                                   cy=cy,
                                                   flip_correction=flip_correction,
                                                   )

            lbl_columns = [r'$z$', r'$z_{tilt}$', r'$z^{f.c.}$', r'$z_{tilt}^{f.c.}$']
            z_columns = ['z_corr', 'z_corr_tilt', 'z_corr_fc', 'z_corr_tilt_fc']
            error_columns = ['error_corr', 'error_corr_tilt', 'error_corr_fc', 'error_corr_tilt_fc']

            # ---

    # ---

# ---

dfm_all = pd.concat(mean_error_results)
dfstd_all = pd.concat(std_error_results)

dfm_all.to_excel(path_results_combined + '/mean_error_combined.xlsx')
dfstd_all.to_excel(path_results_combined + '/std_error_combined.xlsx')

# ----------------------------------------------------------------------------------------------------------------------
# B. ANALYZE CALIBRATION AND TEST COORDS TOGETHER

analyze_calibration_and_test = False
if analyze_calibration_and_test:
    # read coords
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='spct')
    dft = io.read_test_coords(path_test_coords)

    # assign easier name
    dfc = dfcstats

    # get baseline frame
    dfc = dfc[dfc['frame'] == baseline_frame]
    dft = dft[dft['frame'] == baseline_frame]

    fig, (axx, axy) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.2, size_y_inches * 0.8))

    s = 2
    mrk1 = 'o'
    mrk2 = 'd'
    cmap1 = 'viridis'
    cmap2 = 'RdBu'

    axx.scatter(dfc.x, dfc.gauss_A, c=dfc.id, s=s * 6, marker=mrk1, cmap=cmap1)
    axx.scatter(dft.x, dft.gauss_A, c=dft.id, s=s, marker=mrk1, cmap=cmap2)

    axy.scatter(dfc.y, dfc.gauss_A, c=dfc.id, s=s * 6, marker=mrk1, cmap=cmap1)
    axy.scatter(dft.y, dft.gauss_A, c=dft.id, s=s, marker=mrk1, cmap=cmap2)

    axx.set_xlabel('x')
    axx.set_ylabel('I')
    axy.set_xlabel('y')
    plt.tight_layout()
    plt.show()
    plt.close()

# ---

print("Analysis completed without errors")