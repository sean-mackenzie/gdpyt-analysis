# test bin, analyze, and plot functions
import itertools
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, CloughTocher2DInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

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
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])  # 'ieee', 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)
del ax

# ----------------------------------------------------------------------------------------------------------------------
# 00. Effect of a systematic uncertainty of 0.4 um on measurement uncertainty given n=21 samples.

calc_systematic_uncertainty = False
if calc_systematic_uncertainty:
    mu, sigma = 0, 0.4
    n_iterations = 10000

    sum_uncertainty = 0
    for n in range(n_iterations):
        s = np.random.normal(mu, sigma, 21)
        sum_uncertainty += np.mean(np.abs(s))

    sum_uncertainty = sum_uncertainty / n_iterations
    print("Average effect of 0.4 micron systematic uncertainty (n=21) = {} microns".format(sum_uncertainty))

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# TEST COORDS (FINAL)
"""
IDPT:
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/
results-07.29.22-idpt-tmg'

SPCT:
base_dir = ''
"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

method = 'spct'

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_aberrations_induced_errors'

path_test_coords = join(base_dir, 'coords/test-coords')
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# ---

if method != 'spct':
    raise ValueError("This script is only for analyzing 'spct' results. ")

# ---

# processing
calib_baseline_frame = 50
calib_id_from_testset = 42
padding = 5

# experimental
mag_eff = 10.01
NA_eff = 0.45
microns_per_pixel = 1.6
size_pixels = 16  # microns
depth_of_field = functions.depth_of_field(mag_eff, NA_eff, 600e-9, 1.0, size_pixels * 1e-6) * 1e6
print("Depth of field = {}".format(depth_of_field))
num_pixels = 512
area_pixels = num_pixels ** 2
img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding
area_microns = (num_pixels * microns_per_pixel) ** 2


# ----------------------------------------------------------------------------------------------------------------------
# A. EVALUATE STAGE TILT ON CALIBRATION COORDS


def fit_plane_and_bispl(path_figs=None):
    # file paths
    if path_figs is not None:
        path_calib_surface = path_results + '/calibration-surface'
        if not os.path.exists(path_calib_surface):
            os.makedirs(path_calib_surface)

    # read coords
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='spct')
    dfc = dfcpid
    del dfcpid, dfcstats

    # filter
    dfc = dfc[np.abs(dfc['zf_nearest_calib'] - 50) < 4]

    # print mean
    zf_methods = ['zf_from_peak_int', 'zf_from_nsv', 'zf_from_nsv_signal']
    for zfm in zf_methods:
        print("{}: {} +/- {}".format(zfm, np.round(dfc[zfm].mean(), 2), np.round(dfc[zfm].std(), 2)))

    # ---

    # processing

    # mirror y-coordinate
    # dfc['y'] = img_yc * 2 - dfc.y

    # ---

    # fit plane
    param_zf = 'zf_from_nsv'
    # dfc[param_zf] = dfc[param_zf]

    # ---

    # fit spline to 'raw' data
    bispl_raw, rmse = fit.fit_3d_spline(x=dfc.x,
                                        y=dfc.y,  # raw: dfc.y, mirror-y: img_yc * 2 - dfc.y
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
                                                   path_figs=path_figs)

    # return faux flat plane
    faux_zf = 'faux_zf_flat'
    dfc[faux_zf] = dfc['zf_from_nsv'].mean()
    dict_flat_plane = correct.fit_in_focus_plane(dfc, faux_zf, microns_per_pixel, img_xc, img_yc)

    return dict_fit_plane, dict_fit_plane_bspl_corrected, dict_flat_plane, bispl, bispl_raw


# ---


# ----------------------------------------------------------------------------------------------------------------------
# 2. Read calibration coordinates

# get calibration surface
dict_fit_plane, dict_fit_plane_bspl_corrected, dict_flat_plane, bispl, bispl_raw = fit_plane_and_bispl(path_figs=None)

# read coords
dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)
dfc = dfcstats
del dfcstats

# -

# post-process calib coords
dfcpid = dfcpid[np.abs(dfcpid['zf_nearest_calib'] - 50) < 4]
z_f_mean = dfcpid['zf_nearest_calib'].mean()
include_pids_cal = dfcpid['id']

# ----------------------------------------------------------------------------------------------------------------------
# 3. Read meta-test coordinates

# read test coords
dft = io.read_test_coords(path_test_coords)

# ---

# STEP #6: add 'z_no_corr' and 'error_no_corr' column
dft['z_no_corr'] = dft['z']
dft['error_z'] = dft['z'] - dft['z_true']
dft['error_z_no_corr'] = dft['error_z']

#   2.a - calibration particle is "zero" position
dft_baseline = dft[dft['frame'] == calib_baseline_frame]
cx = dft_baseline[dft_baseline['id'] == calib_id_from_testset].x.values[0]
cy = dft_baseline[dft_baseline['id'] == calib_id_from_testset].y.values[0]
cz = dft_baseline[dft_baseline['id'] == calib_id_from_testset].z.values[0]

#   2.b - stage tilt
flip_correction = True
dft = correct.correct_z_by_plane_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                          df=dft,
                                                          dict_fit_plane=dict_fit_plane_bspl_corrected,
                                                          param_z='z',
                                                          param_z_corr='z_corr_tilt',
                                                          param_z_surface='z_tilt',
                                                          flip_correction=flip_correction,
                                                          )

#   2.c - correct 'corr_tilt' for field curvature
dft = correct.correct_z_by_spline_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                           df=dft,
                                                           bispl=bispl,
                                                           param_z='z_corr_tilt',
                                                           param_z_corr='z_corr_tilt_fc',
                                                           param_z_surface='z_tilt_fc',
                                                           flip_correction=flip_correction,
                                                           )

# ---
# ASSIGN "Z_TRUE" TO CALIBRATION PARTICLE'S AVERAGE Z-POSITION PER Z (N=3, FRAMES)
error_relative_calib_particle = True
if error_relative_calib_particle:
    z_trues = dft.z_true.unique()
    dfs = []
    for z_true in z_trues:
        dftz = dft[dft['z_true'] == z_true]

        # z_true = dftz[dftz['id'] == calib_id_from_testset].z_true.iloc[0]
        z_calib = dftz[dftz['id'] == calib_id_from_testset].z.mean()
        dftz['z_true'] = np.round(z_calib, 3)
        dfs.append(dftz)

    dfs = pd.concat(dfs)
    dft = dfs

dft['error_z'] = dft['z'] - dft['z_true']
dft['error_z_corr_tilt'] = dft['z_corr_tilt'] - dft['z_true']
dft['error_z_corr_tilt_fc'] = dft['z_corr_tilt_fc'] - dft['z_true']

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 4. Post-process test coords

pz = 'error_z_corr_tilt'  # 'error_z_corr_tilt' # 'error_z' # 'error_z_corr_tilt_fc'
px = 'gauss_xc'  # 'gauss_xc' 'x'
py = 'gauss_yc'  # 'gauss_yc' 'y'

# Get stats on calib and test coords
z_range = (dfc.z_true.min(), dfc.z_true.max())
measurement_depth = z_range[1] - z_range[0]
error_z_barnkob = measurement_depth / 10

# -

# List of filter variables
filter_min_cm = 0.5
filter_error_z = 5  # error_z_barnkob

# Filter #1: remove low similarity (keep: Cm > Cm_min)
dft = dft[dft['cm'] > filter_min_cm]

# Filter #2: remove high z-errors (keep: error_z < filter_error_z)
dft = dft[dft[pz].abs() < filter_error_z]

# Filter #3: keep only "good" calibration particles
"""print(len(dft))
dft = dft[dft['id'].isin(include_pids_cal)]
print(len(dft))"""

# make r-coordinate
dft['r'] = np.sqrt((dft['x'] - 256) ** 2 + (dft['y'] - 256) ** 2)


# -

# ----------------------------------------------------------------------------------------------------------------------
# 5. Evaluate field dependent z-errors

# evaluate each z_true
eval_by_z = False
if eval_by_z:

    z_trues = dft.z_true.unique()
    inspect_z_trues = z_trues[(z_trues > 45) & (z_trues < 57)]
    for z_true in inspect_z_trues:  # np.arange(20, 29, 1):
        dftz = dft[dft['z_true'] == z_true]

        plot_scatter = True
        if plot_scatter:
            fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(size_x_inches * 2.25, size_y_inches / 1.5))
            ax[0].scatter(dftz['r'], dftz[pz])
            ax[1].scatter(dftz[px], dftz[pz])
            ax[2].scatter(dftz[py], dftz[pz], label=pz)
            ax[0].set_xlabel('r')
            ax[0].set_ylabel(pz)
            ax[1].set_xlabel('x')
            ax[2].set_xlabel('y')
            plt.suptitle(r'$\overline{\epsilon_{z}} = $' + ' {} '.format(np.mean(dftz[pz].abs())) + r'$\mu m$' + '\n' +
                         'z-true={}'.format(np.round(z_true, 1)))
            plt.tight_layout()
            if path_figs:
                plt.savefig(path_figs + '/test_fit-scatter_z={}_z-true={}.png'.format(pz, np.round(z_true, 1)))
            else:
                plt.show()
            plt.close()

        plot_surf = False
        if plot_surf:
            kx = 2
            ky = 2

            # fit spline to data
            bispl, rmse = fit.fit_3d_spline(x=dftz[px],
                                            y=dftz[py],  # raw: dfc.y, mirror-y: img_yc * 2 - dfc.y
                                            z=dftz[pz],
                                            kx=kx,
                                            ky=ky,
                                            s=len(dftz[pz]) * 2.5,
                                            )

            # plot
            fig, ax = plotting.scatter_3d_and_spline(dftz[px],
                                                     dftz[py],
                                                     dftz[pz],
                                                     bispl,
                                                     cmap='RdBu',
                                                     grid_resolution=30,
                                                     view='multi',
                                                     )
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_zlabel(r'$z_{f} \: (\mu m)$')
            plt.suptitle('fit RMSE = {}'.format(np.round(rmse, 3)))
            if path_figs:
                plt.savefig(path_figs + '/test_fit-spline_z={}_z-true={}.png'.format(pz, np.round(z_true, 1)))
            plt.close()

# -

# evaluate z-errors over a z-range
eval_across_z = False
if eval_across_z:

    dz_zfs = [50]

    for dz_zf in dz_zfs:
        z_range_eval = (z_f_mean - dz_zf, z_f_mean + dz_zf)

        dftzz = dft[(dft['z_true'] > z_range_eval[0]) & (dft['z_true'] < z_range_eval[1])]

        dftz = dftzz.groupby('id').mean()
        dftz_std = dftzz.groupby('id').std()

        # plot

        plot_scatter = True
        if plot_scatter:
            fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 2, size_y_inches))
            ax[0].errorbar(dftz[px], dftz[pz], yerr=dftz_std[pz], fmt='o', ms=3, capsize=1, elinewidth=0.5, )
            ax[1].errorbar(dftz[py], dftz[pz], yerr=dftz_std[pz], fmt='o', ms=3, capsize=1, elinewidth=0.5, label=pz)
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('z')
            ax[1].set_xlabel('y')
            plt.suptitle(r'$\overline{\epsilon_{z}} = $' + ' {} '.format(np.mean(dftz[pz].abs())) + r'$\mu m$' + '\n' +
                         'range: ({}, {})'.format(np.round(z_range_eval[0], 1), np.round(z_range_eval[1], 1)))
            plt.tight_layout()
            if path_figs:
                plt.savefig(path_figs +
                            '/plot-error-z_fit-scatter-xy_groupby-pid_{}-z-{}.png'.format(np.round(z_range_eval[0], 1),
                                                                                          np.round(z_range_eval[1], 1)),
                            )
            else:
                plt.show()
            plt.close()

        # -

        plot_surf = True
        if plot_surf:

            kx = 2
            ky = 2

            # fit spline to 'raw' data
            bispl, rmse = fit.fit_3d_spline(x=dftz[px],
                                            y=dftz[py],  # raw: dfc.y, mirror-y: img_yc * 2 - dfc.y
                                            z=dftz[pz],
                                            kx=kx,
                                            ky=ky,
                                            s=len(dftz[pz]) * 1.5,
                                            )

            # plot
            fig, ax = plotting.scatter_3d_and_spline(dftz[px],
                                                     dftz[py],
                                                     dftz[pz],
                                                     bispl,
                                                     cmap='RdBu',
                                                     grid_resolution=30,
                                                     view='multi',
                                                     )
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')
            ax.set_zlabel(r'$z_{f} \: (\mu m)$')
            plt.suptitle('fit RMSE = {}'.format(np.round(rmse, 3)))
            if path_figs:
                plt.savefig(path_figs +
                            '/plot-error-z_fit-spline_groupby-pid_{}-z-{}.png'.format(np.round(z_range_eval[0], 1),
                                                                                      np.round(z_range_eval[1], 1)),
                            )
            plt.close()

        # -

        plot_heatmap = False
        if plot_heatmap:
            fig, ax = plotting.plot_heatmap(dftz, px, py, pz)
            fig.show()

# ---

# evaluate rmse_z
eval_rmse_z = False
if eval_rmse_z:

    true_num_particles_per_frame = 88
    num_frames_per_step = 1
    z_columns = ['z', 'z_corr_tilt', 'z_corr_tilt_fc']
    error_columns = ['error_z', 'error_z_corr_tilt', 'error_z_corr_tilt_fc']

    for zc, ec in zip(z_columns, error_columns):

        # copy dataframe for rmse
        dfrmse = dft.copy()
        dfrmse['z'] = dfrmse[zc]
        dfrmse['bin'] = dfrmse['z_true']

        # calculate squared error
        dfrmse['rmse_z'] = dfrmse[ec] ** 2
        dfrmse = dfrmse[['bin', 'frame', 'id', 'z_true', 'z', 'cm', 'rmse_z']]

        # local rmse

        # groupby 'dz'
        column_to_bin = 'z_true'
        bins = dfrmse.bin.unique()
        num_dz_steps = len(bins)
        round_to_decimal = 1
        dfb = bin.bin_by_list(dfrmse, column_to_bin, bins, round_to_decimal)

        # count
        dfc = dfb.groupby('bin').count()
        dfc['true_num'] = true_num_particles_per_frame * num_frames_per_step
        dfc['num_idd'] = 1  # i_num_rows_per_z_df[i_num_rows_per_z_df['z_true'].isin(dfc.index.values)].id.to_numpy()
        # NOTE: this was changed on 10/27/2022 to filter 'i_num_rows_per_z' according to bins
        dfc['num_meas'] = dfc['z']
        dfc['percent_meas_idd'] = dfc.num_meas / dfc.num_idd
        dfc['true_percent_meas'] = dfc.num_meas / dfc.true_num

        # calculate rmse per column
        dfb = dfb.groupby('bin').mean()
        bin_z_trues = dfb.z_true.to_numpy()
        bin_zs = dfb.z.to_numpy()
        bin_cms = dfb.cm.to_numpy()
        dfb = np.sqrt(dfb)
        dfb['z_true'] = bin_z_trues
        dfb['z'] = bin_zs
        dfb['cm'] = bin_cms
        dfb = dfb.drop(columns=['frame', 'id'])

        # rmse + percent_measure
        dfrmse_bins = pd.concat([dfb,
                                 dfc[['true_num', 'num_idd', 'num_meas', 'percent_meas_idd', 'true_percent_meas']],
                                 ], axis=1, join='inner', sort=False)

        # export
        dfrmse_bins.to_excel(path_results + '/bin-z-true_rmse-z_ec-{}_relative-calib.xlsx'.format(ec))

        # -

        # mean (bins = 1)
        dfbm = bin.bin_by_column(dfrmse, column_to_bin, number_of_bins=1, round_to_decimal=round_to_decimal)
        dfbm['test_id'] = 1

        # count
        dfcm = dfbm.groupby('bin').count()
        dfcm['true_num'] = true_num_particles_per_frame * num_frames_per_step * num_dz_steps
        dfcm['num_idd'] = 1  # np.sum(i_num_rows_per_z)
        dfcm['num_meas'] = dfcm['z']
        dfcm['percent_meas_idd'] = dfcm.num_meas / dfcm.num_idd
        dfcm['true_percent_meas'] = dfcm.num_meas / dfcm.true_num

        # calculate rmse per column
        dfbm = dfbm.groupby('bin').mean()
        mean_z_true = dfbm.iloc[0].z_true
        mean_z = dfbm.iloc[0].z
        mean_cm = dfbm.iloc[0].cm
        dfbm = np.sqrt(dfbm)
        dfbm['z_true'] = mean_z_true
        dfbm['z'] = mean_z
        dfbm['cm'] = mean_cm
        dfbm = dfbm.drop(columns=['frame', 'id'])

        dfrmse_mean = pd.concat([dfbm,
                                 dfcm[['true_num', 'num_idd', 'num_meas', 'percent_meas_idd', 'true_percent_meas']],
                                 ], axis=1, join='inner', sort=False)
        dfrmse_mean = dfrmse_mean.groupby('test_id').mean()
        dfrmse_mean.to_excel(path_results + '/mean-dz_rmse-z_ec-{}_relative-calib.xlsx'.format(ec))

        # ---

        # plot
        save_figs = True
        show_figs = True
        ms = 2

        if save_figs or show_figs:
            fig, (ax0, ax2, ax4) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))

            ax0.plot(dfrmse_bins.index, dfrmse_bins.cm, '-o', ms=ms)
            ax0.set_ylabel(r'$C^{\delta}_{m}$')
            ax0.set_ylim([0.75, 1])
            ax0.set_yticks([0.75, 1])

            ax2.plot(dfrmse_bins.index, dfrmse_bins.rmse_z, '-o', ms=ms)
            ax2.set_ylabel(r'$\sigma^{\delta}_{z} \: (\mu m)$')
            ax2.set_ylim([0, 6])
            ax2.set_yticks([0, 5])

            ax4.plot(dfrmse_bins.index, dfrmse_bins.true_percent_meas, '-o', ms=ms)
            ax4.set_ylabel(r'$\phi^{\delta}$')
            ax4.set_ylim([0, 1.05])
            ax4.set_yticks([0, 1])
            ax4.set_xlabel(r'$z \: (\mu m)$')

            plt.tight_layout()
            plt.suptitle(ec)
            if save_figs:
                plt.savefig(path_figs + '/rmse-z_by_z-true_ec-{}_relative-calib.png'.format(ec))
            if show_figs:
                plt.show()
            plt.close()

        # ---

    # ---

    # ---

# ---

# compare rmse_z
compare_rmse_z = False
if compare_rmse_z:
    error_columns = ['error_z', 'error_z_corr_tilt', 'error_z_corr_tilt_fc']

    fig, ax = plt.subplots()
    ms = 1
    for ec in error_columns:
        dfrmse_bins = pd.read_excel(path_results + '/bin-z-true_rmse-z_ec-{}_relative-calib.xlsx'.format(ec))
        dfrmse_mean = pd.read_excel(path_results + '/mean-dz_rmse-z_ec-{}_relative-calib.xlsx'.format(ec))

        ax.plot(dfrmse_bins.index, dfrmse_bins.rmse_z, '-o', ms=ms,
                label='{}: {}'.format(ec, np.round(dfrmse_mean['rmse_z'].iloc[0], 2)))

    ax.set_ylabel(r'$\sigma^{\delta}_{z} \: (\mu m)$')
    ax.set_ylim([0, 6])
    ax.set_yticks([0, 5])
    ax.legend(title='r.m.s. error(z) from')

    plt.tight_layout()
    plt.savefig(path_figs + '/compare-ecs_rmse-z_by_z-true_relative-calib.png')
    plt.show()
    plt.close()

# ---

# 2d-bin by r and z
bin_r_z = True
if bin_r_z:

    z_trues = dft.z_true.unique()

    pzs = ['error_z_corr_tilt', 'error_z', 'error_z_corr_tilt_fc']
    columns_to_bin = ['r', 'z_true']
    column_to_count = 'id'
    bins = [3, z_trues]
    round_to_decimals = [1, 3]
    min_num_bin = 10
    return_groupby = True
    plot_fit = False

    dftt = dft.copy()

    for pz in pzs:

        dft = dftt.copy()

        plot_rmse = True
        if plot_rmse:
            save_id = pz
            dft['rmse_z'] = dftt[pz] ** 2
            pz = 'rmse_z'
        else:
            save_id = pz

        dfm, dfstd = bin.bin_generic_2d(dft,
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

        mean_rmse_z = np.round(np.mean(np.sqrt(dfm[pz])), 3)

        # plot
        fig, ax = plt.subplots()

        for i, bin_r in enumerate(dfm.bin_tl.unique()):
            dfbr = dfm[dfm['bin_tl'] == bin_r]
            dfbr_std = dfstd[dfstd['bin_tl'] == bin_r]

            if plot_rmse:
                ax.plot(dfbr.bin_ll, np.sqrt(dfbr[pz]), '-o', ms=2.5,
                        label='{}, {}'.format(int(np.round(bin_r, 0)), np.round(np.mean(np.sqrt(dfbr[pz])), 3)))
                ylbl = r'$\sigma_{z}^{\delta} \: (\mu m)$'
            else:
                # scatter: mean +/- std
                ax.errorbar(dfbr.bin_ll, dfbr[pz], yerr=dfbr_std[pz],
                            fmt='-o', ms=2, elinewidth=0.5, capsize=1, label=int(np.round(bin_r, 0)))
                ylbl = r'$\epsilon_{z} \: (\mu m)$'

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([10, 90])
        #ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylim([0, 3.5])
        #ax.set_yticks(ticks=yerr_ticks, labels=yerr_ticks)
        ax.legend(loc='upper left', title=r'$r_{bin}, \overline{\sigma_{z}}$',
                  borderpad=0.2, handletextpad=0.6, borderaxespad=0.25, markerscale=0.75)
        ax.set_ylabel(ylbl)
        ax.set_title('mean rmse-z = {} microns'.format(mean_rmse_z))
        plt.savefig(path_figs + '/bin-r-z_{}-{}_by_z.png'.format(save_id, pz))
        plt.tight_layout()
        plt.show()
        plt.close()


# ---

print("Analysis completed without errors.")