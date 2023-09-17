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

# ---

path_figs = '/Users/mackenzie/Desktop/sm-test'

method = 'spct'

# setup file paths
if method == 'spct':
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
         'results-07.29.22-idpt-tmg/tests/spct_soft-baseline_1/coords/test-coords/publication-test-coords/' \
         'min_cm_0.5_z_is_z-corr-tilt/post-processed/' \
         'test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'
    simple_cols = ['frame', 'id', 'x', 'y', 'cm', 'z_true', 'z_no_corr', 'z_corr_tilt', 'z_corr_tilt_fc']
    z_cols = ['z_no_corr', 'z_corr_tilt', 'z_corr_tilt_fc']

elif method == 'idpt':
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
         'results-07.29.22-idpt-tmg/tests/tm16_cm19/coords/test-coords/post-processed/' \
         'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
    simple_cols = ['frame', 'id', 'x', 'y', 'cm', 'z_true', 'z_no_corr', 'z_corr_tilt', 'z_corr_tilt_fc', 'z_corr_fc']
    z_cols = ['z_no_corr', 'z_corr_tilt', 'z_corr_tilt_fc', 'z_corr_fc']
else:
    raise ValueError()

# ----------------------------------------------------------------------------------------------------------------------
# 1. SPCT

fit_spline = False
if fit_spline:

    dfc_raw = pd.read_excel(fp)

    dfc_z = dfc_raw[dfc_raw['z_true'].abs() < 10]
    dfc_z = dfc_z[dfc_z['error_corr_tilt'].abs() < 4]
    z_trues = dfc_z.z_true.unique()

    for z_true in z_trues:
        dfc = dfc_z[dfc_z['z_true'] == z_true]

        # fit spline to 'raw' data
        kx = 2
        ky = 2
        bispl, rmse = fit.fit_3d_spline(x=dfc.x,
                                        y=dfc.y,  # raw: dfc.y, mirror-y: img_yc * 2 - dfc.y
                                        z=dfc['error_corr_tilt'],
                                        kx=kx,
                                        ky=ky)

        fig, ax = plotting.scatter_3d_and_spline(dfc.x,
                                                 dfc.y,
                                                 dfc['error_corr_tilt'],
                                                 bispl,
                                                 cmap='RdBu',
                                                 grid_resolution=30,
                                                 view='multi')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_zlabel(r'$error_{z} \: (\mu m)$')
        plt.suptitle('fit RMSE = {}'.format(np.round(rmse, 3)))
        path_figs = '/Users/mackenzie/Desktop/sm-test'
        plt.savefig(path_figs + '/zt={}_fit-spline-to-error_kx{}_ky{}_after-tilt-correction.png'.format(np.round(z_true, 1), kx, ky))
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(dfc.r, dfc.error_corr_tilt, 'o')
        ax.set_xlabel('r')
        ax.set_ylabel('error-z')
        plt.savefig(path_figs + '/zt={}_plot-error_kx{}_ky{}_after-tilt-correction.png'.format(np.round(z_true, 1), kx, ky))
        plt.close()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. IDPT

scatter_xy = False  # True False
if scatter_xy:

    dfc_raw = pd.read_excel(fp)
    dfc_raw = dfc_raw[simple_cols]

    dfc_z = dfc_raw[dfc_raw['z_true'].abs() > 10]
    z_trues = dfc_z.z_true.unique()

    for z_col in z_cols:
        for z_true in z_trues:
            dfc = dfc_z[dfc_z['z_true'] == z_true]

            fig, ax = plotting.scatter_z_by_xy(dfc, z_params=z_col)
            plt.savefig(path_figs + '/{}_zt={}_{}_scatter-xy.png'.format(method, np.round(z_true, 1), z_col))
            plt.close()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. COMPARE

compare_scatter_xy_or_ref_model = True  # False True
if compare_scatter_xy_or_ref_model:
    # experimental
    mag_eff = 10.01
    NA_eff = 0.45
    microns_per_pixel = 1.6
    size_pixels = 16  # microns
    depth_of_field = functions.depth_of_field(mag_eff, NA_eff, 600e-9, 1.0, size_pixels * 1e-6) * 1e6
    print("Depth of field = {}".format(depth_of_field))
    num_pixels = 512
    area_pixels = num_pixels ** 2
    padding = 5
    img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding
    area_microns = (num_pixels * microns_per_pixel) ** 2

    fps = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
         'results-07.29.22-idpt-tmg/tests/spct_soft-baseline_1/coords/test-coords/publication-test-coords/' \
         'min_cm_0.5_z_is_z-corr-tilt/post-processed/' \
         'test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'
    calib_id_from_testset_spct = 92

    fpi = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
         'results-07.29.22-idpt-tmg/tests/tm16_cm19/coords/test-coords/post-processed/' \
         'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
    calib_id_from_testset_idpt = 42

    path_figs_dist_plane_flat = path_figs + '/figs_plane_fit_to_idpt'
    if not os.path.exists(path_figs_dist_plane_flat):
        os.makedirs(path_figs_dist_plane_flat)

    simple_cols = ['frame', 'id', 'x', 'y', 'cm', 'z_true', 'z_no_corr', 'z_corr_tilt', 'z_corr_tilt_fc']
    z_cols = ['z_no_corr', 'z_corr_tilt', 'z_corr_tilt_fc']

    dfs = pd.read_excel(fps)
    dfi = pd.read_excel(fpi)

    dfs = dfs[simple_cols]
    dfi = dfi[simple_cols]

    # dfs = dfs[dfs['cm'] > 0.8]

    # dfs = dfs[dfs['z_true'].abs() > 10]
    # dfi = dfi[dfi['z_true'].abs() > 10]
    z_trues = dfi.z_true.unique()

    # ---

    # analyze calibration particle only
    analyze_only_calibration_particle = True  # True False
    if analyze_only_calibration_particle:
        fp_ref = '/Users/mackenzie/Desktop/sm-test/ref/dfdz_icp_z-error-4_in-plane-dist-5.xlsx'
        dfref = pd.read_excel(fp_ref)

        dfcs = dfs[dfs['id'] == calib_id_from_testset_spct]
        dfci = dfi[dfi['id'] == calib_id_from_testset_idpt]

        dfcs = dfcs.groupby('z_true').mean().reset_index()
        dfci = dfci.groupby('z_true').mean().reset_index()

        zcs = dfcs['z_no_corr'].to_numpy()
        zci = dfci['z_no_corr'].to_numpy()
        diff = zci - zcs

        dzx = dfcs[(dfcs['z_true'] < -10) | (dfcs['z_true'] > 0)]['z_true'].to_numpy()[:-1]
        dzcs = dfcs['z_no_corr'].diff().to_numpy()[1:]
        dzci = dfci['z_no_corr'].diff().to_numpy()[1:]

        dzcs = dzcs[np.abs(dzcs - 5) < 2]
        dzci = dzci[np.abs(dzci - 5) < 2]

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.125, size_y_inches * 1.25))
        #ax.plot(dfcs.z_true, dfcs['z_no_corr'], label='spct')
        #ax.plot(dfci.z_true, dfci['z_no_corr'], label='idpt')
        ax1.plot(dfcs.z_true, diff, '-o', ms=2, color='k', label='mean diff = {}'.format(np.round(np.mean(diff), 3)))
        ax1.legend(loc='lower center')

        ax1r = ax1.twinx()
        ax1r.plot(dfcs.z_true, dfcs.cm, color='red')
        ax1r.plot(dfci.z_true, dfci.cm, color='blue')
        ax1r.set_ylabel('Cm', color='blue')

        ax2.plot(dzx, dzcs, '-o', ms=3, label=np.round(np.mean(dzcs), 3), color='red')
        ax2.plot(dzx, dzci, '-^', ms=3, label=np.round(np.mean(dzci), 3), color='blue')
        ax2.plot(dfref['z'], dfref['dz'] * -1, '-*', ms=3, color='k', label='R.T.')
        ax2.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    # ---

    dfresss = []

    for i, z_col in enumerate(z_cols):

        # scatter-xy plot
        plot_scatter_xy = False
        if plot_scatter_xy:

            # iterate through z_true
            for z_true in z_trues:
                dfsz = dfs[dfs['z_true'] == z_true]
                dfiz = dfi[dfi['z_true'] == z_true]

                # figure limits
                ylim_center = dfiz['z_no_corr'].mean()
                ylim_dy = 5

                fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 2, size_y_inches))

                ax[0].scatter(dfiz.x, dfiz['z_no_corr'], s=3, label='IDPT')
                ax[1].scatter(dfiz.y, dfiz['z_no_corr'], s=3, label='z_no_corr')

                ax[0].scatter(dfsz.x, dfsz[z_col], s=3, label='SPCT')
                ax[1].scatter(dfsz.y, dfsz[z_col], s=3, label=z_col)

                ax[0].set_xlabel('x')
                ax[0].set_ylabel(z_col)
                ax[0].set_ylim([ylim_center - ylim_dy, ylim_center + ylim_dy])
                ax[0].legend(ncol=2)

                ax[1].set_xlabel('y')
                ax[1].legend(ncol=2)

                plt.tight_layout()
                plt.savefig(path_figs + '/compare_zt={}_{}_scatter-xy.png'.format(np.round(z_true, 1), z_col))
                plt.close()

            # ---

        # ---

        # calculate SPCT error relative to IDPT
        error_relative_idpt_fitted_plane = True
        if error_relative_idpt_fitted_plane:

            dfress = []

            # iterate through z_true
            for z_true in z_trues:

                dfsz = dfs[dfs['z_true'] == z_true]
                dfiz = dfi[dfi['z_true'] == z_true]

                dict_fit_plane = correct.fit_in_focus_plane(df=dfiz,
                                                            param_zf='z_no_corr',
                                                            microns_per_pixel=microns_per_pixel,
                                                            img_xc=img_xc,
                                                            img_yc=img_yc,
                                                            )
                # print('z_true={}, z center fit={}'.format(np.round(z_true, 2), dict_fit_plane['z_f_fit_plane_image_center']))
                # continue

                ref_xy = np.array([img_xc, img_yc])
                dict_fit_plane.update({'ref_xy': ref_xy})

                model = 'plane'
                dict_model = dict_fit_plane
                df = dfsz
                column_to_bin = 'z_true'
                column_to_fit = z_col  # 'z'
                xy_cols = ['x', 'y']
                std_filter = None
                distance_filter = None  # 4  # None  # depth_of_field / 2

                dfres = analyze.evaluate_reference_model_by_bin(model,
                                                                dict_model,
                                                                df,
                                                                column_to_bin,
                                                                column_to_fit,
                                                                xy_cols,
                                                                path_results=path_figs_dist_plane_flat,
                                                                std_filter=std_filter,
                                                                distance_filter=distance_filter,
                                                                save_figs=False,
                                                                save_every_x_fig=1,
                                                                )
                dfress.append(dfres)

                # ---

            dfress = pd.concat(dfress)
            dfress['zcol'] = i + 1
            dfresss.append(dfress)
            # dfress.to_excel(path_figs_dist_plane_flat + '/spct-{}_ref-idpt-plane.xlsx'.format(z_col))

            # ---

    dfresss = pd.concat(dfresss)
    dfresss.to_excel(path_figs_dist_plane_flat + '/spct-compare_ref-idpt-plane.xlsx')

    # ---

    # ---

# ---

compare_spct_correction_method_rel_idpt_fit_plane = False  # True False
if compare_spct_correction_method_rel_idpt_fit_plane:
    fp = '/Users/mackenzie/Desktop/sm-test/compare_rmse-z_rel-idpt-fit-plane_cm0.8.xlsx'
    corr_methods = ['no corr', 'tilt', 'tilt fc']

    df = pd.read_excel(fp)

    fig, ax = plt.subplots()

    dfnc = df[df['zcol'] == 1]
    arr_yc = dfnc['rmse'].to_numpy()

    for cm in [1, 2, 3]:
        dfc = df[df['zcol'] == cm]
        rmse_mean = dfc.rmse.mean()

        arr_y = dfc['rmse'].to_numpy()

        ax.plot(dfc['z_true'], arr_y, '-o', label='{}: {}'.format(corr_methods[cm - 1], np.round(rmse_mean, 3)))

    ax.set_xlabel('z')
    ax.set_ylabel('rmse-z')
    ax.legend(title='Corr: rmse-z')
    plt.tight_layout()
    plt.show()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. CALCULATE ERROR RELATIVE CALIBRATION PARTICLE AT EACH Z-POSITION AFTER REMOVING TILT AT EACH Z-POSITION

calc_error_relative_calibration_particle_after_removing_tilt = True  # False True
if calc_error_relative_calibration_particle_after_removing_tilt:
    # experimental
    mag_eff = 10.01
    NA_eff = 0.45
    microns_per_pixel = 1.6
    size_pixels = 16  # microns
    depth_of_field = functions.depth_of_field(mag_eff, NA_eff, 600e-9, 1.0, size_pixels * 1e-6) * 1e6
    print("Depth of field = {}".format(depth_of_field))
    num_pixels = 512
    area_pixels = num_pixels ** 2
    padding = 5
    img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding
    area_microns = (num_pixels * microns_per_pixel) ** 2

    fps = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
         'results-07.29.22-idpt-tmg/tests/spct_soft-baseline_1/coords/test-coords/publication-test-coords/' \
         'min_cm_0.5_z_is_z-corr-tilt/post-processed/' \
         'test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'
    calib_id_from_testset_spct = 92

    fpi = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
         'results-07.29.22-idpt-tmg/tests/tm16_cm19/coords/test-coords/post-processed/' \
         'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
    calib_id_from_testset_idpt = 42

    path_figs_dist_plane_flat = path_figs + '/figs_plane_fit_to_xxx'
    if not os.path.exists(path_figs_dist_plane_flat):
        os.makedirs(path_figs_dist_plane_flat)

    simple_cols = ['frame', 'id', 'x', 'y', 'cm', 'z_true', 'z', 'z_no_corr', 'z_corr_tilt', 'z_corr_tilt_fc']
    z_cols = ['z_no_corr']

    dfs = pd.read_excel(fps)
    dfi = pd.read_excel(fpi)

    dfs = dfs[simple_cols]
    dfi = dfi[simple_cols]

    dfs = dfs[dfs['cm'] > 0.5]

    # dfs = dfs[dfs['z_true'].abs() > 10]
    # dfi = dfi[dfi['z_true'].abs() > 10]
    z_trues = dfi.z_true.unique()

    # ---

    # ---

    dfresss = []
    for i, z_col in enumerate(z_cols):

        # calculate SPCT error relative to IDPT
        error_relative_calib_after_tilt_corr = True
        if error_relative_calib_after_tilt_corr:

            dfress = []

            # iterate through z_true
            for z_true in z_trues:

                dfsz = dfs[dfs['z_true'] == z_true]
                dfiz = dfi[dfi['z_true'] == z_true]

                dict_fit_plane = correct.fit_in_focus_plane(df=dfiz,
                                                            param_zf='z_no_corr',
                                                            microns_per_pixel=microns_per_pixel,
                                                            img_xc=img_xc,
                                                            img_yc=img_yc,
                                                            )
                # print('z_true={}, z center fit={}'.format(np.round(z_true, 2), dict_fit_plane['z_f_fit_plane_image_center']))
                # continue

                ref_xy = np.array([img_xc, img_yc])
                dict_fit_plane.update({'ref_xy': ref_xy})

                model = 'plane'
                dict_model = dict_fit_plane
                df = dfsz
                column_to_bin = 'z_true'
                column_to_fit = z_col  # 'z'
                xy_cols = ['x', 'y']
                std_filter = None
                distance_filter = None  # 4  # None  # depth_of_field / 2

                dfres = analyze.evaluate_reference_model_by_bin(model,
                                                                dict_model,
                                                                df,
                                                                column_to_bin,
                                                                column_to_fit,
                                                                xy_cols,
                                                                path_results=path_figs_dist_plane_flat,
                                                                std_filter=std_filter,
                                                                distance_filter=distance_filter,
                                                                save_figs=False,
                                                                save_every_x_fig=1,
                                                                )
                dfress.append(dfres)

                # ---

            dfress = pd.concat(dfress)
            dfress['zcol'] = i + 1
            dfresss.append(dfress)
            # dfress.to_excel(path_figs_dist_plane_flat + '/spct-{}_ref-idpt-plane.xlsx'.format(z_col))

            # ---

    dfresss = pd.concat(dfresss)
    dfresss.to_excel(path_figs_dist_plane_flat + '/spct-compare_ref-idpt-plane.xlsx')

    # ---

    # ---

# ---

# ---