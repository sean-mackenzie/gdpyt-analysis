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

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
           'results-07.29.22-idpt-tmg'

method = 'spct'

if method == 'spct':
    test_dir = base_dir + '/tests/spct_soft-baseline_1'
    test_id = 1
    test_name = 'test_coords_particle_image_stats_spct-{}'.format(test_id)
    padding = 5
    padding_rel_true_x = 0
    padding_rel_true_y = 0
    calib_id_from_testset = 92
    calib_id_from_calibset = 46

    calib_baseline_frame = 12  # NOTE: baseline frame was 'calib_13.tif' but output coords always begin at frame = 0.

elif method == 'gdpt':
    test_dir = base_dir + '/tests/gdpt'
    test_id = 3
    test_name = 'test_coords_v{}'.format(test_id)
    padding = 0
    padding_rel_true_x = -4
    padding_rel_true_y = -4
else:
    test_dir = base_dir + '/tests/tm16_cm19'
    test_id = 19
    test_name = 'test_coords_particle_image_stats_tm16_cm{}'.format(test_id)
    padding = 5
    padding_rel_true_x = 0
    padding_rel_true_y = 0

    calib_id_from_testset = 42
    calib_id_from_calibset = 42

path_test_coords = join(test_dir, 'coords/test-coords')
path_test_coords_corrected = path_test_coords + '/custom-correction'
path_test_coords_post_processed = path_test_coords + '/post-processed'
path_test_coords_corr = path_test_coords_corrected + '/{}.xlsx'.format(test_name)

path_calib_coords = join(test_dir, 'coords/calib-coords')
path_similarity = join(test_dir, 'similarity')
path_results = join(test_dir, 'results')
path_results_original = path_results
path_results_error = path_results + '/error'
path_figs = join(test_dir, 'figs')

path_results_true_positions = join(path_results, 'true_positions')
if not os.path.exists(path_results_true_positions):
    os.makedirs(path_results_true_positions)

path_results_true_coords = join(test_dir, 'coords/true-fiji-coords')  # join(base_dir, 'coords', 'true-fiji')
if not os.path.exists(path_results_true_coords):
    os.makedirs(path_results_true_coords)


# ----------------------------------------------------------------------------------------------------------------------
# 0. SETUP PROCESS CONTROLS

# subject to change
true_num_particles_per_frame = 88
# num_dz_steps = 20
baseline_frame = 39
baseline_frames = [39, 40, 41]

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

# processing
z_range = [-50, 55]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
num_frames_per_step = 3
filter_barnkob = measurement_depth / 10
filter_step_size = 10
min_cm = 0.9
min_percent_layers = 0.5
remove_ids = None

# initialize variables which modify later processing decisions
dict_fit_plane = None
dict_fit_plane_bspl_corrected = None
dict_flat_plane = None
bispl = None
bispl_raw = None

# dataset alignment
z_zero_from_calibration = 49.9  # 50.0
z_zero_of_calib_id_from_calibration = 49.6  # the in-focus position of calib particle in test set.

z_zero_from_test_img_center = 68.6  # 68.51
z_zero_of_calib_id_from_test = 68.1  # the in-focus position of calib particle in calib set.

# ---

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
    if method == 'idpt':
        print("Performing SPCT z_f analysis on IDPT dataset. BE CAREFUL!")
        param_zf = 'zf_from_nsv'
        dfc[param_zf] = dfc[param_zf]

        # param_zf = 'zf_flat'
        # dfc[param_zf] = dfc['zf_from_nsv'].mean()
    else:
        param_zf = 'zf_from_nsv'
        dfc[param_zf] = dfc[param_zf]

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
# C. ANALYZE TEST COORDS AFTER CORRECTION

analyze_test = True
if analyze_test:

    column_for_z = 'z_corr_tilt'  # no correction: 'z'; tilt: 'z_corr_tilt'; tilt + field curvature : 'z_corr_tilt_fc'
    export_processed_coords = True
    correct_idpt = True
    flip_correction = False

    plot_z_corrections = False
    plot_z_corrs_by_xy = False
    plot_dz_corrs_by_xy = False

    # ---

    # ---

    # fit
    dict_fit_plane, dict_fit_plane_bspl_corrected, dict_flat_plane, bispl, bispl_raw = fit_plane_and_bispl(path_figs=None)

    # ---

    # ----------------------------------------------------------------------------------------------------------------------
    # B. CORRECT TEST COORDS

    # read test coords (where z is based on 1-micron steps and z_true is 5-micron steps every 3 frames)

    if not os.path.exists(path_test_coords_corrected + '/{}_dzf.xlsx'.format(test_name)):

        # ------------------------------------------------------------------------------------------------------------------
        # 2. PERFORM CORRECTION

        """
        Old Correction Values:
        
        zf_image_center = 49.988
        z_test_offset = -13.7343
        z_bias = 0.5
        """

        if not os.path.exists(path_test_coords_corr):

            if not os.path.exists(path_test_coords_corrected):
                os.makedirs(path_test_coords_corrected)

            # read test coords (where z is based on 1-micron steps and z_true is 5-micron steps every 3 frames)
            dft = io.read_test_coords(path_test_coords)

            # ---

            # 1. REMOVE PARTICLES BY IMAGE BORDERS
            i_num_rows_pre = len(dft)
            dft = dft[(dft['x'] > 19 + padding_rel_true_x) & (dft['x'] < 501 + padding_rel_true_x) &
                      (dft['y'] > 17 + padding_rel_true_x) & (dft['y'] < 499 + padding_rel_true_x)]
            i_num_rows_post = len(dft)
            print("{} rows removed by image borders filter.".format(i_num_rows_pre - i_num_rows_post))

            # ---

            # PREPARE 'GDPT' coords by matching columns

            if method == 'gdpt':
                dft['frame'] = dft['frame'] - 1
                dft['z_true'] = dft['frame']
                dft['z_norm'] = dft['z']
                dft['z'] = dft['z'] * 106  # - dft['z'] / (2 * 106)
                dft['gauss_xc'] = dft['x']
                dft['gauss_yc'] = dft['y']

            # ---

            # 2. CORRECT 'z' AND 'z_true' ACCORDING TO IN-FOCUS POSITIONS

            """ NEW CORRECTION METHOD """
            # CORRECTION #1: resolve z-position as a function of 'frame' discrepancy
            dft['z_true_corr'] = (dft['z_true'] - dft['z_true'] % 3) / 3 * 5 + 5

            # CORRECTION #2: shift 'z_true' according to z_f (test images)
            dft['z_true_minus_zf_from_test'] = dft['z_true_corr'] - z_zero_of_calib_id_from_test

            # CORRECTION #3: shift 'z' according to z_f (calibration images)
            dft['z_minus_zf_from_calib'] = dft['z'] - z_zero_of_calib_id_from_calibration

            # STEP #4: store "original" 'z' and 'z_true' coordinates
            dft['z_orig'] = dft['z']
            dft['z_true_orig'] = dft['z_true']

            # STEP #5: update 'z' and 'z_true' coordinates & add 'error' column
            dft['z'] = dft['z_minus_zf_from_calib']
            dft['z_true'] = dft['z_true_minus_zf_from_test']
            dft['error'] = dft['z'] - dft['z_true']

            # STEP #6: add 'z_no_corr' and 'error_no_corr' column
            dft['z_no_corr'] = dft['z']
            dft['error_no_corr'] = dft['error']

            # ---

            # old correction
            """ 
            OLD CORRECTION METHOD:

            # CORRECTION #1: resolve z-position as a function of 'frame' discrepancy
            # dft['z_true'] = (dft['z_true'] - dft['z_true'] % 3) / 3 * 5

            # CORRECTION #2: shift 'z' according to z @ z_f (test images)
            # dft['z_true'] = dft['z_true'] - zf_image_center + z_test_offset + z_bias

            dft['z'] = dft['z'] - zf_image_center
            dft['error'] = dft['z'] - dft['z_true']
            """

            # ---

            # 2. FITTED PLANE CORRECTION (local correction)

            if method in ['spct', 'gdpt'] or correct_idpt is True:

                #   2.a - calibration particle is "zero" position
                dft_baseline = dft[dft['frame'] == baseline_frame]
                cx = dft_baseline[dft_baseline['id'] == calib_id_from_testset].x.values[0]
                cy = dft_baseline[dft_baseline['id'] == calib_id_from_testset].y.values[0]
                cz = dft_baseline[dft_baseline['id'] == calib_id_from_testset].z.values[0]

                #   2.b - stage tilt
                dft = correct.correct_z_by_plane_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                          df=dft,
                                                                          dict_fit_plane=dict_fit_plane_bspl_corrected,
                                                                          param_z='z',
                                                                          param_z_corr='z_corr_tilt',
                                                                          param_z_surface='z_tilt',
                                                                          flip_correction=flip_correction,
                                                                          )

                # ---

                #   2.c - correct 'corr_tilt' for field curvature
                dft = correct.correct_z_by_spline_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                           df=dft,
                                                                           bispl=bispl,
                                                                           param_z='z_corr_tilt',
                                                                           param_z_corr='z_corr_tilt_fc',
                                                                           param_z_surface='z_tilt_fc',
                                                                           flip_correction=flip_correction,
                                                                           )

                #   add column for just field-curvature correction
                dft = correct.correct_z_by_spline_relative_to_calib_pid_xy(cx=cx, cy=cy,
                                                                           df=dft,
                                                                           bispl=bispl,
                                                                           param_z='z',
                                                                           param_z_corr='z_corr_fc',
                                                                           param_z_surface='z_fc',
                                                                           flip_correction=flip_correction,
                                                                           )

                # ---

                # check the correction at several z_true positions
                if plot_z_corrections:

                    # file paths

                    path_z_corrections_relative = path_results + '/corrections_relative'
                    if not os.path.exists(path_z_corrections_relative):
                        os.makedirs(path_z_corrections_relative)

                    path_z_corr_positions = path_z_corrections_relative + '/relative_position'
                    if not os.path.exists(path_z_corr_positions):
                        os.makedirs(path_z_corr_positions)

                    path_z_corr_displacement = path_z_corrections_relative + '/relative_displacement'
                    if not os.path.exists(path_z_corr_displacement):
                        os.makedirs(path_z_corr_displacement)

                    # ---

                    # processing

                    # round z_true for easy access
                    dft['bin'] = np.round(dft['z_true'], 0).astype(int)
                    bin_z_trues = [-38, -28, -18, -13, 12, 22, 27, 32, 42]
                    dz_lim = 3

                    # plotting

                    # plot setup
                    figsize = (size_x_inches * 1.75, size_y_inches * 1.5)
                    s = 1.5

                    # plot positions: z(x, y)
                    if plot_z_corrs_by_xy:
                        for bzt in bin_z_trues:
                            dftg = dft[dft['bin'] == bzt]
                            dftg = dftg[(dftg['z_no_corr'] > bzt - dz_lim) & (dftg['z_no_corr'] < bzt + dz_lim)]

                            fig, (axx, axy) = plt.subplots(nrows=2, figsize=figsize)
                            axx.scatter(dftg.x, dftg.z_no_corr, s=s, color=sciblue)
                            axx.scatter(dftg.x, dftg.z_corr_tilt, s=s, marker='^', color=scigreen)
                            axx.scatter(dftg.x, dftg.z_corr_tilt_fc, s=s, marker='D', color=scired)

                            axy.scatter(dftg.y, dftg.z_no_corr, s=s, color=sciblue, label='None')
                            axy.scatter(dftg.y, dftg.z_corr_tilt, s=s, marker='^', color=scigreen, label='Tilt')
                            axy.scatter(dftg.y, dftg.z_corr_tilt_fc, s=s, marker='D', color=scired, label='T+F.C.')

                            axx.set_xlabel('x (pix.)')
                            axx.set_ylabel(r'$z$')
                            axy.set_xlabel('y (pix.)')
                            axx.set_ylabel(r'$z$')
                            axy.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            plt.tight_layout()
                            plt.savefig(path_z_corr_positions + '/z-corr-positions_z-bin_{}.png'.format(bzt))
                            # plt.show()
                            plt.close()

                    # ---

                    # plot correction displacement per particle: dz_corr(x, y)
                    if plot_dz_corrs_by_xy:
                        dft['dz_no_corr'] = dft['z_no_corr'] - dft['z_no_corr']
                        dft['dz_corr_tilt'] = dft['z_corr_tilt'] - dft['z_no_corr']
                        dft['dz_corr_fc'] = dft['z_corr_fc'] - dft['z_no_corr']
                        dft['dz_corr_tilt_fc'] = dft['z_corr_tilt_fc'] - dft['z_no_corr']

                        for bzt in bin_z_trues:
                            dftg = dft[dft['bin'] == bzt]
                            dftg = dftg[(dftg['z_no_corr'] > bzt - dz_lim) & (dftg['z_no_corr'] < bzt + dz_lim)]

                            fig, (axx, axy) = plt.subplots(nrows=2, figsize=figsize)
                            axx.scatter(dftg.x, dftg.dz_no_corr, s=s, color=sciblue)
                            axx.scatter(dftg.x, dftg.dz_corr_tilt, s=s, marker='^', color=scigreen)
                            axx.scatter(dftg.x, dftg.dz_corr_fc, s=s, color=sciorange)
                            axx.scatter(dftg.x, dftg.dz_corr_tilt_fc, s=s, marker='D', color=scired)

                            axy.scatter(dftg.y, dftg.dz_no_corr, s=s, color=sciblue, label='None')
                            axy.scatter(dftg.y, dftg.dz_corr_tilt, s=s, marker='^', color=scigreen, label='Tilt')
                            axy.scatter(dftg.y, dftg.dz_corr_fc, s=s, color=sciorange, label='F.C.')
                            axy.scatter(dftg.y, dftg.dz_corr_tilt_fc, s=s, marker='D', color=scired, label='T.+F.C.')

                            axx.set_xlabel('x (pix.)')
                            axx.set_ylabel(r'$\Delta z$')
                            axy.set_xlabel('y (pix.)')
                            axx.set_ylabel(r'$\Delta z$')
                            axy.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            plt.tight_layout()
                            plt.savefig(path_z_corr_displacement + '/z-corr-rel-dz_z-bin_{}.png'.format(bzt))
                            # plt.show()
                            plt.close()

                # ---

                #   2.d - add error columns

                dft['error_corr_tilt'] = dft['z_corr_tilt'] - dft['z_true']
                dft['error_corr_tilt_fc'] = dft['z_corr_tilt_fc'] - dft['z_true']
                dft['error_corr_fc'] = dft['z_corr_fc'] - dft['z_true']

                if column_for_z == 'z_corr_tilt_fc':
                    dft['z'] = dft['z_corr_tilt_fc']
                    dft['error'] = dft['error_corr_tilt_fc']
                elif column_for_z == 'z_corr_tilt':
                    dft['z'] = dft['z_corr_tilt']
                    dft['error'] = dft['error_corr_tilt']
                elif column_for_z == 'z_corr_fc':
                    dft['z'] = dft['z_corr_fc']
                    dft['error'] = dft['error_corr_fc']

                # ---

            else:
                dft['z_corr_tilt'] = dft['z']
                dft['z_corr_tilt_fc'] = dft['z']
                dft['error_corr_tilt'] = dft['z_corr_tilt'] - dft['z_true']
                dft['error_corr_tilt_fc'] = dft['z_corr_tilt_fc'] - dft['z_true']

            # ---

            # ----------------------------------------------------------------------------------------------------------------------
            # 3. EVALUATE TRUE PARTICLE POSITIONS

            # ---

            plot_and_compute_fiji = False
            if plot_and_compute_fiji:

                # true particle in-focus positions in 5 um test images (from FIJI)
                path_true_particle_locations = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/' \
                                               '11.06.21_z-micrometer-v2/analyses/shared-results/fiji-particle-locations/' \
                                               'true_positions.xlsx'

                df_true = pd.read_excel(path_true_particle_locations)

                # shift locations to match IDPT + padding
                xy_cols = [['x', 'y'], ['gauss_xc', 'gauss_yc']]
                df_true['x'] = df_true['x'] + padding_rel_true_x
                df_true['y'] = df_true['y'] + padding_rel_true_y
                df_true['gauss_xc'] = df_true['gauss_xc'] + padding_rel_true_x
                df_true['gauss_yc'] = df_true['gauss_yc'] + padding_rel_true_y

                fig, ax = plt.subplots()
                ax.scatter(df_true.x, df_true.y, s=df_true.contour_area, c=df_true.id)
                ax.invert_yaxis()
                ax.set_title('integer positions from FIJI')
                plt.savefig(path_results_true_positions + '/fiji_positions.png')
                plt.show()

                if not os.path.exists(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx'):
                    frames_and_ztrues = dft.sort_values('frame')[['frame', 'z_true']].values.tolist()
                    frames_and_ztrues = np.unique(np.array(frames_and_ztrues), axis=0)
                    temp = []
                    for ft, zt in frames_and_ztrues:
                        df_true['z_true'] = zt
                        df_true['z'] = zt
                        df_true['frame'] = ft
                        temp.append(df_true.copy())
                    df_true = pd.concat(temp)
                    df_true.to_excel(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx', index=False)

            else:
                df_true = pd.read_excel(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx')

            # ---

            # COMPARE IN-FOCUS TO FIJI
            compare_in_focus_to_fiji = False
            if compare_in_focus_to_fiji:

                # in-focus positions
                dfzf = dft[dft['frame'] == baseline_frame]

                fig, ax = plt.subplots()
                if 'contour_area' in dfzf.columns:
                    ax.scatter(dfzf.x, dfzf.y, s=dfzf.contour_area, c=dfzf.id, cmap='inferno', alpha=0.5)
                else:
                    ax.scatter(dfzf.x, dfzf.y, color='black', alpha=0.5)
                ax.invert_yaxis()
                ax.set_title('Baseline image (Frame = {})'.format(baseline_frame))
                plt.savefig(path_results_true_positions + '/{}_in-focus_positions.png'.format(method))
                plt.show()

                # compare FIJI to in-focus
                fig, ax = plt.subplots()
                if 'contour_area' in dfzf.columns:
                    ax.scatter(dfzf.x, dfzf.y, s=dfzf.contour_area, color='blue', alpha=0.5, zorder=3.1,
                               label=r'$z_{f}$')
                else:
                    ax.scatter(dfzf.x, dfzf.y, color='blue', alpha=0.5, zorder=3.1, label=r'$z_{f}$')
                ax.scatter(df_true.x, df_true.y, s=df_true.contour_area, color='red', alpha=0.75, zorder=3.2,
                           label='FIJI')
                ax.invert_yaxis()
                ax.legend()
                ax.set_title('{} vs. FIJI'.format(method))
                plt.savefig(path_results_true_positions + '/compare_in_focus_to_fiji_positions.png')
                plt.show()

                # plot big figure with small points to see it better
                for x_col, y_col in xy_cols:
                    fig, ax = plt.subplots(figsize=(size_x_inches * 2, size_y_inches * 2))
                    ax.scatter(dfzf.x, dfzf.y, s=2, color='blue', alpha=0.5, zorder=3.1, label=r'$z_{f}$')
                    ax.scatter(df_true[x_col], df_true[y_col], s=2, color='red', alpha=0.75, zorder=3.2, label='FIJI')
                    ax.invert_yaxis()
                    ax.legend()
                    ax.set_title('{} vs. FIJI'.format(method))
                    plt.savefig(path_results_true_positions +
                                '/compare_in_focus_to_fiji_positions_big_xcol={}.png'.format(x_col))
                    plt.show()

                    # plot big figure with small points to see it better (flip z-order)
                    fig, ax = plt.subplots(figsize=(size_x_inches * 2, size_y_inches * 2))
                    ax.scatter(dfzf.x, dfzf.y, s=2, color='blue', alpha=0.75, zorder=3.2, label=r'$z_{f}$')
                    ax.scatter(df_true[x_col], df_true[y_col], s=2, color='red', alpha=0.5, zorder=3.1, label='FIJI')
                    ax.invert_yaxis()
                    ax.legend()
                    ax.set_title('{} vs. FIJI'.format(method))
                    plt.savefig(path_results_true_positions +
                                '/compare_in_focus_to_fiji_positions_big-flip-z-order_xcol={}.png'.format(x_col))
                    plt.show()

            # ---

            """# --------------------------------------------------------------------------------------------------------------
            # 3. (OPTIONAL) FIND Z_F IN TEST COORDS BY GAUSSIAN AMPLITUDE

            find_zf_in_test_coords = False
            if find_zf_in_test_coords:
                pids = dft.id.unique()

                dft_f = []
                for pid in pids:
                    dfpid = dft[dft['id'] == pid]

                    # data
                    dfpid_fit = dfpid.dropna()
                    dfpid_fit = dfpid_fit[dfpid_fit['gauss_A'] < 2 ** 16]
                    dfpid_fit = dfpid_fit.groupby('z_true').mean().reset_index()
                    x = dfpid_fit.z_true.to_numpy()
                    y = dfpid_fit.gauss_A.to_numpy() - dfpid.gauss_A.min()

                    # fit
                    try:
                        popt, pcov = curve_fit(functions.gauss_1d, x, y)

                        # resample
                        xf = np.linspace(x.min(), x.max(), len(x) * 10)
                        yf = functions.gauss_1d(xf, *popt)

                        # zf (in-focus)
                        zf = xf[np.argmax(yf)]

                        # zf (nearest calib)
                        dz_zf = x - zf
                        zf_nearest_calib = x[np.argmin(np.abs(dz_zf))]

                        if pid % 10 == 0:
                            fig, ax = plt.subplots()
                            ax.plot(x, y, 'o', ms=2, label=pid)
                            ax.plot(xf, yf, '--', linewidth=0.5, color='black', label='Fit')
                            ax.plot(zf, np.max(yf), '*', ms=3, color='red',
                                    label=r'$z_{f}=$' + ' {} '.format(np.round(zf, 2)) + r'$\mu m$')
                            ax.set_xlabel('z')
                            ax.set_ylabel('A')
                            ax.legend()
                            plt.show()

                    except RuntimeError:
                        popt = None
                        zf = np.nan
                        zf_nearest_calib = np.nan

                        fig, ax = plt.subplots()
                        ax.plot(x, y, 'o', ms=2, label=pid)
                        ax.set_xlabel('z')
                        ax.set_ylabel('A')
                        ax.legend()
                        plt.show()

                    # store
                    dfpid['zf'] = zf
                    dfpid['zf_nearest_calib'] = zf_nearest_calib
                    dft_f.append(dfpid)

                # export dataframe
                dft_f = pd.concat(dft_f)
                dft_f.to_excel(path_test_coords_corrected + '/{}_zf.xlsx'.format(test_name))

            # ---"""

            # ---

            # --------------------------------------------------------------------------------------------------------------
            # 4. CALCULATE DISPLACEMENT RELATIVE TO FIJI

            calculate_error_relative_to_fiji = True
            if calculate_error_relative_to_fiji:

                if method == 'gdpt':
                    xy_columns = [['x', 'y'], ['gauss_xc', 'gauss_yc']]
                    name_dist_columns = ['dxy', 'gauss_dxy']
                else:
                    xy_columns = [['x', 'y'], ['xm', 'ym'], ['xg', 'yg'], ['gauss_xc', 'gauss_yc']]
                    name_dist_columns = ['dxy', 'dxym', 'dxyg', 'gauss_dxy']

                ground_truth_xy_columns = ['gauss_xc', 'gauss_yc']

                if method == 'spct':
                    drop_na = True
                    drop_na_cols = ['dist_gauss_dxy']
                else:
                    drop_na = False
                    drop_na_cols = None

                dft = analyze.calculate_distance_from_baseline_positions(dft, xy_columns,
                                                                         df_true, ground_truth_xy_columns,
                                                                         name_dist_columns,
                                                                         error_threshold=20,
                                                                         drop_na=drop_na,
                                                                         drop_na_cols=drop_na_cols,
                                                                         )

                if method == 'gdpt':
                    dft['id'] = dft['nid_dxy']

            else:
                # calculate relative to in-focus position
                df_true = dft[dft['frame'].isin(baseline_frames)].groupby('id').mean().reset_index()

                if method == 'gdpt':
                    ground_truth_xy_columns = ['com_x', 'com_y']
                    xy_columns = [['x', 'y'], ['gauss_xc', 'gauss_yc']]
                    name_dist_columns = ['dxy', 'gauss_dxy']
                else:
                    ground_truth_xy_columns = ['xg', 'yg']
                    xy_columns = [['x', 'y'], ['xm', 'ym'], ['xg', 'yg'], ['gauss_xc', 'gauss_yc']]
                    name_dist_columns = ['dxy', 'dxym', 'dxyg', 'gauss_dxy']

                if method == 'spct':
                    ground_truth_xy_columns = ['gauss_xc', 'gauss_yc']
                    drop_na = True
                    drop_na_cols = ['dist_gauss_dxy']
                else:
                    drop_na = False
                    drop_na_cols = None

                dft = analyze.calculate_distance_from_baseline_positions(dft, xy_columns,
                                                                         df_true, ground_truth_xy_columns,
                                                                         name_dist_columns,
                                                                         error_threshold=20,
                                                                         drop_na=drop_na,
                                                                         drop_na_cols=drop_na_cols,
                                                                         )

                if method == 'gdpt':
                    dft['id'] = dft['nid_dxy']

            # ---

            evaluate_in_focus_error_relative_to_fiji = False
            if evaluate_in_focus_error_relative_to_fiji:
                dft_f = dft[(dft['z_true'] > z_range[0]) & (dft['z_true'] < z_range[1])]
                dfg = dft_f.groupby('z_true').mean()
                dfs = dft_f.groupby('z_true').std()
                dfc = dft_f.groupby('z_true').count()

                # plot
                for ndc in name_dist_columns:
                    y1 = 'error'
                    y2 = 'x_error_' + ndc
                    y3 = 'y_error_' + ndc
                    y4 = 'dist_' + ndc

                    fig, (ax, axr, axs, axt) = plt.subplots(nrows=4, sharex=True,
                                                            figsize=(size_x_inches,
                                                                     size_y_inches * 2))

                    ax.errorbar(dfg.index, dfg[y1], yerr=dfs[y1], fmt='o', ms=2, capsize=1, elinewidth=0.5)
                    axr.errorbar(dfg.index, dfg[y2], yerr=dfs[y2], fmt='o', ms=2, capsize=2, elinewidth=1)
                    axs.errorbar(dfg.index, dfg[y3], yerr=dfs[y3], fmt='o', ms=2, capsize=2, elinewidth=1)
                    axt.errorbar(dfg.index, dfg[y4], yerr=dfs[y4], fmt='o', ms=2, capsize=2, elinewidth=1)

                    ax.axhline(y=0, color='black', alpha=0.25)
                    axr.axhline(y=0, color='black', alpha=0.25)
                    axs.axhline(y=0, color='black', alpha=0.25)
                    ax.set_ylim([-2.5, 2.5])
                    axr.set_ylim([-.5, .5])
                    axs.set_ylim([-.5, .5])

                    ax.set_ylabel(y1)
                    axr.set_ylabel(y2)
                    axs.set_ylabel(y3)
                    axt.set_ylabel(y4)

                    plt.tight_layout()
                    plt.savefig(path_results_true_positions +
                                '/compare_in_focus_to_true_positions_xycols={}.png'.format(ndc))
                    plt.show()

            # ---

            # --------------------------------------------------------------------------------------------------------------
            # 4. CALCULATE DISPLACEMENT RELATIVE TO GROUPBY(Z_F)

            calculate_displacement_relative_to_zf = True
            if calculate_displacement_relative_to_zf:

                i_num_rows_pre = len(dft)

                # compute the radial distance
                if 'r' not in dft.columns:
                    dft['r'] = np.sqrt((img_xc - dft.x) ** 2 + (img_yc - dft.y) ** 2)

                if 'xm' in dft.columns:
                    if 'rm' not in dft.columns:
                        dft['rm'] = np.sqrt((img_xc - dft.xm) ** 2 + (img_yc - dft.ym) ** 2)

                if 'xg' in dft.columns:
                    if 'rg' not in dft.columns:
                        dft['rg'] = np.sqrt((img_xc - dft.xg) ** 2 + (img_yc - dft.yg) ** 2)

                if 'gauss_xc' in dft.columns:
                    if 'gauss_rc' not in dft.columns:
                        dft['gauss_rc'] = np.sqrt((img_xc - dft.gauss_xc) ** 2 + (img_yc - dft.gauss_yc) ** 2)

                # ---

                pids = dft.id.unique()

                dft_f = []
                for pid in pids:

                    # get this dataframe
                    dfpid = dft[dft['id'] == pid]

                    if len(dfpid) < 3:
                        continue

                    # get in-focus position
                    dfgid = dfpid.groupby('z_true').mean().reset_index()
                    idx_min = np.argmin(dfgid.z_true.abs())
                    dfgid_f = dfgid[dfgid['z_true'] == dfgid.iloc[idx_min].z_true]

                    r0 = dfgid_f.r.values
                    if len(r0) > 0:
                        r0 = r0[0]
                        dr = dfpid.r - r0
                    else:
                        r0 = np.nan
                        dr = np.nan

                    # calculate displacement relative to in-focus
                    dfpid['r0'] = r0
                    dfpid['dr'] = dr

                    if 'rm' in dft.columns:
                        rm0 = dfgid_f.rm.values
                        if len(rm0) > 0:
                            xm0 = dfgid_f.xm.values[0]
                            dxm = dfpid.xm - xm0
                            ym0 = dfgid_f.ym.values[0]
                            dym = dfpid.ym - ym0
                            rm0 = rm0[0]
                            drm = dfpid.rm - rm0
                        else:
                            xm0 = np.nan
                            dxm = np.nan
                            ym0 = np.nan
                            dym = np.nan
                            rm0 = np.nan
                            drm = np.nan

                        dfpid['xm0'] = xm0
                        dfpid['dxm'] = dxm
                        dfpid['ym0'] = ym0
                        dfpid['dym'] = dym
                        dfpid['rm0'] = rm0
                        dfpid['drm'] = drm

                    if 'rg' in dft.columns:
                        rg0 = dfgid_f.rg.values
                        if len(rg0) > 0:
                            xg0 = dfgid_f.xg.values[0]
                            dxg = dfpid.xg - xg0
                            yg0 = dfgid_f.yg.values[0]
                            dyg = dfpid.yg - yg0
                            rg0 = rg0[0]
                            drg = dfpid.rg - rg0
                        else:
                            xg0 = np.nan
                            dxg = np.nan
                            yg0 = np.nan
                            dyg = np.nan
                            rm0 = np.nan
                            drg = np.nan

                        dfpid['xg0'] = xg0
                        dfpid['dxg'] = dxg
                        dfpid['yg0'] = yg0
                        dfpid['dyg'] = dyg
                        dfpid['rg0'] = rg0
                        dfpid['drg'] = drg

                    if 'gauss_rc' in dft.columns:
                        gauss_r0 = dfgid_f.gauss_rc.values
                        if len(gauss_r0) > 0:
                            gauss_x0 = dfgid_f.gauss_xc.values[0]
                            gauss_dx = dfpid.gauss_xc - gauss_x0
                            gauss_y0 = dfgid_f.gauss_yc.values[0]
                            gauss_dy = dfpid.gauss_yc - gauss_y0
                            gauss_r0 = gauss_r0[0]
                            gauss_dr = dfpid.gauss_rc - gauss_r0
                        else:
                            gauss_x0 = np.nan
                            gauss_dx = np.nan
                            gauss_y0 = np.nan
                            gauss_dy = np.nan
                            gauss_r0 = np.nan
                            gauss_dr = np.nan

                        dfpid['gauss_x0'] = gauss_x0
                        dfpid['gauss_dx'] = gauss_dx
                        dfpid['gauss_y0'] = gauss_y0
                        dfpid['gauss_dy'] = gauss_dy
                        dfpid['gauss_r0'] = gauss_r0
                        dfpid['gauss_dr'] = gauss_dr

                    # store
                    dft_f.append(dfpid)

                # collect
                dft_f = pd.concat(dft_f)
                dft_f.to_excel(path_test_coords_corrected + '/{}_dzf.xlsx'.format(test_name), index=False)

                # compare # rows before and after
                i_num_rows_post = len(dft)
                print("{} rows difference before and after dr relative to in-focus positions".format(
                    i_num_rows_pre - i_num_rows_post))

                dft_f = dft_f[dft_f['error'].abs() < 3.5]
                print("dfz mean error = {} (where error < 3.5)".format(np.round(dft_f.error.mean(), 3)))
                del dft_f

            # ---

    else:
        # read dataframe
        dft = pd.read_excel(path_test_coords_corrected + '/{}_dzf.xlsx'.format(test_name))

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT COUNTS

    plot_counts = False
    if plot_counts:

        path_results_count = path_results + '/counts'
        if not os.path.exists(path_results_count):
            os.makedirs(path_results_count)

        # copy dataframe and exclude unecessary columns (which may contain NaNs)
        df = dft[['frame', 'z_true', 'id', 'z']].copy()

        # filter z_range
        df = df[(df['z_true'] > z_range[0]) & (df['z_true'] < z_range[1])]

        dftc = df.groupby(['z_true', 'id']).count().reset_index()
        dftc = dftc.groupby('z_true').count()
        fig, ax = plt.subplots()
        ax.plot(dftc.index, dftc.id, 'o', ms=3, label='raw')

        filter_counts = None  # 8
        if filter_counts is not None:
            # filter on counts
            dftc = df.groupby(['z_true', 'id']).count().reset_index()
            dftc = dftc.groupby('id').count().reset_index()
            passing_ids_counts = dftc[dftc['z'] > filter_counts].id.unique()
            dft_fc = df[df.id.isin(passing_ids_counts)]
            dftc = dft_fc.groupby(['z_true', 'id']).count().reset_index()
            dftc = dftc.groupby('z_true').count()
            ax.plot(dftc.index, dftc.id, 'o', ms=3, label=r'counts $>$ {}'.format(filter_counts))

        # filter on ID's in frame 39
        passing_ids_baseline = df[df['frame'] == baseline_frame].id.unique()
        dft_b = df[df.id.isin(passing_ids_baseline)]

        # plot again
        dftc = dft_b.groupby(['z_true', 'id']).count().reset_index()
        dftc = dftc.groupby('z_true').count()
        ax.plot(dftc.index, dftc.id, 'o', ms=3, label='ID(baseline)')
        ax.set_xlabel('z')
        ax.set_ylabel('Counts')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_results_count + '/counts_per_dz.png')
        plt.show()
        plt.close()

    else:
        passing_ids_baseline = dft.id.unique()
        passing_ids_counts = dft.id.unique()

    # ------------------------------------------------------------------------------------------------------------------

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 1. STORE "RAW" STATS (AFTER FILTERING RANGE)

    # filter range so test matches calibration range
    dft = dft[(dft['z_true'] > z_range[0]) & (dft['z_true'] < z_range[1])]

    # number of particles identified
    dft_counts = dft.groupby('z_true').count().reset_index()
    i_num_rows_per_z = dft_counts.id.to_numpy()
    i_num_rows_per_z_df = dft_counts[['z_true', 'id']]
    i_num_rows_per_z_df = i_num_rows_per_z_df.round({'z_true': 0})
    i_num_rows_per_z_df = i_num_rows_per_z_df.astype({'z_true': int})

    i_num_rows = len(dft)
    i_num_pids = len(dft.id.unique())

    # get filtering stats
    num_rows_cm_filter = len(dft[dft['cm'] < min_cm])
    num_rows_barnkob_filter = len(dft[dft['error'].abs() > filter_barnkob])

    print("{} out of {} rows ({}%) below min_cm filter: {}".format(num_rows_cm_filter,
                                                                   i_num_rows,
                                                                   np.round(num_rows_cm_filter / i_num_rows * 100,
                                                                            1),
                                                                   min_cm)
          )

    print("{} out of {} rows ({}%) above error filter: {}".format(num_rows_barnkob_filter,
                                                                  i_num_rows,
                                                                  np.round(
                                                                      num_rows_barnkob_filter / i_num_rows * 100,
                                                                      1),
                                                                  filter_barnkob)
          )

    # C_m filter
    print("C_m = {}  (mean of all pids)".format(np.round(dft.cm.mean(), 3)))
    dft = dft[dft['cm'] > min_cm]
    print("C_m = {}  (mean of all pids passing min_cm filter)".format(np.round(dft.cm.mean(), 3)))

    # Barnkob filter
    dft = dft[dft['error'].abs() < filter_barnkob]
    print("C_m = {}  (mean of all pids passing Barnkob filter)".format(np.round(dft.cm.mean(), 3)))

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 0.5 EXPORT "FINAL" TEST COORDS (THOSE USED FOR PLOTTING AND DISCUSSION)

    if export_processed_coords:

        if not os.path.exists(path_test_coords_post_processed):
            os.makedirs(path_test_coords_post_processed)

        dft.to_excel(path_test_coords_post_processed + '/{}_dzf-post-processed.xlsx'.format(test_name))

        # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE POST-PROCESSED COORDS

evaluate_test = True
if evaluate_test:

    # read coords
    dft = pd.read_excel(path_test_coords_post_processed + '/{}_dzf-post-processed.xlsx'.format(test_name))

    # ------------------------------------------------------------------------------------------------------------------
    # 1. SPLIT DATAFRAME INTO BINS

    true_zs = dft.z_true.unique()
    dft['bin'] = np.round(dft['z_true'], 0).astype(int)
    dzs = dft.bin.unique()
    num_dz_steps = len(dzs)
    print("Number of dz steps = {}".format(num_dz_steps))

    dfs = []
    names = []
    initial_stats = []

    for i, dz in enumerate(dzs):
        dftz = dft[dft['bin'] == dz]
        dfs.append(dftz)
        names.append(i)
        initial_stats.append([i, dz, len(dftz), len(dftz.id.unique())])

    # ---

    # plot 2: per-particle rmse_z (x, y)
    plot_per_pid_rmse_z = False
    if plot_per_pid_rmse_z:
        vmin, vmax = 0.5, 5
        cmap = mpl.cm.coolwarm
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # filters
        cm_lim = min_cm
        err_lim = filter_step_size

        # processing
        dft_pid = dft[['id', 'x', 'y', 'r', 'cm', 'error']]
        dft_pid = dft_pid[dft_pid['error'] < err_lim]
        dft_pid = dft_pid[dft_pid['cm'] > cm_lim]

        dft_pid['error_squared'] = dft_pid['error'] ** 2
        dft_pid = dft_pid.groupby('id').mean()
        dft_pid['rmse_z'] = np.sqrt(dft_pid['error_squared'])
        dft_pid = dft_pid.sort_values('rmse_z')

        sp = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_field-dependence-on-localization'
        dft_pid.to_excel(sp + '/per-particle-rmse_z_errlim{}.xlsx'.format(err_lim))
        # raise ValueError()

        # plot
        fig, ax = plt.subplots(figsize=(size_y_inches, size_y_inches), subplot_kw={'aspect': 'equal'})

        sc = ax.scatter(dft_pid['x'], dft_pid['y'], c=cmap(norm(dft_pid['rmse_z'])), s=10)

        ax.set_xlabel(r'$X (pix.)$')
        ax.set_xlim([padding, 512 + padding])
        ax.set_xticks([padding, 512 + padding], labels=[0, 512])
        ax.set_ylabel(r'$Y (pix.)$')
        ax.set_ylim([padding, 512 + padding])
        ax.set_yticks([padding, 512 + padding], labels=[0, 512])
        ax.invert_yaxis()

        # color bar
        cbar_lbl = r'$\overline{\sigma}_{z}^{i} \: (\mu m)$'
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cax, orientation='vertical', label=cbar_lbl, extend='both')

        # cbar = plt.colorbar(sc)
        # cbar.set_label(r'$\overline{\sigma}_{z}^{i} \: (\mu m)$')

        plt.tight_layout()
        plt.savefig(path_results_error + '/mean-rmse-z-per-pid-colored_by_x-y_cm{}_err{}_.svg'.format(cm_lim, err_lim))
        plt.show()
        plt.close()
        raise ValueError()

    # ------------------------------------------------------------------------------------------------------------------
    # 1. GENERATE IN-FOCUS SURFACE MODELS IF NONE

    if bispl_raw is None:
        dict_fit_plane, dict_fit_plane_bspl_corrected, dict_flat_plane, bispl, bispl_raw = fit_plane_and_bispl(path_figs=None)

    # ------------------------------------------------------------------------------------------------------------------
    # 2. EVALUATE 3D PARTICLE DISTRIBUTION RELATIVE TO FITTED SURFACE

    evaluate_3d_distribution = False
    if evaluate_3d_distribution:

        plot_xy_and_plane = True
        plot_tilt_and_fc = True
        plot_fc = True
        plot_plane_fcorr = True
        plot_plane_raw = True
        plot_flat_plane = True

        save_figs = True

        path_results_distribution = path_results + '/3d-distribution'
        if not os.path.exists(path_results_distribution):
            os.makedirs(path_results_distribution)

        # ---

        # fit field curvature (bispl)
        if method in ['spct', 'gdpt']:
            dft_baseline_calib_id = dft[dft['frame'].isin(baseline_frames)]
            dft_baseline_calib_id = dft_baseline_calib_id[dft_baseline_calib_id['id'] == calib_id_from_testset]
            dft_baseline_calib_id = dft_baseline_calib_id.groupby('z_true').mean()
            ref_xy = dft_baseline_calib_id[['x', 'y']].to_numpy()[0]
        else:
            ref_xy = np.array([img_xc, img_yc])

        # ---

        # dicts for models
        dict_bispl_raw = {'bispl': bispl_raw, 'ref_xy': ref_xy}  # , 'Transform': 'y', 'img_xc': img_xc, 'img_yc': img_yc
        dict_bispl = {'bispl': bispl, 'ref_xy': ref_xy}
        dict_fit_plane_bspl_corrected.update({'ref_xy': ref_xy})
        dict_fit_plane.update({'ref_xy': ref_xy})
        dict_flat_plane.update({'ref_xy': ref_xy})

        # ---

        # plot 'z' by 'x' and 'y' and plot plane
        if plot_xy_and_plane:

            path_figs_dist_xy = path_results_distribution + '/figs_xy'
            if not os.path.exists(path_figs_dist_xy):
                os.makedirs(path_figs_dist_xy)

            # ---

            # setup
            df = dft
            column_to_bin = 'z_true'
            column_to_fit = 'z'
            column_to_color = 'id'
            xy_cols = ['x', 'y']
            dict_plane = dict_fit_plane
            scatter_size = 5
            plane_alpha = 0.2
            path_results_example = path_figs_dist_xy

            # processing
            xmin, xmax = df[xy_cols[0]].min(), df[xy_cols[0]].max()
            ymin, ymax = df[xy_cols[1]].min(), df[xy_cols[1]].max()

            x_data = np.array([xmin, xmax, xmax, xmin, xmin])
            y_data = np.array([ymin, ymin, ymax, ymax, ymin])

            popt_plane = dict_plane['popt_pixels']
            z_data = functions.calculate_z_of_3d_plane(x_data, y_data, popt_plane)

            # iterate
            for cb in df[column_to_bin].unique():
                dfb = df[df[column_to_bin] == cb]

                # plane z-offset
                z_points_mean = dfb[column_to_fit].mean()
                z_plane_mean = np.mean(z_data)
                z_offset = z_plane_mean - z_points_mean

                fig, (axx, axy) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.2, size_y_inches * 0.8))

                axx.scatter(dfb[xy_cols[0]], dfb[column_to_fit], c=dfb[column_to_color], s=scatter_size)
                axx.plot(x_data, z_data - z_offset, color='red', alpha=plane_alpha)

                axy.scatter(dfb[xy_cols[1]], dfb[column_to_fit], c=dfb[column_to_color], s=scatter_size)
                axy.plot(y_data, z_data - z_offset, color='red', alpha=plane_alpha)

                axx.set_xlabel(xy_cols[0])
                axx.set_ylabel(column_to_fit)
                axy.set_xlabel(xy_cols[1])
                plt.tight_layout()
                plt.savefig(path_results_example + '/scatter-xy_bin-{}={}.png'.format(column_to_bin, np.round(cb, 2)))
                plt.close()

        # ---

        # fit field-curvature (bispl)
        if plot_tilt_and_fc:

            path_figs_distribution_tilt_and_fc = path_results_distribution + '/figs_tilt-and-field-curvature'
            if not os.path.exists(path_figs_distribution_tilt_and_fc):
                os.makedirs(path_figs_distribution_tilt_and_fc)

            model = 'bispl'
            dict_model = dict_bispl_raw
            df = dft
            column_to_bin = 'z_true'
            column_to_fit = 'z'
            xy_cols = ['x', 'y']
            std_filter = None
            distance_filter = None  # depth_of_field / 2

            dfres = analyze.evaluate_reference_model_by_bin(model,
                                                            dict_model,
                                                            df,
                                                            column_to_bin,
                                                            column_to_fit,
                                                            xy_cols,
                                                            path_results=path_figs_distribution_tilt_and_fc,
                                                            std_filter=std_filter,
                                                            distance_filter=distance_filter,
                                                            save_figs=save_figs,
                                                            )

            dfres.to_excel(path_results_distribution + '/df_results_reference-bispl-raw_ie_tilt-fc-corr.xlsx')

            # ---

        # ---

        # fit field-curvature (bispl)
        if plot_fc:

            path_figs_dist_fc = path_results_distribution + '/figs_field-curvature'
            if not os.path.exists(path_figs_dist_fc):
                os.makedirs(path_figs_dist_fc)

            model = 'bispl'
            dict_model = dict_bispl
            df = dft
            column_to_bin = 'z_true'
            column_to_fit = 'z'
            xy_cols = ['x', 'y']
            std_filter = None
            distance_filter = None  # depth_of_field / 2

            dfres = analyze.evaluate_reference_model_by_bin(model,
                                                            dict_model,
                                                            df,
                                                            column_to_bin,
                                                            column_to_fit,
                                                            xy_cols,
                                                            path_results=path_figs_dist_fc,
                                                            std_filter=std_filter,
                                                            distance_filter=distance_filter,
                                                            save_figs=save_figs,
                                                            )

            dfres.to_excel(path_results_distribution + '/df_results_reference-bispl.xlsx')

            # ---

        # ---

        # fit plane from 'bispl-corrected' coordinates
        if plot_plane_fcorr:

            path_figs_dist_plane_bispl_corr = path_results_distribution + '/figs_plane_fc-corr'
            if not os.path.exists(path_figs_dist_plane_bispl_corr):
                os.makedirs(path_figs_dist_plane_bispl_corr)

            model = 'plane'
            dict_model = dict_fit_plane_bspl_corrected
            df = dft
            column_to_bin = 'z_true'
            column_to_fit = 'z'
            xy_cols = ['x', 'y']
            std_filter = None
            distance_filter = None  # depth_of_field / 2

            dfres = analyze.evaluate_reference_model_by_bin(model,
                                                            dict_model,
                                                            df,
                                                            column_to_bin,
                                                            column_to_fit,
                                                            xy_cols,
                                                            path_results=path_figs_dist_plane_bispl_corr,
                                                            std_filter=std_filter,
                                                            distance_filter=distance_filter,
                                                            save_figs=save_figs,
                                                            )

            dfres.to_excel(path_results_distribution + '/df_results_reference-plane_bispl-corr.xlsx')

        # ---

        # Fit plane from 'raw' coordinates

        if plot_plane_raw:

            path_figs_dist_plane_raw = path_results_distribution + '/figs_plane_raw'
            if not os.path.exists(path_figs_dist_plane_raw):
                os.makedirs(path_figs_dist_plane_raw)

            model = 'plane'
            dict_model = dict_fit_plane
            df = dft
            column_to_bin = 'z_true'
            column_to_fit = 'z'
            xy_cols = ['x', 'y']
            std_filter = None
            distance_filter = None  # depth_of_field / 2

            dfres = analyze.evaluate_reference_model_by_bin(model,
                                                            dict_model,
                                                            df,
                                                            column_to_bin,
                                                            column_to_fit,
                                                            xy_cols,
                                                            path_results=path_figs_dist_plane_raw,
                                                            std_filter=std_filter,
                                                            distance_filter=distance_filter,
                                                            save_figs=save_figs,
                                                            )

            dfres.to_excel(path_results_distribution + '/df_results_reference-plane_raw.xlsx')

            # ---

        # ---

        # Fit flat plane coordinates

        if plot_flat_plane:

            path_figs_dist_plane_flat = path_results_distribution + '/figs_plane_flat'
            if not os.path.exists(path_figs_dist_plane_flat):
                os.makedirs(path_figs_dist_plane_flat)

            model = 'plane'
            dict_model = dict_flat_plane
            df = dft
            column_to_bin = 'z_true'
            column_to_fit = 'z'
            xy_cols = ['x', 'y']
            std_filter = None
            distance_filter = None  # depth_of_field / 2

            dfres = analyze.evaluate_reference_model_by_bin(model,
                                                            dict_model,
                                                            df,
                                                            column_to_bin,
                                                            column_to_fit,
                                                            xy_cols,
                                                            path_results=path_figs_dist_plane_flat,
                                                            std_filter=std_filter,
                                                            distance_filter=distance_filter,
                                                            save_figs=save_figs,
                                                            )

            dfres.to_excel(path_results_distribution + '/df_results_reference-plane_flat.xlsx')

            # ---

        # ---

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. EVALUATE DISPLACEMENT ERROR (ID)

    analyze_displacement = False
    if analyze_displacement:

        path_results_displacement = path_results + '/displacement'
        if not os.path.exists(path_results_displacement):
            os.makedirs(path_results_displacement)

        # groupby 'z_true'
        dfg_mean = dft.groupby('z_true').mean().reset_index()
        dfg_std = dft.groupby('z_true').std().reset_index()
        dfg_counts = dft.groupby('z_true').count().reset_index()

        # ---

        # plot z (mean +/- std) by z_true
        plot_z_by_z_true_with_and_without_correction = False
        if plot_z_by_z_true_with_and_without_correction:
            x = 'z_true'
            y1 = 'z'
            y2 = 'z_no_corr'

            fig, ax = plt.subplots()
            ax.plot(dfg_mean[x], dfg_mean[x], linestyle='--', linewidth=0.75, alpha=0.5, color='black')
            ax.errorbar(dfg_mean[x], dfg_mean[y2], yerr=dfg_std[y2],
                        fmt='o', ms=3, capsize=3, elinewidth=2,
                        label='NoCorr={}'.format(np.round(dfg_std.z_no_corr.mean(), 3)))

            ax.errorbar(dfg_mean[x], dfg_mean[y1], yerr=dfg_std[y1],
                        fmt='o', ms=1, capsize=1.5, elinewidth=0.5, color=scired,
                        label='Corr={}'.format(np.round(dfg_std.z.mean(), 3)))

            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_xlim([-52.5, 55])
            ax.set_xticks([-50, -25, 0, 25, 50])
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.legend(title=r'$\overline{sigma_{z}} (stdev.)$')
            plt.tight_layout()
            plt.savefig(path_results_displacement + '/z_by_z_true_w-wo-correction.png')
            plt.show()

            print("Mean std z_no_corr = {}".format(np.round(dfg_std.z_no_corr.mean(), 4)))
            print("Mean std z_corr = {}".format(np.round(dfg_std.z.mean(), 4)))

        # ---

        # iterative closest point algorithm
        apply_iterative_closest_point = True
        if apply_iterative_closest_point:
            from utils import iterative_closest_point as icp
            from sklearn.neighbors import NearestNeighbors

            path_results_icp = path_results_displacement + '/icp_rigid_transforms'
            if not os.path.exists(path_results_icp):
                os.makedirs(path_results_icp)

            # ---

            # evaluate between the 3 frames per z
            eval_between_dz_frames = True
            if eval_between_dz_frames:

                # important modifiers
                dft = dft[dft['error'] < filter_step_size]
                in_plane_distance_threshold = 5

                # iterate
                zts = dft.z_true.unique()
                zts.sort()
                data = []

                ddx, ddy, ddz = [], [], []
                z_distances = []
                for ii in range(len(zts) - 1):

                    dfA = dft[dft['z_true'] == zts[ii]].groupby('id').mean().reset_index()
                    dfB = dft[dft['z_true'] == zts[ii + 1]].groupby('id').mean().reset_index()

                    A = dfA[['x', 'y', 'z']].to_numpy()
                    B = dfB[['x', 'y', 'z']].to_numpy()

                    if len(A) > len(B):
                        ground_truth_xy = A[:, :2]
                        ground_truth_pids = dfA.id.to_numpy()
                        locations = B[:, :2]
                    else:
                        ground_truth_xy = B[:, :2]
                        ground_truth_pids = dfB.id.to_numpy()
                        locations = A[:, :2]

                    # calcualte distance using NearestNeighbors
                    nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_xy)
                    distances, indices = nneigh.kneighbors(locations)

                    distances = np.where(distances < in_plane_distance_threshold, distances, np.nan)

                    nearest_pid = ground_truth_pids[indices]
                    nearest_pid = np.where(distances < in_plane_distance_threshold, nearest_pid, np.nan)
                    nearest_pid = nearest_pid[~np.isnan(nearest_pid)]
                    uniq_pids = np.unique(nearest_pid)

                    # iterative closest point algorithm
                    dfAA = dfA[dfA.id.isin(uniq_pids)]
                    dfBB = dfB[dfB.id.isin(uniq_pids)]

                    if len(dfAA) > len(dfBB):
                        uniq_pids = dfBB.id.unique()
                        dfAA = dfAA[dfAA.id.isin(uniq_pids)]
                    elif len(dfAA) < len(dfBB):
                        uniq_pids = dfAA.id.unique()
                        dfBB = dfBB[dfBB.id.isin(uniq_pids)]

                    A = dfAA[['xg', 'yg', 'z']].to_numpy()
                    B = dfBB[['xg', 'yg', 'z']].to_numpy()

                    print(len(A))
                    print(len(B))

                    N = len(A)
                    T, distances, iterations = icp.icp(B, A, tolerance=0.000001)

                    # Make C a homogeneous representation of B
                    C = np.ones((N, 4))
                    C[:, 0:3] = np.copy(B)

                    # Transform C
                    C = np.dot(T, C.T).T

                    # evaluate transformation results
                    # ddx.append(T[0, 3])
                    # ddy.append(T[1, 3])
                    # ddz.append(T[2, 3])
                    deltax, deltay, deltaz = T[0, 3], T[1, 3], T[2, 3]
                    # z_distances.append(distances.tolist())
                    precision_dist = np.std(distances)
                    rmse_dist = np.sqrt(np.mean(np.square(distances)))
                    data.append([zts[ii], precision_dist, rmse_dist, deltax, deltay, deltaz])

                # package
                df_icp = pd.DataFrame(np.array(data), columns=['z', 'precision', 'rmse', 'dx', 'dy', 'dz'])
                df_icp.to_excel(path_results_icp +
                                '/dfdz_icp_z-error-{}_in-plane-dist-{}.xlsx'.format(filter_step_size,
                                                                                    in_plane_distance_threshold))

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                ax1.plot(df_icp.z, df_icp.rmse, label='rmse')
                ax1.plot(df_icp.z, df_icp.precision, label='precision')
                ax1.legend()
                ax2.plot(df_icp.z, df_icp.dx, label='x')
                ax2.plot(df_icp.z, df_icp.dy, label='y')
                ax2.plot(df_icp.z, df_icp.dz, label='z')
                ax2.legend()
                plt.show()

            # ---

            # evaluate between the 3 frames per z
            eval_between_3_frames = True
            if eval_between_3_frames:

                # iterate
                zts = dft.z_true.unique()
                data = []
                for zt in zts:
                    dfzt = dft[dft['z_true'] == zt]

                    frs = dfzt.frame.unique()
                    frs.sort()

                    ddx, ddy, ddz = [], [], []
                    z_distances = []
                    for i, fr in enumerate(frs):
                        if i == len(frs) - 1:
                            ii = 0
                        else:
                            ii = i + 1
                        dfA = dfzt[dfzt['frame'] == fr]
                        dfB = dfzt[dfzt['frame'] == frs[ii]]

                        A = dfA[['x', 'y', 'z']].to_numpy()
                        B = dfB[['x', 'y', 'z']].to_numpy()

                        if len(A) > len(B):
                            ground_truth_xy = A[:, :2]
                            ground_truth_pids = dfA.id.to_numpy()
                            locations = B[:, :2]
                        else:
                            ground_truth_xy = B[:, :2]
                            ground_truth_pids = dfB.id.to_numpy()
                            locations = A[:, :2]

                        # calcualte distance using NearestNeighbors
                        nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_xy)
                        distances, indices = nneigh.kneighbors(locations)

                        distances = np.where(distances < in_plane_distance_threshold, distances, np.nan)

                        nearest_pid = ground_truth_pids[indices]
                        nearest_pid = np.where(distances < in_plane_distance_threshold, nearest_pid, np.nan)
                        nearest_pid = nearest_pid[~np.isnan(nearest_pid)]

                        # nearest_pid = np.ravel(ground_truth_pids[indices])

                        uniq_pids = np.unique(nearest_pid)

                        # iterative closest point algorithm
                        dfAA = dfA[dfA.id.isin(uniq_pids)]
                        dfBB = dfB[dfB.id.isin(uniq_pids)]

                        if len(dfAA) > len(dfBB):
                            uniq_pids = dfBB.id.unique()
                            dfAA = dfAA[dfAA.id.isin(uniq_pids)]
                        elif len(dfAA) < len(dfBB):
                            uniq_pids = dfAA.id.unique()
                            dfBB = dfBB[dfBB.id.isin(uniq_pids)]

                        A = dfAA[['xg', 'yg', 'z']].to_numpy()
                        B = dfBB[['xg', 'yg', 'z']].to_numpy()

                        print(len(A))
                        print(len(B))

                        N = len(A)
                        T, distances, iterations = icp.icp(B, A, tolerance=0.000001)

                        # Make C a homogeneous representation of B
                        C = np.ones((N, 4))
                        C[:, 0:3] = np.copy(B)

                        # Transform C
                        C = np.dot(T, C.T).T

                        # evaluate transformation results
                        ddx.append(T[0, 3])
                        ddy.append(T[1, 3])
                        ddz.append(T[2, 3])
                        # deltax, deltay, deltaz = T[0, 3], T[1, 3], T[2, 3]
                        z_distances.append(distances.tolist())

                    # error analysis
                    z_distances = np.array(list(itertools.chain(*z_distances)))
                    precision_dist = np.std(z_distances)
                    rmse_dist = np.sqrt(np.mean(np.square(z_distances)))
                    std_dx = np.std(ddx)
                    std_dy = np.std(ddy)
                    std_dz = np.std(ddz)

                    data.append([zt, precision_dist, rmse_dist, std_dx, std_dy, std_dz])

                df_icp = pd.DataFrame(np.array(data),
                                      columns=['z', 'precision', 'rmse', 'std_dx', 'std_dy', 'std_dz'])
                df_icp.to_excel(path_results_icp +
                                '/df_icp_z-error-{}_in-plane-dist-{}.xlsx'.format(filter_step_size,
                                                                                  in_plane_distance_threshold))

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                ax1.plot(df_icp.z, df_icp.rmse, label='rmse')
                ax1.plot(df_icp.z, df_icp.precision, label='precision')
                ax1.legend()
                ax2.plot(df_icp.z, df_icp.std_dx, label='x')
                ax2.plot(df_icp.z, df_icp.std_dy, label='y')
                ax2.plot(df_icp.z, df_icp.std_dz, label='z')
                ax2.legend()
                plt.show()

        # ---

        # fit a 2D plane at each z (to the raw data) and evaluate RMSE + other metrics
        fit_plane_by_z_true_and_evaluate = True
        if fit_plane_by_z_true_and_evaluate:

            plot_fitted_plane_at_each_z = True
            plot_fitted_plane_escalator = True

            # setup
            column_to_bin = 'z_true'
            columns_to_fit = ['z', 'z_corr_tilt', 'z_corr_tilt_fc', 'z_corr_fc', 'z_no_corr']
            xy_cols = ['x', 'y']

            # setup esclator
            vmin, vmax = -0.075, 0.075
            cmap = mpl.cm.coolwarm
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


            for cf in columns_to_fit:

                path_results_fit_plane_eval = path_results_displacement + '/fit-plane-{}-by-z_true'.format(cf)
                if not os.path.exists(path_results_fit_plane_eval):
                    os.makedirs(path_results_fit_plane_eval)

                if plot_fitted_plane_escalator:
                    fig, (axx, axy) = plt.subplots(ncols=2, sharey=True,
                                                   figsize=(size_x_inches * 1.2, size_y_inches * 2))

                # iterate
                zts = dft.z_true.unique()
                data = []
                for zt in zts:
                    dfzt = dft[dft['z_true'] == zt]

                    dict_fit_plane = correct.fit_in_focus_plane(dfzt,
                                                                cf,
                                                                microns_per_pixel,
                                                                img_xc, img_yc)

                    z_plane_center = dict_fit_plane['z_f_fit_plane_image_center']
                    z_plane_num_meas = dict_fit_plane['num_locations']
                    z_plane_rmse = dict_fit_plane['rmse']
                    z_plane_r_squared = dict_fit_plane['r_squared']
                    z_plane_tilt_x = dict_fit_plane['tilt_x_degrees']
                    z_plane_tilt_y = dict_fit_plane['tilt_y_degrees']

                    datum = [test_id, zt, z_plane_center, z_plane_num_meas, z_plane_rmse, z_plane_r_squared,
                             z_plane_tilt_x, z_plane_tilt_y]
                    data.append(datum)

                    # ---

                    if plot_fitted_plane_at_each_z:

                        # setup
                        df = dfzt
                        # column_to_bin = 'z_true'
                        column_to_fit = cf  # 'z'
                        column_to_color = 'id'
                        xy_cols = ['x', 'y']
                        dict_plane = dict_fit_plane
                        scatter_size = 5
                        plane_alpha = 1
                        path_results_example = path_results_fit_plane_eval

                        # processing
                        xmin, xmax = df[xy_cols[0]].min(), df[xy_cols[0]].max()
                        ymin, ymax = df[xy_cols[1]].min(), df[xy_cols[1]].max()

                        x_data = np.array([xmin, xmax, xmax, xmin, xmin])
                        y_data = np.array([ymin, ymin, ymax, ymax, ymin])

                        popt_plane = dict_plane['popt_pixels']
                        z_data = functions.calculate_z_of_3d_plane(x_data, y_data, popt_plane)

                        if plot_fitted_plane_escalator:
                            axx.plot(x_data, z_data, color=cmap(norm(z_plane_tilt_x)), alpha=plane_alpha)
                            axy.plot(y_data, z_data, color=cmap(norm(z_plane_tilt_y)), alpha=plane_alpha)
                        else:
                            fig, (axx, axy) = plt.subplots(ncols=2, sharey=True,
                                                           figsize=(size_x_inches * 1.2, size_y_inches * 0.8))

                            axx.scatter(df[xy_cols[0]], df[column_to_fit], c=df[column_to_color], s=scatter_size)
                            axx.plot(x_data, z_data, color='red', alpha=plane_alpha)

                            axy.scatter(df[xy_cols[1]], df[column_to_fit], c=df[column_to_color], s=scatter_size)
                            axy.plot(y_data, z_data, color='red', alpha=plane_alpha)

                            axx.set_xlabel(xy_cols[0])
                            axx.set_ylabel(column_to_fit)
                            axy.set_xlabel(xy_cols[1])
                            plt.tight_layout()
                            plt.savefig(path_results_example +
                                        '/scatter-xy_bin-{}={}.png'.format(column_to_bin, np.round(zt, 2)))
                            plt.close()

                        # ---

                    # ---

                # fitted plane escalator
                if plot_fitted_plane_escalator:
                    axx.set_xlabel(xy_cols[0])
                    axx.set_ylabel(column_to_fit)
                    axy.set_xlabel(xy_cols[1])

                    plot_colorbar = False
                    cbar_lbl = 'Tilt'
                    if plot_colorbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("bottom", size="5%", pad=0.25)
                        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                                     cax=cax, orientation='horizontal', label=cbar_lbl, extend='both')

                    plt.tight_layout()
                    plt.savefig(path_results_example +
                                '/fitted-plane-escalator_bin-{}.png'.format(column_to_bin))
                    plt.close()
                    plt.show()

                # package
                dffp = pd.DataFrame(np.array(data), columns=['test_id', 'z_true', 'z_plane_mean', 'num_meas',
                                                             'rmse', 'r_squared',
                                                             'tilt_x_degrees', 'tilt_y_degrees',
                                                             ],
                                    )
                dffp.to_excel(path_results_fit_plane_eval + '/local-fit-plane-to-{}_by_{}.xlsx'.format(cf, column_to_bin))
                dffpm = dffp.groupby('test_id').mean()
                dffpm.to_excel(
                    path_results_fit_plane_eval + '/mean-fit-plane-to-{}_by_{}.xlsx'.format(cf, column_to_bin))

            # ---

        # ---

        # fit a 2D bivariate spline to particle distribution at each z_true
        fit_spline_by_z_true_with_and_without_correction = True
        if fit_spline_by_z_true_with_and_without_correction:

            column_to_bin = 'z_true'
            columns_to_fit = ['z', 'z_no_corr']
            xy_cols = ['x', 'y']
            kx, ky = 2, 2

            for cf in columns_to_fit:

                path_results_fit_spline = path_results_displacement + '/spline-{}-by-z_true'.format(cf)
                if not os.path.exists(path_results_fit_spline):
                    os.makedirs(path_results_fit_spline)

                analyze.evaluate_error_from_fitted_bispl(df=dft,
                                                         column_to_bin=column_to_bin,
                                                         column_to_fit=cf,
                                                         xy_cols=xy_cols,
                                                         kx=kx, ky=ky,
                                                         path_results=path_results_fit_spline,
                                                         std_filter=2,
                                                         min_num_counts=20,
                                                         save_figs=False,
                                                         )

        # ---

        # plot particle positions on top of "true" (FIJI) positions for several frames
        plot_rel_FIJI = False
        plot_every_z_steps = 3
        if plot_rel_FIJI:

            path_results_pos_rel_fiji = path_results_displacement + '/relative_true_fiji'
            if not os.path.exists(path_results_pos_rel_fiji):
                os.makedirs(path_results_pos_rel_fiji)

            # read true coords
            df_true = pd.read_excel(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx')
            df_true = df_true.groupby('id').mean().reset_index()

            # plot columns
            if method == 'idpt':
                xy_cols = ['xg', 'yg']
            elif method == 'gdpt':
                xy_cols = ['x', 'y']
            else:
                xy_cols = ['gauss_xc', 'gauss_yc']

            xy_cols_true = ['gauss_xc', 'gauss_yc']

            # plot for select frames
            frois = dft.frame.unique()[::plot_every_z_steps]
            for froi in frois:
                dfr = dft[dft['frame'] == froi]

                # plot big figure with small points to see it better (flip z-order)
                fig, ax = plt.subplots(figsize=(size_x_inches * 2, size_y_inches * 2.1))
                ax.scatter(dfr[xy_cols[0]], dfr[xy_cols[1]], s=2, color='blue', alpha=0.75, zorder=3.2,
                           label=r'$z_{f}$')
                ax.scatter(df_true[xy_cols_true[0]], df_true[xy_cols_true[1]], s=5, marker='s', color='red', alpha=0.5,
                           zorder=3.1, label='FIJI')
                ax.plot([0, num_pixels + padding * 2, num_pixels + padding * 2, 0, 0],
                        [0, 0, num_pixels + padding * 2, num_pixels + padding * 2, 0],
                        '-', color='gray', linewidth=1, alpha=0.5, label='FoV')
                ax.set_xlim([-padding - 5, num_pixels + padding * 3 + 5])
                ax.set_ylim([-padding - 5, num_pixels + padding * 3 + 5])
                ax.invert_yaxis()
                ax.legend()
                ax.set_title("Note, missing particles due to focal plane bias errors", fontsize=6)
                plt.suptitle('{} vs. FIJI - Frame={}'.format(method, froi))
                plt.savefig(path_results_pos_rel_fiji +
                            '/compare-positions-to-true_frame={}.png'.format(froi))
                plt.close()

            # ---

        # ---

        # plot dx, dy, dz by z
        plot_dr_relative_fiji_by_z = False
        if plot_dr_relative_fiji_by_z:

            if method == 'idpt':
                xy_col = 'dist_dxyg'
            elif method == 'gdpt':
                xy_col = 'dist_dxy'
            else:
                xy_col = 'dist_gauss_dxy'

            # plot
            for i in [0, 1]:

                fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True,
                                                    figsize=(size_x_inches * 1.25, size_y_inches * 1.5))

                # number of measurements
                ax0.plot(dfg_counts.z_true, dfg_counts.z, 'o', ms=3)
                ax0.set_ylabel(r'$N_{p} \: (\#)$')

                # displacement relative in-focus positions (FIJI)
                if method == 'idpt':
                    ax1.errorbar(dfg_mean.z_true, dfg_mean[xy_col] * microns_per_pixel,
                                 yerr=dfg_std[xy_col] * microns_per_pixel,
                                 fmt='o', ms=2, capsize=2, elinewidth=1,
                                 label=r'$x_{g}$')
                else:
                    ax1.errorbar(dfg_mean.z_true, dfg_mean[xy_col] * microns_per_pixel,
                                 yerr=dfg_std[xy_col] * microns_per_pixel,
                                 fmt='o', ms=2, capsize=2, elinewidth=1, alpha=0.3,
                                 label=r'$G_{x}$')

                ax1.set_ylabel(r'$\Delta x \: (\mu m)$')
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

                # raw error
                ax2.errorbar(dfg_mean.z_true, dfg_mean.z, yerr=dfg_std.z, fmt='o', ms=3, capsize=2, elinewidth=1)
                ax2.set_ylabel(r'$\Delta z \: (\mu m)$')
                ax2.set_xlabel(r'$z \: (\mu m)$')

                if i == 1:
                    ax1.set_ylim([-0.25, 3.25])

                plt.tight_layout()
                plt.savefig(path_results_displacement + '/displacement-rel-FIJI-by-z_true_errorbars_{}.svg'.format(i))
                plt.show()

            # ---

        # ---

        # plot dx, dy, dz by z
        plot_dxyz_by_z = False
        if plot_dxyz_by_z:
            for i in [0, 1]:

                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                    figsize=(size_x_inches * 1.25, size_y_inches * 2))

                # x-displacement
                if method == 'idpt':
                    ax1.errorbar(dfg_mean.z_true, dfg_mean.dxm * microns_per_pixel,
                                 yerr=dfg_std.dxm * microns_per_pixel,
                                 fmt='o', ms=2, capsize=2, elinewidth=1, alpha=0.6,
                                 label=r'$x_{m}$')
                    ax1.errorbar(dfg_mean.z_true, dfg_mean.dxg * microns_per_pixel,
                                 yerr=dfg_std.dxg * microns_per_pixel,
                                 fmt='o', ms=2, capsize=2, elinewidth=1,
                                 label=r'$x_{g}$')
                ax1.errorbar(dfg_mean.z_true, dfg_mean.gauss_dx * microns_per_pixel,
                             yerr=dfg_std.gauss_dx * microns_per_pixel,
                             fmt='o', ms=2, capsize=2, elinewidth=1, alpha=0.3,
                             label=r'$G_{x}$')

                ax1.set_ylabel(r'$\Delta x \: (\mu m)$')
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

                # y-displacement
                if method == 'idpt':
                    ax2.errorbar(dfg_mean.z_true, dfg_mean.dym * microns_per_pixel,
                                 yerr=dfg_std.dym * microns_per_pixel,
                                 fmt='s', ms=2, capsize=2, elinewidth=1, alpha=0.6,
                                 label=r'$y_{m}$')
                    ax2.errorbar(dfg_mean.z_true, dfg_mean.dyg * microns_per_pixel,
                                 yerr=dfg_std.dyg * microns_per_pixel,
                                 fmt='s', ms=2, capsize=2, elinewidth=1,
                                 label=r'$y_{g}$')
                ax2.errorbar(dfg_mean.z_true, dfg_mean.gauss_dy * microns_per_pixel,
                             yerr=dfg_std.gauss_dy * microns_per_pixel,
                             fmt='s', ms=2, capsize=2, elinewidth=1, alpha=0.3,
                             label=r'$G_{y}$')
                ax2.set_ylabel(r'$\Delta y \: (\mu m)$')
                ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

                # raw error
                ax3.errorbar(dfg_mean.z_true, dfg_mean.z, yerr=dfg_std.z, fmt='o', ms=3, capsize=2, elinewidth=1)
                ax3.set_ylabel(r'$\Delta z \: (\mu m)$')
                ax3.set_xlabel(r'$z \: (\mu m)$')

                if i == 1:
                    ax1.set_ylim([-2.1, 2.1])
                    ax2.set_ylim([-3.1, 1.1])

                plt.tight_layout()
                plt.savefig(path_results_displacement + '/displacement-by-z_true_errorbars_{}.svg'.format(i))
                plt.show()

            # ---

        # ---

        # plot dx, dy by frame (quiver)
        plot_quiver = False
        if plot_quiver:
            import numpy.ma as ma

            path_quiver = path_results_displacement + '/quiver'
            if not os.path.exists(path_quiver):
                os.makedirs(path_quiver)

            # frames of interest
            frois = np.arange(10, 39, 3)

            for froi in frois:
                dfr = dft[dft['frame'] == froi]

                if method == 'idpt':

                    # Here, we must correct for padding
                    x = dfr.xg.to_numpy() - padding
                    y = dfr.yg.to_numpy() - padding

                    # x-y-r displacement
                    dx = dfr.dxg.to_numpy() * microns_per_pixel
                    dy = dfr.dyg.to_numpy() * microns_per_pixel
                    dr = dfr.drg.to_numpy() * microns_per_pixel

                else:
                    # method = SPCT
                    x = dfr.gauss_xc.to_numpy() - padding
                    y = dfr.gauss_yc.to_numpy() - padding

                    dx = dfr.gauss_dx.to_numpy() * microns_per_pixel
                    dy = dfr.gauss_dy.to_numpy() * microns_per_pixel
                    dr = dfr.gauss_dr.to_numpy() * microns_per_pixel

                X = np.linspace(0, 512, 512 + 1)
                Y = np.linspace(0, 512, 512 + 1)
                X, Y = np.meshgrid(X, Y)

                U = np.zeros_like(X)
                V = np.zeros_like(Y)
                UV = np.zeros_like(X)
                for px, py, pdx, pdy, pdr in zip(x, y, dx, dy, dr):
                    U[int(np.round(py, 0)), int(np.round(px, 0))] = pdx
                    V[int(np.round(py, 0)), int(np.round(px, 0))] = pdy
                    UV[int(np.round(py, 0)), int(np.round(px, 0))] = pdr

                # mask zeros
                mUV = ma.masked_equal(UV, 0)
                mU = ma.array(U, mask=mUV.mask)
                mV = ma.array(V, mask=mUV.mask)

                fig, ax = plt.subplots()
                q = ax.quiver(X, Y, mU, mV, angles='xy', scale_units='xy', scale=1 / (microns_per_pixel * 5))
                ax.quiverkey(q, X=0.85, Y=1.05, U=5, label=r'$1 \mu m$', labelpos='E', labelsep=0.05)
                ax.scatter(x, y, s=1, color='red', alpha=0.1)

                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('x (pixels)')
                ax.set_ylabel('y (pixels)')
                ax.set_title('Frame = {}'.format(froi))
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig(path_quiver + '/quiver_frame{}.png'.format(froi), dpi=300)
                plt.close()

            # ---

        # ---

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. EVALUATE DISPLACEMENT ERROR (ID)

    analyze_error = False
    if analyze_error:
        """
        NOTE:
            i. this adds columns to the 'dft' dataframe.
            ii. the primary purpose of this function is to calculate error/rmse relative to a fitted, flat plane.
            iii. error/rmse is also calculated per-particle.
            
        Columns that are added to dataframe:
        
        1. Columns relating 'z' to 'z_true' (i.e., "regular" evaluation of error/rmse):
            > 'z_squared_error': squared error of 'error' (i.e., 'z' relative to 'z_true').
        
        2. Columns relating 'z' to 'z_plane' (i.e., error relative to a flat, fitted plane):
            > 'z_plane': evaluated plane height at particle position (x, y)
            > 'error_z_plane': error of particle position (z) relative to plane height
            > 'z_plane_squared_error': squared error of 'error_z_plane' (i.e., 'z' relative to 'z_plane').
            
        3. Columns relating 'z_plane' to 'z_true' (i.e., error of flat, fitted plane relative to 'z_true'):
            > 'dz_plane': mean position (z) of plane @ image center (i.e., dict_fit_plane['z_f_fit_plane_image_center'])
            > 'dz_plane_error': error of fitted, flat plane ('dz_plane') relative to 'z_true'.
            
        """

        # modifiers
        calc_error_relative_to_plane = True  # ALWAYS TRUE

        # file paths
        if not os.path.exists(path_results_error):
            os.makedirs(path_results_error)

        # ---

        # calculate error - relative to fitted plane
        plot_figs_plane = False
        if calc_error_relative_to_plane:

            # file paths
            if plot_figs_plane:
                path_results_error_plane_figs = path_results_error + '/plane-error-by-z'
                if not os.path.exists(path_results_error_plane_figs):
                    os.makedirs(path_results_error_plane_figs)

                path_results_error_xy_figs = path_results_error + '/xy-error-by-z'
                if not os.path.exists(path_results_error_xy_figs):
                    os.makedirs(path_results_error_xy_figs)

            # ---

            # setup

            # copy dataframe
            dfrs = dft.copy()

            # assign columns for evaluation
            eval_col = 'z'  # --> should always be 'z'. To analyze other columns, assign them as 'z' and repeat.

            if method == 'idpt':
                xy_cols = ['xg', 'yg']
            else:
                xy_cols = ['gauss_xc', 'gauss_yc']

            # iterate - processing
            dftfrs = []
            for fr in dfrs.frame.unique():
                """
                NOTE:
                    > columns are added to the dataframe. 
                    > the name of the column depends on the name of 'eval_col'.
                    > an example is provided below for 'eval_col' = 'z':
                    
                        'z_plane': evaluated plane height at particle position (x, y)
                        'error_z_plane': error of particle position (z) relative to plane height
                        'dz_plane': mean position (z) of plane
                                    > i.e., @ image center: dict_fit_plane['z_f_fit_plane_image_center']
                """
                dftfr = dfrs[dfrs['frame'] == fr]
                dftfr, rmse, r_squared = analyze.evaluate_error_from_fitted_plane(dftfr,
                                                                                  dict_flat_plane,
                                                                                  xy_cols=xy_cols,
                                                                                  eval_col=eval_col)

                # store results
                dftfrs.append(dftfr.copy())

                # plot
                if plot_figs_plane:
                    # 2. apply std filter
                    std_filter = 2
                    dftfr = dftfr[
                        np.abs(dftfr[eval_col] - dftfr[eval_col].mean()) < dftfr[eval_col].std() * std_filter]

                    fig = analyze.helper_plot_fitted_plane_and_points(dftfr,
                                                                      dict_fit_plane_bspl_corrected,
                                                                      xy_cols, eval_col,
                                                                      rmse, r_squared,
                                                                      )
                    fig.savefig(path_results_error_plane_figs +
                                '/plane-3d_std-filter-{}_frame={}.png'.format(std_filter, fr))
                    plt.close(fig)

                    # scatter x-y
                    fig, (axx, axy) = plt.subplots(ncols=2, sharey=True,
                                                   figsize=(size_x_inches * 1.2, size_y_inches * 0.8))
                    axx.scatter(dftfr[xy_cols[0]], dftfr[eval_col], c=dftfr['id'])
                    axx.axhline(dftfr[eval_col].mean(), ls='--')
                    axx.axhline(dftfr['z_true'].mean(), ls='--', color='black')
                    axx.set_xlabel(xy_cols[0])
                    axx.set_ylabel(eval_col)

                    axy.scatter(dftfr[xy_cols[1]], dftfr[eval_col], c=dftfr['id'])
                    axy.axhline(dftfr[eval_col].mean(), ls='--', label='Mean')
                    axy.axhline(dftfr['z_true'].mean(), ls='--', color='black', label='True')
                    axy.set_xlabel(xy_cols[1])
                    axy.legend()

                    plt.tight_layout()
                    plt.savefig(path_results_error_xy_figs +
                                '/scatter-xy_std-filter-{}_frame={}.png'.format(std_filter, fr))
                    plt.close()

                    # ---

                # ---

            dftfrs = pd.concat(dftfrs)

            # ---

            # add error + rmse columns

            # rmse_z relative to z_true (nothing to do with plane)
            dftfrs['z_squared_error'] = dftfrs['error'] ** 2

            # rmse_z relative to fitted plane
            dftfrs['z_plane_squared_error'] = dftfrs['error_z_plane'] ** 2
            dftfrs['dz_plane_error'] = dftfrs['dz_plane'] - dftfrs['z_true']
            dftfrs['dz_plane_squared_error'] = dftfrs['dz_plane_error'] ** 2

            # copy to main test dataframe
            dft = dftfrs.copy()
            dft = dft.drop(columns=['z_squared_error', 'z_plane_squared_error', 'dz_plane_squared_error'])

            # ---

            # export mean rmse (bins = 1):
            dfbm = bin.bin_by_column(dftfrs, column_to_bin='z_true', number_of_bins=1, round_to_decimal=2)
            dfbm = dfbm[['bin', 'z_true', 'z', 'cm', 'z_squared_error', 'z_plane_squared_error', 'dz_plane_squared_error']]
            dfbm = dfbm.groupby('bin').mean()
            dfbm['rmse_z'] = np.sqrt(dfbm['z_squared_error'])
            dfbm['rmse_z_plane'] = np.sqrt(dfbm['z_plane_squared_error'])
            dfbm['rmse_dz_plane'] = np.sqrt(dfbm['dz_plane_squared_error'])
            dfbm.to_excel(path_results_error + '/global-rmse-z-relative-plane_by_z.xlsx')

            # ---

            # plot 1: z_plane error, z_error + std, rmse_z

            # calculate mean + std + rmse_z
            dftfrs_mean = dftfrs.groupby('z_true').mean()
            dftfrs_std = dftfrs.groupby('z_true').std()

            dftfrs_mean['rmse_z'] = np.sqrt(dftfrs_mean['z_squared_error'])
            dftfrs_mean['rmse_z_plane'] = np.sqrt(dftfrs_mean['z_plane_squared_error'])
            dftfrs_mean['rmse_dz_plane'] = np.sqrt(dftfrs_mean['dz_plane_squared_error'])

            # ---

            # plot
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))
            ax1.plot(dftfrs_mean.index, dftfrs_mean['dz_plane_error'], '-o')
            ax2.errorbar(dftfrs_mean.index, dftfrs_mean['error_z_plane'], yerr=dftfrs_std['error_z_plane'],
                         fmt='o', capsize=2, elinewidth=1)
            ax3.plot(dftfrs_mean.index, dftfrs_mean['rmse_z_plane'], '-o')

            ax1.set_ylabel(r'$\overline{\epsilon_{z}} \: (\mu m)$')
            ax2.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
            ax3.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
            ax3.set_xlabel(r'$z \: (\mu m)$')
            plt.tight_layout()
            plt.savefig(path_results_error + '/local-rmse-z-relative-plane_by_z.svg')
            plt.show()
            plt.close()

            # ---

            # export
            export_columns = ['z', 'cm', 'rmse_z', 'rmse_z_plane', 'rmse_dz_plane',
                              'z_squared_error', 'z_plane_squared_error', 'dz_plane_squared_error']
            dftfrs_mean = dftfrs_mean[export_columns]
            dftfrs_mean.to_excel(path_results_error + '/local-rmse-z-relative-plane_by_z.xlsx')

            # ---

            # ---

            # plot 2: per-particle rmse_z (x, y)

            dftfrs_id = dftfrs.groupby('id').mean()

            dftfrs_id['rmse_z'] = np.sqrt(dftfrs_id['z_squared_error'])
            dftfrs_id['rmse_z_plane'] = np.sqrt(dftfrs_id['z_plane_squared_error'])
            dftfrs_id['rmse_dz_plane'] = np.sqrt(dftfrs_id['dz_plane_squared_error'])

            # plot
            fig, ax = plt.subplots()

            sc = ax.scatter(dftfrs_id[xy_cols[0]], dftfrs_id[xy_cols[1]], c=dftfrs_id['rmse_z_plane'])

            cbar = plt.colorbar(sc)
            cbar.set_label(r'$\overline{\sigma}_{z}^{i} \: (\mu m)$')

            ax.set_xlabel('x (pix.)')
            ax.set_xlim([0, 512 + padding * 2])
            ax.set_ylabel('y (pix.)')
            ax.set_ylim([0, 512 + padding * 2])

            plt.tight_layout()
            plt.savefig(path_results_error + '/mean-rmse-z-per-pid-relative-plane_by_x-y.svg')
            plt.show()
            plt.close()

            # ---

            # export
            export_columns = ['z_true', 'z', 'cm', 'rmse_z', 'rmse_z_plane', 'rmse_dz_plane',
                              'z_squared_error', 'z_plane_squared_error', 'dz_plane_squared_error']
            dftfrs_id = dftfrs_id[export_columns]
            dftfrs_id.to_excel(path_results_error + '/mean-rmse-z-per-pid-relative-plane.xlsx')

            # ---

        # ---

        # get only necessary columns
        df = dft[['frame', 'id', 'z_true', 'z', 'error', 'error_z_plane', 'x', 'y', 'r']].copy()

        # plot error - z dependence
        plot_error_relative_plane_by_z = False
        if plot_error_relative_plane_by_z:

            # processing

            # absolute error
            df['error_abs'] = df['error'].abs()
            df['error_plane_abs'] = df['error_z_plane'].abs()

            # groupby 'z_true'
            dfg_mean = df.groupby('z_true').mean().reset_index()
            dfg_std = df.groupby('z_true').std().reset_index()

            # ---

            # plot
            for i in [0, 1]:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))

                # raw error
                ax1.errorbar(dfg_mean.z_true, dfg_mean.error_z_plane, yerr=dfg_std.error_z_plane,
                             fmt='o', ms=3, capsize=2, elinewidth=1)
                ax1.axhline(df.error_z_plane.mean(), linestyle='--', linewidth=0.75, color='black',
                            label=r'$\overline{\epsilon_{z}}$' +
                                  ' {} '.format(np.round(df.error_z_plane.mean(), 3)) +
                                  r'$\mu m$'
                            )
                # line (error = 0)
                ax1.axhline(0, linestyle='-', linewidth=0.5, color='black', alpha=0.25)

                ax1.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                ax1.legend()

                # abs error
                ax2.errorbar(dfg_mean.z_true, dfg_mean.error_plane_abs, yerr=dfg_std.error_plane_abs,
                             fmt='o', ms=3, capsize=2, elinewidth=1)
                ax2.axhline(df.error_plane_abs.mean(), linestyle='--', linewidth=0.75, color='black',
                            label=r'$\overline{|\epsilon_{z}|}=$' +
                                  ' {} '.format(np.round(df.error_plane_abs.mean(), 3)) +
                                  r'$\mu m$'
                            )
                ax2.set_xlabel(r'$z \: (\mu m)$')
                ax2.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
                ax2.legend()

                if i == 1:
                    ax1.set_ylim([-1.65, 1.65])
                    ax2.set_ylim([-0.45, 1.95])

                plt.tight_layout()
                plt.savefig(path_results_error + '/error-relative-plane_by_z_true_errorbars_{}_calib-id.svg'.format(i))
                plt.show()

            # ---

        # ---

        # plot error - radial dependence
        plot_error_by_r = False
        if plot_error_by_r:
            for ec in ['error', 'error_z_plane']:
                plot_collections.plot_error_analysis(dft=df,
                                                     error_column=ec, error_threshold=filter_step_size,
                                                     img_center=(img_xc, img_yc),
                                                     xy_cols=xy_cols,
                                                     microns_per_pixels=microns_per_pixel,
                                                     path_results=path_results_error,
                                                     r_bins_microns=[100, 300, 500],
                                                     z_bins=num_dz_steps,
                                                     )

        # ---

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 7. EVALUATE DISPLACEMENT PRECISION (ID)

    analyze_precision_per_id_for_all_locations = False
    if analyze_precision_per_id_for_all_locations:

        save_figs = True
        show_figs = False
        if method == 'idpt':
            every_x_figs = 24
        else:
            every_x_figs = 16
        deg_poly = 0

        # ---

        path_results_precision = path_results + '/precision'
        path_results_precision_figs_z = path_results_precision + '/z'
        path_results_precision_figs_xy = path_results_precision + '/xy'
        path_results_precision_figs_r = path_results_precision + '/r'
        if not os.path.exists(path_results_precision):
            os.makedirs(path_results_precision)

            if not os.path.exists(path_results_precision_figs_z):
                os.makedirs(path_results_precision_figs_z)

            if not os.path.exists(path_results_precision_figs_xy):
                os.makedirs(path_results_precision_figs_xy)

            if not os.path.exists(path_results_precision_figs_r):
                os.makedirs(path_results_precision_figs_r)

        # ---

        precision_data = []

        for i, pid in enumerate(dft.id.unique()):

            dfpid = dft[dft['id'] == pid]

            if len(dfpid) < 4:
                continue

            pid_precision_data = [pid]

            # ---

            # axial (z) precision
            if method in ['spct', 'gdpt']:
                precision_columns = ['z', 'z_corr_tilt']
            else:
                precision_columns = ['z']

            for pc in precision_columns:

                # data
                x = dfpid['z_true'].to_numpy()
                y = dfpid[pc].to_numpy()
                fit_func = functions.line

                # fit line
                popt, pcov = curve_fit(fit_func, x, y)
                yf = fit_func(x, *popt)
                rmse, r_squared = fit.calculate_fit_error(fit_results=yf, data_fit_to=y)
                rmse_r = np.round(rmse, 2)

                # store
                pid_precision_data.extend([rmse_r])

                # plot
                if save_figs or show_figs:
                    if i % every_x_figs == 0:
                        fig, ax = plt.subplots()
                        ax.plot(x, y, 'o', ms=3, label=pid)
                        ax.plot(x, yf, '--', label=r'$\overline {\sigma_{z}}=$' +
                                                   ' {} '.format(rmse_r) + r'$\mu m$' + '\n' +
                                                   '{}z + {}'.format(np.round(popt[0], 3), np.round(popt[1], 2)))

                        ax.set_xlabel(r'$z \: (\mu m)$')
                        ax.set_ylabel(pc)
                        ax.legend()
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(
                                path_results_precision_figs_z + '/pid{}_{}-precision={}.png'.format(pid, pc, rmse_r))
                        if show_figs:
                            plt.show()
                        plt.close()

            # ---

            # lateral (xy) precision
            if method in ['spct', 'gdpt']:
                precision_columns = ['gauss_xc', 'gauss_yc']
            else:
                precision_columns = ['xg', 'yg']  # ['xm', 'ym', 'xg', 'yg', 'gauss_xc', 'gauss_yc']
            for pc in precision_columns:

                # data
                x = dfpid['z_true'].to_numpy()
                y = dfpid[pc].to_numpy() * microns_per_pixel

                # fit line
                if deg_poly == 0:
                    yf = np.ones_like(y) * np.mean(y)
                    xf = x
                    rmse, r_squared = fit.calculate_fit_error(fit_results=yf, data_fit_to=y)
                    rmse_r = np.round(rmse, 2)

                else:
                    p4 = np.poly1d(np.polyfit(x, y, deg_poly))
                    rmse, r_squared = fit.calculate_fit_error(fit_results=p4(x), data_fit_to=y)
                    rmse_r = np.round(rmse, 2)

                    # resample
                    xf = np.linspace(x.min(), x.max(), 250)
                    yf = p4(xf)

                # store
                pid_precision_data.extend([rmse_r])

                # plot
                if save_figs or show_figs:
                    if i % (every_x_figs * 3) == 0:
                        fig, ax = plt.subplots()
                        ax.plot(x, y, 'o', ms=3, label=pid)
                        ax.plot(xf, yf, '--', label=r'$\sigma_{fit}$' + '({})={} '.format(pc, rmse_r) + r'$\mu m$')
                        ax.set_xlabel(r'$z \: (\mu m)$')
                        ax.set_ylabel(pc)
                        ax.legend()
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(
                                path_results_precision_figs_xy + '/pid{}_{}-precision={}.png'.format(pid, pc, rmse_r))
                        if show_figs:
                            plt.show()
                        plt.close()

            # ---

            # radial (r) precision
            if method in ['spct', 'gdpt']:
                precision_columns = ['gauss_rc']  # , 'gauss_dr']
            else:
                precision_columns = ['rg']  # ['rm', 'drm', 'rg', 'drg', 'gauss_rc', 'gauss_dr']
            for pc in precision_columns:

                # data
                x = dfpid['z_true'].to_numpy()
                y = dfpid[pc].to_numpy() * microns_per_pixel

                # fit line
                if deg_poly == 0:
                    yf = np.ones_like(y) * np.mean(y)
                    xf = x
                    rmse, r_squared = fit.calculate_fit_error(fit_results=yf, data_fit_to=y)
                    rmse_r = np.round(rmse, 2)

                else:
                    p4 = np.poly1d(np.polyfit(x, y, deg_poly))
                    rmse, r_squared = fit.calculate_fit_error(fit_results=p4(x), data_fit_to=y)
                    rmse_r = np.round(rmse, 2)

                    # resample
                    xf = np.linspace(x.min(), x.max(), 250)
                    yf = p4(xf)

                # store
                pid_precision_data.extend([rmse_r])

                # plot
                if save_figs or show_figs:
                    if i % every_x_figs == 0:
                        if pc in ['drm', 'drg', 'gauss_dr']:
                            fig, ax = plt.subplots()
                            ax.plot(x, y, 'o', ms=3, label=pid)
                            ax.plot(xf, yf, '--', label=r'$\sigma_{fit}$' + '({})={} '.format(pc, rmse_r) + r'$\mu m$')
                            ax.set_xlabel(r'$z \: (\mu m)$')
                            ax.set_ylabel(pc)
                            ax.legend()
                            plt.tight_layout()
                            if save_figs:
                                plt.savefig(
                                    path_results_precision_figs_r + '/pid{}_{}-precision={}.png'.format(pid, pc,
                                                                                                        rmse_r))
                            if show_figs:
                                plt.show()
                            plt.close()

            # ---

            # store results
            precision_data.append(pid_precision_data)

        # ---

        # structure

        if method in ['spct', 'gdpt']:
            df_precision = pd.DataFrame(np.array(precision_data),
                                        columns=['id',
                                                 'z', 'z_corr_tilt_fc',
                                                 'gauss_xc', 'gauss_yc',
                                                 'gauss_rc', 'gauss_dr',
                                                 ],
                                        )
        else:
            df_precision = pd.DataFrame(np.array(precision_data),
                                        columns=['id',
                                                 'z',
                                                 'xg', 'yg', 'rg',
                                                 ]
                                        )
        df_precision['test_id'] = test_id

        # export precision
        df_precision.to_excel(path_results_precision + '/df_precision_z-x-y-r_units=microns.xlsx', index=False)

        # export mean precision
        dfm_precision = df_precision.groupby('test_id').mean()
        dfm_precision.to_excel(path_results_precision + '/df_mean-precision_z-x-y-r_units=microns.xlsx')

        # ---

    # ---

    analyze_precision_for_all_pids_locations = False
    if analyze_precision_for_all_pids_locations:

        path_results_precision = path_results + '/precision'
        if not os.path.exists(path_results_precision):
            os.makedirs(path_results_precision)

        path_results_precision_figs_all = path_results_precision + '/all'
        if not os.path.exists(path_results_precision_figs_all):
            os.makedirs(path_results_precision_figs_all)

        # ---

        save_figs = True
        show_figs = False

        # ---
        precision_data = []

        # axial (z) precision
        precision_columns = ['z', 'z_corr_tilt']
        for pc in precision_columns:

            # data
            x = dft['z_true'].to_numpy()
            y = dft[pc].to_numpy()
            fit_func = functions.line

            # fit line
            popt, pcov = curve_fit(fit_func, x, y)
            yf = fit_func(x, *popt)
            rmse, r_squared = fit.calculate_fit_error(fit_results=yf, data_fit_to=y)
            rmse_r = np.round(rmse, 2)

            # plot
            if save_figs or show_figs:
                fig, ax = plt.subplots()
                ax.plot(x, y, 'o', ms=1)
                ax.plot(x, yf,
                        linestyle='--', linewidth=1, color='black',
                        label=r'$\overline {\sigma_{z}}=$' +
                              ' {} '.format(rmse_r) + r'$\mu m$' + '\n' +
                              '{}z + {}'.format(np.round(popt[0], 3), np.round(popt[1], 2)))
                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(pc)
                ax.legend()
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_precision_figs_all + '/all-pids_{}-precision={}.png'.format(pc, rmse_r))
                if show_figs:
                    plt.show()
                plt.close()

            # ---
            dft['z_fit'] = yf
            dft['z_fit_error'] = dft['z'] - dft['z_fit']
            dft['error_abs'] = dft['z_fit_error'].abs()

            # get only necessary columns
            df = dft[['frame', 'id', 'z_true', 'z', 'cm', 'error', 'z_fit', 'z_fit_error', 'error_abs']]

            # groupby 'z_true'
            dfg_mean = df.groupby('z_true').mean().reset_index()
            dfg_std = df.groupby('z_true').std().reset_index()

            # plot
            if save_figs or show_figs:
                fig, ax = plt.subplots()
                ax.errorbar(dfg_mean.z_true, dfg_mean.error_abs, yerr=dfg_std.error_abs,
                            fmt='o', capsize=2, elinewidth=1, label=r'$\overline{\epsilon_{z}}(z) + \sigma_{z}$')
                ax.axhline(dfg_mean.error_abs.mean(), linestyle='--', color='black', label=r'$\overline{\epsilon_{z}}$')
                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                ax.legend()
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_precision_figs_all + '/precision-error-by-z_true_errorbars.svg')
                if show_figs:
                    plt.show()
                plt.close()

            # ---

            # RMSE

            path_results_rmse = path_results_precision + '/rmse'
            if not os.path.exists(path_results_rmse):
                os.makedirs(path_results_rmse)

            dfrmse_mean = bin.bin_local_rmse_z(df, column_to_bin='z_true', bins=1, min_cm=min_cm,
                                               z_range=None, round_to_decimal=1, df_ground_truth=None, dropna=True,
                                               error_column='z_fit_error')
            dfrmse_mean.to_excel(path_results_rmse + '/mean-rmse-z.xlsx')

            dfrmse_bin = bin.bin_local_rmse_z(df, column_to_bin='z_true', bins=dzs, min_cm=min_cm,
                                              z_range=None, round_to_decimal=1, df_ground_truth=None, dropna=True,
                                              error_column='z_fit_error')
            dfrmse_bin['true_num_per_bin'] = true_num_particles_per_frame * 3
            dfrmse_bin['true_percent_meas'] = dfrmse_bin['num_meas'] / dfrmse_bin['true_num_per_bin']
            dfrmse_bin.to_excel(path_results_rmse + '/binned-by-z_true-rmse-z.xlsx')

            # plot
            save_figs = True
            show_figs = True

            if save_figs or show_figs:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                ax1.plot(dfrmse_bin.index, dfrmse_bin.true_percent_meas, '-o')
                ax1.set_ylabel(r'$\phi^{\delta}_{z}(z) \: (\%)$')

                ax2.plot(dfrmse_bin.index, dfrmse_bin.rmse_z, '-o')
                ax2.set_xlabel(r'$z \: (\mu m)$')
                ax2.set_ylabel(r'$\sigma^{\delta}_{z}(z) \: (\mu m)$')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_rmse + '/precision-rmse-percent-meas-by-z_true.svg')
                if show_figs:
                    plt.show()
                plt.close()

            # ---

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 7. EVALUATE RMSE-XYZ

    analyze_rmse_for_all_locations = True
    if analyze_rmse_for_all_locations:

        path_results_rmse = path_results_original + '/rmse/min_cm_{}'.format(min_cm)
        if not os.path.exists(path_results_rmse):
            os.makedirs(path_results_rmse)

        # compute rmse_z via multiple errors
        """
        NOTE:
            > 'error_no_corr' will always be the uncorrected error (i.e., 'z' relative to 'z_true').
            > 'error' will be identical to either: 'error_corr_tilt_fc' or 'error_no_corr'.
            > 'error_z_plane' is relative to a flat, fitted plane.
        """
        error_columns = ['error_no_corr', 'error', 'error_corr_tilt', 'error_corr_tilt_fc', 'error_z_plane']

        for ec in error_columns:

            # copy dataframe for rmse
            dfrmse = dft.copy()
            dfrmse['test_id'] = test_id

            # calculate squared error
            if method in ['spct', 'gdpt']:
                dfrmse['rmse_z'] = dfrmse[ec] ** 2
                
                # relative to "true" positions (external document)
                dfrmse['xy_error_gauss_dxy'] = np.sqrt(dfrmse['x_error_gauss_dxy'] ** 2 + dfrmse['y_error_gauss_dxy'] ** 2)
                dfrmse['rmse_xy_gauss'] = (dfrmse['xy_error_gauss_dxy'] * microns_per_pixel) ** 2
                dfrmse['rmse_x_gauss'] = (dfrmse['x_error_gauss_dxy'] * microns_per_pixel) ** 2
                dfrmse['rmse_y_gauss'] = (dfrmse['y_error_gauss_dxy'] * microns_per_pixel) ** 2

                # relative to "in-focus" positions (self-referencing)
                dfrmse['rmse_dr'] = (dfrmse['dr'] * microns_per_pixel) ** 2
                dfrmse['rmse_gauss_dr'] = (dfrmse['gauss_dr'] * microns_per_pixel) ** 2
                dfrmse['rmse_gauss_drf'] = (dfrmse['dist_gauss_dxy'] * microns_per_pixel) ** 2

                dfrmse = dfrmse[['test_id', 'bin', 'frame', 'id', 'z_true', 'z', 'cm',
                                 'rmse_xy_gauss', 'rmse_x_gauss', 'rmse_y_gauss',
                                 'rmse_gauss_dr', 'rmse_gauss_drf',
                                 'rmse_z']]

            else:
                dfrmse['rmse_z'] = dfrmse[ec] ** 2

                # relative to "true" positions (external document)
                dfrmse['xy_error_dxyg'] = np.sqrt(dfrmse['x_error_dxyg'] ** 2 + dfrmse['y_error_dxyg'] ** 2)
                dfrmse['rmse_xyg'] = (dfrmse['xy_error_dxyg'] * microns_per_pixel) ** 2
                dfrmse['rmse_xg'] = (dfrmse['x_error_dxyg'] * microns_per_pixel) ** 2
                dfrmse['rmse_yg'] = (dfrmse['y_error_dxyg'] * microns_per_pixel) ** 2
                dfrmse['xy_error_gauss_dxy'] = np.sqrt(dfrmse['x_error_gauss_dxy'] ** 2 + dfrmse['y_error_gauss_dxy'] ** 2)
                dfrmse['rmse_xy_gauss'] = (dfrmse['xy_error_gauss_dxy'] * microns_per_pixel) ** 2
                dfrmse['rmse_x_gauss'] = (dfrmse['x_error_gauss_dxy'] * microns_per_pixel) ** 2
                dfrmse['rmse_y_gauss'] = (dfrmse['y_error_gauss_dxy'] * microns_per_pixel) ** 2

                # relative to "in-focus" positions (self-referencing)
                # dfrmse['rmse_dr'] = (dfrmse['dr'] * microns_per_pixel) ** 2
                # dfrmse['rmse_drm'] = (dfrmse['drm'] * microns_per_pixel) ** 2
                # dfrmse['rmse_drg'] = (dfrmse['drg'] * microns_per_pixel) ** 2
                dfrmse['rmse_drgf'] = (dfrmse['dist_dxyg'] * microns_per_pixel) ** 2
                # dfrmse['rmse_gauss_dr'] = (dfrmse['gauss_dr'] * microns_per_pixel) ** 2
                dfrmse['rmse_gauss_drf'] = (dfrmse['dist_gauss_dxy'] * microns_per_pixel) ** 2

                dfrmse = dfrmse[['test_id', 'bin', 'frame', 'id', 'z_true', 'z', 'cm',
                                 'rmse_xyg', 'rmse_xg', 'rmse_yg',
                                 'rmse_drgf', 'rmse_gauss_drf',
                                 'rmse_z']]

            # groupby 'dz'
            column_to_bin = 'z_true'
            bins = dzs
            round_to_decimal = 1
            dfb = bin.bin_by_list(dfrmse, column_to_bin, bins, round_to_decimal)

            # count
            dfc = dfb.groupby('bin').count()
            dfc['true_num'] = true_num_particles_per_frame * num_frames_per_step
            dfc['num_idd'] = i_num_rows_per_z_df[i_num_rows_per_z_df['z_true'].isin(dfc.index.values)].id.to_numpy()
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
            dfrmse_bins.to_excel(path_results_rmse +
                                 '/bin-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-{}.xlsx'.format(ec))

            # ---

            # mean (bins = 1)
            dfbm = bin.bin_by_column(dfrmse, column_to_bin, number_of_bins=1, round_to_decimal=round_to_decimal)

            # count
            dfcm = dfbm.groupby('bin').count()
            dfcm['true_num'] = true_num_particles_per_frame * num_frames_per_step * num_dz_steps
            dfcm['num_idd'] = np.sum(i_num_rows_per_z)
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
            dfrmse_mean.to_excel(path_results_rmse +
                                 '/mean-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-{}.xlsx'.format(
                                     ec))

            # ---

            # plot
            save_figs = True
            show_figs = True

            if save_figs or show_figs:
                fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, sharex=True,
                                                              figsize=(size_x_inches, size_y_inches * 1.5),
                                                              )

                ax0.plot(dfrmse_bins.index, dfrmse_bins.cm, '-o')
                ax0.set_ylabel(r'$C^{\delta}_{m}$')
                ax0.set_ylim([0.75, 1])
                ax0.set_yticks([0.75, 1])

                if method == 'idpt':
                    ax1.plot(dfrmse_bins.index, dfrmse_bins.rmse_xyg, '-o')
                    # ax1.plot(dfrmse_bins.index, dfrmse_bins.rmse_drgf, '-o')
                else:
                    ax1.plot(dfrmse_bins.index, dfrmse_bins.rmse_xy_gauss, '-o')
                    # ax1.plot(dfrmse_bins.index, dfrmse_bins.rmse_gauss_drf, '-o')
                ax1.set_ylabel(r'$\sigma^{\delta}_{xy} \: (\mu m)$')
                ax1.set_ylim([0, 2.2])
                ax1.set_yticks([0, 1, 2])

                ax2.plot(dfrmse_bins.index, dfrmse_bins.rmse_z, '-o')
                ax2.set_ylabel(r'$\sigma^{\delta}_{z} \: (\mu m)$')
                ax2.set_ylim([0, 6])
                ax2.set_yticks([0, 5])

                ax3.plot(dfrmse_bins.index, dfrmse_bins.percent_meas_idd, '-o')
                ax3.set_ylabel(r'$\phi^{\delta}_{ID}$')
                ax3.set_ylim([0, 1.05])
                ax3.set_yticks([0, 1])

                ax4.plot(dfrmse_bins.index, dfrmse_bins.true_percent_meas, '-o')
                ax4.set_ylabel(r'$\phi^{\delta}$')
                ax4.set_ylim([0, 1.05])
                ax4.set_yticks([0, 1])
                ax4.set_xlabel(r'$z \: (\mu m)$')

                plt.tight_layout()
                if save_figs:
                    plt.savefig(
                        path_results_rmse + '/precision-xy-rmse-percent-meas-by-z_true_relFIJI_ec-{}.svg'.format(ec))
                if show_figs:
                    plt.show()
                plt.close()

            # ---

        # ---

# ---
raise ValueError()
# ----------------------------------------------------------------------------------------------------------------------
# D. COMPARE SPCT AND IDPT RMSE

compare_idpt_and_spct_rmse = False
if compare_idpt_and_spct_rmse:

    # file paths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
               'results-07.29.22-idpt-tmg'
    path_compare = join(base_dir, 'compare/rmse/min_cm_0.5_error_corr_tilt_relative-plane')

    # file names
    fni = 'bin-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-error_z_plane'
    fns = 'bin-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-error_z_plane'
    fng = None

    # error columns
    error_columns = ['error']

    for ec in error_columns:

        fpi = join(path_compare, 'idpt', fni + '.xlsx')
        fps = join(path_compare, 'spct', fns + '.xlsx')

        dfi = pd.read_excel(fpi)
        dfs = pd.read_excel(fps)

        # setup
        save_id = 'IDPT_SPCT'

        include_gdpt = False
        if include_gdpt:
            save_id = save_id + '_GDPT'
            ms = 3
            fp_gdpt = join(path_compare, 'gdpt', fng + '.xlsx')
            dfgdpt = pd.read_excel(fp_gdpt)

        # ---

        # plot setup
        save_figs = True
        show_figs = True
        rmse_xy_cols = [['rmse_xyg', 'rmse_xy_gauss'], ['rmse_drg', 'rmse_gauss_dr'], ['rmse_drgf', 'rmse_gauss_drf']]
        ms = 4

        if min_cm < 0.9:
            ylim_cm = [0.7, 1.01]
            yticks_cm = [0.7, 1]
        else:
            ylim_cm = [0.9, 1.01]
            yticks_cm = [0.9, 1]

        # plot

        if save_figs or show_figs:

            for h, ps in zip([1, measurement_depth], [1, microns_per_pixel]):

                for r_col in rmse_xy_cols:

                    """fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True,
                                                             figsize=(size_x_inches, size_y_inches * 1.5),
                                                             )"""

                    # figure.constrained_layout.w_pad:  0.04167  # inches. Default is 3/72 inches (3 points)
                    plt.rcParams['figure.constrained_layout.w_pad'] = 0.15

                    fig = plt.figure(constrained_layout=True,
                                     figsize=(size_x_inches * 2, size_y_inches * 1.25))
                    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])  # create a 1x2 grid of axes (1 row, 2 columns)
                    gsl = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                                           hspace=0.2, wspace=0.25)  # split the left axis into 3 rows
                    gsr = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1],
                                                           hspace=0.2,
                                                           wspace=0.25)  # split the right axis into a top and bottom

                    ax0 = fig.add_subplot(gsl[0, 0])  # left axis - top
                    ax1 = fig.add_subplot(gsl[1, 0])  # left axis - middle
                    ax2 = fig.add_subplot(gsl[2, 0])  # left axis - bottom
                    ax3 = fig.add_subplot(gsr[0, 0])  # right axis - top
                    ax4 = fig.add_subplot(gsr[1, 0])  # right axis - bottom

                    # left column
                    ax0.plot(dfi.bin / h, dfi.cm, '-o', ms=ms, label='IDPT', zorder=4)
                    ax0.plot(dfs.bin / h, dfs.cm, '-o', ms=ms, label='SPCT')
                    ax0.set_ylabel(r'$C^{\delta}_{m}$')
                    ax0.set_ylim(ylim_cm)
                    ax0.set_yticks(yticks_cm)

                    ax1.plot(dfi.bin / h, dfi.percent_meas_idd, '-o', ms=ms, label='IDPT', zorder=4)
                    ax1.plot(dfs.bin / h, dfs.percent_meas_idd, '-o', ms=ms, label='SPCT', zorder=3.5)
                    ax1.set_ylabel(r'$\phi^{\delta}_{ID}$')
                    ax1.set_ylim([0, 1.05])
                    ax1.set_yticks([0, 1])

                    ax2.plot(dfi.bin / h, dfi.true_percent_meas, '-o', ms=ms, label='IDPT', zorder=4)
                    ax2.plot(dfs.bin / h, dfs.true_percent_meas, '-o', ms=ms, label='SPCT', zorder=3.5)
                    ax2.set_ylabel(r'$\phi^{\delta}$')
                    ax2.set_ylim([0, 1.05])
                    ax2.set_yticks([0, 1])
                    ax2.set_xlabel(r'$z / h$')

                    # right column
                    ax3.plot(dfi.bin / h, dfi[r_col[0]] / ps, '-o', ms=ms, label='IDPT', zorder=4)
                    ax3.plot(dfs.bin / h, dfs[r_col[1]] / ps, '-o', ms=ms, label='SPCT', zorder=3.5)

                    ax4.plot(dfi.bin / h, dfi.rmse_z / h, '-o', ms=ms, label='IDPT', zorder=4)
                    ax4.plot(dfs.bin / h, dfs.rmse_z / h, '-o', ms=ms, label='SPCT', zorder=3.5)

                    if include_gdpt:
                        ax0.plot(dfgdpt.bin / h, dfgdpt.cm, '-o', ms=ms, label='GDPT')
                        ax1.plot(dfgdpt.bin / h, dfgdpt.percent_meas_idd, '-o', ms=ms, label='GDPT', zorder=3.3)
                        ax2.plot(dfgdpt.bin / h, dfgdpt.true_percent_meas, '-o', ms=ms, label='GDPT', zorder=3.3)
                        ax3.plot(dfgdpt.bin / h, dfgdpt[r_col[1]] / ps, '-o', ms=ms, label='GDPT', zorder=3.3)
                        ax4.plot(dfgdpt.bin / h, dfgdpt.rmse_z / h, '-o', ms=ms, label='GDPT', zorder=3.5)

                    ax4.legend(loc='upper left')

                    if h == 1:
                        ax3.set_ylabel(r'$\sigma^{\delta}_{xy} \: (\mu m)$')
                        ax4.set_ylabel(r'$\sigma^{\delta}_{z} \: (\mu m)$')
                        ax4.set_xlabel(r'$z \: (\mu m)$')
                        xlim_norm = [-50, 0, 50]
                        ax0.set_xticks(xlim_norm, labels=[])
                        ax1.set_xticks(xlim_norm, labels=[])
                        ax2.set_xticks(xlim_norm)
                        ax3.set_xticks(xlim_norm, labels=[])
                        ax4.set_xticks(xlim_norm)
                    else:
                        ax3.set_ylabel(r'$\sigma^{\delta}_{xy} / w$')
                        ax4.set_ylabel(r'$\sigma^{\delta}_{z} / h$')
                        ax4.set_xlabel(r'$z / h$')
                        xlim_norm = [-0.5, 0, 0.5]
                        ax0.set_xticks(xlim_norm, labels=[])
                        ax1.set_xticks(xlim_norm, labels=[])
                        ax2.set_xticks(xlim_norm)
                        ax3.set_xticks(xlim_norm, labels=[])
                        ax4.set_xticks(xlim_norm)

                    # fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.0, wspace=0.0)

                    # plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_compare +
                                    '/results/{}-xyz-percent-meas-by-z_true_norm-{}_{}.svg'.format(r_col[0],
                                                                                                         h,
                                                                                                         save_id))
                    if show_figs:
                        plt.show()
                    plt.close()

        # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# D. COMPARE SPCT AND IDPT - RADIAL DEPENDENCE BY PER-ID PRECISION ANALYSIS

compare_idpt_and_spct_radial_per_id = False
if compare_idpt_and_spct_radial_per_id:

    method = 'spct'

    # import stats package
    from scipy.stats import pearsonr, spearmanr
    """
    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([0, 0, 0, 1, 1, 1, 1])
    >>> b = np.arange(7)
    >>> stats.pearsonr(a, b)
    (0.8660254037844386, 0.011724811003954649)
    
    >>> stats.spearmanr([1,2,3,4,5], [5,6,7,8,7])
    SpearmanrResult(correlation=0.82078..., pvalue=0.08858...)
    """

    path_compare = base_dir + '/compare'
    path_compare_spearman = path_compare + '/spearman'

    if not os.path.exists(path_compare_spearman):
        os.makedirs(path_compare_spearman)

    fn_idpt_res = 'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed'
    fn_spct_res = 'test_coords_particle_image_stats_spct-1_dzf-post-processed'

    # ---

    # file paths
    fp_idpt_res = '/tests/tm16_cm19/coords/test-coords/post-processed/'
    fp_spct_res = '/tests/spct_soft-baseline_1/coords/test-coords/min_cm_0.5_z_is_z-corr-tilt/post-processed/'

    # ---

    # setup
    bins_r = np.array([130, 250, 340, 430]) / microns_per_pixel
    bins_rr = 22

    if method == 'spct':
        df = pd.read_excel(base_dir + fp_spct_res + fn_spct_res + '.xlsx')
        ylim = [0, 10]
    else:
        df = pd.read_excel(base_dir + fp_idpt_res + fn_idpt_res + '.xlsx')
        ylim = [-0.5, 8.25]

    # processing
    df['error_squared'] = df['error'] ** 2

    # 1D binning
    column_to_bin = 'r'
    column_to_count = 'id'
    round_to_decimal = 1
    return_groupby = False

    df = bin.bin_generic(df,
                         column_to_bin,
                         column_to_count,
                         bins_r,
                         round_to_decimal,
                         return_groupby
                         )

    compute_t_test = True
    if compute_t_test:
        # parameters
        params = ['error_squared']

        for p in params:
            # groups
            a = df[df['bin'] == bins_r[0]][p].to_numpy()
            b = df[df['bin'] == bins_r[-1]][p].to_numpy()

            # t-test
            t, dof = functions.calculate_t_test(a, b)
            print("{} {}: t-value = {}, degrees-of-freedom = {}".format(method, p, np.round(t, 4), dof))

    # ---

    compute_correlation_table = False
    if compute_correlation_table:
        dfg = df.copy()

        dfb1 = dfg[dfg['bin'] == bins_r[0]]
        dfb2 = dfg[dfg['bin'] == bins_r[1]]
        dfb3 = dfg[dfg['bin'] == bins_r[2]]
        dfb4 = dfg[dfg['bin'] == bins_r[3]]

        min_length = np.min([len(dfb1), len(dfb2), len(dfb3), len(dfb4)])

        data = []
        for i, df1 in enumerate([dfb1, dfb2, dfb3, dfb4]):
            for j, df2 in enumerate([dfb1, dfb2, dfb3, dfb4]):

                prs, pvs = [], []
                srs, svs = [], []

                for k in range(10):
                    a = np.random.choice(df1.error.to_numpy(), size=min_length, replace=False)
                    b = np.random.choice(df2.error.to_numpy(), size=min_length, replace=False)

                    pearson_r, pearson_pval = pearsonr(a, b)
                    spearman_r, spearman_pval = spearmanr(a, b)

                    prs.append(pearson_r)
                    pvs.append(pearson_pval)
                    srs.append(spearman_r)
                    svs.append(spearman_pval)

                data.append([i, j, np.mean(prs), np.mean(pvs), np.mean(srs), np.mean(svs)])

        res = np.array(data)
        dfres = pd.DataFrame(res, columns=['i', 'j', 'p_r', 'p_p', 's_r', 's_p'])
        dfres.to_excel(path_compare_spearman + '/correlation_table_param=error.xlsx', index=False)

        # plot
        p_r = np.reshape(res[:, 2], (4, 4))
        p_p = np.reshape(res[:, 3], (4, 4))
        s_r = np.reshape(res[:, 4], (4, 4))
        s_p = np.reshape(res[:, 5], (4, 4))

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(p_r, cmap='RdBu', vmin=-0.15, vmax=0.15)
        ax[0, 1].imshow(p_p, cmap='RdBu', vmin=0, vmax=0.75)
        ax[1, 0].imshow(s_r, cmap='RdBu', vmin=-0.15, vmax=0.15)
        ax[1, 1].imshow(s_p, cmap='RdBu', vmin=0, vmax=0.75)
        plt.show()

    # ---

    compute_correlation_1d = True
    if compute_correlation_1d:

        save_fig = True

        dependencies = ['r']
        params = ['rmse_z']
        error_threshold = 10
        exclude_DoF = 0

        for d in dependencies:
            for p in params:

                dfcorr = df[df['error'].abs() < error_threshold]
                dfcorr = dfcorr[dfcorr['z_true'].abs() > exclude_DoF / 2]

                if p == 'rmse_z':
                    dfg = dfcorr.copy().groupby('id').mean()
                    dfg['rmse_z'] = np.sqrt(dfg['error_squared'])
                else:
                    dfg = dfcorr.copy()

                a = dfg[d].to_numpy() * microns_per_pixel
                b = dfg[p].to_numpy()

                # pearson + spearman
                pearson_r, pearson_pval = np.round(pearsonr(a, b), 4)
                spearman_r, spearman_pval = np.round(spearmanr(a, b, alternative='greater'), 4)

                # plot
                fig, ax = plt.subplots()

                ax.plot(a, b, 'o', ms=2, alpha=0.75)

                # fit line
                popt, pcov, ffunc = fit.fit(a, b, fit_function=functions.line)
                ax.plot(a, ffunc(a, *popt), linestyle='-', color='black', alpha=0.5,
                        label='Fit: ' + r'$d\sigma_{z}/dr$' + ' = {} \n'
                                                              'Pearson(r, p): {}, {}\n'
                                                              'Spearman(r, p): {}, {}\n'
                                                              'Degrees of Freedom: {}'.format(
                            np.format_float_scientific(popt[0],
                                                       precision=4,
                                                       exp_digits=4),
                            pearson_r, pearson_pval,
                            spearman_r, spearman_pval,
                            len(b) - 2,
                        )
                        )
                ax.set_xlabel(r'$r \: (\mu m)$')
                ax.set_ylabel(r'$\overline{\sigma_{z}^{i}} \: (\mu m)$')
                # ax.set_ylim(ylim)
                ax.legend(loc='lower right', bbox_to_anchor=(1, 1))
                plt.suptitle('{}'.format(method))
                plt.tight_layout()
                if save_fig:
                    plt.savefig(path_compare_spearman +
                                '/{}-correlation_{}_by_{}_errlim{}_DoF{}.svg'.format(method,
                                                                                     p,
                                                                                     d,
                                                                                     error_threshold,
                                                                                     exclude_DoF),
                                )
                plt.show()

    # ---

# ----------------------------------------------------------------------------------------------------------------------
# D. COMPARE SPCT AND IDPT - RADIAL DEPENDENCE

compare_idpt_and_spct_radial = False
if compare_idpt_and_spct_radial:

    # file paths
    path_results_radial = base_dir + '/compare/radial'

    fpi_post_processed = base_dir + '/tests/tm16_cm19/coords/test-coords/post-processed/test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
    fps_post_processed = base_dir + '/tests/spct_soft-baseline_1/coords/test-coords/min_cm_0.5_z_is_z-corr-tilt/post-processed/test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'

    # ---

    # setup
    error_threshold = 10
    bins_r = np.array([150, 300, 450]) / microns_per_pixel
    bins_rr = 15
    min_num_bin = 20

    # ---

    # ---

    path_results_radial = path_results_radial + '/error_threshold_{}'.format(error_threshold)
    if not os.path.exists(path_results_radial):
        os.makedirs(path_results_radial)

    path_results_radial_figs_1d = path_results_radial + '/1d_figs_errlim{}'.format(error_threshold)
    if not os.path.exists(path_results_radial_figs_1d):
        os.makedirs(path_results_radial_figs_1d)

    path_results_radial_coords_1d = path_results_radial + '/1d_groupby-coords_errlim{}'.format(error_threshold)
    if not os.path.exists(path_results_radial_coords_1d):
        os.makedirs(path_results_radial_coords_1d)

    path_results_radial_figs_2d = path_results_radial + '/2d_figs_errlim{}'.format(error_threshold)
    if not os.path.exists(path_results_radial_figs_2d):
        os.makedirs(path_results_radial_figs_2d)

    path_results_radial_coords_2d = path_results_radial + '/2d_groupby-coords_errlim{}'.format(error_threshold)
    if not os.path.exists(path_results_radial_coords_2d):
        os.makedirs(path_results_radial_coords_2d)

    if not os.path.exists(path_results_radial_coords_2d + '/dfim.xlsx'):
        # read
        fpi = fpi_post_processed
        fps = fps_post_processed

        dfi = pd.read_excel(fpi)
        dfs = pd.read_excel(fps)

        # create an absolute error column
        dfi['error_abs'] = dfi['error'].abs()
        dfs['error_abs'] = dfs['error'].abs()

        # filter errors
        dfi = dfi[dfi['error_abs'] < error_threshold]
        dfs = dfs[dfs['error_abs'] < error_threshold]

        # rmse_z
        dfi['error_squared'] = dfi['error'] * dfi['error']
        dfs['error_squared'] = dfs['error'] * dfs['error']

        # BIN - 1D

        column_to_bin = 'r'
        column_to_count = 'id'
        round_to_decimal = 1
        return_groupby = True

        # IDPT
        dfibm, dfibstd = bin.bin_generic(dfi,
                                         column_to_bin,
                                         column_to_count,
                                         bins_r,
                                         round_to_decimal,
                                         return_groupby
                                         )

        # rmse-z
        dfibm['rmse_z'] = np.sqrt(dfibm['error_squared'])

        # resolve floating point bin selecting
        dfibm = dfibm.round({'bin': 0})
        dfibstd = dfibstd.round({'bin': 0})

        dfibm = dfibm.sort_values(['bin'])
        dfibstd = dfibstd.sort_values(['bin'])

        # export
        dfibm.to_excel(path_results_radial_coords_1d + '/dfim.xlsx')
        dfibstd.to_excel(path_results_radial_coords_1d + '/dfistd.xlsx')

        # SPCT
        dfibm, dfibstd = bin.bin_generic(dfs,
                                         column_to_bin,
                                         column_to_count,
                                         bins_r,
                                         round_to_decimal,
                                         return_groupby
                                         )

        # rmse-z
        dfibm['rmse_z'] = np.sqrt(dfibm['error_squared'])

        # resolve floating point bin selecting
        dfibm = dfibm.round({'bin': 0})
        dfibstd = dfibstd.round({'bin': 0})

        dfibm = dfibm.sort_values(['bin'])
        dfibstd = dfibstd.sort_values(['bin'])

        # export
        dfibm.to_excel(path_results_radial_coords_1d + '/dfsm.xlsx')
        dfibstd.to_excel(path_results_radial_coords_1d + '/dfsstd.xlsx')

        # ---

        # BIN - 1D (many r-values)

        # IDPT
        dfibm, dfibstd = bin.bin_generic(dfi,
                                         column_to_bin,
                                         column_to_count,
                                         bins_rr,
                                         round_to_decimal,
                                         return_groupby
                                         )

        # rmse-z
        dfibm['rmse_z'] = np.sqrt(dfibm['error_squared'])

        # resolve floating point bin selecting
        dfibm = dfibm.round({'bin': 0})
        dfibstd = dfibstd.round({'bin': 0})

        dfibm = dfibm.sort_values(['bin'])
        dfibstd = dfibstd.sort_values(['bin'])

        # export
        dfibm.to_excel(path_results_radial_coords_1d + '/dfim_rr.xlsx')
        dfibstd.to_excel(path_results_radial_coords_1d + '/dfistd_rr.xlsx')

        # ---

        # SPCT
        dfibm, dfibstd = bin.bin_generic(dfs,
                                         column_to_bin,
                                         column_to_count,
                                         bins_rr,
                                         round_to_decimal,
                                         return_groupby
                                         )

        # rmse-z
        dfibm['rmse_z'] = np.sqrt(dfibm['error_squared'])

        # resolve floating point bin selecting
        dfibm = dfibm.round({'bin': 0})
        dfibstd = dfibstd.round({'bin': 0})

        dfibm = dfibm.sort_values(['bin'])
        dfibstd = dfibstd.sort_values(['bin'])

        # export
        dfibm.to_excel(path_results_radial_coords_1d + '/dfsm_rr.xlsx')
        dfibstd.to_excel(path_results_radial_coords_1d + '/dfsstd_rr.xlsx')

        # ---

        # BIN - 2D

        # processing
        columns_to_bin = ['r', 'z_true']
        column_to_count = 'id'
        bins_z = np.round(dfi[columns_to_bin[1]].unique(), 2)
        bins = [bins_r, bins_z]
        round_to_decimals = [1, 1]
        return_groupby = True

        dfim, dfistd = bin.bin_generic_2d(dfi,
                                          columns_to_bin,
                                          column_to_count,
                                          bins,
                                          round_to_decimals,
                                          min_num_bin,
                                          return_groupby
                                          )

        dfsm, dfsstd = bin.bin_generic_2d(dfs,
                                          columns_to_bin,
                                          column_to_count,
                                          bins,
                                          round_to_decimals,
                                          min_num_bin,
                                          return_groupby
                                          )

        # rmse-z
        dfim['rmse_z'] = np.sqrt(dfim['error_squared'])
        dfsm['rmse_z'] = np.sqrt(dfsm['error_squared'])

        # resolve floating point bin selecting
        dfim = dfim.round({'bin_tl': 1, 'bin_ll': 1})
        dfistd = dfistd.round({'bin_tl': 1, 'bin_ll': 1})
        dfsm = dfsm.round({'bin_tl': 1, 'bin_ll': 1})
        dfsstd = dfsstd.round({'bin_tl': 1, 'bin_ll': 1})

        dfim = dfim.sort_values(['bin_tl', 'bin_ll'])
        dfistd = dfistd.sort_values(['bin_tl', 'bin_ll'])

        dfsm = dfsm.sort_values(['bin_tl', 'bin_ll'])
        dfsstd = dfsstd.sort_values(['bin_tl', 'bin_ll'])

        # export
        dfim.to_excel(path_results_radial_coords_2d + '/dfim.xlsx')
        dfistd.to_excel(path_results_radial_coords_2d + '/dfistd.xlsx')
        dfsm.to_excel(path_results_radial_coords_2d + '/dfsm.xlsx')
        dfsstd.to_excel(path_results_radial_coords_2d + '/dfsstd.xlsx')

    # ---

    # pubfig
    plot_pubfig = True
    if plot_pubfig:

        pf_error_threshold = 10

        path_results_radial_pubfigs = path_results_radial + '/pubfigs_errlim{}'.format(pf_error_threshold)
        if not os.path.exists(path_results_radial_pubfigs):
            os.makedirs(path_results_radial_pubfigs)

        # read
        fpi = fpi_post_processed
        fps = fps_post_processed

        dfi = pd.read_excel(fpi)
        dfs = pd.read_excel(fps)

        # create an absolute error column
        dfi['error_abs'] = dfi['error'].abs()
        dfs['error_abs'] = dfs['error'].abs()

        # filter errors
        dfi = dfi[dfi['error_abs'] < error_threshold]
        dfs = dfs[dfs['error_abs'] < error_threshold]

        # rmse_z
        dfi['error_squared'] = dfi['error'] * dfi['error']
        dfs['error_squared'] = dfs['error'] * dfs['error']

        # ---

        # BIN - 1D (many r-values)
        bins_r_pf = 5
        column_to_bin = 'r'
        column_to_count = 'id'
        round_to_decimal = 1
        return_groupby = True

        # ---

        compute_t_test = True
        if compute_t_test:
            dfib = bin.bin_generic(dfi,
                                   column_to_bin,
                                   column_to_count,
                                   bins_r_pf,
                                   round_to_decimal,
                                   return_groupby=False,
                                   )
            dfsb = bin.bin_generic(dfs,
                                   column_to_bin,
                                   column_to_count,
                                   bins_r_pf,
                                   round_to_decimal,
                                   return_groupby=False,
                                   )

            # parameters
            params = ['error']

            for p in params:
                for mthd, df in zip(['IDPT', 'SPCT'], [dfib, dfsb]):
                    bins_t_test = df.bin.unique()
                    bins_t_test.sort()

                    # groups
                    b = df[df['bin'] == bins_t_test[0]][p].to_numpy()
                    a = df[df['bin'] == bins_t_test[-1]][p].to_numpy()

                    # t-test
                    t, dof = functions.calculate_t_test(a, b)
                    print("{}: T-test = {}, DoF = {}".format(mthd, np.round(t, 4), dof))

        # ---

        # IDPT
        dfibm, dfibstd = bin.bin_generic(dfi,
                                         column_to_bin,
                                         column_to_count,
                                         bins_r_pf,
                                         round_to_decimal,
                                         return_groupby
                                         )

        # rmse-z
        dfibm['rmse_z'] = np.sqrt(dfibm['error_squared'])

        # resolve floating point bin selecting
        dfibm = dfibm.round({'bin': 0})
        dfibstd = dfibstd.round({'bin': 0})

        dfibm = dfibm.sort_values(['bin'])
        dfibstd = dfibstd.sort_values(['bin'])

        # ---

        # SPCT
        dfsbm, dfsbstd = bin.bin_generic(dfs,
                                         column_to_bin,
                                         column_to_count,
                                         bins_r_pf,
                                         round_to_decimal,
                                         return_groupby
                                         )

        # rmse-z
        dfsbm['rmse_z'] = np.sqrt(dfsbm['error_squared'])

        # resolve floating point bin selecting
        dfsbm = dfsbm.round({'bin': 0})
        dfsbstd = dfsbstd.round({'bin': 0})

        dfsbm = dfsbm.sort_values(['bin'])
        dfsbstd = dfsbstd.sort_values(['bin'])

        # ---

        # export
        dfibm.to_excel(path_results_radial_pubfigs + '/dfibm.xlsx')
        dfibstd.to_excel(path_results_radial_pubfigs + '/dfibstd.xlsx')
        dfsbm.to_excel(path_results_radial_pubfigs + '/dfsbm.xlsx')
        dfsbstd.to_excel(path_results_radial_pubfigs + '/dfsbstd.xlsx')

        # ---

        # Pearson and Spearman
        compute_pearson = True
        if compute_pearson:
            from scipy.stats import pearsonr, spearmanr

            dependent = 'bin'
            params = ['rmse_z']
            for p in params:
                for mthd, df in zip(['IDPT', 'SPCT'], [dfibm, dfsbm]):
                    a = df[dependent].to_numpy()
                    b = df[p].to_numpy()

                    plt.scatter(a, b)
                    plt.show()

                    # pearson + spearman
                    pearson_r, pearson_pval = np.round(pearsonr(a, b), 4)
                    spearman_r, spearman_pval = np.round(spearmanr(a, b), 4)

                    print("{}: Pearson r = {}; p = {}".format(mthd, pearson_r, pearson_pval))
                    print("{}: Spearman r = {}; p = {}".format(mthd, spearman_r, spearman_pval))

        # ---

        print("IDPT Metrics:")
        print("Cm = {} +/- {}".format(np.round(dfibm.cm.mean(), 3), np.round(dfibm.cm.std(), 3)))
        print("Error = {} +/- {}".format(np.round(dfibm.error.mean(), 3), np.round(dfibm.error.std(), 3)))
        print("RMSE = {} +/- {}".format(np.round(dfibm.rmse_z.mean(), 3), np.round(dfibm.rmse_z.std(), 3)))
        print(" ---------- ")
        print("SPCT Metrics:")
        print("Cm = {} +/- {}".format(np.round(dfsbm.cm.mean(), 3), np.round(dfsbm.cm.std(), 3)))
        print("Error = {} +/- {}".format(np.round(dfsbm.error.mean(), 3), np.round(dfsbm.error.std(), 3)))
        print("RMSE = {} +/- {}".format(np.round(dfsbm.rmse_z.mean(), 3), np.round(dfsbm.rmse_z.std(), 3)))

        # plot

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True, figsize=(size_x_inches, size_y_inches * 1.75))

        # bin-rr (all-z, many r's)

        ax0.plot(dfibm.bin * microns_per_pixel, dfibm.cm, '-o', color=sciblue, label='IDPT')
        ax0.plot(dfsbm.bin * microns_per_pixel, dfsbm.cm, '-o', color=scigreen, label='SPCT')
        ax0.set_ylabel(r'$\overline{C_{m}}(r)$')
        ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax1.plot(dfibm.bin * microns_per_pixel, dfibm.error, '-o', color=sciblue, label='IDPT')
        ax1.plot(dfsbm.bin * microns_per_pixel, dfsbm.error, '-o', color=scigreen, label='SPCT')
        ax1.set_ylabel(r'$\overline{\epsilon_{z}}(r) \: (\mu m)$')

        ax2.plot(dfibstd.bin * microns_per_pixel, dfibstd.error ** 2, '-o', color=sciblue, label='IDPT')
        ax2.plot(dfsbstd.bin * microns_per_pixel, dfsbstd.error ** 2, '-o', color=scigreen, label='SPCT')
        ax2.set_ylabel(r'$\overline{var}_{\epsilon_{z}}(r) \: (\mu m)$')

        ax3.plot(dfibm.bin * microns_per_pixel, dfibm.rmse_z, '-o', color=sciblue, label='IDPT')
        ax3.plot(dfsbm.bin * microns_per_pixel, dfsbm.rmse_z, '-o', color=scigreen, label='SPCT')
        ax3.set_xlabel(r'$r \: (\mu m)$')
        ax3.set_xlim([60, 530])
        ax3.set_ylabel(r'$\overline{\sigma_{z}}(r) \: (\mu m)$')

        if error_threshold > 5:
            ax0.set_ylim([0.885, 1.005])
            ax1.set_ylim([-1.5, 1])
            ax2.set_ylim([1, 8.25])

            ax3.set_ylim([1, 3.25])
        else:
            ax0.set_ylim([0.885, 1.005])
            ax1.set_ylim([-1.25, 0.75])
            ax2.set_ylim([0, 4])
            ax3.set_ylim([0, 2.25])

        plt.tight_layout()
        plt.savefig(
            path_results_radial_pubfigs + '/pubfig_bin-r_plot-cm-error-var-rmse-z_errlim{}.svg'.format(error_threshold))
        plt.show()
        plt.close()

    # ---

    # compare - groupby r
    plot_bin_r = True
    if plot_bin_r:
        dfim = pd.read_excel(path_results_radial_coords_1d + '/dfim.xlsx')
        dfistd = pd.read_excel(path_results_radial_coords_1d + '/dfistd.xlsx')

        dfsm = pd.read_excel(path_results_radial_coords_1d + '/dfsm.xlsx')
        dfsstd = pd.read_excel(path_results_radial_coords_1d + '/dfsstd.xlsx')

        # ---

        # plot rmse_z by r

        # ONE FIGURE
        fig, ax = plt.subplots()

        ax.plot(dfim.bin * microns_per_pixel, dfim.rmse_z, '-o', color=sciblue, label='IDPT')
        ax.plot(dfsm.bin * microns_per_pixel, dfsm.rmse_z, '-o', color=scigreen, label='SPCT')

        ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim([100, 500])
        if error_threshold < 5:
            ax.set_ylim([0.4, 2.1])
            ax.set_yticks([0.5, 1, 1.5, 2])
        else:
            ax.set_ylim([0.9, 3.1])
            ax.set_yticks([1, 2, 3])
            ax.set_xticks([200, 300, 400])
        ax.legend(loc='upper left')

        plt.savefig(path_results_radial_figs_1d + '/bin-r_rmse-z_by_r_1subplot.svg')
        plt.tight_layout()
        plt.show()
        plt.close()

        # ---

        # TWO FIGURES
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(dfim.bin * microns_per_pixel, dfim.rmse_z,
                 '-o', color=sciblue,
                 label='IDPT',
                 )
        ax2.plot(dfsm.bin * microns_per_pixel, dfsm.rmse_z,
                 '-o', color=scigreen,
                 label='SPCT',
                 )

        ax1.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        # ax1.set_ylim([0.4, 2])
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax2.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        # ax2.set_ylim([0.4, 2])
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.set_xlabel(r'$r \: (\mu m)$')

        plt.savefig(path_results_radial_figs_1d + '/bin-r_rmse-z_by_r_2subplots.svg')
        plt.tight_layout()
        plt.show()
        plt.close()

        # ---

        # TWO FIGURE - plot error bars by r
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.errorbar(dfim.bin * microns_per_pixel, dfim.error, yerr=dfistd.error,
                     fmt='-o', elinewidth=1, capsize=2, color=sciblue,
                     label='IDPT',
                     )
        ax2.errorbar(dfsm.bin * microns_per_pixel, dfsm.error, yerr=dfsstd.error,
                     fmt='-o', elinewidth=1, capsize=2, color=scigreen,
                     label='SPCT',
                     )

        ax1.set_ylabel(r'$\epsilon_{z}^{\delta} \: (\mu m)$')
        # ax1.set_ylim([-2.95, 1.45])
        # ax1.set_yticks([-2, -1, 0, 1])
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax2.set_ylabel(r'$\epsilon_{z}^{\delta} \: (\mu m)$')
        # ax2.set_ylim([-2.95, 1.45])
        # ax2.set_yticks([-2, -1, 0, 1])
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax2.set_xlabel(r'$r \: (\mu m)$')
        # ax2.set_xlim([0, 600])
        # ax2.set_xticks(ticks=[100, 300, 500])

        plt.savefig(path_results_radial_figs_1d + '/bin-r_normalized-z-errors_by_z_mean+std.svg')
        plt.tight_layout()
        plt.show()
        plt.close()

        # ---

        # ONE FIGURE - plot error bars by r
        fig, ax = plt.subplots()

        ax.errorbar(dfim.bin * microns_per_pixel, dfim.error, yerr=dfistd.error,
                    fmt='-o', elinewidth=1, capsize=2, color=sciblue,
                    label='IDPT',
                    )
        ax.errorbar(dfsm.bin * microns_per_pixel, dfsm.error, yerr=dfsstd.error,
                    fmt='-o', elinewidth=1, capsize=2, color=scigreen,
                    label='SPCT',
                    )

        ax.set_ylabel(r'$\epsilon_{z}^{\delta} \: (\mu m)$')
        ax2.set_xlabel(r'$r \: (\mu m)$')
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.savefig(path_results_radial_figs_1d + '/bin-r_normalized-z-errors_by_z_mean+std_1FIG.svg')
        plt.tight_layout()
        plt.show()
        plt.close()

    # compare - groupby r (many r-values)
    plot_bin_rr = True
    if plot_bin_rr:
        dfim = pd.read_excel(path_results_radial_coords_1d + '/dfim_rr.xlsx')
        dfistd = pd.read_excel(path_results_radial_coords_1d + '/dfistd_rr.xlsx')

        dfsm = pd.read_excel(path_results_radial_coords_1d + '/dfsm_rr.xlsx')
        dfsstd = pd.read_excel(path_results_radial_coords_1d + '/dfsstd_rr.xlsx')

        # ---

        # plot error bars by r
        plot_columns = [['rmse_z', 'rmse_z'],
                        ['error', 'error'],
                        ['cm', 'cm'],
                        ['dist_dxyg', 'dist_gauss_dxy'],
                        ['drg', 'gauss_dr'],
                        ['count_id', 'count_id']
                        ]

        for pc in plot_columns:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

            if pc[0] in ['rmse_z', 'count_id']:
                ax1.plot(dfim.bin * microns_per_pixel, dfim[pc[0]], '-o',
                         ms=1, color=sciblue, label='IDPT')
                ax2.plot(dfsm.bin * microns_per_pixel, dfsm[pc[0]], '-o',
                         ms=1, color=scigreen, label='SPCT')
            else:
                ax1.errorbar(dfim.bin * microns_per_pixel, dfim[pc[0]], yerr=dfistd[pc[0]],
                             fmt='-o', elinewidth=1, capsize=2, color=sciblue,
                             label='IDPT',
                             )
                ax2.errorbar(dfsm.bin * microns_per_pixel, dfsm[pc[1]], yerr=dfsstd[pc[1]],
                             fmt='-o', elinewidth=1, capsize=2, color=scigreen,
                             label='SPCT',
                             )

            ax1.set_ylabel(pc[0])
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax2.set_ylabel(pc[1])
            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax2.set_xlabel(r'$r \: (\mu m)$')

            plt.savefig(path_results_radial_figs_1d + '/bin-rr_plot-{}_by_r.svg'.format(pc[0]))
            plt.tight_layout()
            plt.show()
            plt.close()

        # ---

        # ---

        # plot error bars by r
        plot_columns = [
            ['error', 'error'],
            ['cm', 'cm'],
            ['dist_dxyg', 'dist_gauss_dxy'],
            ['drg', 'gauss_dr'],
        ]

        for pc in plot_columns:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

            ax1.plot(dfistd.bin * microns_per_pixel, dfistd[pc[0]],
                     '-o', color=sciblue, label='IDPT',
                     )
            ax2.plot(dfsstd.bin * microns_per_pixel, dfsstd[pc[1]],
                     '-o', color=scigreen, label='SPCT',
                     )

            ax1.set_ylabel(pc[0])
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax2.set_ylabel(pc[1])
            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax2.set_xlabel(r'$r \: (\mu m)$')

            plt.savefig(path_results_radial_figs_1d + '/bin-rr_plot-variance-{}_by_r.svg'.format(pc[0]))
            plt.tight_layout()
            plt.show()
            plt.close()

        # ---

    # compare - bin(r, z)
    plot_bin_2d_r_z = True
    if plot_bin_2d_r_z:

        dfim = pd.read_excel(path_results_radial_coords_2d + '/dfim.xlsx')
        dfistd = pd.read_excel(path_results_radial_coords_2d + '/dfistd.xlsx')
        dfsm = pd.read_excel(path_results_radial_coords_2d + '/dfsm.xlsx')
        dfsstd = pd.read_excel(path_results_radial_coords_2d + '/dfsstd.xlsx')

        dfim = dfim.sort_values(['bin_tl', 'bin_ll'])
        dfistd = dfistd.sort_values(['bin_tl', 'bin_ll'])

        bins_r_idpt = dfim['bin_tl'].unique()
        bins_r_idpt.sort()

        dfsm = dfsm.sort_values(['bin_tl', 'bin_ll'])
        dfsstd = dfsstd.sort_values(['bin_tl', 'bin_ll'])

        bins_r_spct = dfsm['bin_tl'].unique()
        bins_r_spct.sort()

        # define
        plot_fits = [False, True]
        fit_func = functions.quadratic_slide
        xlim = [-52.5, 52.5]
        xticks = [-50, -25, 0, 25, 50]
        ylims = ([-4.45, 3.15], [-2.25, 2.25])
        linestyles = ['solid', 'solid', 'solid', 'solid']
        markers = ['o', 'o', 'o', 'o']
        ms = 1.5
        clr_mods = [0.8, 1, 1.2, 1.4]

        # plot - error by z
        plot_radial_error = True
        if plot_radial_error:

            # rmse-z
            for plot_fit, ylim in zip(plot_fits, ylims):

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))

                # IDPT - FIGURE 1
                for i, bin_r, ls, mkr in zip(clr_mods, bins_r_idpt, linestyles, markers):

                    dfbr = dfim[dfim['bin_tl'] == bin_r]
                    dfbr_std = dfistd[dfistd['bin_tl'] == bin_r]

                    # plot: fit
                    if plot_fit:
                        popt, pcov = curve_fit(fit_func, dfbr.z_true, dfbr.rmse_z)
                        fit_bin_ll = np.linspace(dfbr.bin_ll.min(), dfbr.bin_ll.max(), len(dfbr.z_true) + 1)
                        ax1.plot(fit_bin_ll, fit_func(fit_bin_ll, *popt),
                                 linestyle=ls,
                                 marker=mkr,
                                 markersize=ms,
                                 color=lighten_color(sciblue, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )

                    else:
                        # scatter: mean +/- std
                        ax1.plot(dfbr.bin_ll, dfbr.rmse_z, '-o',
                                 ms=ms,
                                 color=lighten_color(sciblue, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )

                ax1.set_xlim(xlim)
                ax1.set_xticks(ticks=xticks)
                ax1.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                # ax1.set_ylim([0, 2.75])
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')

                # ---

                del dfbr, dfbr_std

                # ---

                # SPCT - FIGURE 2
                for i, bin_r, ls, mkr in zip(clr_mods, bins_r_spct, linestyles, markers):

                    dfbr = dfsm[dfsm['bin_tl'] == bin_r]
                    dfbr_std = dfsstd[dfsstd['bin_tl'] == bin_r]

                    # plot: fit
                    if plot_fit:
                        if len(dfbr.error) < 5:
                            continue
                        popt, pcov = curve_fit(fit_func, dfbr.z_true, dfbr.rmse_z)
                        fit_bin_ll = np.linspace(dfbr.bin_ll.min(), dfbr.bin_ll.max(), len(dfbr.z_true) + 1)
                        ax2.plot(fit_bin_ll, fit_func(fit_bin_ll, *popt),
                                 linestyle=ls,
                                 marker=mkr,
                                 markersize=ms,
                                 color=lighten_color(scigreen, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )
                    else:
                        # scatter: mean +/- std
                        ax2.plot(dfbr.bin_ll, dfbr.rmse_z,
                                 '-o', ms=ms,
                                 color=lighten_color(scigreen, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )

                ax2.set_xlabel(r'$z \: (\mu m)$')
                ax2.set_xlim(xlim)
                ax2.set_xticks(ticks=xticks)
                ax2.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                # ax2.set_ylim([0, 2.75])
                ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')

                plt.tight_layout()
                plt.savefig(path_results_radial_figs_2d + '/bin-r-z_plot_rmse-z_fit{}_errlim{}.svg'.format(plot_fit,
                                                                                                           error_threshold),
                            )
                plt.show()
                plt.close()

            # ---

            # error
            for plot_fit, ylim in zip(plot_fits, ylims):

                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))

                # IDPT - FIGURE 1
                for i, bin_r, ls, mkr in zip(clr_mods, bins_r_idpt, linestyles, markers):

                    dfbr = dfim[dfim['bin_tl'] == bin_r]
                    dfbr_std = dfistd[dfistd['bin_tl'] == bin_r]

                    # plot: fit
                    if plot_fit:
                        popt, pcov = curve_fit(fit_func, dfbr.z_true, dfbr.error)
                        fit_bin_ll = np.linspace(dfbr.bin_ll.min(), dfbr.bin_ll.max(), len(dfbr.z_true) + 1)
                        ax1.plot(fit_bin_ll, fit_func(fit_bin_ll, *popt),
                                 linestyle=ls,
                                 marker=mkr,
                                 markersize=ms,
                                 color=lighten_color(sciblue, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )

                    else:
                        # scatter: mean +/- std
                        ax1.errorbar(dfbr.bin_ll, dfbr.error, yerr=dfbr_std.error,
                                     fmt='-o', ms=ms, elinewidth=0.5, capsize=1,
                                     color=lighten_color(sciblue, amount=i),
                                     label=int(np.round(bin_r * microns_per_pixel, 0)),
                                     )

                ax1.set_xlim(xlim)
                ax1.set_xticks(ticks=xticks)
                ax1.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')
                # ax1.set_ylim(ylim)

                # ---

                del dfbr, dfbr_std

                # ---

                # SPCT - FIGURE 2
                for i, bin_r, ls, mkr in zip(clr_mods, bins_r_spct, linestyles, markers):

                    dfbr = dfsm[dfsm['bin_tl'] == bin_r]
                    dfbr_std = dfsstd[dfsstd['bin_tl'] == bin_r]

                    # plot: fit
                    if plot_fit:
                        if len(dfbr.error) < 5:
                            continue
                        popt, pcov = curve_fit(fit_func, dfbr.z_true, dfbr.error)
                        fit_bin_ll = np.linspace(dfbr.bin_ll.min(), dfbr.bin_ll.max(), len(dfbr.z_true) + 1)
                        ax2.plot(fit_bin_ll, fit_func(fit_bin_ll, *popt),
                                 linestyle=ls,
                                 marker=mkr,
                                 markersize=ms,
                                 color=lighten_color(scigreen, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )
                    else:
                        # scatter: mean +/- std
                        ax2.errorbar(dfbr.bin_ll, dfbr.error, yerr=dfbr_std.error,
                                     fmt='-o', ms=ms, elinewidth=0.5, capsize=1,
                                     color=lighten_color(scigreen, amount=i),
                                     label=int(np.round(bin_r * microns_per_pixel, 0)),
                                     )

                ax2.set_xlabel(r'$z \: (\mu m)$')
                ax2.set_xlim(xlim)
                ax2.set_xticks(ticks=xticks)
                ax2.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                # ax2.set_ylim(ylim)
                ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')

                plt.tight_layout()
                plt.savefig(path_results_radial_figs_2d + '/bin-r-z_plot_err-z_fit{}_errlim{}.svg'.format(plot_fit,
                                                                                                          error_threshold),
                            )
                plt.show()
                plt.close()

            # ---

        # ---

        # plot other columns
        plot_columns = [['count_id', 'count_id'],
                        ['cm', 'cm'],
                        ['gauss_sigma_x', 'gauss_sigma_x'],
                        ['gauss_sigma_y', 'gauss_sigma_y'],
                        ['dist_dxyg', 'dist_gauss_dxy'],
                        ['drg', 'gauss_dr'],
                        ]

        plot_scaling_factors = [1, 1,
                                microns_per_pixel, microns_per_pixel, microns_per_pixel, microns_per_pixel,
                                ]

        plot_lbls = [[r'$N_{p} \: (\#)$', r'$N_{p} \: (\#)$'],
                     [r'$C_{m}$', r'$C_{m}$'],
                     [r'$w_{x} \: (\mu m)$', r'$w_{x} \: (\mu m)$'],
                     [r'$w_{y} \: (\mu m)$', r'$w_{y} \: (\mu m)$'],
                     [r'$\Delta xy \: (\mu m)$', r'$\Delta xy \: (\mu m)$'],
                     [r'$\Delta r \: (\mu m)$', r'$\Delta r \: (\mu m)$'],
                     ]

        ylims = [None,
                 None,  # [0.68, 1.025]
                 None,
                 None,
                 None,  # [0, 2.55],
                 None,  # [-1.05, 1.05],
                 ]

        elinewidth = 0.35
        capsize = 1.25

        for pc, pcsf, pclbl, ylim in zip(plot_columns, plot_scaling_factors, plot_lbls, ylims):

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))

            # IDPT - FIGURE 1
            for i, bin_r, in zip(clr_mods, bins_r_idpt):
                dfbr = dfim[dfim['bin_tl'] == bin_r]
                dfbr_std = dfistd[dfistd['bin_tl'] == bin_r]
                if pc[0] == 'count_id':
                    ax1.plot(dfbr.bin_ll, dfbr[pc[0]] * pcsf, '-o',
                             ms=ms, color=lighten_color(sciblue, amount=i),
                             label=int(np.round(bin_r * microns_per_pixel, 0)), )
                else:
                    ax1.errorbar(dfbr.bin_ll, dfbr[pc[0]] * pcsf, yerr=dfbr_std[pc[0]] * pcsf,
                                 fmt='-o', ms=ms, elinewidth=elinewidth, capsize=capsize,
                                 color=lighten_color(sciblue, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )

            # SPCT - FIGURE 2
            for i, bin_r, in zip(clr_mods, bins_r_spct):
                dfbr = dfsm[dfsm['bin_tl'] == bin_r]
                dfbr_std = dfsstd[dfsstd['bin_tl'] == bin_r]
                if pc[1] == 'count_id':
                    ax2.plot(dfbr.bin_ll, dfbr[pc[1]] * pcsf, '-o',
                             ms=ms, color=lighten_color(scigreen, amount=i),
                             label=int(np.round(bin_r * microns_per_pixel, 0)), )
                else:
                    ax2.errorbar(dfbr.bin_ll, dfbr[pc[1]] * pcsf, yerr=dfbr_std[pc[1]] * pcsf,
                                 fmt='-o', ms=ms, elinewidth=elinewidth, capsize=capsize,
                                 color=lighten_color(scigreen, amount=i),
                                 label=int(np.round(bin_r * microns_per_pixel, 0)),
                                 )

            if ylim is not None:
                ax1.set_ylim(ylim)
                ax2.set_ylim(ylim)

            ax1.set_ylabel(pclbl[0])
            # ax1.set_xlabel(r'$z \: (\mu m)$')
            # ax1.set_xlim(xlim)
            # ax1.set_xticks(ticks=xticks)
            ax2.set_ylabel(pclbl[1])
            ax2.set_xlabel(r'$z \: (\mu m)$')
            ax2.set_xlim(xlim)
            ax2.set_xticks(ticks=xticks)
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')
            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')

            plt.tight_layout()
            plt.savefig(path_results_radial_figs_2d + '/bin-r-z_plot-{}.svg'.format(pc[0]),
                        )
            plt.show()
            plt.close()

        # ---

    # ---

# ----------------------------------------------------------------------------------------------------------------------
# D. COMPARE SPCT AND IDPT SIMILARITY

compare_idpt_and_spct_similarity = False
if compare_idpt_and_spct_similarity:

    path_compare = base_dir + '/compare'
    path_compare_groupby = path_compare + '/similarity/groupby_coords'
    path_compare_figs = path_compare + '/similarity/figs'

    fn_idpt_raw = 'test_coords_tm16_cm19_dzf'
    lbl_i = 'IDPT'

    fn_idpt_res = 'test_coords_tm16_cm19_dzf-post-processed'
    lbl_ii = 'IDPT'

    fn_spct_raw = 'test_coords_spct-1_dzf'
    lbl_s = 'SPCT'

    fn_spct_res = 'test_coords_spct-1_dzf-post-processed'
    lbl_ss = 'SPCT'

    # ---

    # plot setup
    x = 'z_true'
    y = 'cm'

    # ---

    # read coords
    if not os.path.exists(join(path_compare_groupby, fn_idpt_res + '.xlsx')):

        if not os.path.exists(path_compare_groupby):
            os.makedirs(path_compare_groupby)

        # export groupby stats
        fp_idpt_raw = '/tests/tm16_cm19/coords/test-coords/custom-correction/'
        dfi = pd.read_excel(base_dir + fp_idpt_raw + fn_idpt_raw + '.xlsx')

        fp_idpt_res = '/tests/tm16_cm19/coords/test-coords/post-processed/'
        dfii = pd.read_excel(base_dir + fp_idpt_res + fn_idpt_res + '.xlsx')

        fp_spct_raw = '/tests/spct_soft-baseline_1/coords/test-coords/custom-correction/'
        dfs = pd.read_excel(base_dir + fp_spct_raw + fn_spct_raw + '.xlsx')

        fp_spct_res = '/tests/spct_soft-baseline_1/coords/test-coords/post-processed/'
        dfss = pd.read_excel(base_dir + fp_spct_res + fn_spct_res + '.xlsx')

        # processing - z range
        dfi = dfi[(dfi[x] > z_range[0]) & (dfi[x] < z_range[1])]
        dfs = dfs[(dfs[x] > z_range[0]) & (dfs[x] < z_range[1])]

        # processing - groupby
        dfgi = dfi.groupby(x).mean().reset_index()
        dfgii = dfii.groupby(x).mean().reset_index()
        dfgs = dfs.groupby(x).mean().reset_index()
        dfgss = dfss.groupby(x).mean().reset_index()

        dfgi.to_excel(join(path_compare_groupby, fn_idpt_raw + '.xlsx'))
        dfgii.to_excel(join(path_compare_groupby, fn_idpt_res + '.xlsx'))
        dfgs.to_excel(join(path_compare_groupby, fn_spct_raw + '.xlsx'))
        dfgss.to_excel(join(path_compare_groupby, fn_spct_res + '.xlsx'))
    else:
        dfgi = pd.read_excel(join(path_compare_groupby, fn_idpt_raw + '.xlsx'))
        dfgii = pd.read_excel(join(path_compare_groupby, fn_idpt_res + '.xlsx'))
        dfgs = pd.read_excel(join(path_compare_groupby, fn_spct_raw + '.xlsx'))
        dfgss = pd.read_excel(join(path_compare_groupby, fn_spct_res + '.xlsx'))

    # ---

    # plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches))

    ax1.plot(dfgi[x], dfgi[y], '--o', ms=2, label=lbl_i)
    ax1.plot(dfgs[x], dfgs[y], '--o', ms=2, label=lbl_s)

    ax2.plot(dfgii[x], dfgii[y], '-o', ms=2, label=lbl_ii)
    ax2.plot(dfgss[x], dfgss[y], '-o', ms=2, label=lbl_ss)

    ax1.set_ylabel(r'$C_{m}$')
    ax1.set_ylim([0.75, 1.01])
    ax1.set_yticks([0.8, 0.9, 1.0])
    ax1.legend(title='Raw', loc='upper left', bbox_to_anchor=(1, 1))

    ax2.set_ylabel(r'$C_{m}$')
    ax2.set_ylim([0.75, 1.01])
    ax2.set_yticks([0.8, 0.9, 1.0])
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_xticks([-50, -25, 0, 25, 50])
    ax2.legend(title='Filtered', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(path_compare_figs + '/cm_by_z.svg')
    plt.show()
    plt.close()

    # ---

    # processing
    dfgi['gauss_sigma'] = np.sqrt(dfgi['gauss_sigma_x'] ** 2 + dfgi['gauss_sigma_y'] ** 2)
    dfgii['gauss_sigma'] = np.sqrt(dfgii['gauss_sigma_x'] ** 2 + dfgii['gauss_sigma_y'] ** 2)
    dfgs['gauss_sigma'] = np.sqrt(dfgs['gauss_sigma_x'] ** 2 + dfgs['gauss_sigma_y'] ** 2)
    dfgss['gauss_sigma'] = np.sqrt(dfgss['gauss_sigma_x'] ** 2 + dfgss['gauss_sigma_y'] ** 2)

    y1 = 'gauss_A'
    y2 = 'contour_area'
    y31 = 'gauss_sigma_x'
    y32 = 'gauss_sigma_y'

    # plotting
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.25))

    ax1.plot(dfgs[x], dfgs[y1], '-o', ms=1, label=lbl_i)

    ax2.plot(dfgs[x], dfgs[y2] * microns_per_pixel ** 2, '-o', ms=1, label=lbl_ii)

    ax3.plot(dfgii[x], dfgii[y31] * microns_per_pixel, 'o', ms=2, color=sciblue, label='x')
    ax3.plot(dfgii[x], dfgii[y32] * microns_per_pixel, 'o', ms=2, color=scired, label='y')

    ax1.set_ylabel(r'$I_{0} \: (A.U.)$')
    ax2.set_ylabel(r'$A \: (\mu m^2)$')
    ax3.set_ylabel(r'$w \: (\mu m)$')
    ax3.set_xlabel(r'$z \: (\mu m)$')
    ax3.set_xticks([-50, -25, 0, 25, 50])
    ax3.legend(loc='lower left', borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    plt.tight_layout()
    plt.savefig(path_compare_figs + '/particle-image-stats_by_z.svg')
    plt.show()
    plt.close()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 16. COMPUTE SIMILARITY SWEEP

compute_cm_sweep = False
if compute_cm_sweep:

    # modifier
    method_ = 'spct'

    # file paths
    path_results_cm_sweep = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/' \
                            'analyses/shared-results/cm-sweep'

    fpi_post_processed = base_dir + '/tests/tm16_cm19/coords/test-coords/post-processed/test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
    fps_post_processed = base_dir + '/tests/spct_soft-baseline_1/coords/test-coords/min_cm_0.5_z_is_z-corr-tilt/post-processed/test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'

    # ---

    # read dataframe
    if method_ == 'idpt':
        fp = fpi_post_processed
    else:
        fp = fps_post_processed
    dft = pd.read_excel(fp)

    # get only necessary columns
    dft = dft[['frame', 'id', 'z_true', 'z', 'error', 'cm']]

    # get stats
    true_zs = dft.z_true.unique()
    dft['bin'] = np.round(dft['z_true'], 0).astype(int)
    dzs = dft.bin.unique()
    num_dz_steps = len(dzs)

    # ---

    # compute for each: cm_i, z
    compute_2d_cm_sweep = False

    if compute_2d_cm_sweep:
        cm_sweep = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]
        true_num_per_bin = true_num_particles_per_frame * num_frames_per_step * num_dz_steps
        res_cm = []

        # binning
        columns_to_bin = ['']
        for cm_i in cm_sweep:
            dfrmse_cm = bin.bin_local_rmse_z(dft, column_to_bin='bin', bins=dzs, min_cm=cm_i, z_range=None,
                                             round_to_decimal=0, df_ground_truth=None, dropna=True,
                                             error_column='error')
            dfcm = dfrmse_cm[['cm', 'num_bind', 'num_meas', 'percent_meas', 'rmse_z']]
            dfcm['cm_i'] = cm_i
            res_cm.append(dfcm)

        dfcm = pd.concat(res_cm)  # DataFrame(, columns=['cm', 'num_bind', 'num_meas', 'percent_meas', 'rmse_z', 'cmi'])
        dfcm['true_num'] = true_num_per_bin / num_dz_steps
        dfcm['true_percent_meas'] = dfcm['num_meas'] / dfcm['true_num'] * 100
        dfcm.to_excel(path_results_cm_sweep + '/{}-dfrmse_2d-cm-sweep.xlsx'.format(method_), index_label='bin')

    # ---

    # compute average(z) for each cm_i
    compute_1d_cm_sweep = True

    if compute_1d_cm_sweep:
        cm_sweep = np.linspace(0.5, 0.995, 250)
        true_num_per_bin = true_num_particles_per_frame * num_frames_per_step * num_dz_steps

        res_cm = []
        for cm_i in cm_sweep:
            dfrmse_cm = bin.bin_local_rmse_z(dft, column_to_bin='frame', bins=1, min_cm=cm_i, z_range=None,
                                             round_to_decimal=0, df_ground_truth=None, dropna=True,
                                             error_column='error')
            cm_ii = [cm_i]
            if len(dfrmse_cm) < 1:
                cm_ii.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
            else:
                cm_ii.extend(dfrmse_cm[['cm', 'num_bind', 'num_meas', 'percent_meas', 'rmse_z']].values.tolist()[0])
            res_cm.append(cm_ii)

        res_cm = np.array(res_cm)
        dfcm = pd.DataFrame(res_cm, columns=['cmi', 'cm', 'num_bind', 'num_meas', 'percent_meas', 'rmse_z'])
        dfcm['true_num'] = true_num_per_bin
        dfcm['true_percent_meas'] = dfcm['num_meas'] / dfcm['true_num'] * 100
        dfcm.to_excel(path_results_cm_sweep + '/dfrmse_cm-sweep-{}.xlsx'.format(method_), index_label='i')

        # ---

# ----------------------------------------------------------------------------------------------------------------------
# 16. COMPARE CM SWEEP OF IDPT AND SPCT

compare_cm_sweep = False
if compare_cm_sweep:

    plot_2d_cm_sweep = False
    plot_1d_cm_sweep = True

    # shared
    path_figs = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                'shared-results/cm-sweep'
    save_plots = True
    show_plots = True

    # ---

    if plot_2d_cm_sweep:
        # filepaths
        dfi = pd.read_excel(
            '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/shared-results/'
            'cm-sweep/idpt-dfrmse_2d-cm-sweep.xlsx'
        )
        dfs = pd.read_excel(
            '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/shared-results/'
            'cm-sweep/spct-dfrmse_2d-cm-sweep.xlsx'
        )

        # processing
        dfi['mdx'] = np.sqrt(area_microns / dfi['num_meas'])
        dfs['mdx'] = np.sqrt(area_microns / dfs['num_meas'])

        # plot percent measure per z-step ('bin'): cms = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]

        # setup
        cmis = [0.5, 0.85, 0.95]
        ms = 3
        markers = ['o', '^', 's', 'D', 'v', 'P', '*', 'd', 'X', 'p']
        xylim = 62.5
        xyticks = [-50, -25, 0, 25, 50]

        plot_columns = ['mdx', 'percent_meas', 'true_percent_meas']
        plot_column_labels = [r'$\overline{\delta x} \: (\mu m)$', r'$\phi_{ID} \: (\%)$', r'$\phi \: (\%)$']

        for pc, pl in zip(plot_columns, plot_column_labels):

            fig, axr = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))
            iclrs = iter(cm.Blues(np.linspace(0.5, 0.85, len(cmis))))
            sclrs = iter(cm.RdPu(np.linspace(0.5, 0.85, len(cmis))))
            ps = []
            for cmi, mk in zip(cmis, markers):
                dficm = dfi[dfi['cm_i'] == cmi]
                dfscm = dfs[dfs['cm_i'] == cmi]

                pi1, = axr.plot(dficm.bin, dficm[pc], marker=mk, ms=ms, ls='-', c=next(iclrs), label=np.round(cmi, 3))
                ps1, = axr.plot(dfscm.bin, dfscm[pc], marker=mk, ms=ms, ls='dotted', c=next(sclrs))
                ps.append((pi1, ps1))

            axr.set_xlim([-xylim, xylim])
            axr.set_xticks(ticks=xyticks)
            axr.set_xlabel(r'$z \: (\mu m)$')
            axr.set_ylabel(pl)

            if pc == 'mdx':
                axr.set_ylim([47.5, 125])
                # axr.set_yscale('log')

            l = axr.legend(ps, cmis, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
                           loc='upper left', bbox_to_anchor=(1, 1), title=r'$c_{m, min}$')

            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs + '/cm-sweep_{}_by_cmi-2d_dotted.png'.format(pc))
            if show_plots:
                plt.show()
            plt.close()

    # ---

    if plot_1d_cm_sweep:
        # filepaths
        dfi = pd.read_excel(
            '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/shared-results/'
            'cm-sweep/dfrmse_cm-sweep-idpt.xlsx'
        )
        dfs = pd.read_excel(
            '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/shared-results/'
            'cm-sweep/dfrmse_cm-sweep-spct.xlsx'
        )

        # processing
        dfi['mdx'] = np.sqrt(area_microns / dfi['num_meas'])
        dfs['mdx'] = np.sqrt(area_microns / dfs['num_meas'])

        # processing
        i_idxhalf = np.argmin(np.abs(dfi.true_percent_meas - 50))
        i_half_rmse = dfi.iloc[i_idxhalf].rmse_z
        i_half_mdx = dfi.iloc[i_idxhalf].mdx
        s_idxhalf = np.argmin(np.abs(dfs.true_percent_meas - 50))
        s_half_rmse = dfs.iloc[s_idxhalf].rmse_z
        s_half_mdx = dfs.iloc[s_idxhalf].mdx

        # setup figures
        sciblue_mod = 0.85
        scigreen_mod = 1.25
        save_plots = True
        show_plots = True

        # figure 1. rmse-z, phi-ID (cm_input)
        xlim_lefts = [0.475, 0.675]
        lbls = [['IDPT: ' + r'$\sigma_{z}(\phi=50\%)=$' + '{}'.format(np.round(i_half_rmse, 3)),
                 'SPCT: ' + r'$\sigma_{z}(\phi=50\%)=$' + '{}'.format(np.round(s_half_rmse, 3))],
                ['IDPT', 'SPCT']]

        print(lbls)
        print("1/2 percent measure: IDPT mdx = {}, SPCT mdx = {}".format(np.round(i_half_mdx, 3),
                                                                         np.round(s_half_mdx, 3)))

        for xleft, lbl in zip(xlim_lefts, lbls):
            fig, [axr, axm, ax] = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.5))

            # top - percent measure
            pi1, = axr.plot(dfi.cmi, dfi.true_percent_meas, color=sciblue, linestyle='-')
            ps1, = axr.plot(dfs.cmi, dfs.true_percent_meas, color=scigreen, linestyle='-')
            axr.set_ylabel(r'$\overline{\phi} \: (\%)$')
            axr.set_ylim(top=107.5)

            axrr = axr.twinx()
            pi2, = axrr.plot(dfi.cmi, dfi.percent_meas, color=sciblue, linestyle='dotted')
            ps2, = axrr.plot(dfs.cmi, dfs.percent_meas, color=scigreen, linestyle='dotted')
            axrr.set_ylabel(r'$\cdots \: \overline{\phi_{ID}} \: (\%)$')
            axrr.set_ylim(top=107.5)

            # middle - mean lateral spacing
            axm.plot(dfi.cmi, dfi.mdx, color=sciblue, linestyle='-')
            axm.plot(dfs.cmi, dfs.mdx, color=scigreen, linestyle='-')
            axm.set_ylabel(r'$\overline{\delta x} \: (\mu m)$')
            axm.set_ylim(bottom=5, top=35)
            # axm.set_yscale('log')

            # bottom - rmse_z
            ax.plot(dfi.cmi, dfi.rmse_z, color=sciblue)  # , label=lbl[0]
            ax.plot(dfs.cmi, dfs.rmse_z, color=scigreen)  # , label=lbl[1]
            ax.set_xlabel(r'$c_{m,min}$')
            ax.set_xlim(left=xleft)
            ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            ax.set_ylim(top=2.8)

            l = axm.legend([(pi1, pi2), (ps1, ps2)], ['IDPT', 'SPCT'], numpoints=1,
                           handler_map={tuple: HandlerTuple(ndivide=None)},
                           loc='upper left', handletextpad=0.2, borderaxespad=0.25)

            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs + '/cm-sweep_by_cmi_i{}_dots.svg'.format(xleft))
            if show_plots:
                plt.show()
            plt.close()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 16. PLOT SPCT STATS

plot_spct_stats = False
if plot_spct_stats:
    path_results = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                   'results-07.29.22-idpt-tmg/results/spct_stats'

    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
         'results-07.29.22-idpt-tmg/tests/spct_hard-baseline_1/tracking/calibration-spct_micrometer_5um-spct-cal/' \
         'calib_spct_stats_11.06.21_z-micrometer-v2_5umMS__spct-cal.xlsx'

    df = pd.read_excel(fp)

    # filters
    df = df[(df['z_corr'] > -55) & (df['z_corr'] < 53)]

    # processing
    dfc = df.groupby('id').count().reset_index()
    max_counts = dfc.z_true.max()
    passing_ids = dfc[dfc['z_true'] > max_counts * 0.01].id.values

    # filters
    df = df[df['id'].isin(passing_ids)]

    # bin to smooth
    column_to_bin = 'z_true'
    column_to_count = 'id'
    bins = len(df.z_true.unique())
    round_to_decimal = 1
    return_groupby = True

    dfm, dfstd = bin.bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

    dfm['snr'] = (dfm['mean_int'] - dfm['bkg_mean'].mean()) / dfm['bkg_noise'].mean()
    # dfm['z_corr'] = dfm['z_corr'] - 1

    # export dataframe
    dfm.to_excel(path_results + '/groupby_z-true_mean.xlsx')
    dfstd.to_excel(path_results + '/groupby_z-true_std.xlsx')

    # ---

    # plot - snr, contour_area ~ f(z_corr)
    x = 'z_corr'
    y1 = 'snr'
    y2 = 'contour_area'
    ms = 4

    fig, ax = plt.subplots()

    p1, = ax.plot(dfm[x], dfm[y1] * microns_per_pixel ** 2, '-', ms=ms)
    ax.set_ylabel(r'$SNR$', color=p1.get_color())
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_xticks([-50, -25, 0, 25, 50])

    axr = ax.twinx()
    p2, = axr.plot(dfm[x], dfm[y2] * microns_per_pixel ** 2, '-', ms=ms, color=scigreen)
    axr.set_ylabel(r'$A_{p} \: (\mu m^2)$', color=p2.get_color())

    plt.tight_layout()
    # plt.savefig(path_results + '/snr_and_area_by_z.svg')
    plt.show()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 16. PLOT SPCT STATS

plot_self_similarity = False
if plot_self_similarity:
    path_results = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-07.29.22-idpt-tmg/compare/self-similarity'

    fpi = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-07.29.22-idpt-tmg/tests/tm16_cm19/tracking/calibration-idpt_micrometer_5um-3/calib_stacks_forward_self-similarity_11.06.21_z-micrometer-v2_5umMS__3.xlsx'
    dfi = pd.read_excel(fpi)

    fps = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-07.29.22-idpt-tmg/tests/spct_soft-baseline_1/tracking/calibration-spct_micrometer_5um-spct-cal/calib_stacks_forward_self-similarity_11.06.21_z-micrometer-v2_5umMS__spct-cal.xlsx'
    dfs = pd.read_excel(fps)
    spct_calib_id = 46
    dfs = dfs[dfs['id'] == spct_calib_id]

    # setup
    zf = 50.0
    x = 'z_corr'
    y = 'cm'

    # processing

    # center z on focal plane
    dfi['z_corr'] = dfi['z'] - zf
    dfs['z_corr'] = dfs['z'] - zf

    # groupby stats
    dfm = dfi.groupby('z').mean()
    dfstd = dfi.groupby('z').std()

    z_shift = 1
    dfm_shift = dfi[dfi['z'] > z_shift].groupby('z').mean()
    dfstd_shift = dfi[dfi['z'] > z_shift].groupby('z').std()

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(dfs[x], dfs[y], 'o', ms=1, color=scigreen, label='SPCT')

    ax2.errorbar(dfm_shift[x], dfm_shift[y], yerr=dfstd_shift[y], fmt='o', ms=1, color=sciblue,
                 capsize=2, ecolor='silver', elinewidth=1, errorevery=5,
                 label='IDPT')
    ax2.plot(dfm[x], dfm[y], 'o', ms=1, color=sciblue)

    ax1.set_ylabel(r'$S \left( z_{i}, z_{i+1} \right)$')
    ax1.set_ylim(bottom=0.905, top=1.0075)
    ax1.set_yticks([0.95, 1.00])
    ax1.legend(loc='lower center')

    ax2.set_ylabel(r'$\overline{S} \left( z_{i}, z_{i+1} \right) $')
    ax2.set_ylim(bottom=0.905, top=1.0075)
    ax2.set_yticks([0.95, 1.00])
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_xticks([-50, -25, 0, 25, 50])
    ax2.legend(loc='lower center')

    plt.tight_layout()
    plt.savefig(path_results + '/compare_SPCT_IDPT_forward-self-similarity.svg')
    plt.show()

    j = 1

# ---


# ----------------------------------------------------------------------------------------------------------------------
# 16. PLOT CORRELATION COEFFICIENT DEPENDENCE ON RADIAL POSITION

plot_cm_radial_dependence = False
if plot_cm_radial_dependence:
    path_results = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-07.29.22-idpt-tmg/compare/self-similarity'

    # fpi = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-07.29.22-idpt-tmg/tests/tm16_cm19/tracking/calibration-idpt_micrometer_5um-3/calib_stacks_forward_self-similarity_11.06.21_z-micrometer-v2_5umMS__3.xlsx'
    # dfi = pd.read_excel(fpi)

    fps = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-07.29.22-idpt-tmg/tests/spct_soft-baseline_1/coords/test-coords/custom-correction/test_coords_particle_image_stats_spct-1_dzf.xlsx'
    dfs = pd.read_excel(fps)
    # spct_calib_id = 46
    # dfs = dfs[dfs['id'] == spct_calib_id]

    # setup
    error_threshold = 10
    bins_r = np.array([150, 300, 450]) / microns_per_pixel
    bins_rr = 15
    min_num_bin = 20

    # BIN - 1D (many r-values)

    # IDPT
    dfibm, dfibstd = bin.bin_generic(dfi,
                                     column_to_bin,
                                     column_to_count,
                                     bins_rr,
                                     round_to_decimal,
                                     return_groupby
                                     )

    # rmse-z
    dfibm['rmse_z'] = np.sqrt(dfibm['error_squared'])

    # resolve floating point bin selecting
    dfibm = dfibm.round({'bin': 0})
    dfibstd = dfibstd.round({'bin': 0})

    dfibm = dfibm.sort_values(['bin'])
    dfibstd = dfibstd.sort_values(['bin'])

    # export
    dfibm.to_excel(path_results_radial_coords_1d + '/dfim_rr.xlsx')
    dfibstd.to_excel(path_results_radial_coords_1d + '/dfistd_rr.xlsx')

    # ---

    # SPCT
    dfibm, dfibstd = bin.bin_generic(dfs,
                                     column_to_bin,
                                     column_to_count,
                                     bins_rr,
                                     round_to_decimal,
                                     return_groupby
                                     )

    # rmse-z
    dfibm['rmse_z'] = np.sqrt(dfibm['error_squared'])

    # resolve floating point bin selecting
    dfibm = dfibm.round({'bin': 0})
    dfibstd = dfibstd.round({'bin': 0})

    dfibm = dfibm.sort_values(['bin'])
    dfibstd = dfibstd.sort_values(['bin'])

    # export
    dfibm.to_excel(path_results_radial_coords_1d + '/dfsm_rr.xlsx')
    dfibstd.to_excel(path_results_radial_coords_1d + '/dfsstd_rr.xlsx')

    j = 1

# ---


# ----------------------------------------------------------------------------------------------------------------------
# D. COMPARE SPCT AND IDPT RMSE

compare_idpt_and_spct_dft = False
if compare_idpt_and_spct_dft:

    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
               'results-07.29.22-idpt-tmg'
    path_compare = join(base_dir, 'compare/rmse/min_cm_0.5_error_corr_tilt')
    path_results = join(base_dir, 'compare/defocused-tracking-efficacy')

    fni = 'bin-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-error'
    fns = 'bin-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-error_corr_tilt'
    fng = None

    fpi = join(path_compare, 'idpt', fni + '.xlsx')
    fps = join(path_compare, 'spct', fns + '.xlsx')
    # fp_gdpt = join(path_compare, 'gdpt', fng + '.xlsx')

    dfi = pd.read_excel(fpi)
    dfs = pd.read_excel(fps)
    # dfgdpt = pd.read_excel(fp_gdpt)

    # ---

    # processing
    dfi['dft'] = dfi['percent_meas_idd'] ** 2 / dfi['true_percent_meas']
    dfs['dft'] = dfs['percent_meas_idd'] ** 2 / dfs['true_percent_meas']
    # dfgdpt['dft'] = dfgdpt['percent_meas_idd'] ** 2 / dfgdpt['true_percent_meas']

    # ---

    # plotting

    # setup
    save_figs = True
    show_figs = True

    # plot
    x = 'bin'
    y = 'dft'

    # ---

    # plot: DFT by z/h for IDPT, SPCT, and GDPT
    if save_figs or show_figs:
        fig, ax = plt.subplots(figsize=(size_x_inches * 1, size_y_inches))

        ax.plot(dfi[x], dfi[y], '-o', ms=4, label='IDPT', zorder=3.6)
        ax.plot(dfs[x], dfs[y], '-o', ms=4, label='SPCT', zorder=3.5)
        # ax.plot(dfgdpt[x], dfgdpt[y], '-o', ms=3, label='GDPT', zorder=3.4)

        ax.axhline(y=1.0, linewidth=0.5, linestyle='--', color='gray', alpha=0.75, zorder=3.3)

        ax.set_xlabel(r'$z/h$')
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.set_ylabel(r'$\epsilon_{DFT}$')
        # ax.set_ylim([0.7, 1.3])
        ax.legend()  # loc='upper left', bbox_to_anchor=(1, 1)

        plt.tight_layout()
        if save_figs:
            plt.savefig(path_results + '/compare_DFT_idpt-spct.svg')
        if show_figs:
            plt.show()
        plt.close()

    # ---

    # fill between 1 and DFT by z/h for IDPT, SPCT, and GDPT

    # setup
    save_figs = True
    show_figs = True

    ylim = [0.7, 1.3]
    yticks = [0.8, 1, 1.2]

    # plot
    if save_figs or show_figs:
        fig, (ax2, ax3) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches))

        ax3.fill_between(dfi[x], dfi[y], 1, color=sciblue, edgecolor='black', linewidth=0.25, label='IDPT')
        ax3.plot(dfi[x], dfi[y], 'o', ms=1, color='black')

        ax2.fill_between(dfs[x], dfs[y], 1, color=scigreen, edgecolor='black', linewidth=0.25, label='SPCT')
        ax2.plot(dfs[x], dfs[y], 'o', ms=1, color='black')

        # ax1.fill_between(dfgdpt[x], dfgdpt[y], 1, color=sciorange, edgecolor='black', linewidth=0.25, label='GDPT')
        # ax1.plot(dfgdpt[x], dfgdpt[y], 'o', ms=1, color='black')

        ax3.set_xlabel(r'$z/h$')
        ax3.set_xticks([-50, -25, 0, 25, 50])
        ax3.set_ylabel(r'$\epsilon_{DFT}$')
        ax3.set_ylim(ylim)
        ax3.set_yticks(yticks)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax2.set_ylabel(r'$\epsilon_{DFT}$')
        ax2.set_ylim(ylim)
        ax2.set_yticks(yticks)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # ax1.set_ylabel(r'$\epsilon_{DFT}$')
        # ax1.set_ylim(ylim)
        # ax1.set_yticks(yticks)
        # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        if save_figs:
            plt.savefig(path_results + '/compare_DFT_idpt-spct_fill-between_zoom.svg')
        if show_figs:
            plt.show()
        plt.close()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------


print("analysis completed without errors.")