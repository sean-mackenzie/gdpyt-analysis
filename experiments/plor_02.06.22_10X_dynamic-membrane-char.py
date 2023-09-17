# test bin, analyze, and plot functions

# imports
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, griddata

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import filter
import analyze
from correction import correct
from utils import fit, functions, bin, io, plotting, modify, plot_collections, boundary
from utils.plotting import lighten_color
from utils.functions import fSphericalUniformLoad, fNonDimensionalNonlinearSphericalUniformLoad

# A note on SciencePlots colors
"""
Blue: #0C5DA5
Green: #00B945
Red: #FF2C00
Orange: #FF9500

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
del ax

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 0. VERY IMPORTANT PARAMETERS!

""" THESE PARAMETERS ARE SHARED BETWEEN CALIBRATION AND TEST PROCESSING. MAKE SURE THEY ARE CORRECT! """

mag_eff = 5.0
numerical_aperture = 0.3
pixel_size = 16
depth_of_focus = 17.3  # functions.depth_of_field(mag_eff, numerical_aperture, 600e-9, 1.0, pixel_size=pixel_size * 1e-6) * 1e6
microns_per_pixel = 3.2
frame_rate = 24.444
E_silpuran = 500e3
poisson = 0.5
t_membrane = 20e-6

"""NOTE: instead of 'adding 10' for template padding. I'm going to 'subtract 10' so that fitted plane is equiv."""
padding_during_idpt_test_calib = 15  # 10
padding_during_calib = 10
padding_during_spct_calib = padding_during_calib

image_length = 512
img_xc = 256
img_yc = 256

""" ----------------- CONFIRM THE CORRECTNESS OF THE ABOVE PARAMETERS! ----------------- """

# ---

""" --- MEMBRANE SPECIFIC PARAMETERS --- """

# mask lower right membrane
xc_lr, yc_lr, r_edge_lr = 423, 502, 252
circle_coords_lr = [xc_lr, yc_lr, r_edge_lr]

# mask upper left membrane
xc_ul, yc_ul, r_edge_ul = 167, 35, 157
circle_coords_ul = [xc_ul, yc_ul, r_edge_ul]

# mask left membrane
xc_ll, yc_ll, r_edge_ll = 12, 289, 78
circle_coords_ll = [xc_ll, yc_ll, r_edge_ll]

# mask middle
xc_mm, yc_mm, r_edge_mm = 177, 261, 31
circle_coords_mm = [xc_mm, yc_mm, r_edge_mm]

# ---

# test specific parameters
start_frame = 39
tid_1_start_time = start_frame / frame_rate  # 1.6


# ---

# ----------------------------------------------------------------------------------------------------------------------
# A. DEFINE FUNCTIONS


def calculate_correction_for_test_coords_from_focus(path_figs, param_zf='zf_from_gauss_nsv', zf_limits=None):
    """
    dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
    calculate_correction_for_test_coords_from_focus(path_figs_initial_calibration, param_zf='zf_from_gauss_A')

    :param path_figs:
    :param param_zf:
    :return:
    """

    # setup file paths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
               'analyses/shared-results/initial-surface'

    # used for test results: 'analyses/results-04.16.22_10X-spct-idpt-meta-assessment/spct'

    path_calib_coords = join(base_dir, 'coords/calib-coords')
    # path_similarity = join(base_dir, 'similarity')
    # path_results = join(base_dir, 'results')
    # path_figs = join(base_dir, 'figs')

    method = 'idpt'

    # ------------------------------------------------------------------------------------------------------------------
    # 1. READ CALIB COORDS
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)

    # ------------------------------------------------------------------------------------------------------------------
    # 1.1 COMPUTE INITIAL SURFACE CORRECTION

    kx = 2
    ky = 2

    # 1.0 DISCARD OUTLIERS
    i_num_pids_c = len(dfcpid)

    if zf_limits is not None:
        dfcpid = dfcpid[(dfcpid[param_zf] > zf_limits[0]) & (dfcpid[param_zf] < zf_limits[1])]
    else:
        values = np.abs(dfcpid[param_zf] - dfcpid[param_zf].mean())
        stds = dfcpid[param_zf].std() * 2
        dfcpid = dfcpid[np.abs(dfcpid[param_zf] - dfcpid[param_zf].mean()) < dfcpid[param_zf].std() * 2]

    f_num_pids_c = len(dfcpid)

    if i_num_pids_c != f_num_pids_c:
        print("{} pids were dropped!".format(i_num_pids_c - f_num_pids_c))

    dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
        correct.fit_plane_correct_plane_fit_spline(dfcal=dfcpid,
                                                   param_zf=param_zf,
                                                   microns_per_pixel=microns_per_pixel,
                                                   img_xc=img_xc + padding_during_calib,
                                                   img_yc=img_yc + padding_during_calib,
                                                   kx=kx,
                                                   ky=ky,
                                                   path_figs=path_figs)

    return dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl


# ---


def get_pids_on_features(df, start_time, path_results, path_boundary, export_pids_per_membrane, return_full):
    """
    dflr, dful, dfll, dfmm, dfbd, lr_pids, ul_pids, ll_pids, mm_pids, boundary_pids, z_i_mean_lr, \
    z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary, z_zero, z_initial_fpb_filter =
    get_pids_on_features(df, path_results, path_boundary, export_pids_per_membrane, return_full=True)

    z_i_mean_lr, z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary, z_zero, z_initial_fpb_filter =
    get_pids_on_features(df, path_results, path_boundary, export_pids_per_membrane, return_full=False)

    :param df:
    :param path_results:
    :param path_boundary:
    :param return_full:
    :return:
    """

    # --- DEFINE LOWER, UPPER, LEFT, MIDDLE MEMBRANES
    x_bounds_lr, y_bounds_lr = 200, 240
    y_bounds_ul = 210
    x_bounds_ll, y_bounds_ll = 100, 300
    x_bounds_mm, y_bounds_mm = [140, 200], [220, 300]

    boundary_pids = io.read_txt_file_to_list(join(path_boundary, 'boundary_pids.txt'), data_type='int')
    interior_pids = io.read_txt_file_to_list(join(path_boundary, 'interior_pids.txt'), data_type='int')

    boundary_pids.sort()
    interior_pids.sort()

    # split interior pids into (1) lower right membrane and (2) upper left membrane
    df_interior_pids = df[df['id'].isin(interior_pids)]

    dflr = df_interior_pids[(df_interior_pids['x'] > x_bounds_lr) & (df_interior_pids['y'] > y_bounds_lr)]
    dful = df_interior_pids[df_interior_pids['y'] < y_bounds_ul]
    dfll = df_interior_pids[(df_interior_pids['x'] < x_bounds_ll) & (df_interior_pids['y'] > y_bounds_ll)]
    dfmm = df_interior_pids[(df_interior_pids['x'] > x_bounds_mm[0]) &
                            (df_interior_pids['x'] < x_bounds_mm[1]) &
                            (df_interior_pids['y'] > y_bounds_mm[0]) &
                            (df_interior_pids['y'] < y_bounds_mm[1])]

    lr_pids = sorted(dflr.id.unique())
    ul_pids = sorted(dful.id.unique())
    ll_pids = sorted(dfll.id.unique())
    mm_pids = sorted(dfmm.id.unique())

    # ---

    # export particle ID's for each membrane
    if export_pids_per_membrane:
        dict_memb_pids = {'lr': lr_pids,
                          'ul': ul_pids,
                          'll': ll_pids,
                          'mm': mm_pids,
                          'boundary': boundary_pids,
                          }

        export_memb_pids = pd.DataFrame.from_dict(data=dict_memb_pids, orient='index')
        export_memb_pids = export_memb_pids.rename(columns={0: 'pids'})
        export_memb_pids.to_excel(path_results + '/df_pids_per_membrane.xlsx')

    # ---

    # --- Z TO FILTER OUT FOCAL PLANE BIAS ERRORS IN INITIAL Z-HEIGHT
    z_initial_fpb_filter = 0
    # ---

    # get mean z-coordinate of interior particles prior to 'start time'
    z_i_mean_lr = dflr[(dflr['t'] < start_time) & (dflr['z_corr'] < z_initial_fpb_filter)].z_corr.mean()
    z_i_mean_ul = dful[(dful['t'] < start_time) & (dful['z_corr'] < z_initial_fpb_filter)].z_corr.mean()
    z_i_mean_ll = dfll[(dfll['t'] < start_time) & (dfll['z_corr'] < z_initial_fpb_filter)].z_corr.mean()
    z_i_mean_mm = dfmm[(dfmm['t'] < start_time) & (dfmm['z_corr'] < z_initial_fpb_filter)].z_corr.mean()

    # ---
    # CAREFUL! MANUAL CORRECTION
    if z_i_mean_lr > -6 or np.isnan(z_i_mean_lr):
        print("Manually correcting z_i_mean_lr from {} to -8.".format(np.round(z_i_mean_lr, 2)))
        z_i_mean_lr = -12.26

    if z_i_mean_ul > -6 or np.isnan(z_i_mean_ul):
        print("Manually correcting z_i_mean_lr from {} to -8.".format(np.round(z_i_mean_ul, 2)))
        z_i_mean_ul = -8.3

    if z_i_mean_ll > -6 or np.isnan(z_i_mean_ll):
        print("Manually correcting z_i_mean_ll from {} to -8.".format(np.round(z_i_mean_ll, 2)))
        z_i_mean_ll = -9.2

    if z_i_mean_mm > -6 or np.isnan(z_i_mean_mm):
        print("Manually correcting z_i_mean_mm from {} to -8.".format(np.round(z_i_mean_mm, 2)))
        z_i_mean_mm = -11

    # ---

    # get mean z-coordinate of boundary particles
    dfbd = df[df['id'].isin(boundary_pids)]
    z_mean_boundary = dfbd[(dfbd['t'] < start_time) & (dfbd['z_corr'] < z_initial_fpb_filter)].z_corr.mean()

    # MANUAL CORRECTION
    z_i_mean_lr = -5
    z_i_mean_ul = -2
    z_i_mean_ll = -4
    z_i_mean_mm = -4
    z_mean_boundary = -4
    print("MANUALLY CORRECTING ALL Z-INITIALS: (lr, ul, ll, mm, bd) = "
          "({}, {}, {}, {}, {})".format(z_i_mean_lr, z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary))
    # ----

    # get mean z-zero
    z_zero = np.mean([z_mean_boundary, z_i_mean_lr, z_i_mean_ul, z_i_mean_ll, z_i_mean_mm])

    # ---

    if return_full:
        return dflr, dful, dfll, dfmm, dfbd, lr_pids, ul_pids, ll_pids, mm_pids, boundary_pids, z_i_mean_lr, \
               z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary, z_zero, z_initial_fpb_filter
    else:
        return z_i_mean_lr, z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary, z_zero, z_initial_fpb_filter


# ---


# ----------------------------------------------------------------------------------------------------------------------
# A. SPCT-DEPENDENT FUNCTIONS

# ---

# EVALUATE INITIAL SURFACE (FOCUS POSITIONS)
evaluate_initial_surface = False

if evaluate_initial_surface:
    path_figs_initial_calibration = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/' \
                                    '02.06.22_membrane_characterization/analyses/shared-results/initial-surface'
    for param_zf in ['zf_from_peak_int', 'zf_from_gauss_A', 'zf_from_nsv', 'zf_from_nsv_signal']:
        dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
            calculate_correction_for_test_coords_from_focus(path_figs_initial_calibration, param_zf,
                                                            zf_limits=[134, 146])

# ---

# ANALYZE SPCT STATS
analyze_spct = False

if analyze_spct:

    # 1. Setup

    # setup file paths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
               'analyses/shared-results/initial-surface'

    # used for test results: 'analyses/results-04.16.22_10X-spct-idpt-meta-assessment/spct'

    path_calib_coords = join(base_dir, 'coords/calib-coords')
    path_similarity = join(base_dir, 'similarity')
    path_results = join(base_dir, 'results')
    path_figs = join(base_dir, 'figs')

    method = 'idpt'

    # ------------------------------------------------------------------------------------------------------------------
    # 1. READ CALIB COORDS
    read_calib_coords = True

    if read_calib_coords:
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)

        # ------------------------------------------------------------------------------------------------------------------
        # 1.1 COMPUTE INITIAL SURFACE CORRECTION
        compute_initial_surface_correction = True

        if compute_initial_surface_correction:
            param_zf = 'zf_from_gauss_A'
            kx = 2
            ky = 2

            # 1.0 DISCARD OUTLIERS
            i_num_pids_c = len(dfcpid)

            values = np.abs(dfcpid[param_zf] - dfcpid[param_zf].mean())
            stds = dfcpid[param_zf].std() * 2
            dfcpid = dfcpid[np.abs(dfcpid[param_zf] - dfcpid[param_zf].mean()) < dfcpid[param_zf].std() * 2]

            f_num_pids_c = len(dfcpid)

            if i_num_pids_c != f_num_pids_c:
                print("{} pids were dropped!".format(i_num_pids_c - f_num_pids_c))

            dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
                correct.fit_plane_correct_plane_fit_spline(dfcal=dfcpid,
                                                           param_zf=param_zf,
                                                           microns_per_pixel=microns_per_pixel,
                                                           img_xc=img_xc + padding_during_calib,
                                                           img_yc=img_yc + padding_during_calib,
                                                           kx=kx,
                                                           ky=ky,
                                                           path_figs=None)

            # --------------------------------------------------------------------------------------------------------------
            # 1.2 CORRECT THE SPCT_STATS DATAFRAME

            correct_spct = False

            if correct_spct:
                # step 1. correct coordinates using field curvature spline
                dfcstats_field_curvature_corrected = correct.correct_z_by_spline(dfcstats, bispl, param_z='z')

                # step 2. correct coordinates using fitted plane
                dfcstats_field_curvature_tilt_corrected = correct.correct_z_by_plane_tilt(dfcal=None,
                                                                                          dftest=dfcstats_field_curvature_corrected,
                                                                                          param_zf='none',
                                                                                          param_z='z_corr',
                                                                                          param_z_true='none',
                                                                                          popt_calib=None,
                                                                                          params_correct=None,
                                                                                          dict_fit_plane=dict_fit_plane_bspl_corrected,
                                                                                          )

                # export the corrected dfcstats
                dfcstats_field_curvature_tilt_corrected.to_excel(
                    path_results + '/calib_{}_stats_field-curvature-and-tilt-corrected_{}.xlsx'.format(method,
                                                                                                       param_zf),
                    index=False)

            # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. MASK IMAGE TO GET SURFACE AND MEMBRANE PARTICLES
    get_mask = False

    if get_mask:
        # setup

        # ------------------------------------------------------------------------------------------------------------------
        # image used for boundary identification
        path_image_to_mask = join(base_dir, 'images/AVG_calib_67.tif')

        path_results = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/results-05.03.22_10X-idpt-FINAL'
        path_mask_boundary = join(path_results, 'boundary')
        path_figs_boundary = path_mask_boundary
        fp_mask_boundary = join(path_mask_boundary, 'mask_boundary')
        fp_mask_edge = join(path_mask_boundary, 'mask_edge')

        # ---------------- THERE ARE TWO AREAS TO MASK

        # mask lower right membrane
        xc, yc, r_edge = 423, 502, 251
        circle_coords_lr = [xc, yc, r_edge]

        # mask upper left membrane
        xc, yc, r_edge = 167, 35, 157
        circle_coords_ul = [xc, yc, r_edge]

        # mask left membrane
        xc, yc, r_edge = 12, 289, 78
        circle_coords_ll = [xc, yc, r_edge]

        # mask middle
        xc, yc, r_edge = 177, 261, 31
        circle_coords_mm = [xc, yc, r_edge]

        # combine masks into list
        circle_coords = [circle_coords_lr, circle_coords_ul, circle_coords_ll, circle_coords_mm]

        # ----------------

        # approx. edge space to allow edge particles "in"
        acceptance_boundary_pixels = 6

        save_mask_boundary = True
        save_boundary_images = True
        show_boundary_images = True

        padding_during_spct_calib = 0

        mask_dict = {
            'path_mask_boundary': path_mask_boundary,
            'path_image_to_mask': path_image_to_mask,
            'padding_during_gdpyt': padding_during_spct_calib,
            'circle_coords': circle_coords,
            'acceptance_boundary_pixels': acceptance_boundary_pixels,
            'save_mask_boundary': save_mask_boundary,
            'save_boundary_images': save_boundary_images,
            'show_boundary_images': show_boundary_images,
        }

        # compute mask
        mask_dict = boundary.compute_boundary_mask(mask_dict)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. SPCT STATS

    analyze_spct_stats = False

    if analyze_spct_stats:
        # read
        plot_collections.plot_spct_stats(base_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. SPCT SIMILARITIES

    analyze_similarities = False

    if analyze_similarities:
        plot_collections.plot_similarity_analysis(base_dir, method='idpt', mean_min_dx=None)

    # ---

# ---


# ----------------------------------------------------------------------------------------------------------------------
# B. EVALUATE TESTS


# setup file paths
top_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/'
analysis_dir = 'results-09.15.22_idpt-subpix'  # 'results-09.15.22_idpt-sweep-ttemp'  # 'results-09.15.22_idpt-subpix'
base_dir = top_dir + analysis_dir

if analysis_dir.endswith('ttemp'):
    fn_test_coords = 'test_coords_stats_dz1_ttemp{}'
elif analysis_dir.endswith('subpix'):
    fn_test_coords = 'test_coords_stats_dz{}'
else:
    raise ValueError('Analysis dir not understood.')

# IMPORTANT FILTERING: - REMOVE PARTICLES ON THE "WRONG" SIDE OF THE FOCAL PLANE
apply_z_corr_directional_filters = True
apply_z_corr_directional_filters_ll_mm = False
always_edge_limit = 1.0
apply_radius_edge_filter = True
radius_edge_limit = 0.9
z_i_mean_allowance = 2.5

# define plate theory model to fit
plate_model = 'nonlinear'
lva = 'no'
boundary_compliance = 1.0
boundary_compliance_large_pos = 1.0

analyze_test_coords = True
if analyze_test_coords:

    # --- IMPORTANT: THIS SHOULD BE DONE ONLY ONCE - BUT - THEY NEED TO BE CAREFULLY INSPECTED FOR CORRECTNESS
    perform_distortion_correction = False
    compute_interior_and_boundary_pids = True
    plot_interior_and_boundary_pids = True
    plot_particle_collections = False
    export_pids_per_membrane = True
    calculate_stats_and_corrected_test_coords = True
    plot_boundary_plane_initially = False
    export_boundary_plane_dict = False
    plot_lr_particles_wrt_radial_coordinate = False
    plot_ul_particles_wrt_radial_coordinate = False
    plot_ll_particles_wrt_radial_coordinate = False
    plot_mm_particles_wrt_radial_coordinate = False

    # --- PRIMARY FOR-LOOP PLOTTING MODIFIERS
    plot_widefield = True  # --> True to run the for loop
    plot_frames_of_interest = True  # --> True to run the for loop

    # analyze other membranes
    analyze_ellipsoid_ul = True
    analyze_ellipsoid_ll = True
    analyze_ellipsoid_mm = True

    # plot per-frame radial z(r)
    plot_per_frame_membrane_radial_profile_all = False
    plot_per_frame_membrane_radial_profile_lr = False
    plot_per_frame_membrane_radial_profile_ul = False
    plot_per_frame_membrane_radial_profile_ll = False
    plot_per_frame_membrane_radial_profile_mm = False

    # plot figures on the 5 'frames of interest'
    plot_froi_fit_plate_theory = True  # z(r) + plate theory + rmse_z

    # plot 3D topography
    plot_3d_topography = False   # I think this is deprecated?
    plot_boundary_plane = False  # topographic map - spherical

    # post analysis plots
    plot_pressure_relationship = False
    analyze_per_particle = True
    analyze_uncertainty = True
    analyze_fit_plate_theory = True
    calculate_non_dimensional = False

    # ---

    # save results
    export_all_results = True

    # show/save figures
    save_figs = True
    show_figs = False

    # ------------------------------------------------------------------------------------------------------------------
    # 1. SETUP

    # NOTE: adjust the test_coords path to read corrected dataframe
    path_test_coords = join(base_dir, 'coords/test-coords')  # /fc_and_tilt_corrected
    path_boundary = join(base_dir, 'boundary')
    path_mask_boundary_npy = join(path_boundary, 'mask/mask_boundary.npy')

    method = 'idpt'

    # ---

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. READ TEST COORDS

    # setup file paths
    fp1 = 'test_coords_particle_image_stats'  # 'test_coords_particle_image_stats'  # 'test_coords_id1_dynamic_neg_first'
    fp2 = 'test_coords_stats_dz{}'.format('whatever dz step is')
    fp3 = 'test_coords_id3_dynamic_pos_first'
    fp4 = 'test_coords_id4_large_pos'
    fps = ['blah', 'blah', 'blah', 'blah']  # , fp2, fp3, fp4]
    filetype = '.xlsx'

    # test ID's
    tid_groups = [['15']] # [['2', '3', '4'], ['5', '6', '7', '8'], ['11', '15']]  # , '2', '3', '4'],]  #
    fr_width_groups = [15, 51, 10]  # 52,
    """
    Saturated pids for ttemp <= 11: [12, 13, 18, 34, 39, 49, 66, 78]
    Saturated pids for ttemp == 13: [11, 12, 17, 33, 38, 48, 65, 77]
    """
    exclude_saturated_gids = [[], [], []]  # [[12, 13, 18, 34, 39, 49, 66, 78]]

    for tids, fr_width, exclude_saturated_pids in zip(tid_groups, fr_width_groups, exclude_saturated_gids):

        # choose frames where peak deflection approximately occurs (need exactly 6 per test)
        froi1 = [51, 86, 115, 145, 170, 197]
        froi2 = [45, 50, 65, 75, 80, 85]
        froi3 = [50, 72, 101, 123, 163, 180]
        froi4 = [55, 75, 90, 100, 120, 130]
        frames_of_interests = [froi1, froi1, froi1, froi1]  # froi2, froi3, froi4]

        # frame width analyzes all of the frames in a 2 * fr_width bin around frames of interest (froi)
        # fr_width = 5  # 52
        fr_lb = 0  # greater than or equal to
        fr_ub = 199  # less than or equal to, NOTE: frame should never be 200. It will cause an error.

        # start times
        start_times = [tid_1_start_time, tid_1_start_time, tid_1_start_time, tid_1_start_time]  # 2, 1.525, 1.8]

        # ----

        # BEGIN PROCESSING

        for fp, tid, frames_of_interest, start_time in zip(fps, tids, frames_of_interests, start_times):

            # setup test-specific file paths
            path_similarity = join(base_dir, 'similarity', 'dz{}'.format(tid))
            path_results = join(base_dir, 'results', 'dz{}'.format(tid))
            path_figs = join(base_dir, 'figs', 'dz{}'.format(tid))
            path_figs_initial_calibration = join(path_figs, 'initial-correction')
            path_boundary = join(base_dir, 'boundary', 'dz{}'.format(tid))

            if not os.path.exists(path_results):
                os.makedirs(path_results)
            if not os.path.exists(path_figs):
                os.makedirs(path_figs)
            if not os.path.exists(path_boundary):
                os.makedirs(path_boundary)

            # ---

            # 0. TEST COORDS PRE-PROCESSING
            if calculate_stats_and_corrected_test_coords:

                # 0.0 read dataframe
                fp = fn_test_coords.format(tid)  # fp = 'test_coords_stats_dz{}'.format(tid)
                df = pd.read_excel(join(path_test_coords, fp + filetype))

                # filters

                # filter originally used to remove faux baseline but it no longer has a purpose
                # because test_X1.tif is used as a true baseline.
                df = df[df['frame'] >= 0]

                # remove saturated pids
                df = df[~df.id.isin(exclude_saturated_pids)]

                # ---

                # --------------------------------------------------------------------------------------------------------------

                # fit Gaussian to intensity profile
                """
    
    
                def gauss_1d(x, a, x0, sigma):
                    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    
                # 1. read calib coords to get particle's peak intensity
                fpc = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/shared-results/initial-surface/coords/calib-coords/calib_idpt_pid_defocus_stats_02.06.22_ttemp11_dzc1.xlsx'
                dfc = pd.read_excel(fpc)
    
                # 2. get idpt stats
                fpcs = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/shared-results/initial-surface/coords/calib-coords/calib_idpt_stats_02.06.22_ttemp11_dzc1.xlsx'
                dfcs = pd.read_excel(fpcs)
    
                # ---
    
                # modify test coords
                zf = 140
                zdof = 15
                zlim_c = 20
                zlim = 50
    
                df['z_corr'] = df['z'] - zf
                df['t'] = df['frame'] / frame_rate
                dfst = df[df['t'] < start_time]
    
                # exclude particle's in the DoF
                df = df[df['z_corr'].abs() > zdof]
                df = df[df['z_corr'].abs() < zlim]
    
                # -
    
                # modify calib coords
                dfcs_fit = dfcs[dfcs['z_corr'].abs() < zlim_c]
                dfcs = dfcs[dfcs['z_corr'].abs() < zlim]
    
    
                p_ints = [7, 10, 24, 41, 53, 57, 60, 47, 50, 52, 66, 68, 71, 72, 94, 80]
                p_ints_not_great = [7, 10, 60, 47, 68]
    
                for pid in p_ints:
                    if pid in p_ints_not_great:
                        continue
    
                    # data to plot later
                    # dfcid = dfcs[dfcs['id'] == pid][['z', 'z_corr', 'peak_int', 'gauss_A', 'pdf_A']]
                    xfcp = dfcs[dfcs['id'] == pid]['z_corr'].to_numpy()
                    yfcp = dfcs[dfcs['id'] == pid]['pdf_A'].to_numpy()
    
                    # fit function
                    dfcid = dfcs_fit[dfcs_fit['id'] == pid][['z', 'z_corr', 'peak_int', 'gauss_A', 'pdf_A']]
                    xfc = dfcid['z_corr'].to_numpy()
                    yfc = dfcid['pdf_A'].to_numpy()
                    popc, pcov = curve_fit(gauss_1d, xfc, yfc)  # , p0=guess, bounds=bounds)
                    xfitc = np.linspace(-zlim, zlim, 200)
                    yfitc = gauss_1d(xfitc, *popc)
    
                    # for matching test particle
                    xfitc_n = np.linspace(-zlim_c, 0, 1001)
                    yfitc_n = gauss_1d(xfitc_n, *popc)
    
                    show_cal_fit = False
                    if show_cal_fit:
                        fig, ax = plt.subplots()
                        ax.plot(xfc, yfc, 'o', ms=1)
                        ax.plot(xfitc, yfitc, '-', label='A, x0, w = {}, {}, {}'.format(np.round(popc[0], -1),
                                                                                        np.round(popc[1], 1),
                                                                                        np.round(popc[2], 1),
                                                                                        )
                                )
                        plt.show()
                        plt.close()
    
    
                    def fit_intensity(xx, dA, dx):
                        return dA * gauss_1d(xx, popc[0], popc[1] + dx, popc[2])
    
    
                    # ---
    
                    # ---
    
                    # get particle dataframe and df before start time
                    dfpid = df[df['id'] == pid][['frame', 'z_corr', 'gauss_A', 'pdf_A']]
                    dfst_pid = dfst[dfst['id'] == pid][['frame', 'z_corr', 'gauss_A', 'pdf_A']]
    
                    # get particle's peak intensity
                    pid_max_int = dfc[dfc['id'] == pid]['zf_peak_int'].values
    
                    # ---
    
                    # fit function
                    xf = dfpid['z_corr'].to_numpy()
                    yf = dfpid['pdf_A'].to_numpy()
    
                    xfit = np.linspace(-zlim, zlim, 200)
    
                    # guess = [65000, 0, 10]
                    # bounds = ([2000, -8, 4], [350000, 8, 30])
                    # popt, pcov = curve_fit(gauss_1d, xf, yf, p0=guess, bounds=bounds)
                    # yfit = gauss_1d(xfit, *popt)
                    #
                    # only negative intensity and more dense sampling
                    # xfit_n = np.linspace(-20, 0, 1001)
                    # yfit_n = gauss_1d(xfit_n, *popt)
    
                    # fit calib intensity profile
                    bounds = ([0.8, -8], [3, 8])
                    popt, pcovv = curve_fit(fit_intensity, xf, yf, bounds=bounds)  # (xx, dx, dA)
                    yfit = fit_intensity(xfit, *popt)
                    # xfit_n = np.linspace(-20, 0, 1001)
                    # yfit_n = fit_intensity(xfit_n, *popt)
    
                    # calculate z that matches particle's intensity
                    pid_i0 = dfst_pid['pdf_A'].mean()
                    pid_i0_sub = yfitc_n - pid_i0
                    pid_i0_sub_idx = np.argmin(np.abs(pid_i0_sub))
                    pid_z0 = xfitc_n[pid_i0_sub_idx]
    
                    # fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(size_x_inches, size_y_inches * 1.5), gridspec_kw={'height_ratios': [5, 1]}, )
                    fig, ax0 = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches * 1.5))
    
                    # plot test
                    ax0.plot(dfpid['z_corr'], dfpid['pdf_A'], 'D', ms=3, color='magenta', label='pdf_A')
                    # ax0.plot(xfit, yfit, '-',
                    #         label='(dA={}, \n dx={}, \n x0={})'.format(np.round(popt[0], 2), np.round(popt[1], 2), np.round(popc[1], 2)))
    
                    # plot intensity prior to start time
                    ax0.plot(dfst_pid['z_corr'], dfst_pid['pdf_A'], 'o', ms=2, color='red',
                             label='zi={}, \n zf={}'.format(np.round(dfst_pid['z_corr'].mean(), 1), np.round(pid_z0, 2)))
    
                    # plot estimated z position
                    ax0.axhline(dfst_pid['pdf_A'].mean(), linestyle='--', linewidth=0.5, color='red', alpha=0.5)
                    ax0.axvline(pid_z0, linestyle='--', linewidth=0.5, color='black', alpha=0.75)
    
                    # plot calibration
                    ax0.plot(xfcp, yfcp, 'o', ms=2, color='purple')
                    ax0.plot(xfitc, yfitc, '--', color='darkgreen', linewidth=0.5,
                             label='A={},\n x0={},\n w={}'.format(np.round(popc[0], -1),
                                                                  np.round(popc[1], 1),
                                                                  np.round(popc[2], 1),
                                                                  )
                             )
    
                    ax0.set_xlabel('z_corr')
                    ax0.set_ylabel(r'$I_{o}$')
                    ax0.legend(title='pID: {}'.format(pid))
    
                    # ax1.plot(dfpid['frame'], dfpid['z_corr'], 'o', ms=2, label='gauss_A')
                    # ax1.plot(dfpid['frame'], dfpid['z_corr'], 'o', ms=2, label='pdf_A')
                    # ax1.set_ylabel(r'$z_{corr}$')
                    # x1.set_xlabel('Frame')
    
                    # ax2.plot(dfpid['frame'], dfpid['gauss_A'], 'o', ms=2, label='gauss_A')
                    # ax2.plot(dfpid['frame'], dfpid['pdf_A'], 'o', ms=2, label='pdf_A')
                    # ax2.set_xlabel('Frame')
                    # ax2.set_ylabel(r'$I_{o}$')
    
                    plt.tight_layout()
                    plt.show()
    
                """

                # --------------------------------------------------------------------------------------------------------------

                # ---

                # --------------------------------------------------------------------------------------------------------------
                # CORRECT IMAGE DISTORTION AND STAGE TILT

                # NOTE: can correct for only field curvature, only stage tilt, both, none, or,
                # perform a quasi manual correction.

                if perform_distortion_correction:

                    # 1. DETERMINE THE NECESSARY CORRECTION

                    if not os.path.exists(path_figs_initial_calibration):
                        os.makedirs(path_figs_initial_calibration)

                    dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
                        calculate_correction_for_test_coords_from_focus(path_figs_initial_calibration,
                                                                        param_zf='zf_from_nsv',
                                                                        zf_limits=[134, 146],
                                                                        )

                    # ---

                    # 2. PERFORM CORRECTION

                    how_to_correct = ['field curvature', 'tilt']

                    if 'field curvature' in how_to_correct:
                        print("Performing field curvature correction!")

                        # step 1. correct coordinates using field curvature spline
                        df = correct.correct_z_by_spline(df, bispl, param_z='z')

                    if 'tilt' in how_to_correct:
                        print("Performing tilt correction!")

                        # set correct z variable handle
                        if 'field curvature' in how_to_correct:
                            param_z = 'z_corr'
                        else:
                            param_z = 'z'

                        # step 2. correct coordinates using fitted plane
                        df = correct.correct_z_by_plane_tilt(dfcal=None,
                                                             dftest=df,
                                                             param_zf='none',
                                                             param_z='z_corr',
                                                             param_z_true='none',
                                                             popt_calib=None,
                                                             params_correct=None,
                                                             dict_fit_plane=dict_fit_plane_bspl_corrected,
                                                             )

                    elif 'manual' in how_to_correct:
                        # 3. READ FIELD-CURVATURE CORRECTED FITTED PLANE FOR TILT CORRECTION

                        # setup file paths
                        fp_tilt_corr = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/' \
                                       '02.06.22_membrane_characterization/analyses/' \
                                       'results-04.16.22_10X-spct-idpt-meta-assessment/spct/figs/calibration-surface/' \
                                       'calib-coords_tilt_from-field-curvature-corrected.xlsx'

                        # read tilt-correction dataframe
                        df_tilt_corr = pd.read_excel(fp_tilt_corr, index_col=0)
                        """NOTE: manually entering 'popt_calib' instead of parsing string."""
                        popt_calib = [-0.002801094622188694,
                                      -0.01192551812107627,
                                      143.66113373198064,
                                      -143.66113373198064,
                                      np.array([0.00280109, 0.01192552, 1.])
                                      ]

                        # ----------------------------------------------------------------------------------------------------------
                        # 4. PERFORM TILT-CORRECTION
                        df = correct.correct_z_by_plane_tilt(dfcal=None,
                                                             dftest=df,
                                                             param_zf='none',
                                                             param_z='z',
                                                             param_z_true='none',
                                                             popt_calib=popt_calib,
                                                             params_correct=None,
                                                             dict_fit_plane=None,
                                                             )

                    # ---

                    # export the corrected coords

                    path_test_coords_corrected = join(path_test_coords, 'fc_and_tilt_corrected')
                    if not os.path.exists(path_test_coords_corrected):
                        os.makedirs(path_test_coords_corrected)

                    df.to_excel(join(path_test_coords_corrected, fp + filetype), index=False)

                    # ---

                else:
                    if 'z_corr' not in df.columns:
                        df['z_corr'] = df['z'] - 140
                        print("WARNING - DATASET SPECIFIC CORRECTION TO Z-COOR!!!!")

                # ---

                # --------------------------------------------------------------------------------------------------------------
                # SIMPLE ADJUSTMENT

                # 0.1 remove template_padding from coordinates
                if padding_during_idpt_test_calib != 0:
                    print("Adjusting X, Y coordinates by {} to account for padding!".format(padding_during_idpt_test_calib))
                    df['x'] = df['x'] - padding_during_idpt_test_calib
                    df['y'] = df['y'] - padding_during_idpt_test_calib

                    if 'xg' in df.columns:
                        df['xg'] = df['xg'] - padding_during_idpt_test_calib
                        df['yg'] = df['yg'] - padding_during_idpt_test_calib

                    if 'gauss_xc' in df.columns:
                        df['gauss_xc'] = df['gauss_xc'] - padding_during_idpt_test_calib
                        df['gauss_yc'] = df['gauss_yc'] - padding_during_idpt_test_calib

                    if 'pdf_xc' in df.columns:
                        df['pdf_xc'] = df['pdf_xc'] - padding_during_idpt_test_calib
                        df['pdf_yc'] = df['pdf_yc'] - padding_during_idpt_test_calib

                # 0.2 drop unnecessary columns for readability
                for drop_col in ['stack_id', 'z_true', 'max_sim', 'error']:
                    if drop_col in df.columns:
                        df = df.drop(columns=[drop_col])

                # 0.3 add a column for 'time' + start time
                df['t'] = df['frame'] / frame_rate
                df['r'] = functions.calculate_radius_at_xy(df.x, df.y, xc=img_xc, yc=img_yc)

                # --------------------------------------------------------------------------------------------------------------
                # STRUCTURE PARTICLES INTO INTERIOR (1) AND (2) AND BOUNDARY

                # ---

                # get boundary and interior particles

                # ONE-TIME ONLY - compute image mask, boundary, and interior particle ID's
                if compute_interior_and_boundary_pids:
                    mask_boundary = np.load(path_mask_boundary_npy)

                    flip_xy_coords = True
                    boundary_pids, interior_pids = boundary.get_boundary_particles(mask_boundary,
                                                                                   df,
                                                                                   return_interior_particles=True,
                                                                                   flip_xy_coords=flip_xy_coords,
                                                                                   flip_xy_coords_minus_mask_size=False,
                                                                                   )

                    # package and export to avoid computing later
                    io.write_list_to_txt_file(list_values=boundary_pids, filename='boundary_pids.txt',
                                              directory=path_boundary)
                    io.write_list_to_txt_file(list_values=interior_pids, filename='interior_pids.txt',
                                              directory=path_boundary)

                    # ---

                    """ NOTE: IT'S VERY IMPORTANT TO CONFIRM THE BOUNDARY PARTICLES ARE STATIONARY! """

                    # plot to confirm

                    dfbpid = df[df['id'].isin(boundary_pids)]
                    i_z_mean_boundary = dfbpid.z_corr.mean()

                    # ---

                    if plot_interior_and_boundary_pids:

                        # scatter x-y
                        fig, ax = plt.subplots()
                        ax.imshow(mask_boundary)
                        ax.scatter(dfbpid.x, dfbpid.y, s=2, color='blue', label='boundary')
                        dfipid = df[df['id'].isin(interior_pids)]
                        ax.scatter(dfipid.x, dfipid.y, s=2, color='red', label='interior')
                        ax.set_title('Flip xy coords: {}'.format(flip_xy_coords))
                        plt.savefig(path_boundary + '/mask_interior_and_boundary_pids.png')
                        # plt.show()
                        plt.close()

                        # ---

                        # plot x-y scatter w/ particle ID labels + x-z scatter + z_mean

                        fig, (ax, axz) = plt.subplots(nrows=2, sharex=True,
                                                      figsize=(size_x_inches * 1.125, size_y_inches * 1.5),
                                                      gridspec_kw={'height_ratios': [3, 1]})

                        for df_collection, label, zcm in zip([dfbpid], ['boundary'], [i_z_mean_boundary]):
                            df_collection = df_collection[df_collection['t'] < start_time]  # df_collection['frame'] == 1]
                            ax.scatter(df_collection.x, df_collection.y, c=df_collection.id, s=2, label=label)
                            axz.scatter(df_collection.x, df_collection.z_corr, c=df_collection.id, s=2,
                                        label=np.round(zcm, 1))
                            for x, y, bpid in zip(df_collection.x.to_numpy(), df_collection.y.to_numpy(),
                                                  df_collection.id.to_numpy()):
                                ax.text(x, y, str(bpid), color='black', fontsize=8)

                        ax.set_ylabel('y')
                        ax.set_ylim([0, image_length])
                        ax.set_yticks([0, 256, 512])
                        ax.invert_yaxis()
                        ax.set_aspect('equal', adjustable='box')
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                        axz.set_xlabel('x')
                        axz.set_xlim([0, image_length])
                        axz.set_xticks([0, 256, 512])
                        axz.set_ylabel(r'$z_{corr} \: (\mu m)$')
                        axz.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\overline{z_{corr}}(t<t_{start})$')

                        plt.tight_layout()
                        plt.savefig(path_boundary + '/raw_boundary_pids_before-start-time.png')
                        # plt.show()
                        plt.close()

                        # ---

                        # plot trajectories: z(y)
                        num_plots_per_fig = 6
                        num_figs = int(np.ceil(len(boundary_pids) / num_plots_per_fig))

                        for i in range(num_figs):
                            pids_this_plot = boundary_pids[i * num_plots_per_fig:(i + 1) * num_plots_per_fig]

                            fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches))
                            for ptp in pids_this_plot:
                                dfbd_ptp = dfbpid[dfbpid['id'] == ptp]
                                ax.plot(dfbd_ptp.t, dfbd_ptp.z_corr, '-o', ms=2, label=ptp)

                            ax.set_xlabel(r'$t \: (s)$')
                            ax.set_ylabel(r'$z_{corr} \: (\mu m)$')
                            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            plt.tight_layout()
                            plt.savefig(path_boundary + '/boundary-pids-group-{}.png'.format(i))
                            # plt.show()
                            plt.close()

                    # raise ValueError()

                # ---

                # get dfs, pids, and z_i_means for each feature
                dflr, dful, dfll, dfmm, dfbd, lr_pids, ul_pids, ll_pids, mm_pids, boundary_pids, z_i_mean_lr, \
                z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary, z_zero, z_initial_fpb_filter = \
                    get_pids_on_features(df, start_time, path_results, path_boundary, export_pids_per_membrane,
                                         return_full=True)

                # ---

                # plot to confirm
                if plot_particle_collections:

                    fig, (ax, axz) = plt.subplots(nrows=2, sharex=True,
                                                  figsize=(size_x_inches * 1.35, size_y_inches * 2),
                                                  gridspec_kw={'height_ratios': [3, 1]}
                                                  )

                    for df_collection, label, zcm in zip(
                            [dflr, dful, dfll, dfmm, dfbd],
                            ['lower right', 'upper left', 'left', 'middle', 'boundary'],
                            [z_i_mean_lr, z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary],
                    ):

                        df_collection = df_collection[df_collection['t'] < start_time]  # df_collection['frame'] == 1]
                        ax.scatter(df_collection.x, df_collection.y, s=2, label=label)
                        for x, y, bpid in zip(df_collection.x.to_numpy(), df_collection.y.to_numpy(),
                                              df_collection.id.to_numpy()):
                            ax.text(x, y, str(bpid), color='black', fontsize=6)
                        axz.scatter(df_collection.x, df_collection.z_corr, s=2, label=np.round(zcm, 1))

                    axz.axhline(y=z_initial_fpb_filter, linestyle='--', linewidth=0.75, color='red',
                                label=r'$\epsilon_{fpb}$')

                    ax.set_ylabel('y')
                    ax.set_ylim([0, image_length])
                    ax.set_yticks([0, 256, 512])
                    ax.invert_yaxis()
                    ax.set_aspect('equal', adjustable='box')
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                    axz.set_xlabel('x')
                    axz.set_xlim([0, image_length])
                    axz.set_xticks([0, 256, 512])
                    axz.set_ylabel(r'$z_{corr} \: (\mu m)$')
                    axz.set_title(r'$\overline{z_{corr}} < $' + ' {} '.format(z_initial_fpb_filter) + r'$\mu m$')
                    axz.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\overline{z_{corr}}(t<t_{start})$')

                    plt.tight_layout()
                    plt.savefig(path_boundary + '/collections_interior-lr-ul_and_boundary_pids.png')
                    # plt.show()
                    plt.close()

                    # ---

                    # FIT PLANE TO BOUNDARY PARTICLES ONLY

                    # --- filter out focal plane bias errors before fitting
                    dfbd_fit_plane = dfbd[(dfbd['t'] < start_time) & (dfbd['z_corr'] < z_initial_fpb_filter)]

                    # fit plane to boundary particles only
                    dict_fit_plane_bd = correct.fit_in_focus_plane(df=dfbd_fit_plane,
                                                                   param_zf='z_corr',
                                                                   microns_per_pixel=microns_per_pixel,
                                                                   img_xc=img_xc,
                                                                   img_yc=img_yc,
                                                                   )
                    if plot_boundary_plane_initially:
                        fig = plotting.plot_fitted_plane_and_points(dfbd_fit_plane, dict_fit_plane_bd)
                        plt.savefig(path_boundary + '/plane-fitted-to-fpb-filtered-boundary-pids.png')
                        # plt.show()
                        plt.close()
                    if export_boundary_plane_dict:
                        export_dict_fit_plane_bd = pd.DataFrame.from_dict(data=dict_fit_plane_bd, orient='index')
                        export_dict_fit_plane_bd = export_dict_fit_plane_bd.rename(columns={0: 'quantity'})
                        export_dict_fit_plane_bd.to_excel(path_boundary + '/dict_fit-plane-to-boundary-pids.xlsx')

                # ---

                # --------------------------------------------------------------------------------------------------------------
                # 6. CALCULATE GENERAL TRACKING STATISTICS

                if calculate_stats_and_corrected_test_coords:

                    # general
                    z_calib_min = 2.0
                    z_calib_max = 272.0
                    z_calib_clip = 0.0  # 2.5
                    cm_threshold = 0.5

                    # number of measurements
                    num_frames = len(df.frame.unique())
                    i_num_pids = len(df.id.unique())
                    i_num_rows = len(df)

                    # FILTERS

                    # 1. filter out max/min z (edge of calib stack)
                    df = df[df['z'] > z_calib_min + z_calib_clip]
                    df = df[df['z'] < z_calib_max - z_calib_clip]
                    ii_num_rows_calib_extents = i_num_rows - len(df)

                    # 2. cm filter
                    df = df[df['cm'] > cm_threshold]
                    iii_num_rows_cm_filter = ii_num_rows_calib_extents - len(df)

                    # ---

                    # ---

                    # 2. CALCULATE IN-PLANE DISPLACEMENT

                    # ---

                    # read particle ID's per membrane
                    df_memb_pids = pd.read_excel(path_results + '/df_pids_per_membrane.xlsx', index_col=0)
                    pids_lr = df_memb_pids.loc['lr', :].dropna().values
                    pids_ul = df_memb_pids.loc['ul', :].dropna().values
                    pids_ll = df_memb_pids.loc['ll', :].dropna().values
                    pids_mm = df_memb_pids.loc['mm', :].dropna().values

                    # ---

                    # STEP 0.0 - PLOT EACH PARTICLE 'xg', 'yg', 'gauss_xc' 'gauss_yc', 'z'
                    plot_one_at_a_time = False
                    if plot_one_at_a_time:
                        path_oaats = path_results + '/pids_trajectories'
                        if not os.path.exists(path_oaats):
                            os.makedirs(path_oaats)

                        oaat_cols = ['xg', 'yg', 'gauss_xc', 'gauss_yc', 'pdf_xc', 'pdf_yc', 'z']
                        dfpid_froaat = df[df['frame'] > 0]
                        for pids_one_at_a_time in df.id.unique():
                            dfpid_oaat = dfpid_froaat[dfpid_froaat['id'] == pids_one_at_a_time]
                            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches,
                                                                                               size_y_inches * 1.25))
                            ax1.plot(dfpid_oaat.frame, dfpid_oaat[oaat_cols[0]], label=oaat_cols[0])
                            ax1.plot(dfpid_oaat.frame, dfpid_oaat[oaat_cols[2]], label=oaat_cols[2])
                            ax1.plot(dfpid_oaat.frame, dfpid_oaat[oaat_cols[4]], label=oaat_cols[4])

                            ax2.plot(dfpid_oaat.frame, dfpid_oaat[oaat_cols[1]], label=oaat_cols[1])
                            ax2.plot(dfpid_oaat.frame, dfpid_oaat[oaat_cols[3]], label=oaat_cols[3])
                            ax2.plot(dfpid_oaat.frame, dfpid_oaat[oaat_cols[5]], label=oaat_cols[5])

                            ax3.plot(dfpid_oaat.frame, dfpid_oaat[oaat_cols[6]], label=oaat_cols[6])

                            ax1.set_ylabel('x')
                            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            ax2.set_ylabel('y')
                            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            ax3.set_ylabel('z')
                            ax3.set_xlabel('Frame')
                            plt.tight_layout()
                            plt.savefig(path_oaats + '/pid{}_xg-yg-xy-gauss-z.png'.format(pids_one_at_a_time),
                                        dpi=200)
                            plt.close()

                    # ---

                    # STEP 1. COMPUTE IN-PLANE DISPLACEMENT
                    compute_relative_displacement_method = 'old'

                    if compute_relative_displacement_method == 'new':

                        """
                        THESE FUNCTIONS ARE 'READY' BUT NOT SETUP YET.
                        
                        xy_cols = [['x', 'y'], ['xg', 'yg'], ['gauss_xc', 'gauss_yc'], ['pdf_xc', 'pdf_yc']]
                        r_cols = ['r', 'rg', 'gauss_r', 'pdf_r']
                        start_time_col = 't'
                        min_num_measurements = 3
                        memb_id_col = 'memb_id'
    
                        # mask lower right membrane
                        xc_lr, yc_lr, r_edge_lr = 423, 502, 252
                        circle_coords_lr = [xc_lr, yc_lr, r_edge_lr]
    
                        # mask upper left membrane
                        xc_ul, yc_ul, r_edge_ul = 167, 35, 157
                        circle_coords_ul = [xc_ul, yc_ul, r_edge_ul]
    
                        # mask left membrane
                        xc_ll, yc_ll, r_edge_ll = 12, 289, 78
                        circle_coords_ll = [xc_ll, yc_ll, r_edge_ll]
    
                        # mask middle
                        xc_mm, yc_mm, r_edge_mm = 177, 261, 31
                        circle_coords_mm = [xc_mm, yc_mm, r_edge_mm]
    
                        dfd_membs = []
    
                        for memb_id, df_memb, disc_coords in zip([1, 2],
                                                                 [dflr, dful],
                                                                 [circle_coords_lr, circle_coords_ul],
                                                                 ):
                            
                            dfd_memb = analyze.disc_calculate_local_displacement(df_memb,
                                                                                 xy_cols,
                                                                                 r_cols,
                                                                                 disc_coords,
                                                                                 start_time,
                                                                                 start_time_col,
                                                                                 min_num_measurements,
                                                                                 memb_id,
                                                                                 memb_id_col,
                                                                                 )
    
                            dfd_membs.append(dfd_memb)
                        """

                    else:

                        # STEP 1. COMPUTE AVERAGE INITIAL POSITION PRIOR TO START TIME
                        # dfgi = df[df['t'] < start_time * 0.9].groupby('id').mean()
                        dfim, dfistd = bin.bin_generic(df[df['t'] < start_time],
                                                       column_to_bin='id',
                                                       column_to_count='frame',
                                                       bins=df.id.unique(),
                                                       round_to_decimal=1,
                                                       return_groupby=True,
                                                       )

                        if 'xg' in dfim.columns:
                            xy_method = 'subpix'
                            xy_cols = ['id', 'x', 'y', 'xg', 'yg', 'gauss_xc', 'gauss_yc', 'pdf_xc', 'pdf_yc']
                            x0, y0 = 'xg', 'yg'

                        elif 'gauss_xc' in dfim.columns:
                            xy_method = 'gauss'
                            xy_cols = ['id', 'x', 'y', 'gauss_xc', 'gauss_yc']
                            x0, y0 = 'gauss_xc', 'gauss_yc'

                        else:
                            xy_method = 'ccorr'
                            xy_cols = ['id', 'x', 'y', 'xm', 'ym']
                            x0, y0 = 'x', 'y'

                        dfim = dfim[xy_cols]

                        # ---

                        # STEP B. CALCULATE PER-PARTICLE DISPLACEMENTS FROM AVERAGE INITIAL
                        df_temps = []
                        for pid in df.id.unique():

                            # get initial coords df per particle
                            dfpid_im = dfim[dfim['id'] == pid].reset_index()

                            # get test coords df per particle
                            dfpid = df[df['id'] == pid]
                            dfpid = dfpid.sort_values('t')

                            if len(dfpid_im) < 1:
                                print("pid {} in only {} frames".format(pid, len(dfpid)))
                                dfpid_im = dfpid[xy_cols].copy().reset_index()

                            # calculate per-frame template displacement from initial
                            if xy_method == 'subpix':
                                dfpid['dxm'] = dfpid_im.iloc[0]['xg'] - dfpid['xg']  # positive is right
                                dfpid['dym'] = dfpid['yg'] - dfpid_im.iloc[0]['yg']  # positive is up
                                dfpid['drm'] = np.sqrt(dfpid['dxm'] ** 2 + dfpid['dym'] ** 2)

                                # add per-frame position from per-frame template displacement
                                dfpid['xxm'] = dfpid['x'] + dfpid['dxm']
                                dfpid['yym'] = dfpid['y'] - dfpid['dym']
                                # raise ValueError("I'm uncertain about the above")

                            elif xy_method == 'gauss':
                                dfpid['dxm'] = dfpid['gauss_xc'] - dfpid_im.iloc[0]['gauss_xc']  # positive is right
                                dfpid['dym'] = dfpid_im.iloc[0]['gauss_yc'] - dfpid['gauss_yc']  # positive is up
                                dfpid['drm'] = np.sqrt(dfpid['dxm'] ** 2 + dfpid['dym'] ** 2)

                                # add per-frame position from per-frame template displacement
                                dfpid['xxm'] = dfpid['gauss_xc']
                                dfpid['yym'] = dfpid['gauss_yc']

                            else:
                                """ Note, +xm actually indicates particle moved -xm (b/c how template matching is performed) """
                                dfpid['dxm'] = dfpid_im.iloc[0]['xm'] - dfpid['xm']  # positive is right
                                dfpid['dym'] = dfpid['ym'] - dfpid_im.iloc[0]['ym']  # positive is up
                                dfpid['drm'] = np.sqrt(dfpid['dxm'] ** 2 + dfpid['dym'] ** 2)

                                # add per-frame position from per-frame template displacement
                                dfpid['xxm'] = dfpid['x'] + dfpid['dxm']  # x = 0 is left image edge
                                dfpid['yym'] = dfpid['y'] - dfpid[
                                    'dym']  # y = 0 is top image edge (maintain original coord. axes)

                            # get membrane center corresponding to pid
                            if pid in pids_lr:
                                memb_xc, memb_yc = xc_lr, yc_lr
                                memb_r = r_edge_lr
                                memb_id = 1

                            elif pid in pids_ul:
                                memb_xc, memb_yc = xc_ul, yc_ul
                                memb_r = r_edge_ul
                                memb_id = 2

                            elif pid in pids_ll:
                                memb_xc, memb_yc = xc_ll, yc_ll
                                memb_r = r_edge_ll
                                memb_id = 3

                            elif pid in pids_mm:
                                memb_xc, memb_yc = xc_mm, yc_mm
                                memb_r = r_edge_mm
                                memb_id = 4

                            else:
                                memb_xc, memb_yc = img_xc, img_yc
                                memb_r = img_xc
                                memb_id = 0

                            # add column for membrane center coordinates and radius
                            dfpid['memb_id'] = memb_id
                            dfpid['mmb_xc'] = memb_xc
                            dfpid['mmb_yc'] = memb_yc
                            dfpid['mmb_r'] = memb_r

                            # calculate initial radial distance
                            dfpid['rr0'] = functions.calculate_radius_at_xy(dfpid_im.iloc[0][x0],
                                                                            dfpid_im.iloc[0][y0],
                                                                            xc=memb_xc, yc=memb_yc)

                            # calculate per-frame signed radial displacement and radial position
                            dfpid['rx'] = dfpid.xxm - memb_xc
                            dfpid['ry'] = memb_yc - dfpid.yym  # flipped b/c y = 0 is top of image
                            dfpid['rr'] = functions.calculate_radius_at_xy(dfpid.xxm, dfpid.yym, xc=memb_xc, yc=memb_yc)

                            # calculate signed radial displacement
                            dfpid['drr'] = dfpid['rr'] - dfpid['rr0']

                            # ---

                            # store
                            df_temps.append(dfpid)

                        # ---

                        # overwrite the old test_coords dataframe
                        dfn = pd.concat(df_temps)
                        del df, df_temps
                        df = dfn.copy()
                        del dfn

                        # 2.2 filter out
                        i_rows_before_drm_filter = len(df)
                        drm_threshold = 4.5
                        df = df[df['drm'] < drm_threshold]
                        iv_num_rows_template_displacement = i_rows_before_drm_filter - len(df)
                        if iv_num_rows_template_displacement > 0:
                            print(
                                "{} rows exceeded {} drm filter.".format(iv_num_rows_template_displacement, drm_threshold))
                            # raise ValueError("Need to investigate why {} rows exceed this unclear filter.".format(iii_num_rows_template_displacement))

                        # ---

                    # CALCULATE PROCESSING STATISTICS - RELATIVE NUMBER OF PARTICLES MEASURED

                    # post filter stats
                    calib_measurement_depth = z_calib_max - z_calib_min
                    corr_measurement_depth = df.z_corr.max() - df.z_corr.min()
                    f_num_pids = len(df.id.unique())
                    f_num_rows = len(df)
                    f_percent_rows = f_num_rows / i_num_rows * 100
                    f_cm_mean = df.cm.mean()
                    f_cm_std = df.cm.std()

                    # particle density
                    p_density_pixels_per_particle = image_length ** 2 / f_num_pids
                    p_density_microns_per_particle = (image_length * microns_per_pixel) ** 2 / f_num_pids
                    f_mean_particle_spacing_microns = np.sqrt(p_density_microns_per_particle)

                    # ---

                    # BOUNDARY PARTICLES

                    # boundary particles
                    dfbd = df[df['id'].isin(boundary_pids)]
                    dfbd_st = dfbd[dfbd['t'] < start_time]
                    xparam = 'id'
                    particle_ids = dfbd_st[xparam].unique()
                    count_column = 'counts'
                    pos = ['z_corr', 'drm', 'xxm', 'yym']
                    dfp_id, dfpm = analyze.evaluate_1d_static_precision(dfbd_st,
                                                                        column_to_bin=xparam,
                                                                        precision_columns=pos,
                                                                        bins=particle_ids,
                                                                        round_to_decimal=0)
                    z_mean_boundary_method2 = dfp_id.z_corr_m.mean()
                    dbd_mean_rm = dfp_id.drm_m.mean()
                    dbd_mean_xm = dfp_id.xxm_m.mean()
                    dbd_mean_ym = dfp_id.yym_m.mean()

                    dbd_precision_z_corr = dfp_id.z_corr.mean()
                    dbd_precision_rm = dfp_id.drm.mean()
                    dbd_precision_xm = dfp_id.xxm.mean()
                    dbd_precision_ym = dfp_id.yym.mean()

                    # LOWER RIGHT MEMBRANE PARTICLES

                    pids_lr = dflr.id.unique()
                    f_num_pids_lr = len(pids_lr)

                    # lower right membrane particles
                    dflr = df[df['id'].isin(pids_lr)]
                    dflr_st = dflr[dflr['t'] < start_time]
                    xparam = 'id'
                    pos = ['z_corr', 'drm', 'xxm', 'yym']
                    particle_ids = dflr_st[xparam].unique()
                    count_column = 'counts'

                    dfp_id, dfpm = analyze.evaluate_1d_static_precision(dflr_st,
                                                                        column_to_bin=xparam,
                                                                        precision_columns=pos,
                                                                        bins=particle_ids,
                                                                        round_to_decimal=0)
                    zp_i_mean_lr = dfp_id.z_corr_m.mean()
                    dflr_st_mean_rm = dfp_id.drm_m.mean()
                    dflr_st_mean_xm = dfp_id.xxm_m.mean()
                    dflr_st_mean_ym = dfp_id.yym_m.mean()

                    dflr_st_precision_z_corr = dfp_id.z_corr.mean()
                    dflr_st_precision_rm = dfp_id.drm.mean()
                    dflr_st_precision_xm = dfp_id.xxm.mean()
                    dflr_st_precision_ym = dfp_id.yym.mean()

                    # UPPER LEFT MEMBRANE PARTICLES

                    pids_ul = dful.id.unique()
                    f_num_pids_ul = len(pids_ul)

                    # lower right membrane particles
                    dful = df[df['id'].isin(pids_ul)]
                    dful_st = dful[dful['t'] < start_time]
                    xparam = 'id'
                    pos = ['z_corr', 'drm', 'xxm', 'yym']
                    particle_ids = dful_st[xparam].unique()
                    count_column = 'counts'

                    dfp_id, dfpm = analyze.evaluate_1d_static_precision(dful_st,
                                                                        column_to_bin=xparam,
                                                                        precision_columns=pos,
                                                                        bins=particle_ids,
                                                                        round_to_decimal=0)
                    zp_i_mean_ul = dfp_id.z_corr_m.mean()
                    dful_st_mean_rm = dfp_id.drm_m.mean()
                    dful_st_mean_xm = dfp_id.xxm_m.mean()
                    dful_st_mean_ym = dfp_id.yym_m.mean()

                    dful_st_precision_z_corr = dfp_id.z_corr.mean()
                    dful_st_precision_rm = dfp_id.drm.mean()
                    dful_st_precision_xm = dfp_id.xxm.mean()
                    dful_st_precision_ym = dfp_id.yym.mean()

                    # PACKAGE AND EXPORT

                    # export the results
                    dict_results = {'method': method,
                                    'test_id': fp,
                                    'key_stats': '-----------------------------',
                                    'z_corr_measurement_depth': corr_measurement_depth,
                                    'num_frames': num_frames,
                                    'i_num_pids': i_num_pids,
                                    'f_num_pids': f_num_pids,
                                    'i_num_rows': i_num_rows,
                                    'f_num_rows': f_num_rows,
                                    'f_percent_rows': f_percent_rows,
                                    'cm_threshold': cm_threshold,
                                    'f_cm_mean': f_cm_mean,
                                    'f_cm_std': f_cm_std,
                                    'p_density_pixels_per_particle': p_density_pixels_per_particle,
                                    'p_density_microns_per_particle': p_density_microns_per_particle,
                                    'f_mean_particle_spacing_microns': f_mean_particle_spacing_microns,
                                    'num_rows_filtered_calib_extents': ii_num_rows_calib_extents,
                                    'num_rows_filtered_cm_threshold': iii_num_rows_cm_filter,
                                    'drm_threshold_pixels': drm_threshold,
                                    'drm_threshold_microns': drm_threshold * microns_per_pixel,
                                    'num_rows_filtered_temp_disp': iv_num_rows_template_displacement,
                                    'percent_rows_filtered_temp_disp': iv_num_rows_template_displacement / f_num_rows * 100,
                                    'bd_num_particles': len(dfbd.id.unique()),
                                    'bd_st_z_corr_mean': z_mean_boundary,
                                    'bd_st_precision_z_corr': dbd_precision_z_corr,
                                    'bd_st_precision_rm': dbd_precision_rm,
                                    'dflr_num_particles': f_num_pids_lr,
                                    'dflr_st_z_corr_mean': z_i_mean_lr,
                                    'dflr_st_z_corr_mean_from_precision': zp_i_mean_lr,
                                    'dflr_st_precision_z_corr': dflr_st_precision_z_corr,
                                    'dflr_st_precision_rm': dflr_st_precision_rm,
                                    'dful_num_particles': f_num_pids_ul,
                                    'dful_st_z_corr_mean': z_i_mean_ul,
                                    'dful_st_z_corr_mean_from_precision': zp_i_mean_ul,
                                    'dful_st_precision_z_corr': dful_st_precision_z_corr,
                                    'dful_st_precision_rm': dful_st_precision_rm,
                                    'other_stats': '-----------------------------',
                                    'mag_eff': mag_eff,
                                    'numerical_aperture': numerical_aperture,
                                    'depth_of_focus': depth_of_focus,
                                    'pixel_size': pixel_size,
                                    'microns_per_pixel': microns_per_pixel,
                                    'frame_rate': frame_rate,
                                    'image_length': image_length,
                                    'youngs_modulus_silpuran': E_silpuran,
                                    't_membrane': t_membrane * 1e6,
                                    'padding_during_testing': padding_during_idpt_test_calib,
                                    'padding_during_calib': padding_during_calib,
                                    'z_calib_min': z_calib_min,
                                    'z_calib_max': z_calib_max,
                                    'calib_measurement_depth': calib_measurement_depth,
                                    'z_calib_clip': z_calib_clip,
                                    'bd_mean_rm': dbd_mean_rm,
                                    'bd_mean_xm': dbd_mean_xm,
                                    'bd_mean_ym': dbd_mean_ym,
                                    'bd_precision_xm': dbd_precision_xm,
                                    'bd_precision_ym': dbd_precision_ym,
                                    'dflr_st_mean_rm': dflr_st_mean_rm,
                                    'dflr_st_mean_xm': dflr_st_mean_xm,
                                    'dflr_st_mean_ym': dflr_st_mean_ym,
                                    'dflr_st_precision_xm': dflr_st_precision_xm,
                                    'dflr_st_precision_ym': dflr_st_precision_ym,
                                    'dful_st_mean_rm': dful_st_mean_rm,
                                    'dful_st_mean_xm': dful_st_mean_xm,
                                    'dful_st_mean_ym': dful_st_mean_ym,
                                    'dful_st_precision_xm': dful_st_precision_xm,
                                    'dful_st_precision_ym': dful_st_precision_ym,
                                    }

                    dfict_results = pd.DataFrame.from_dict(dict_results, orient='index', columns=['value'])
                    dfict_results.to_excel(path_results + '/id{}_results_overview.xlsx'.format(tid))

                    # export the new dataframe
                    df.to_excel(path_results + '/test_coords_id{}_corrected.xlsx'.format(tid))

                else:
                    pass

                # ---

            # --------------------------------------------------------------------------------------------------------------
            # 5. PLOT ALL PARTICLES

            # ---

            if plot_widefield:

                # ---

                # use filtered coords with dx, dy, dr columns
                df = pd.read_excel(path_results + '/test_coords_id{}_corrected.xlsx'.format(tid))

                # get mean axial position of each feature
                z_i_mean_lr, z_i_mean_ul, z_i_mean_ll, z_i_mean_mm, z_mean_boundary, z_zero, z_initial_fpb_filter = \
                    get_pids_on_features(df, start_time, path_results, path_boundary, export_pids_per_membrane,
                                         return_full=False)

                # create df per feature
                dflr = df[df['memb_id'] == 1]
                dful = df[df['memb_id'] == 2]
                dfll = df[df['memb_id'] == 3]
                dfmm = df[df['memb_id'] == 4]
                dfbd = df[df['memb_id'] == 0]

                # ---

                if plot_frames_of_interest:

                    # make file paths
                    if plot_3d_topography:
                        path_figs_3d_topography = join(path_figs, '3d-topography')
                        if not os.path.exists(path_figs_3d_topography):
                            os.makedirs(path_figs_3d_topography)

                    if plot_froi_fit_plate_theory:
                        path_figs_fit_profile = join(path_figs, 'fit-profile')
                        if not os.path.exists(path_figs_fit_profile):
                            os.makedirs(path_figs_fit_profile)

                    if plot_per_frame_membrane_radial_profile_lr:
                        path_figs_plot_per_frame_lr = join(path_figs, 'fit-profile-lr')
                        if not os.path.exists(path_figs_plot_per_frame_lr):
                            os.makedirs(path_figs_plot_per_frame_lr)

                    if plot_per_frame_membrane_radial_profile_ul:
                        path_figs_plot_per_frame_ul = join(path_figs, 'fit-profile-ul')
                        if not os.path.exists(path_figs_plot_per_frame_ul):
                            os.makedirs(path_figs_plot_per_frame_ul)

                    if plot_per_frame_membrane_radial_profile_all:
                        path_figs_plot_per_frame_all = join(path_figs, 'fit-profile-all')
                        if not os.path.exists(path_figs_plot_per_frame_all):
                            os.makedirs(path_figs_plot_per_frame_all)

                    # ---

                    # choose frames where "interest" approximately occurs
                    fr1 = np.arange(frames_of_interest[0] - fr_width, frames_of_interest[0] + fr_width)
                    fr2 = np.arange(frames_of_interest[1] - fr_width, frames_of_interest[1] + fr_width)
                    fr3 = np.arange(frames_of_interest[2] - fr_width, frames_of_interest[2] + fr_width)
                    fr4 = np.arange(frames_of_interest[3] - fr_width, frames_of_interest[3] + fr_width)
                    fr5 = np.arange(frames_of_interest[4] - fr_width, frames_of_interest[4] + fr_width)
                    fr6 = np.arange(frames_of_interest[5] - fr_width, frames_of_interest[5] + fr_width)
                    frames_of_interest_for_analysis = np.unique(np.hstack([fr1, fr2, fr3, fr4, fr5, fr6]))
                    frames_of_interest_for_analysis = frames_of_interest_for_analysis[
                        frames_of_interest_for_analysis >= fr_lb]
                    frames_of_interest_for_analysis = frames_of_interest_for_analysis[
                        frames_of_interest_for_analysis <= fr_ub]

                    # reconstruction dataframe
                    df_reconstruction = []
                    df_pids_and_surface = []

                    # results
                    rz_lr_fitted_sphere = []
                    fit_lr_pressure = []
                    fit_lr_pretension = []
                    fit_lr_rmse = []
                    fit_lr_percent_meas = []
                    fit_lr_r_squared = []
                    theta_lr_deg = []
                    rz_ul_fitted_sphere = []
                    fit_ul_pressure = []
                    fit_ul_pretension = []
                    fit_ul_rmse = []
                    fit_ul_percent_meas = []
                    fit_ul_r_squared = []
                    theta_ul_deg = []
                    rz_ll_fitted_sphere = []
                    fit_ll_pressure = []
                    fit_ll_rmse = []
                    fit_ll_percent_meas = []
                    fit_ll_r_squared = []
                    theta_ll_deg = []
                    rz_mm_fitted_sphere = []
                    fit_mm_pressure = []
                    fit_mm_rmse = []
                    fit_mm_percent_meas = []
                    fit_mm_r_squared = []
                    theta_mm_deg = []
                    i_num_pids_lr = []
                    f_num_pids_lr = []
                    i_num_pids_ul = []
                    f_num_pids_ul = []
                    i_num_pids_ll = []
                    f_num_pids_ll = []
                    i_num_pids_mm = []
                    f_num_pids_mm = []

                    # pids not filtered
                    fr_pids = []

                    # coords + error
                    stack_fit_results_lr = []
                    stack_data_fit_to_lr = []
                    stack_data_r_lr = []
                    stack_data_fr_lr = []
                    stack_fit_results_ul = []
                    stack_data_fit_to_ul = []
                    stack_data_r_ul = []
                    stack_data_fr_ul = []
                    stack_fit_results_ll = []
                    stack_data_fit_to_ll = []
                    stack_data_r_ll = []
                    stack_data_fr_ll = []
                    stack_fit_results_mm = []
                    stack_data_fit_to_mm = []
                    stack_data_r_mm = []
                    stack_data_fr_mm = []

                    for fr_idx, fr in enumerate(frames_of_interest_for_analysis):

                        # ---

                        # -

                        # --- LOWER RIGHT MEMBRANE ---------------------------------------------------------------------

                        # -

                        # get this frame only
                        dfrlr = dflr[dflr['frame'] == fr]
                        dfrlr = dfrlr[dfrlr['rr'] < r_edge_lr * always_edge_limit]

                        # store original
                        dfrlr_o = dfrlr[['rr', 'z_corr']]

                        # initialize
                        i_num_lr = len(dfrlr)

                        # plot radial position
                        if plot_lr_particles_wrt_radial_coordinate and fr_idx == 0:
                            fig, (ax, axz) = plt.subplots(nrows=2, sharex=True,
                                                          figsize=(size_x_inches * 1.125, size_y_inches * 1.5),
                                                          gridspec_kw={'height_ratios': [3, 1]}
                                                          )
                            sc = ax.scatter(dfrlr.x, dfrlr.y, c=dfrlr.rr)
                            ax.add_patch(plt.Circle((xc_lr, yc_lr), r_edge_lr, color='black', fill=False))
                            ax.add_patch(plt.Circle((xc_lr, yc_lr), 5, color='red', fill=True))
                            plt.colorbar(sc, ax=ax)
                            axz.scatter(dfrlr.x, dfrlr.z_corr, c=dfrlr.rr)

                            ax.set_ylabel('y')
                            ax.set_ylim([0, image_length])
                            ax.invert_yaxis()
                            ax.set_aspect('equal', adjustable='box')

                            axz.set_xlabel('x')
                            axz.set_xlim([0, image_length])
                            axz.set_ylabel(r'$z_{corr} \: (\mu m)$')
                            plt.tight_layout()
                            plt.savefig(path_figs + '/lr_particles_wrt_radial-coordinate.png')
                            # plt.show()

                        # ---

                        # OPTIONAL: APPLY FILTERS

                        # FILTER #1: how to determine +/-z
                        if apply_z_corr_directional_filters:

                            # "indicator" particles are those that deflect most (i.e. the center of the membrane)
                            indicator_p_z = dfrlr[dfrlr['rr'] < 410 / microns_per_pixel].z_corr.mean()
                            indicator_p_z_std = dfrlr[dfrlr['rr'] < 410 / microns_per_pixel].z_corr.std()

                            # all particles
                            # all_p_z = dfrlr.z_corr.mean()
                            all_p_z = dfrlr[dfrlr['rr'] < r_edge_lr * radius_edge_limit].z_corr.mean()
                            # all_p_z_std = dfrlr.z_corr.std()
                            all_p_z_std = dfrlr[dfrlr['rr'] < r_edge_lr * radius_edge_limit].z_corr.std()

                            # weight the indicator particles and take the mean
                            indicator_weighted_mean_p_z = np.mean([indicator_p_z, all_p_z])

                            if indicator_p_z < -depth_of_focus:
                                # if below DoF, take everything below z_i_mean
                                # dfrlr = dfrlr[dfrlr['z_corr'] < z_i_mean_lr + z_i_mean_allowance]
                                dfrlr['z_corr'] = dfrlr['z_corr'].where(dfrlr['z_corr'] < z_i_mean_lr + z_i_mean_allowance,
                                                                        np.nan)
                                lr_dir = -1

                            elif indicator_p_z > depth_of_focus:
                                # if above DoF, take everything above z_i_mean
                                # dfrlr = dfrlr[dfrlr['z_corr'] > z_i_mean_lr - z_i_mean_allowance]
                                dfrlr['z_corr'] = dfrlr['z_corr'].where(dfrlr['z_corr'] > z_i_mean_lr - z_i_mean_allowance,
                                                                        np.nan)

                                # lva = 'lva'
                                boundary_compliance = boundary_compliance_large_pos
                                lr_dir = 1

                            else:

                                if len(rz_lr_fitted_sphere) > 0:
                                    if rz_lr_fitted_sphere[-1] > depth_of_focus * 1.5:
                                        # dfrlr = dfrlr[dfrlr['z_corr'] > z_i_mean_lr - z_i_mean_allowance]
                                        dfrlr['z_corr'] = dfrlr['z_corr'].where(
                                            dfrlr['z_corr'] > z_i_mean_lr - z_i_mean_allowance,
                                            np.nan)
                                        lr_dir = 1

                                    elif rz_lr_fitted_sphere[-1] < -depth_of_focus * 1.5:  # 2
                                        # dfrlr = dfrlr[dfrlr['z_corr'] < z_i_mean_lr + z_i_mean_allowance]
                                        dfrlr['z_corr'] = dfrlr['z_corr'].where(
                                            dfrlr['z_corr'] < z_i_mean_lr + z_i_mean_allowance,
                                            np.nan)
                                        lr_dir = -1

                                    elif all_p_z_std < depth_of_focus / 1.5:
                                        plate_model = 'linear'
                                        #dfrlr = dfrlr[(dfrlr['z_corr'] < indicator_weighted_mean_p_z + depth_of_focus / 1.5) &
                                        #              (dfrlr['z_corr'] > indicator_weighted_mean_p_z - depth_of_focus / 1.5)]

                                        dfrlr['z_corr'] = dfrlr['z_corr'].where(
                                            np.abs(dfrlr['z_corr'] - indicator_weighted_mean_p_z) < depth_of_focus / 1.5,
                                            np.nan)

                                        lr_dir = 0
                                    else:
                                        plate_model = 'linear'
                                        lr_dir = 0
                                else:
                                    plate_model = 'linear'
                                    lr_dir = 0

                        else:

                            # set the proper angle: phi
                            v_spherical_angle = np.linspace(0, np.pi / 2, 100)

                            # format the plot viewing angle and limits
                            view_height = 15
                            zlim_bot, zlim_top = -150, 150
                            zticks = [-100, 0, 100]

                            # format the zorder
                            zorder_membrane = 2.1
                            zorder_plane = 1.2

                            # format the colormap
                            cmap = cm.coolwarm
                            vmin = -150
                            vmax = 150

                        # store data used in reconstruction
                        df_reconstruction.append(dfrlr)

                        # can drop NaNs after appending to reconstruction
                        dfrlr = dfrlr.dropna(subset=['z_corr'])

                        # FILTER #2: exclude edge particles in deflection fitting
                        dfrlr = dfrlr.sort_values('rr', ascending=False)
                        if apply_radius_edge_filter:
                            dfrlr_fit = dfrlr[dfrlr['rr'] < r_edge_lr * radius_edge_limit]
                        else:
                            dfrlr_fit = dfrlr

                        # ---

                        # store data used in reconstruction
                        # df_reconstruction.append(dfrlr_fit)  --> (change 9/22/22)

                        # calculate the relative number of particles evaluated
                        f_pids_lr = dfrlr.id.unique()
                        f_num_lr = len(dfrlr)

                        if f_num_lr * i_num_lr == 0:
                            fig, ax = plt.subplots()
                            ax.scatter(dfrlr_fit.rr.to_numpy() * microns_per_pixel,
                                       dfrlr_fit.z_corr.to_numpy() - z_i_mean_lr, color='blue', label='Fit')
                            ax.scatter(dfrlr.rr.to_numpy() * microns_per_pixel,
                                       dfrlr.z_corr.to_numpy() - z_i_mean_lr, color='red', label='raw')
                            ax.legend()
                            ax.set_xlabel('r')
                            ax.set_ylabel('z')
                            ax.set_title('L.R. frame = {}, z_i_mean = {}'.format(fr, np.round(z_i_mean_lr, 2)))
                            plt.show()

                        lr_percent_meas = f_num_lr / i_num_lr

                        # ---

                        # z-offset term (the z=0 plane for the membrane) (options: z_i_mean_lr, z_mean_boundary)
                        z_lr_offset = z_i_mean_lr

                        # ---

                        # MODEL PLATE AS: (A) NONLINEAR or (B) LINEAR WITH S.S. BOUNDARIES

                        if plate_model == 'nonlinear':

                            # fit plate nonlinear plate theory
                            a = r_edge_lr * microns_per_pixel * 1e-6 * boundary_compliance
                            fND_lr = fNonDimensionalNonlinearSphericalUniformLoad(r=a,
                                                                                  h=t_membrane,
                                                                                  youngs_modulus=E_silpuran,
                                                                                  poisson=poisson)
                            # assign fit function
                            fitfunc_lr = fND_lr

                            # data to fit on
                            r = dfrlr_fit.rr.to_numpy() * microns_per_pixel * 1e-6
                            z = (dfrlr_fit.z_corr.to_numpy() - z_lr_offset) * 1e-6
                            nd_r = r / a
                            nd_z = z / t_membrane

                            # data to evaluate rmse
                            r_ev = dfrlr.rr.to_numpy() * microns_per_pixel * 1e-6
                            z_ev = (dfrlr.z_corr.to_numpy() - z_lr_offset) * 1e-6
                            nd_r_ev = r_ev / a
                            nd_z_ev = z_ev / t_membrane

                            # guess and bounds
                            if np.max(nd_z) > 2:
                                nd_k_lower_bound = 0.01  # 1
                                nd_k_guess = 5
                                nd_p_guess = 100
                            else:
                                nd_k_lower_bound = 0.01
                                nd_k_guess = 0.5
                                nd_p_guess = -1

                            fit_d_r, fit_d_z, d_p0, d_n0, rmse, r_squared = fND_lr.fit_nd_nonlinear(lva,
                                                                                                    nd_r,
                                                                                                    nd_z,
                                                                                                    nd_r_eval=nd_r_ev,
                                                                                                    nd_z_eval=nd_z_ev,
                                                                                                    guess=(
                                                                                                        nd_p_guess,
                                                                                                        nd_k_guess),
                                                                                                    bounds=(
                                                                                                        [-1e9,
                                                                                                         nd_k_lower_bound],
                                                                                                        [1e9, 1e9])
                                                                                                    )

                            # scale
                            fit_d_r = fit_d_r * 1e6
                            fit_d_z = fit_d_z * 1e6

                            # store z-radius of fitted sphere
                            rz_lr = fit_d_z[1]
                            rz_lr_fitted_sphere.append(rz_lr)
                            fit_lr_pressure.append(d_p0)
                            fit_lr_pretension.append(d_n0)
                            fit_lr_rmse.append(rmse * 1e6)
                            fit_lr_percent_meas.append(lr_percent_meas)
                            fit_lr_r_squared.append(r_squared)

                            # calculate theta at 1/4 radius
                            idx_qr = len(fit_d_r) // 4
                            theta_lr_deg.append(np.rad2deg(np.arctan2(fit_d_z[-idx_qr * 2] - fit_d_z[-idx_qr],
                                                                      fit_d_r[-idx_qr] - fit_d_r[-idx_qr * 2],
                                                                      )))

                            # ---

                            # match some variables names for simplicity

                            # fit space: r, z
                            r_fit_lr = fit_d_r
                            z_fit_lr = fit_d_z + z_lr_offset

                            # ---
                            rmse_lr = rmse

                            # get fit on data for later error analysis
                            # non-dimensionalize
                            nd_P, nd_k = fND_lr.non_dimensionalize_p_k(d_p0, d_n0)
                            # evaluate
                            fit_nd_z = fND_lr.nd_nonlinear_clamped_plate_p_k(nd_r_ev, nd_P, nd_k)
                            # dimensionalize
                            fit_results_lr = fit_nd_z * t_membrane
                            data_fit_to_lr = nd_z_ev * t_membrane

                            # ---

                        else:
                            d_p0 = None
                            d_n0 = None

                            # fit linear plate theory
                            fsphere_lr = fSphericalUniformLoad(r=r_edge_lr * microns_per_pixel * 1e-6,
                                                               h=t_membrane,
                                                               youngs_modulus=E_silpuran,
                                                               poisson=poisson)

                            # assign fit function
                            fitfunc_lr = fsphere_lr

                            # fit
                            r_fit_lr = np.linspace(0, r_edge_lr * microns_per_pixel)
                            popt_lr, pcov_lr = curve_fit(fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p,
                                                         dfrlr_fit.rr.to_numpy() * microns_per_pixel * 1e-6,
                                                         (dfrlr_fit.z_corr.to_numpy() - z_lr_offset) * 1e-6,
                                                         )
                            popt_cc_lr, pcov_cc_lr = curve_fit(fsphere_lr.spherical_uniformly_loaded_clamped_plate_r_p,
                                                               dfrlr_fit.rr.to_numpy() * microns_per_pixel * 1e-6,
                                                               (dfrlr_fit.z_corr.to_numpy() - z_lr_offset) * 1e-6,
                                                               )

                            # fit error
                            fit_results_lr = fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(
                                dfrlr.rr.to_numpy() * microns_per_pixel * 1e-6, *popt_lr)
                            data_fit_to_lr = (dfrlr.z_corr.to_numpy() - z_lr_offset) * 1e-6

                            rmse_lr, r_squared_lr = fit.calculate_fit_error(
                                fit_results=fit_results_lr,
                                data_fit_to=data_fit_to_lr,
                            )

                            # fit profile
                            z_fit_lr = fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit_lr * 1e-6,
                                                                                                        *popt_lr) * 1e6 + z_lr_offset
                            z_fit_cc_lr = fsphere_lr.spherical_uniformly_loaded_clamped_plate_r_p(r_fit_lr * 1e-6,
                                                                                                  *popt_cc_lr) * 1e6 + z_lr_offset

                            # ---

                            # store results
                            rz_lr = fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(0, *popt_lr) * 1e6
                            rz_lr_fitted_sphere.append(rz_lr)
                            fit_lr_pressure.append(popt_lr[0])
                            fit_lr_pretension.append(0.0)
                            fit_lr_rmse.append(rmse_lr * 1e6)
                            fit_lr_percent_meas.append(lr_percent_meas)
                            fit_lr_r_squared.append(r_squared_lr)

                            # calculate theta at 1/4 radius
                            idx_qr = len(r_fit_lr) // 4
                            theta_lr_deg.append(np.rad2deg(np.arctan2(z_fit_lr[-idx_qr * 2] - z_fit_lr[-idx_qr],
                                                                      r_fit_lr[-idx_qr] - r_fit_lr[-idx_qr * 2],
                                                                      )))

                            # ---

                        # ---

                        # plot
                        if plot_per_frame_membrane_radial_profile_lr:
                            fig, ax = plt.subplots()
                            ax.plot(dfrlr_o.rr * microns_per_pixel, dfrlr_o.z_corr,
                                    '^', ms=2, color='black', alpha=0.75, label='raw')
                            ax.plot(dfrlr.rr * microns_per_pixel, dfrlr.z_corr, 'o', ms=2, color='blue', label='data')
                            ax.plot(r_fit_lr, z_fit_lr, '-', label='fit')
                            ax.axhline(z_lr_offset, linestyle='--', alpha=0.5, label=r'$z_{B.C.}$')
                            ax.set_xlabel(r'$r \: (\mu m)$')
                            ax.set_xlim([-5, (r_edge_lr + 5) * microns_per_pixel])
                            ax.set_ylabel(r'$z \: (\mu m)$')
                            ax.set_ylim([-155, 155])
                            ax.legend()
                            ax.set_title('Frame = {}'.format(fr))
                            plt.tight_layout()
                            plt.savefig(path_figs_plot_per_frame_lr + '/memb-lr_fit-spherical_frame={}.png'.format(fr),
                                        dpi=100)
                            # plt.show()
                            plt.close()

                        # ---

                        # number of measurements
                        i_num_pids_lr.append(i_num_lr)
                        f_num_pids_lr.append(f_num_lr)

                        # ---

                        # --- UPPER LEFT MEMBRANE ----------------------------------------------------------------------

                        if analyze_ellipsoid_ul:

                            # get this frame only
                            dfrul = dful[dful['frame'] == fr]
                            dfrul = dfrul[dfrul['rr'] < r_edge_ul * always_edge_limit]

                            # store raw
                            dfrul_o = dfrul[['rr', 'z_corr']]

                            # get raw number of particles
                            i_num_ul = len(dfrul)

                            # assign model to fit
                            plate_model = 'nonlinear'
                            lva = 'no'

                            # plot radial position
                            if plot_ul_particles_wrt_radial_coordinate and fr_idx == 0:
                                figg, (axg, axzg) = plt.subplots(nrows=2, sharex=True,
                                                                 figsize=(size_x_inches * 1.125, size_y_inches * 1.5),
                                                                 gridspec_kw={'height_ratios': [3, 1]}
                                                                 )
                                sc = axg.scatter(dfrul.x, dfrul.y, c=dfrul.rr)
                                axg.add_patch(plt.Circle((xc_ul, yc_ul), r_edge_ul, color='black', fill=False))
                                axg.add_patch(plt.Circle((xc_ul, yc_ul), 5, color='red', fill=True))
                                plt.colorbar(sc, ax=axg)
                                axzg.scatter(dfrul.x, dfrul.z_corr, c=dfrul.rr)

                                axg.set_ylabel('y')
                                axg.set_ylim([0, image_length])
                                axg.invert_yaxis()
                                axg.set_aspect('equal', adjustable='box')

                                axzg.set_xlabel('x')
                                axzg.set_xlim([0, image_length])
                                axzg.set_ylabel(r'$z_{corr} \: (\mu m)$')
                                figg.tight_layout()
                                figg.savefig(path_figs + '/ul_particles_wrt_radial-coordinate.png')
                                figg.show()
                                plt.close(figg)

                            # --- APPLY FILTERS

                            # FILTER #2: how to determine +/-z
                            if apply_z_corr_directional_filters:
                                indicator_p_z = dfrul[dfrul['rr'] < 100 / microns_per_pixel].z_corr.mean()
                                indicator_p_z_std = dfrul[dfrul['rr'] < 100 / microns_per_pixel].z_corr.std()

                                # most particles
                                # all_p_z = dfrul.z_corr.mean()
                                all_p_z = dfrul[dfrul['rr'] < r_edge_ul * radius_edge_limit].z_corr.mean()

                                indicator_weighted_mean_p_z = np.mean([indicator_p_z, all_p_z])

                                if indicator_p_z < -depth_of_focus:
                                    # if below DoF, take everything below z_i_mean
                                    # dfrul = dfrul[dfrul['z_corr'] < z_i_mean_ul + z_i_mean_allowance]
                                    dfrul['z_corr'] = dfrul['z_corr'].where(
                                        dfrul['z_corr'] < z_i_mean_ul + z_i_mean_allowance,
                                        np.nan)

                                    # an odd filter: maybe need to reconsider
                                    # dfrul = dfrul[(dfrul['z_corr'] < -15) | (dfrul['rr'] > r_edge_ul * 0.95)]

                                elif indicator_p_z > depth_of_focus:
                                    # if above DoF, take everything above z_i_mean
                                    # dfrul = dfrul[dfrul['z_corr'] > z_i_mean_ul - z_i_mean_allowance]
                                    dfrul['z_corr'] = dfrul['z_corr'].where(
                                        dfrul['z_corr'] > z_i_mean_ul - z_i_mean_allowance,
                                        np.nan)

                                    boundary_compliance = boundary_compliance_large_pos
                                    # lva = 'lva'

                                    """
                                    NOTE: this code was originally here
                                        > but, to run the dzc = 11, I moved it to the top of this 'if' statement series. 
                                        > honestly, it probably should be at the top because it's the most 'true to deflection'
                                    """
                                elif lr_dir != 0:
                                    if lr_dir == 1:
                                        # dfrul = dfrul[dfrul['z_corr'] > z_i_mean_ul - z_i_mean_allowance]
                                        dfrul['z_corr'] = dfrul['z_corr'].where(
                                            dfrul['z_corr'] > z_i_mean_ul - z_i_mean_allowance,
                                            np.nan)
                                    else:
                                        # dfrul = dfrul[dfrul['z_corr'] < z_i_mean_ul + z_i_mean_allowance]
                                        dfrul['z_corr'] = dfrul['z_corr'].where(
                                            dfrul['z_corr'] < z_i_mean_ul + z_i_mean_allowance,
                                            np.nan)

                                else:
                                    # Use linear plate model which is less prone to errors when variance is high
                                    plate_model = 'linear'
                                    # dfrul = dfrul[(dfrul['z_corr'] < indicator_weighted_mean_p_z + depth_of_focus / 2) &
                                    #              (dfrul['z_corr'] > indicator_weighted_mean_p_z - depth_of_focus / 2)]

                                    dfrul['z_corr'] = dfrul['z_corr'].where(
                                        np.abs(dfrul['z_corr'] - indicator_weighted_mean_p_z) < depth_of_focus / 1.5,
                                        np.nan)

                                # lots of focal plane bias errors at these beyond-edge particles
                                # dfrul = dfrul[(dfrul['rr'] < r_edge_ul * 0.95) | (np.abs(dfrul['z_corr'] - z_i_mean_ul) < depth_of_focus / 2)]

                            else:
                                pass

                            # store data used in reconstruction
                            df_reconstruction.append(dfrul)

                            # can drop NaNs after appending to reconstruction
                            dfrul = dfrul.dropna(subset=['z_corr'])

                            # FILTER #2: exclude outer radial particles from deflection fitting
                            dfrul = dfrul.sort_values('rr', ascending=False)
                            if apply_radius_edge_filter:
                                dfrul_fit = dfrul[dfrul['rr'] < r_edge_ul * radius_edge_limit]
                            else:
                                dfrul_fit = dfrul

                            # ---

                            # calculate the relative number of particles evaluated
                            f_pids_ul = dfrul.id.unique()
                            f_num_ul = len(dfrul)
                            ul_percent_meas = f_num_ul / i_num_ul

                            # ---

                            # --- MODEL USING: (A) NONLINEAR or (B) LINEAR PLATE THEORY

                            # z-offset term (the z=0 plane for the membrane) (options: z_i_mean_lr, z_mean_boundary)
                            z_ul_offset = z_i_mean_ul

                            # ---

                            # NON LINEAR PLATE MODEL
                            if plate_model == 'nonlinear':

                                # fit nonlinear plate theory
                                a = r_edge_ul * microns_per_pixel * 1e-6 * boundary_compliance
                                fND_ul = fNonDimensionalNonlinearSphericalUniformLoad(r=a,
                                                                                      h=t_membrane,
                                                                                      youngs_modulus=E_silpuran,
                                                                                      poisson=poisson)
                                # assign fit function
                                fitfunc_ul = fND_ul

                                # data to fit on
                                r = dfrul_fit.rr.to_numpy() * microns_per_pixel * 1e-6
                                z = (dfrul_fit.z_corr.to_numpy() - z_ul_offset) * 1e-6
                                nd_r = r / a
                                nd_z = z / t_membrane

                                # data to evaluate rmse
                                r_ev = dfrul.rr.to_numpy() * microns_per_pixel * 1e-6
                                z_ev = (dfrul.z_corr.to_numpy() - z_ul_offset) * 1e-6
                                nd_r_ev = r_ev / a
                                nd_z_ev = z_ev / t_membrane

                                # guess and bounds
                                if d_p0 is not None:
                                    nd_P_lr_ul, nd_k_del = fND_ul.non_dimensionalize_p_k(d_p0, d_n0)
                                else:
                                    nd_P_lr_ul, nd_k_del = fND_ul.non_dimensionalize_p_k(fit_lr_pressure[-1],
                                                                                         fit_lr_pretension[-1])

                                if np.max(nd_z) > 2:
                                    nd_k_lower_bound = 0.00005  # 1
                                    nd_k_guess = 1  # 5
                                    nd_P_lower_bound = -1e9  # nd_P_lr_ul - np.abs(nd_P_lr_ul) * 0.2  # -1e9
                                    nd_P_upper_bound = 1e9  # nd_P_lr_ul + np.abs(nd_P_lr_ul) * 0.2  # 1e9
                                    nd_p_guess = nd_P_lr_ul  # 100
                                else:
                                    nd_k_lower_bound = 0.00005
                                    nd_k_guess = 1  # 0.25
                                    nd_P_lower_bound = -1e9  # nd_P_lr_ul - np.abs(nd_P_lr_ul) * 0.4  # -1e9
                                    nd_P_upper_bound = 1e9  # nd_P_lr_ul + np.abs(nd_P_lr_ul) * 0.4  # 1e9
                                    nd_p_guess = nd_P_lr_ul  # nd_P  # -1
                                guess = (nd_p_guess, nd_k_guess)
                                bounds = ([nd_P_lower_bound, nd_k_lower_bound], [nd_P_upper_bound, 1e9])

                                fit_d_r, fit_d_z, d_p0, d_n0, rmse, r_squared = fND_ul.fit_nd_nonlinear(lva,
                                                                                                        nd_r,
                                                                                                        nd_z,
                                                                                                        nd_r_eval=nd_r_ev,
                                                                                                        nd_z_eval=nd_z_ev,
                                                                                                        guess=guess,
                                                                                                        bounds=bounds
                                                                                                        )

                                # scale
                                fit_d_r = fit_d_r * 1e6
                                fit_d_z = fit_d_z * 1e6

                                # store z-radius of fitted sphere
                                rz_ul = fit_d_z[1]
                                rz_ul_fitted_sphere.append(rz_ul)
                                fit_ul_pressure.append(d_p0)
                                fit_ul_pretension.append(d_n0)
                                fit_ul_rmse.append(rmse * 1e6)
                                fit_ul_percent_meas.append(ul_percent_meas)
                                fit_ul_r_squared.append(r_squared)

                                # calculate theta at 1/4 radius
                                idx_qr = len(fit_d_r) // 4
                                theta_ul_deg.append(np.rad2deg(np.arctan2(fit_d_z[-idx_qr * 2] - fit_d_z[-idx_qr],
                                                                          fit_d_r[-idx_qr] - fit_d_r[-idx_qr * 2],
                                                                          )))

                                # ---

                                # match some variables names for simplicity

                                # fit space: r, z
                                r_fit_ul = fit_d_r
                                z_fit_ul = fit_d_z + z_ul_offset

                                # get fit on data for later error analysis
                                rmse_ul = rmse
                                nd_P, nd_k = fND_ul.non_dimensionalize_p_k(d_p0, d_n0)
                                fit_nd_z = fND_ul.nd_nonlinear_clamped_plate_p_k(nd_r_ev, nd_P, nd_k)
                                fit_results_ul = fit_nd_z * t_membrane
                                data_fit_to_ul = nd_z_ev * t_membrane

                                # ---

                            else:

                                # fit plate theory
                                fsphere_ul = fSphericalUniformLoad(r=r_edge_ul * microns_per_pixel * 1e-6,
                                                                   h=t_membrane,
                                                                   youngs_modulus=E_silpuran,
                                                                   poisson=poisson)
                                # assign fit function
                                fitfunc_ul = fsphere_ul

                                # fit
                                r_fit_ul = np.linspace(0, r_edge_ul * microns_per_pixel)
                                dfrul = dfrul.sort_values('rr', ascending=False)
                                popt_ul, pcov_ul = curve_fit(
                                    fsphere_ul.spherical_uniformly_loaded_simply_supported_plate_r_p,
                                    dfrul_fit.rr.to_numpy() * microns_per_pixel * 1e-6,
                                    (dfrul_fit.z_corr.to_numpy() - z_ul_offset) * 1e-6,
                                )
                                popt_cc_ul, pcov_cc_ul = curve_fit(fsphere_ul.spherical_uniformly_loaded_clamped_plate_r_p,
                                                                   dfrul_fit.rr.to_numpy() * microns_per_pixel * 1e-6,
                                                                   (dfrul_fit.z_corr.to_numpy() - z_ul_offset) * 1e-6,
                                                                   )

                                # fit error
                                fit_results_ul = fsphere_ul.spherical_uniformly_loaded_simply_supported_plate_r_p(
                                    dfrul.rr.to_numpy() * microns_per_pixel * 1e-6, *popt_ul)
                                data_fit_to_ul = (dfrul.z_corr.to_numpy() - z_ul_offset) * 1e-6

                                rmse_ul, r_squared_ul = fit.calculate_fit_error(
                                    fit_results=fit_results_ul,
                                    data_fit_to=data_fit_to_ul,
                                )

                                # fit profile
                                z_fit_ul = fsphere_ul.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit_ul * 1e-6,
                                                                                                            *popt_ul) * 1e6 + z_ul_offset
                                z_fit_cc_ul = fsphere_ul.spherical_uniformly_loaded_clamped_plate_r_p(r_fit_ul * 1e-6,
                                                                                                      *popt_cc_ul) * 1e6 + z_ul_offset

                                # ---

                                # store z-radius of fitted sphere
                                rz_ul = fsphere_ul.spherical_uniformly_loaded_simply_supported_plate_r_p(0, *popt_ul) * 1e6
                                rz_ul_fitted_sphere.append(rz_ul)
                                fit_ul_pressure.append(popt_ul[0])
                                fit_ul_pretension.append(0.0)
                                fit_ul_rmse.append(rmse_ul * 1e6)
                                fit_ul_percent_meas.append(ul_percent_meas)
                                fit_ul_r_squared.append(r_squared_ul)

                                # calculate theta at 1/4 radius
                                idx_qr = len(r_fit_ul) // 4
                                theta_ul_deg.append(np.rad2deg(np.arctan2(z_fit_ul[-idx_qr * 2] - z_fit_ul[-idx_qr],
                                                                          r_fit_ul[-idx_qr] - r_fit_ul[-idx_qr * 2])))

                                # ---

                            # ---

                            # plot
                            if plot_per_frame_membrane_radial_profile_ul:
                                figg, axg = plt.subplots()
                                axg.plot(dfrul_o.rr * microns_per_pixel, dfrul_o.z_corr, '^',
                                         ms=2, color='black', alpha=0.75, label='raw u.l.')
                                axg.plot(dfrul.rr * microns_per_pixel, dfrul.z_corr, 'o', ms=2, color='blue', label='data')
                                axg.plot(r_fit_ul, z_fit_ul, '-', label='fit')
                                axg.axhline(z_ul_offset, linestyle='--', alpha=0.5, label=r'$z_{B.C.}$')
                                axg.set_xlabel(r'$r \: (\mu m)$')
                                axg.set_xlim([-5, (r_edge_ul + 5) * microns_per_pixel])
                                axg.set_ylabel(r'$z \: (\mu m)$')
                                axg.set_ylim([-80, 80])
                                axg.legend()
                                axg.set_title('Frame = {}'.format(fr))
                                figg.tight_layout()
                                figg.savefig(path_figs_plot_per_frame_ul + '/memb-ul_fit-spherical_frame={}.png'.format(fr),
                                             dpi=100)
                                # figg.show()
                                plt.close(figg)

                            # ---

                            # number of measurements
                            i_num_pids_ul.append(i_num_ul)
                            f_num_pids_ul.append(f_num_ul)

                            # ---

                        # ---

                        # --- LEFT LEFT MEMBRANE -----------------------------------------------------------------------

                        if analyze_ellipsoid_ll:

                            dfrll = dfll[dfll['frame'] == fr]

                            i_num_ll = len(dfrll)

                            # plot radial position
                            if plot_ll_particles_wrt_radial_coordinate and fr_idx == 0:
                                figg, (axg, axzg) = plt.subplots(nrows=2, sharex=True,
                                                                 figsize=(size_x_inches * 1.125, size_y_inches * 1.5),
                                                                 gridspec_kw={'height_ratios': [3, 1]}
                                                                 )
                                sc = axg.scatter(dfrll.x, dfrll.y, c=dfrll.rr)
                                axg.add_patch(plt.Circle((xc_ll, yc_ll), r_edge_ll, color='black', fill=False))
                                axg.add_patch(plt.Circle((xc_ll, yc_ll), 5, color='red', fill=True))
                                plt.colorbar(sc, ax=axg)
                                axzg.scatter(dfrll.x, dfrll.z_corr, c=dfrll.rr)

                                axg.set_ylabel('y')
                                axg.set_ylim([0, image_length])
                                axg.invert_yaxis()
                                axg.set_aspect('equal', adjustable='box')

                                axzg.set_xlabel('x')
                                axzg.set_xlim([0, image_length])
                                axzg.set_ylabel(r'$z_{corr} \: (\mu m)$')
                                figg.tight_layout()
                                figg.savefig(path_figs + '/ll_particles_wrt_radial-coordinate.png')
                                figg.show()
                                plt.close(figg)

                            if apply_z_corr_directional_filters_ll_mm:
                                if -10 < dfrll.z_corr.mean() < 10:
                                    pass
                                elif dfrll.z_corr.mean() > z_i_mean_ll:
                                    dfrll = dfrll[dfrll['z_corr'] > z_i_mean_ll - z_i_mean_allowance]
                                else:
                                    dfrll = dfrll[dfrll['z_corr'] < z_i_mean_ll + z_i_mean_allowance]
                            else:
                                pass

                            # ---

                            # store data used in reconstruction
                            df_reconstruction.append(dfrll)
                            f_pids_ll = dfrll.id.unique()

                            # calculate the relative number of particles evaluated
                            f_num_ll = len(dfrll)
                            ll_percent_meas = f_num_ll / i_num_ll

                            # ---

                            # z-offset term (the z=0 plane for the membrane) (options: z_i_mean_lr, z_mean_boundary)
                            z_ll_offset = z_i_mean_ll

                            # fit plate theory
                            fsphere_ll = fSphericalUniformLoad(r=r_edge_ll * microns_per_pixel * 1e-6,
                                                               h=t_membrane,
                                                               youngs_modulus=E_silpuran)
                            # assign fit function
                            fitfunc_ll = fsphere_ll

                            # fit
                            r_fit_ll = np.linspace(0, r_edge_ll * microns_per_pixel)
                            dfrll = dfrll.sort_values('rr', ascending=False)
                            rll_fit_r = dfrll.rr.to_numpy() * microns_per_pixel * 1e-6
                            rll_fit_z = (dfrll.z_corr.to_numpy() - z_ll_offset) * 1e-6

                            popt_ll, pcov_ll = curve_fit(
                                fsphere_ll.spherical_uniformly_loaded_simply_supported_plate_r_p,
                                rll_fit_r,
                                rll_fit_z,
                            )

                            # fit error
                            fit_results_ll = fsphere_ll.spherical_uniformly_loaded_simply_supported_plate_r_p(
                                dfrll.rr.to_numpy() * microns_per_pixel * 1e-6, *popt_ll)
                            data_fit_to_ll = (dfrll.z_corr.to_numpy() - z_ll_offset) * 1e-6

                            rmse_ll, r_squared_ll = fit.calculate_fit_error(
                                fit_results=fit_results_ll,
                                data_fit_to=data_fit_to_ll,
                            )

                            # fit profile
                            z_fit_ll = fsphere_ll.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit_ll * 1e-6,
                                                                                                        *popt_ll) * 1e6 + z_ll_offset

                            # ---

                            # plot
                            if plot_per_frame_membrane_radial_profile_ll:
                                figg, axg = plt.subplots()
                                axg.plot(dfrll.rr * microns_per_pixel, dfrll.z_corr, 'o', ms=2, label='l.l.')
                                axg.plot(r_fit_ll, z_fit_ll, '-', label='fit')
                                axg.axhline(z_ll_offset, linestyle='--', alpha=0.5, label=r'$z_{B.C.}$')
                                axg.set_xlabel(r'$r \: (\mu m)$')
                                axg.set_xlim([-5, (r_edge_ll + 5) * microns_per_pixel])
                                axg.set_ylabel(r'$z \: (\mu m)$')
                                axg.set_ylim([-35, 35])
                                axg.legend()
                                axg.set_title('Frame = {}'.format(fr))
                                figg.tight_layout()
                                figg.savefig(path_figs + '/memb-ll_fit-spherical_frame={}.png'.format(fr), dpi=100)
                                # figg.show()
                                plt.close(figg)

                            # ---

                            # store z-radius of fitted sphere
                            rz_ll = fsphere_ll.spherical_uniformly_loaded_simply_supported_plate_r_p(0, *popt_ll) * 1e6
                            rz_ll_fitted_sphere.append(rz_ll)
                            fit_ll_pressure.append(popt_ll[0])
                            fit_ll_rmse.append(rmse_ll * 1e6)
                            fit_ll_percent_meas.append(ll_percent_meas)
                            fit_ll_r_squared.append(r_squared_ll)

                            # calculate theta at 1/4 radius
                            idx_qr = len(r_fit_ll) // 4
                            theta_ll_deg.append(np.rad2deg(np.arctan2(z_fit_ll[-idx_qr] - z_fit_ll[-1],
                                                                      r_fit_ll[-1] - r_fit_ll[-idx_qr])))

                            # number of measurements
                            i_num_pids_ll.append(i_num_ll)
                            f_num_pids_ll.append(f_num_ll)

                            # ---

                        # ---

                        # --- LEFT LEFT MEMBRANE -----------------------------------------------------------------------

                        if analyze_ellipsoid_mm:

                            dfrmm = dfmm[dfmm['frame'] == fr]

                            i_num_mm = len(dfrmm)

                            # plot radial position
                            if plot_mm_particles_wrt_radial_coordinate and fr_idx == 0:
                                figg, (axg, axzg) = plt.subplots(nrows=2, sharex=True,
                                                                 figsize=(size_x_inches * 1.125, size_y_inches * 1.5),
                                                                 gridspec_kw={'height_ratios': [3, 1]}
                                                                 )
                                sc = axg.scatter(dfrmm.x, dfrmm.y, c=dfrmm.rr)
                                axg.add_patch(plt.Circle((xc_mm, yc_mm), r_edge_mm, color='black', fill=False))
                                axg.add_patch(plt.Circle((xc_mm, yc_mm), 5, color='red', fill=True))
                                plt.colorbar(sc, ax=axg)
                                axzg.scatter(dfrmm.x, dfrmm.z_corr, c=dfrmm.rr)

                                axg.set_ylabel('y')
                                axg.set_ylim([0, image_length])
                                axg.invert_yaxis()
                                axg.set_aspect('equal', adjustable='box')

                                axzg.set_xlabel('x')
                                axzg.set_xlim([0, image_length])
                                axzg.set_ylabel(r'$z_{corr} \: (\mu m)$')
                                figg.tight_layout()
                                figg.savefig(path_figs + '/mm_particles_wrt_radial-coordinate.png')
                                figg.show()
                                plt.close(figg)

                            if apply_z_corr_directional_filters_ll_mm:
                                if -10 < dfrmm.z_corr.mean() < 10:
                                    pass
                                elif dfrmm.z_corr.mean() > z_i_mean_mm:
                                    dfrmm = dfrmm[dfrmm['z_corr'] > z_i_mean_mm - z_i_mean_allowance]
                                else:
                                    dfrmm = dfrmm[dfrmm['z_corr'] < z_i_mean_mm + z_i_mean_allowance]
                            else:
                                pass

                            # ---

                            # store data used in reconstruction
                            df_reconstruction.append(dfrmm)
                            f_pids_mm = dfrmm.id.unique()

                            # calculate the relative number of particles evaluated
                            f_num_mm = len(dfrmm)
                            mm_percent_meas = f_num_mm / i_num_mm

                            # ---

                            # z-offset term (the z=0 plane for the membrane) (options: z_i_mean_lr, z_mean_boundary)
                            z_mm_offset = z_i_mean_mm

                            # fit plate theory
                            fsphere_mm = fSphericalUniformLoad(r=r_edge_mm * microns_per_pixel * 1e-6, h=t_membrane,
                                                               youngs_modulus=E_silpuran)
                            # assign fit function
                            fitfunc_mm = fsphere_mm

                            # fit
                            r_fit_mm = np.linspace(0, r_edge_mm * microns_per_pixel)
                            dfrmm = dfrmm.sort_values('rr', ascending=False)
                            popt_mm, pcov_mm = curve_fit(
                                fsphere_mm.spherical_uniformly_loaded_simply_supported_plate_r_p,
                                dfrmm.rr.to_numpy() * microns_per_pixel * 1e-6,
                                (dfrmm.z_corr.to_numpy() - z_mm_offset) * 1e-6,
                            )

                            # fit error
                            fit_results_mm = fsphere_mm.spherical_uniformly_loaded_simply_supported_plate_r_p(
                                dfrmm.rr.to_numpy() * microns_per_pixel * 1e-6, *popt_mm)
                            data_fit_to_mm = (dfrmm.z_corr.to_numpy() - z_mm_offset) * 1e-6

                            rmse_mm, r_squared_mm = fit.calculate_fit_error(
                                fit_results=fit_results_mm,
                                data_fit_to=data_fit_to_mm,
                            )

                            # fit profile
                            z_fit_mm = fsphere_mm.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit_mm * 1e-6,
                                                                                                        *popt_mm) * 1e6 + z_mm_offset

                            # ---

                            # plot
                            if plot_per_frame_membrane_radial_profile_mm:
                                figg, axg = plt.subplots()
                                axg.plot(dfrmm.rr * microns_per_pixel, dfrmm.z_corr, 'o', ms=2, label='m.m.')
                                axg.plot(r_fit_mm, z_fit_mm, '-', label='fit')
                                axg.axhline(z_mm_offset, linestyle='--', alpha=0.5, label=r'$z_{B.C.}$')
                                axg.set_xlabel(r'$r \: (\mu m)$')
                                axg.set_xlim([-5, (r_edge_mm + 5) * microns_per_pixel])
                                axg.set_ylabel(r'$z \: (\mu m)$')
                                axg.set_ylim([-20, 20])
                                axg.legend()
                                axg.set_title('Frame = {}'.format(fr))
                                figg.tight_layout()
                                figg.savefig(path_figs + '/memb-mm_fit-spherical_frame={}.png'.format(fr), dpi=100)
                                # figg.show()
                                plt.close(figg)

                            # ---

                            # store z-radius of fitted sphere
                            rz_mm = fsphere_mm.spherical_uniformly_loaded_simply_supported_plate_r_p(0, *popt_mm) * 1e6
                            rz_mm_fitted_sphere.append(rz_mm)
                            fit_mm_pressure.append(popt_mm[0])
                            fit_mm_rmse.append(rmse_mm * 1e6)
                            fit_mm_percent_meas.append(mm_percent_meas)
                            fit_mm_r_squared.append(r_squared_mm)

                            # calculate theta at 1/4 radius
                            idx_qr = len(r_fit_mm) // 4
                            theta_mm_deg.append(np.rad2deg(np.arctan2(z_fit_mm[-idx_qr] - z_fit_mm[-1],
                                                                      r_fit_mm[-1] - r_fit_mm[-idx_qr])))

                            # number of measurements
                            i_num_pids_mm.append(i_num_mm)
                            f_num_pids_mm.append(f_num_mm)

                            # ---

                        # ---

                        # --- STACK PIDS USED TO RECONSTRUCT SURFACE IN EACH FRAME
                        fr_pids.append(np.hstack([f_pids_lr, f_pids_ul, f_pids_ll, f_pids_mm]))

                        # --- STACK DATA + FIT FOR LOCALIZATION ERROR BY Z ANALYSIS ------------------------------------

                        stack_fit_results_lr.append(fit_results_lr * 1e6 + z_lr_offset)
                        stack_data_fit_to_lr.append(data_fit_to_lr * 1e6 + z_lr_offset)
                        stack_data_r_lr.append(dfrlr.rr.to_numpy() * microns_per_pixel)
                        stack_data_fr_lr.append(np.ones_like(dfrlr.rr.to_numpy()) * fr)
                        stack_fit_results_ul.append(fit_results_ul * 1e6 + z_ul_offset)
                        stack_data_fit_to_ul.append(data_fit_to_ul * 1e6 + z_ul_offset)
                        stack_data_r_ul.append(dfrul.rr.to_numpy() * microns_per_pixel)
                        stack_data_fr_ul.append(np.ones_like(dfrul.rr.to_numpy()) * fr)

                        if analyze_ellipsoid_ll:
                            stack_fit_results_ll.append(fit_results_ll * 1e6 + z_ll_offset)
                            stack_data_fit_to_ll.append(data_fit_to_ll * 1e6 + z_ll_offset)
                            stack_data_r_ll.append(dfrll.rr.to_numpy() * microns_per_pixel)
                            stack_data_fr_ll.append(np.ones_like(dfrll.rr.to_numpy()) * fr)

                        if analyze_ellipsoid_mm:
                            stack_fit_results_mm.append(fit_results_mm * 1e6 + z_mm_offset)
                            stack_data_fit_to_mm.append(data_fit_to_mm * 1e6 + z_mm_offset)
                            stack_data_r_mm.append(dfrmm.rr.to_numpy() * microns_per_pixel)
                            stack_data_fr_mm.append(np.ones_like(dfrmm.rr.to_numpy()) * fr)

                        # ---

                        # --- PLOT ALL FRAMES --------------------------------------------------------------------------

                        # ---

                        # plot radial profile

                        if plot_per_frame_membrane_radial_profile_all:
                            figg, axg = plt.subplots()

                            # lower right
                            axg.plot(dfrlr.rr * microns_per_pixel, dfrlr.z_corr, 'o', ms=2, color=sciblue, label='l.r.')
                            axg.plot(r_fit_lr, z_fit_lr, '-', color=lighten_color(sciblue, 1.15))
                            axg.axhline(z_lr_offset, linestyle='--', linewidth=0.5, color=lighten_color(sciblue, 1.15),
                                        alpha=0.5)

                            # upper left
                            axg.plot(dfrul.rr * microns_per_pixel, dfrul.z_corr, 'o', ms=2, color=scigreen, label='u.l.')
                            axg.plot(r_fit_ul, z_fit_ul, '-', color=lighten_color(scigreen, 1.15))
                            axg.axhline(z_ul_offset, linestyle='--', linewidth=0.5, color=lighten_color(scigreen, 1.15),
                                        alpha=0.5)

                            # left left
                            axg.plot(dfrll.rr * microns_per_pixel, dfrll.z_corr, 'o', ms=2, color=scired, label='l.l.')
                            axg.plot(r_fit_ll, z_fit_ll, '-', color=lighten_color(scired, 1.15))
                            axg.axhline(z_ll_offset, linestyle='--', linewidth=0.5, color=lighten_color(scired, 1.15),
                                        alpha=0.5)

                            # middle middle
                            axg.plot(dfrmm.rr * microns_per_pixel, dfrmm.z_corr, 'o', ms=2, color=sciorange, label='m.m.')
                            axg.plot(r_fit_mm, z_fit_mm, '-', color=lighten_color(sciorange, 1.15))
                            axg.axhline(z_mm_offset, linestyle='--', linewidth=0.5, color=lighten_color(sciorange, 1.15),
                                        alpha=0.5)

                            axg.set_xlabel(r'$r \: (\mu m)$')
                            axg.set_xlim([-5, (r_edge_lr + 5) * microns_per_pixel])
                            axg.set_ylabel(r'$z \: (\mu m)$')
                            axg.set_ylim([-150, 150])
                            axg.legend()
                            axg.set_title('Frame = {}'.format(fr))
                            figg.tight_layout()
                            figg.savefig(path_figs_plot_per_frame_all + '/memb-all_fit-spherical_frame={}.png'.format(fr),
                                         dpi=100)
                            # figg.show()
                            plt.close(figg)

                        # ---

                        # plot 3D topography
                        if fr in frames_of_interest:
                            if plot_3d_topography:
                                # particle positions
                                dfr = df[df['frame'] == fr]
                                xp = dfr.x.to_numpy() * microns_per_pixel
                                yp = dfr.y.to_numpy() * microns_per_pixel
                                zp = dfr.z_corr.to_numpy()

                                # create pressure variable for simplicity
                                lrp = fit_lr_pressure[-1]
                                ulp = fit_ul_pressure[-1]
                                llp = 0  # fit_ll_pressure[-1]
                                mmp = 0  # fit_mm_pressure[-1]

                                # create pretension variable for simplicity
                                lrk = fit_lr_pretension[-1]
                                ulk = fit_ul_pretension[-1]
                                llk = 0
                                mmk = 0

                                # setup
                                cmap = plt.cm.coolwarm  # plt.cm.viridis  # full range: plt.cm.coolwarm
                                cmap_lr = cm.Blues
                                cmap_ul = cm.Greens
                                cmap_ll = cm.Purples
                                cmap_mm = cm.Oranges
                                max_r_z_all_membranes = 135

                                pos_range_only = True
                                if pos_range_only:
                                    vmin, vmax = z_zero - max_r_z_all_membranes, z_zero + max_r_z_all_membranes
                                    zticks = [0, 50, 100]
                                    z_limits = [z_zero - 2.5, 135]
                                else:
                                    vmin, vmax = z_zero - max_r_z_all_membranes, z_zero + max_r_z_all_membranes
                                    zticks = [-100, -50, 0, 50, 100]
                                    z_limits = [-130, 135]

                                alpha = 0.75
                                elev, azim = 20, 30
                                z_axis_scale = 0.015

                                # figure
                                fig = plt.figure(figsize=(size_x_inches * 1.2, size_y_inches * 1.2))
                                ax = fig.add_subplot(projection='3d')

                                # Create the mesh in polar coordinates and compute corresponding Z.
                                p = np.linspace(0, 2 * np.pi, 50)
                                R_lr, P_lr = np.meshgrid(r_fit_lr, p)
                                R_ul, P_ul = np.meshgrid(r_fit_ul, p)
                                R_ll, P_ll = np.meshgrid(r_fit_ll, p)
                                R_mm, P_mm = np.meshgrid(r_fit_mm, p)

                                # Compute corresponding Z
                                Z_lr = fitfunc_lr.plot_dimensional_z_by_r_p_k(d_r=R_lr * 1e-6, d_p0=lrp, d_n0=lrk) * 1e6
                                Z_ul = fitfunc_ul.plot_dimensional_z_by_r_p_k(d_r=R_ul * 1e-6, d_p0=ulp, d_n0=ulk) * 1e6
                                Z_ll = fitfunc_ll.plot_dimensional_z_by_r_p_k(d_r=R_ll * 1e-6, d_p0=llp, d_n0=llk) * 1e6
                                Z_mm = fitfunc_mm.plot_dimensional_z_by_r_p_k(d_r=R_mm * 1e-6, d_p0=mmp, d_n0=mmk) * 1e6

                                """ Old Method:
                                Z_lr = fitfunc_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(r=R_lr * 1e-6,
                                                                                                        P=lrp) * 1e6
                                Z_ul = fitfunc_ul.spherical_uniformly_loaded_simply_supported_plate_r_p(r=R_ul * 1e-6,
                                                                                                        P=ulp) * 1e6
                                Z_ll = fitfunc_ll.spherical_uniformly_loaded_simply_supported_plate_r_p(r=R_ll * 1e-6,
                                                                                                        P=llp) * 1e6
                                Z_mm = fitfunc_mm.spherical_uniformly_loaded_simply_supported_plate_r_p(r=R_mm * 1e-6,
                                                                                                        P=mmp) * 1e6
                                """

                                # Express the mesh in the cartesian system.
                                X_lr, Y_lr = R_lr * np.cos(P_lr), R_lr * np.sin(P_lr)
                                X_ul, Y_ul = R_ul * np.cos(P_ul), R_ul * np.sin(P_ul)
                                X_ll, Y_ll = R_ll * np.cos(P_ll), R_ll * np.sin(P_ll)
                                X_mm, Y_mm = R_mm * np.cos(P_mm), R_mm * np.sin(P_mm)

                                # shift x-y coordinates to membrane centers
                                X_lr, Y_lr = X_lr + xc_lr * microns_per_pixel, Y_lr + yc_lr * microns_per_pixel
                                X_ul, Y_ul = X_ul + xc_ul * microns_per_pixel, Y_ul + yc_ul * microns_per_pixel
                                X_ll, Y_ll = X_ll + xc_ll * microns_per_pixel, Y_ll + yc_ll * microns_per_pixel
                                X_mm, Y_mm = X_mm + xc_mm * microns_per_pixel, Y_mm + yc_mm * microns_per_pixel

                                # plot the surface
                                surf = ax.plot_surface(X_lr, Y_lr, Z_lr + z_i_mean_lr, cmap=cmap_lr, vmin=vmin, vmax=vmax,
                                                       alpha=alpha)
                                ax.plot_surface(X_ul, Y_ul, Z_ul + z_i_mean_ul, cmap=cmap_ul, vmin=vmin, vmax=vmax, alpha=alpha)
                                ax.plot_surface(X_ll, Y_ll, Z_ll + z_i_mean_ll, cmap=cmap_ll, vmin=vmin, vmax=vmax, alpha=alpha)
                                ax.plot_surface(X_mm, Y_mm, Z_mm + z_i_mean_mm, cmap=cmap_mm, vmin=vmin, vmax=vmax, alpha=alpha)

                                # scatter plot points
                                ax.scatter(xp, yp, zp, color='black', marker='.', s=2)

                                # plot the radii centers
                                """
                                ax.scatter(xc_lr * microns_per_pixel, yc_lr * microns_per_pixel, z_i_mean_lr, color='r', marker='*', s=3)
                                ax.scatter(xc_ul * microns_per_pixel, yc_ul * microns_per_pixel, z_i_mean_ul, color='r', marker='*', s=3)
                                ax.scatter(xc_ll * microns_per_pixel, yc_ll * microns_per_pixel, z_i_mean_ll, color='r', marker='*', s=3)
                                ax.scatter(xc_mm * microns_per_pixel, yc_mm * microns_per_pixel, z_i_mean_mm, color='r', marker='*', s=3)
                                """

                                # plot field-of-view box
                                xfov = np.array([0, 512, 512, 0, 0]) * microns_per_pixel
                                yfov = np.array([0, 0, 512, 512, 0]) * microns_per_pixel
                                zfov = np.ones_like(xfov) * z_zero
                                ax.plot(xfov, yfov, zfov, color='black', alpha=0.125)

                                # set_3d_axes_equal(ax, z_axis_scale=z_axis_scale)
                                ax.view_init(elev=elev, azim=azim)
                                ax.set_xlim([-50, 2000])
                                ax.set_ylim([-100, 2400])
                                ax.set_zlim3d(z_limits)
                                ax.invert_yaxis()

                                ax.set_xlabel(r"$x \: (\mu m)$", labelpad=-15)
                                ax.set_ylabel(r"$y \: (\mu m)$", labelpad=-15)
                                ax.set_zlabel(r"$z \: (\mu m)$", labelpad=0)

                                ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                                ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                                ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                                ax.grid(False)

                                """for line in ax.xaxis.get_ticklines():
                                    line.set_visible(False)
                                for line in ax.yaxis.get_ticklines():
                                    line.set_visible(False)
                                for line in ax.zaxis.get_ticklines():
                                    line.set_visible(False)"""

                                ax.xaxis.set_ticks([500, 1000, 1500])
                                ax.xaxis.set_ticklabels([])
                                ax.yaxis.set_ticklabels([])
                                ax.zaxis.set_ticks(zticks, label=zticks, labelpad=-50)
                                # ax.zaxis.set_ticks(zticks_pos_range, label=zticks_pos_range, labelpad=-50)

                                # fig.colorbar(surf, ax=ax, aspect=25, shrink=0.5, location='left', pad=0.15, panchor=(0, 0.75))
                                # ax.dist = 8.15

                                ax.set_title(r'$t = $' + ' {} '.format(np.round(fr / frame_rate, 2)) + r'$s$')

                                plt.savefig(path_figs_3d_topography + '/plot_3d_topography_frame{}.png'.format(fr))
                                plt.close()

                        # ---

                        # --- PLOT ONLY FRAMES OF INTEREST -------------------------------------------------------------

                        # ---

                        if fr in frames_of_interest:

                            # --- plot: z(r) + plate theory + rmse
                            if plot_froi_fit_plate_theory:
                                custom_colors = True
                                # setup
                                cmap = plt.cm.coolwarm  # plt.cm.viridis  # full range: plt.cm.coolwarm
                                cmap_lr = cm.Blues
                                cmap_ul = cm.Greens
                                cmap_ll = cm.Purples
                                cmap_mm = cm.Oranges
                                max_r_z_all_membranes = 150

                                pos_range_only = False
                                if pos_range_only:
                                    vmin, vmax = -30, None  # z_zero, z_zero + max_r_z_all_membranes
                                    zticks = [0, 50, 100, 150]
                                    z_limits = [z_zero - 2.5, 155]
                                else:
                                    vmin, vmax = z_zero - max_r_z_all_membranes, z_zero + max_r_z_all_membranes
                                    zticks = [-100, -50, 0, 50, 100, 150]
                                    z_limits = [-120, 155]

                                figf, axf = plt.subplots(figsize=(size_x_inches, size_y_inches))

                                if custom_colors:
                                    # lower right
                                    axf.scatter(dfrlr.rr * microns_per_pixel, dfrlr.z_corr, s=15,
                                                color=lighten_color(sciblue, 0.9),
                                                cmap=cmap_lr, vmin=vmin, vmax=vmax,
                                                label='800')
                                    # label=r'$r_{z,lr} = $' + ' {} '.format(np.round(rz_lr, 1)) + r'$\mu m$'

                                    axf.plot(r_fit_lr, z_fit_lr, color=lighten_color(sciblue, 1.1))
                                    # label=r'$r.m.s.e.=$' + ' {} '.format(np.round(rmse_lr * 1e6, 1)) + r'$\mu m$'
                                    # axf.plot(r_fit_lr, z_fit_cc_lr, color='navy', linestyle='--')
                                    axf.axvline(r_edge_lr * radius_edge_limit * microns_per_pixel,
                                                linestyle='dotted', linewidth=0.5, color='darkred', alpha=0.15,
                                                label='p0={}, n0={}'.format(np.round(fit_lr_pressure[-1], 1),
                                                                            np.round(fit_lr_pretension[-1], 2)))
                                    # axf.axhline(z_lr_offset, linestyle='--', color='gray', linewidth=0.75, alpha=0.75)

                                    # upper left
                                    axf.scatter(dfrul.rr * microns_per_pixel, dfrul.z_corr, s=15,
                                                color=lighten_color(scigreen, 0.9),
                                                cmap=cmap_ul, vmin=vmin, vmax=vmax,
                                                label='500')
                                    # label=r'$r_{z,ul} = $' + ' {} '.format(np.round(rz_ul, 1)) + r'$\mu m$'

                                    axf.plot(r_fit_ul, z_fit_ul, color=lighten_color(scigreen, 1.1))
                                    # label = r'$r.m.s.e.=$' + ' {} '.format(np.round(rmse_ul * 1e6, 1)) + r'$\mu m$'
                                    # axf.plot(r_fit_ul, z_fit_cc_ul, color='darkgreen', linestyle='--')

                                    axf.axvline(r_edge_ul * radius_edge_limit * microns_per_pixel,
                                                linestyle='dotted', linewidth=0.5, color='darkred', alpha=0.15,
                                                label='p0={}, n0={}'.format(np.round(fit_ul_pressure[-1], 1),
                                                                            np.round(fit_ul_pretension[-1], 2)))
                                    axf.axhline(z_ul_offset, linestyle='--', color='gray', linewidth=0.75, alpha=0.5)
                                else:
                                    # lower right
                                    p1, = axf.plot(dfrlr.rr * microns_per_pixel, dfrlr.z_corr, 'o')
                                    # label=r'$r_{z,lr} = $' + ' {} '.format(np.round(rz_lr, 1)) + r'$\mu m$'

                                    axf.plot(r_fit_lr, z_fit_lr, color=lighten_color(p1.get_color(), 1.125))
                                    # label=r'$r.m.s.e.=$' + ' {} '.format(np.round(rmse_lr * 1e6, 2)) + r'$\mu m$'

                                    axf.axhline(z_lr_offset, linestyle='--', color='blue')

                                    # upper left
                                    p2, = axf.plot(dfrul.rr * microns_per_pixel, dfrul.z_corr, 'o')
                                    # label=r'$r_{z,ul} = $' + ' {} '.format(np.round(rz_ul, 1)) + r'$\mu m$'

                                    axf.plot(r_fit_ul, z_fit_ul, color=lighten_color(p2.get_color(), 1.125))
                                    # label=r'$r.m.s.e.=$' + ' {} '.format(np.round(rmse_ul * 1e6, 2)) + r'$\mu m$'

                                    # axf.axhline(z_ul_offset, linestyle='--', color='green')

                                axf.set_xlabel(r'$r \: (\mu m)$')
                                axf.set_xlim([-15, r_fit_lr.max() + 15])
                                axf.set_ylabel(r'$z \: (\mu m)$')
                                axf.legend(title=r'$r \: (\mu m)$')
                                # axf.set_title('Frame = {}'.format(fr))
                                # axf.legend(loc='upper left', bbox_to_anchor=(1, 1),
                                #           markerscale=0.75, labelspacing=0.35, handletextpad=0.5, borderaxespad=0.35)
                                figf.tight_layout()
                                if save_figs:
                                    figf.savefig(
                                        path_figs_fit_profile + '/id{}_frame{}_fit_ss_spherical_plate.png'.format(tid, fr),
                                        dpi=150,
                                    )
                                if show_figs:
                                    figf.show()
                                plt.close(figf)

                                # ---

                                figf, axf = plt.subplots(figsize=(size_x_inches, size_y_inches))

                                axf.scatter(dfrlr.rr / r_edge_lr,
                                            (dfrlr.z_corr - z_lr_offset) / np.abs(z_fit_lr[0] - z_lr_offset),
                                            s=15, color=lighten_color(sciblue, 0.9), label='800')

                                axf.plot(r_fit_lr / (r_edge_lr * microns_per_pixel),
                                         (z_fit_lr - z_lr_offset) / np.abs(z_fit_lr[0] - z_lr_offset),
                                         color=lighten_color(sciblue, 1.1))

                                # upper left
                                axf.scatter(dfrul.rr / r_edge_ul,
                                            (dfrul.z_corr - z_ul_offset) / np.abs(z_fit_ul[0] - z_ul_offset),
                                            s=15, color=lighten_color(scigreen, 0.9), label='500')

                                axf.plot(r_fit_ul / (r_edge_ul * microns_per_pixel),
                                         (z_fit_ul - z_ul_offset) / np.abs(z_fit_ul[0] - z_ul_offset),
                                         color=lighten_color(scigreen, 1.1))

                                axf.set_xlabel(r'$r/a$')
                                axf.set_xlim([-0.03, 1.03])
                                axf.set_ylabel(r'$\Delta z / \Delta z_{0}$')
                                axf.legend(title=r'$r \: (\mu m)$')
                                # markerscale=0.75, labelspacing=0.35, handletextpad=0.5, borderaxespad=0.35)
                                figf.tight_layout()
                                if save_figs:
                                    figf.savefig(
                                        path_figs_fit_profile + '/id{}_frame{}_normalized-deflection.png'.format(tid, fr),
                                        dpi=150,
                                    )
                                if show_figs:
                                    figf.show()
                                plt.close(figf)

                            # ---

                            # ---

                        # ---

                    # ---

                    # --- SAVE RESULTS TO EXCEL SPREADSHEET
                    if export_all_results:

                        # save reconstruction dataframe
                        df_reconstruction = pd.concat(df_reconstruction)
                        if export_all_results:
                            df_reconstruction.to_excel(path_results + '/df_reconstruction.xlsx')

                        # save pids used to reconstruct per-frame
                        dict_fr_pids = dict([(key, value) for key, value in zip(frames_of_interest_for_analysis, fr_pids)])
                        export_fr_pids = pd.DataFrame.from_dict(data=dict_fr_pids, orient='index')
                        export_fr_pids = export_fr_pids.rename(columns={0: 'pids'})
                        if export_all_results:
                            export_fr_pids.to_excel(path_results + '/pids_per_frame.xlsx', index_label='frame')

                        # coords + fit error
                        stack_data_fr_lr = [item for sublist in stack_data_fr_lr for item in sublist]
                        stack_data_fr_ul = [item for sublist in stack_data_fr_ul for item in sublist]
                        stack_data_fr_ll = [item for sublist in stack_data_fr_ll for item in sublist]
                        stack_data_fr_mm = [item for sublist in stack_data_fr_mm for item in sublist]

                        stack_data_r_lr = [item for sublist in stack_data_r_lr for item in sublist]
                        stack_data_r_ul = [item for sublist in stack_data_r_ul for item in sublist]
                        stack_data_r_ll = [item for sublist in stack_data_r_ll for item in sublist]
                        stack_data_r_mm = [item for sublist in stack_data_r_mm for item in sublist]

                        stack_data_fit_to_lr = [item for sublist in stack_data_fit_to_lr for item in sublist]
                        stack_data_fit_to_ul = [item for sublist in stack_data_fit_to_ul for item in sublist]
                        stack_data_fit_to_ll = [item for sublist in stack_data_fit_to_ll for item in sublist]
                        stack_data_fit_to_mm = [item for sublist in stack_data_fit_to_mm for item in sublist]

                        stack_fit_results_lr = [item for sublist in stack_fit_results_lr for item in sublist]
                        stack_fit_results_ul = [item for sublist in stack_fit_results_ul for item in sublist]
                        stack_fit_results_ll = [item for sublist in stack_fit_results_ll for item in sublist]
                        stack_fit_results_mm = [item for sublist in stack_fit_results_mm for item in sublist]

                        memb_id1 = np.ones_like(stack_data_r_lr)
                        memb_id2 = np.ones_like(stack_data_r_ul) * 2
                        memb_id3 = np.ones_like(stack_data_r_ll) * 3
                        memb_id4 = np.ones_like(stack_data_r_mm) * 4

                        # save all membranes together
                        stack_fr_data = np.hstack([stack_data_fr_lr, stack_data_fr_ul, stack_data_fr_ll, stack_data_fr_mm])
                        stack_memb_id = np.hstack([memb_id1, memb_id2, memb_id3, memb_id4])
                        stack_r_data = np.hstack([stack_data_r_lr, stack_data_r_ul, stack_data_r_ll, stack_data_r_mm])
                        stack_z_data = np.hstack(
                            [stack_data_fit_to_lr, stack_data_fit_to_ul, stack_data_fit_to_ll, stack_data_fit_to_mm])
                        stack_z_fit = np.hstack(
                            [stack_fit_results_lr, stack_fit_results_ul, stack_fit_results_ll, stack_fit_results_mm])

                        stack_z_data_fit = np.vstack([stack_fr_data, stack_memb_id, stack_r_data, stack_z_data, stack_z_fit]).T
                        df_data_fit = pd.DataFrame(data=stack_z_data_fit, columns=['frame', 'memb_id', 'r', 'z', 'z_fit'])
                        df_data_fit['error'] = df_data_fit['z'] - df_data_fit['z_fit']
                        if export_all_results:
                            df_data_fit.to_excel(join(path_results, 'id{}_z_data_and_fit.xlsx'.format(tid)), index=False)

                        # ---

                        # stack data horizontally
                        times_of_interest_for_analysis = np.array(frames_of_interest_for_analysis) / frame_rate
                        data = np.array([frames_of_interest_for_analysis,
                                         times_of_interest_for_analysis.tolist(),
                                         rz_lr_fitted_sphere,
                                         fit_lr_pressure,
                                         fit_lr_pretension,
                                         fit_lr_rmse,
                                         fit_lr_percent_meas,
                                         fit_lr_r_squared,
                                         theta_lr_deg,
                                         rz_ul_fitted_sphere,
                                         fit_ul_pressure,
                                         fit_ul_pretension,
                                         fit_ul_rmse,
                                         fit_ul_percent_meas,
                                         fit_ul_r_squared,
                                         theta_ul_deg,
                                         rz_ll_fitted_sphere,
                                         fit_ll_pressure,
                                         fit_ll_rmse,
                                         fit_ll_percent_meas,
                                         fit_ll_r_squared,
                                         theta_ll_deg,
                                         rz_mm_fitted_sphere,
                                         fit_mm_pressure,
                                         fit_mm_rmse,
                                         fit_mm_percent_meas,
                                         fit_mm_r_squared,
                                         theta_mm_deg,
                                         i_num_pids_lr,
                                         f_num_pids_lr,
                                         i_num_pids_ul,
                                         f_num_pids_ul,
                                         i_num_pids_ll,
                                         f_num_pids_ll,
                                         i_num_pids_mm,
                                         f_num_pids_mm,
                                         ]
                                        ).T
                        df_rz = pd.DataFrame(data,
                                             columns=['frame', 'time',
                                                      'rz_lr', 'fit_lr_pressure', 'fit_lr_pretension', 'fit_lr_rmse',
                                                      'fit_lr_percent_meas',
                                                      'fit_lr_r_squared', 'theta_lr_deg',
                                                      'rz_ul', 'fit_ul_pressure', 'fit_ul_pretension', 'fit_ul_rmse',
                                                      'fit_ul_percent_meas',
                                                      'fit_ul_r_squared', 'theta_ul_deg',
                                                      'rz_ll', 'fit_ll_pressure', 'fit_ll_rmse', 'fit_ll_percent_meas',
                                                      'fit_ll_r_squared', 'theta_ll_deg',
                                                      'rz_mm', 'fit_mm_pressure', 'fit_mm_rmse', 'fit_mm_percent_meas',
                                                      'fit_mm_r_squared', 'theta_mm_deg',
                                                      'i_num_pids_lr', 'f_num_pids_lr',
                                                      'i_num_pids_ul', 'f_num_pids_ul',
                                                      'i_num_pids_ll', 'f_num_pids_ll',
                                                      'i_num_pids_mm', 'f_num_pids_mm',
                                                      ])

                        df_rz['r_lr'] = r_edge_lr * microns_per_pixel
                        df_rz['r_ul'] = r_edge_ul * microns_per_pixel
                        df_rz['r_ll'] = r_edge_ll * microns_per_pixel
                        df_rz['r_mm'] = r_edge_mm * microns_per_pixel

                        df_rz['rz_ul_lr_norm'] = df_rz.rz_ul / df_rz.rz_lr
                        df_rz['mean_pressure'] = (df_rz.fit_lr_pressure + df_rz.fit_ul_pressure) / 2

                        df_rz['theta_lr_peak_deg'] = np.rad2deg(np.arctan2(df_rz.rz_lr, df_rz.r_lr))
                        df_rz['theta_ul_peak_deg'] = np.rad2deg(np.arctan2(df_rz.rz_ul, df_rz.r_ul))

                        # absolute values
                        df_rz['rz_lr_abs'] = df_rz['rz_lr'].abs()
                        df_rz['rz_ul_abs'] = df_rz['rz_ul'].abs()

                        if export_all_results:
                            df_rz.to_excel(
                                join(path_results, 'id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(tid)),
                                index=False)

                        # ---

                        # stack data vertically
                        df_rz_lr = pd.DataFrame(np.array([frames_of_interest_for_analysis, rz_lr_fitted_sphere, fit_lr_rmse,
                                                          fit_lr_percent_meas, i_num_pids_lr, f_num_pids_lr]).T,
                                                columns=['frame', 'rz', 'fit_rmse', 'fit_percent_meas', 'i_num_pids',
                                                         'f_num_pids'])
                        df_rz_lr['r_microns'] = r_edge_lr * microns_per_pixel
                        df_rz_lr['memb_id'] = 1

                        df_rz_ul = pd.DataFrame(np.array([frames_of_interest_for_analysis, rz_ul_fitted_sphere, fit_ul_rmse,
                                                          fit_ul_percent_meas, i_num_pids_ul, f_num_pids_ul]).T,
                                                columns=['frame', 'rz', 'fit_rmse', 'fit_percent_meas', 'i_num_pids',
                                                         'f_num_pids'])
                        df_rz_ul['r_microns'] = r_edge_ul * microns_per_pixel
                        df_rz_ul['memb_id'] = 2

                        df_rz_ll = pd.DataFrame(np.array([frames_of_interest_for_analysis, rz_ll_fitted_sphere, fit_ll_rmse,
                                                          fit_ll_percent_meas, i_num_pids_ll, f_num_pids_ll]).T,
                                                columns=['frame', 'rz', 'fit_rmse', 'fit_percent_meas', 'i_num_pids',
                                                         'f_num_pids'])
                        df_rz_ll['r_microns'] = r_edge_ll * microns_per_pixel
                        df_rz_ll['memb_id'] = 3

                        df_rz_mm = pd.DataFrame(np.array([frames_of_interest_for_analysis, rz_mm_fitted_sphere, fit_mm_rmse,
                                                          fit_mm_percent_meas, i_num_pids_mm, f_num_pids_mm]).T,
                                                columns=['frame', 'rz', 'fit_rmse', 'fit_percent_meas', 'i_num_pids',
                                                         'f_num_pids'])
                        df_rz_mm['r_microns'] = r_edge_mm * microns_per_pixel
                        df_rz_mm['memb_id'] = 4

                        # concat
                        df_membs = pd.concat([df_rz_lr, df_rz_ul, df_rz_ll, df_rz_mm])

                        # export
                        if export_all_results:
                            df_membs.to_excel(
                                join(path_results,
                                     'id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest_vertical.xlsx'.format(tid)),
                                index=False)

                        # ---

                    # ---

                    # --- plot relationship: rz_ul ~ f(rz_lr)
                    if plot_pressure_relationship:

                        # plot pressure and pretension as a function of rz
                        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                                       figsize=(size_x_inches * 1.25, size_y_inches * 1.25),
                                                       )
                        ax1.plot(df_rz['rz_lr'], df_rz['fit_lr_pressure'].abs(), 'o', ms=2, label='LR')
                        ax1.plot(df_rz['rz_ul'], df_rz['fit_ul_pressure'].abs(), 'o', ms=2, label='UL')
                        ax2.plot(df_rz['rz_lr'], df_rz['fit_lr_pretension'], 'o', ms=2, label='LR')
                        ax2.plot(df_rz['rz_ul'], df_rz['fit_ul_pretension'], 'o', ms=2, label='UL')

                        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        ax1.set_ylabel(r'$P_{o}$')
                        ax1.set_yscale('log')

                        ax2.set_ylabel(r'$k$')
                        ax2.set_xlabel(r'$\Delta z_{o} \: (\mu m)$')
                        ax2.set_yscale('log')

                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(join(path_figs, 'id{}_P0-k_by_peak-dz_rz-ul_rz-lr.png'.format(tid)))
                        elif show_figs:
                            plt.show()
                        plt.close()

                        # ---

                        # filters
                        df_rz = df_rz[df_rz['rz_lr'].abs() > df_rz['rz_ul'].abs()]
                        df_rz = df_rz[df_rz['rz_ul_lr_norm'] > 0]

                        # PLOT ALL
                        fig, ax = plt.subplots()
                        ax.scatter(df_rz.rz_lr, df_rz.rz_ul, s=5, label='Data')
                        # df_rz_linear = df_rz[df_rz['rz_lr'] < 30]
                        fx = df_rz.rz_lr.to_numpy()
                        fy = df_rz.rz_ul.to_numpy()
                        popt, pcov = curve_fit(functions.line, fx, fy)
                        fit_space = np.linspace(df_rz.rz_lr.min(), df_rz.rz_lr.max())
                        ax.plot(fit_space, functions.line(fit_space, *popt),
                                linestyle='--', color='black',
                                label=r'$d r_{z, ul} / d r_{z, lr}=$' + ' {}'.format(np.round(popt[0], 3)),
                                )
                        ax.set_xlabel(r'$r_{z, lr}$')
                        ax.set_ylabel(r'$r_{z, ul}$')
                        ax.legend()
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(join(path_figs, 'id{}_fitted_sphere_rz-ul_by_rz-lr.png'.format(tid)))
                        elif show_figs:
                            plt.show()
                        plt.close()

                        # ---

                        # PLOT NORMALIZED DEFLECTION VS. PRESSURE
                        fig, ax = plt.subplots()
                        ax.scatter(df_rz.mean_pressure, df_rz.rz_ul_lr_norm, s=5)
                        ax.set_xlabel(r'$\overline{P} \: (Pa)$')
                        ax.set_ylabel(r'$r_{z, ul}/r_{z, lr}$')
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(join(path_figs, 'id{}_fitted_sphere_rz-ul-lr-norm_by_pressure.png'.format(tid)))
                        elif show_figs:
                            plt.show()
                        plt.close()

                        # ---

                        # SEPARATE PLOTS FOR +Z AND -Z
                        df_rz_neg = df_rz[df_rz.frame.isin(np.hstack([fr1, fr3, fr5]))]
                        df_rz_pos = df_rz[df_rz.frame.isin(np.hstack([fr2, fr4, fr6]))]
                        """if tid in ['1', '3']:
                            if tid == '1':
                                df_rz_neg = df_rz[df_rz.frame.isin(np.hstack([fr1, fr3, fr5]))]
                                df_rz_pos = df_rz[df_rz.frame.isin(np.hstack([fr2, fr4, fr6]))]
                            else:
                                df_rz_pos = df_rz[df_rz.frame.isin(np.hstack([fr1, fr3, fr5]))]
                                df_rz_neg = df_rz[df_rz.frame.isin(np.hstack([fr2, fr4, fr6]))]"""

                        fig, ax = plt.subplots()

                        # NEGATIVE PRESSURE: plot data and fit line
                        sc1, = ax.plot(df_rz_neg.rz_lr, df_rz_neg.rz_ul, 'o', ms=3, color=sciblue, label=r'$-P$')

                        popt1, pcov = curve_fit(functions.line, df_rz_neg.rz_lr, df_rz_neg.rz_ul)
                        fit_space = np.linspace(df_rz_neg.rz_lr.min(), df_rz_neg.rz_lr.max())
                        pf1, = ax.plot(fit_space, functions.line(fit_space, *popt1),
                                       linestyle='--', color=lighten_color(sc1.get_color(), 1.25),
                                       label=r'$-P=$' + ' {}'.format(np.round(popt1[0], 2)),
                                       )

                        # POSITIVE PRESSURE: plot data and fit line
                        sc2, = ax.plot(df_rz_pos.rz_lr, df_rz_pos.rz_ul, 'o', ms=3, color=scired, label=r'$+P$')

                        popt2, pcov = curve_fit(functions.line, df_rz_pos.rz_lr, df_rz_pos.rz_ul)
                        fit_space = np.linspace(df_rz_pos.rz_lr.min(), df_rz_pos.rz_lr.max())
                        pf2, = ax.plot(fit_space, functions.line(fit_space, *popt2),
                                       linestyle='--', color=lighten_color(sc2.get_color(), 1.25),
                                       label=r'$+P=$' + ' {}'.format(np.round(popt2[0], 2)),
                                       )

                        ax.set_xlabel(r'$r_{z, lr}$')
                        ax.set_ylabel(r'$r_{z, ul}$')
                        # ax.legend(title=r'$d r_{z, ul} / d r_{z, lr}$')

                        l = ax.legend([(sc1, pf1), (sc2, pf2)],
                                      [r'$-P=$' + ' {}'.format(np.round(popt1[0], 2)),
                                       r'$+P=$' + ' {}'.format(np.round(popt2[0], 2))
                                       ],
                                      numpoints=1,
                                      handler_map={tuple: HandlerTuple(ndivide=None)},
                                      title=r'$d r_{z, ul} / d r_{z, lr}$',
                                      )

                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(path_figs + '/id{}_fitted_sphere_rz-ul_by_rz-lr_pos-neg-P.png'.format(tid))
                        elif show_figs:
                            plt.show()
                        plt.close()

                # ---

            # ---

            # --------------------------------------------------------------------------------------------------------------

            # BREAK ----------------- END SURFACE RECONSTRUCTION PROCESSING

            # --------------------------------------------------------------------------------------------------------------

            # ---

            # --------------------------------------------------------------------------------------------------------------

            # BREAK ----------------- BEGIN POST-PROCESSING

            # --------------------------------------------------------------------------------------------------------------
            # --- PER-PARTICLE PRECISION
            """ 
            This function set includes:
                    1. Per-particle precision by fitting 12th order poly to static and dynamic frames
                    2. 2D binning of axial and radial displacements
                    3. Fitting a function to model the radial displacement as a function of radial position.
                    4. Quiver plot
                    5. Surface plot via interpolation.
            """

            if analyze_per_particle:

                # setup file paths
                # tid = 1
                # for tid in [1, 2, 3, 4]:
                path_results = join(base_dir, 'results', 'dz{}'.format(tid))
                path_figs = join(base_dir, 'figs', 'dz{}'.format(tid))

                # ---

                # read particle ID's per membrane
                df_memb_pids = pd.read_excel(path_results + '/df_pids_per_membrane.xlsx', index_col=0)
                pids_lr = df_memb_pids.loc['lr', :].dropna().values
                pids_ul = df_memb_pids.loc['ul', :].dropna().values
                pids_lrul = np.concatenate((pids_lr, pids_ul))

                # -

                # read test coords dataframe
                df_raw = pd.read_excel(path_results + '/test_coords_id{}_corrected.xlsx'.format(tid), index_col=0)

                # read reconstruction dataframe
                df_reconstruction = pd.read_excel(path_results + '/df_reconstruction.xlsx')

                # -

                for df_original, filtering in zip([df_reconstruction, df_raw], ['reconstruction', 'raw']):

                    # processing: "static" == before start time; "dynamic" == after start time
                    dfi = df_original[df_original['t'] <= tid_1_start_time]
                    dff = df_original[df_original['t'] > tid_1_start_time]

                    # -

                    # file paths
                    path_filtering = join(path_results, filtering)
                    if not os.path.exists(path_filtering):
                        os.makedirs(path_filtering)

                    # -

                    # for loop

                    for df, time_zone in zip([dff, dfi], ['dynamic', 'static']):  #

                        if time_zone == 'static':
                            continue

                        # file paths
                        path_timezone = join(path_filtering, time_zone)
                        if not os.path.exists(path_timezone):
                            os.makedirs(path_timezone)

                        path_results_pp = join(path_timezone, 'per-particle')
                        if not os.path.exists(path_results_pp):
                            os.makedirs(path_results_pp)

                        path_results_precision = join(path_timezone, 'precision')
                        if not os.path.exists(path_results_precision):
                            os.makedirs(path_results_precision)

                        path_results_precision_figs = join(path_results_precision, 'figs')
                        if not os.path.exists(path_results_precision_figs):
                            os.makedirs(path_results_precision_figs)

                        path_figs_2d = path_timezone + '/bin-2d'
                        if not os.path.exists(path_figs_2d):
                            os.makedirs(path_figs_2d)

                        path_results_2d = path_timezone + '/bin-2d/results'
                        if not os.path.exists(path_results_2d):
                            os.makedirs(path_results_2d)

                        path_figs_quiver = path_timezone + '/quiver'
                        if not os.path.exists(path_figs_quiver):
                            os.makedirs(path_figs_quiver)

                        # ---

                        # df = df[~df.id.isin([39, 71])]
                        dfo = df.copy()

                        # get particles in dataframe
                        pids = sorted(df.id.unique())

                        # -

                        # --- PER PARTICLE

                        analyze_per_particle = False
                        if analyze_per_particle is True and time_zone == 'dynamic':

                            # ---

                            for pid in pids_lrul:

                                # get dataframe
                                dfpid = df[df['id'] == pid]

                                if len(dfpid) < 12:
                                    continue

                                # ---

                                # A. PLOT ALL
                                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.2,
                                                                                              size_y_inches * 1.05),
                                                               gridspec_kw={'height_ratios': [1, 1]})

                                ax1.scatter(dfpid.t, dfpid.z_corr - dfpid.iloc[0].z_corr,
                                            c=dfpid.drr, cmap='coolwarm', vmin=-7.5, vmax=7.5,
                                            s=3, marker='o',
                                            )
                                ax2.scatter(dfpid.t, dfpid.drr * microns_per_pixel,
                                            c=dfpid.z_corr, cmap='coolwarm', vmin=-150, vmax=150,
                                            s=3, marker='o',
                                            )

                                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                                ax2.set_ylabel(r'$\Delta r^{\delta} \: (\mu m)$')
                                ax2.set_xlabel(r'$t \: (s)$')

                                plt.tight_layout()
                                plt.savefig(path_results_pp + '/pid{}_plot-dz-dr.svg'.format(pid))
                                plt.close()

                                # ---

                        # ---

                        # --- PRECISION

                        analyze_precision = False
                        if analyze_precision:

                            # modifiers
                            poly_deg = 12
                            save_figs = False
                            show_figs = False

                            # ---

                            # setup
                            column_to_fit = 't'
                            precision_columns = ['z_corr', 'drm']
                            scale_columns_units = [1, microns_per_pixel]
                            precision_data = []
                            precision_data_columns = ['tid', 'id', 'precision_z', 'precision_drm_microns']

                            # ---

                            for pid in pids:

                                # get dataframe
                                dfpid = df[df['id'] == pid]

                                if len(dfpid) < 12:
                                    continue

                                # list of results
                                pid_precision = [tid, pid]

                                for pc, scale_units in zip(precision_columns, scale_columns_units):
                                    x = dfpid[column_to_fit].to_numpy()
                                    y = dfpid[pc].to_numpy() * scale_units

                                    # fit polynomial
                                    pc_precision = analyze.evaluate_precision_from_polyfit(x,
                                                                                           y,
                                                                                           poly_deg,
                                                                                           )
                                    # store this precisions
                                    pid_precision.extend([pc_precision])

                                    # plot
                                    if pid in pids_lr:  # save_figs or show_figs:

                                        # ----------------------------------------------------------------------------------------------
                                        # NOTE - this is duplicate of the above function; only here to get polynomial coefficients
                                        # Could modify function but not worth the time right now
                                        pcoeff, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg=poly_deg,
                                                                                                     full=True)
                                        pf = np.poly1d(pcoeff)
                                        y_model = pf(x)
                                        y_residuals = y_model - y
                                        y_precision = np.mean(np.std(y_residuals))
                                        y_rmse = np.sqrt(np.mean(y_residuals ** 2))
                                        # ----------------------------------------------------------------------------------------------

                                        # resample fit space
                                        y_fit = np.linspace(x.min(), x.max(), len(y) * 10)

                                        # plot
                                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches / 2))

                                        ax.scatter(x, y, s=2, label=r'$p_{ID}$' + ': {}'.format(pid))
                                        ax.plot(y_fit, pf(y_fit),
                                                linestyle='--', color='black', alpha=0.5,
                                                label=r'$\overline{\sigma}_{y}=$' + '{}'.format(np.round(y_precision, 2)))

                                        ax.set_xlabel(r'$x \: (s)$')
                                        ax.set_ylabel(r'$y \: (\mu m)$')
                                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                                        plt.tight_layout()
                                        if save_figs:
                                            plt.savefig(
                                                path_results_precision_figs +
                                                '/id{}_pid{}_precision-{}_by_{}_equals_{}.png'.format(tid,
                                                                                                      pid,
                                                                                                      pc,
                                                                                                      column_to_fit,
                                                                                                      np.round(y_precision, 2)),
                                            )
                                        elif show_figs:
                                            plt.show()
                                        plt.close()

                                    # ---

                                # store all precisions
                                precision_data.append(pid_precision)

                            # ---

                            # export per-particle precision
                            dfp = pd.DataFrame(np.array(precision_data), columns=precision_data_columns)
                            dfp.to_excel(path_results_precision + '/precision_all-particles-and-frames.xlsx', index=False)

                            # export mean precision
                            dfpg = dfp.groupby('tid').mean()
                            dfpg.to_excel(path_results_precision + '/mean-precision_all-particles-and-frames.xlsx',
                                          index_label='tid')

                            # ---

                        # ---

                        # --- 2D BINNING

                        analyze_2d = False
                        if analyze_2d:
                            plot_lr = False
                            plot_ul = False
                            plot_quiver = False

                            # -

                            # 1. PLOT LOWER RIGHT MEMBRANE
                            if plot_lr:

                                export_dfb_2d = True

                                # get dataframe of particles on the lower right membrane
                                df = df[df.id.isin(pids_lr)]

                                # 2d binning
                                columns_to_bin = ['frame', 'rr0']
                                column_to_count = 'id'
                                bin_frames = df.frame.unique()
                                nd_r_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                                d_r_lbls = [0, 160, 320, 480, 640, 800]
                                bin_r = np.round(np.array(nd_r_bins) * r_edge_lr, 0)
                                # np.round(np.linspace(30, r_edge_lr, 5), 0)  # 7; [42.0, 125.0, 167.0, 208.0]
                                bins = [bin_frames, bin_r]
                                round_to_decimals = [0, 1]
                                min_num_bin = 1
                                return_groupby = True

                                dfm, dfstd = bin.bin_generic_2d(df,
                                                                columns_to_bin,
                                                                column_to_count,
                                                                bins,
                                                                round_to_decimals,
                                                                min_num_bin,
                                                                return_groupby
                                                                )

                                dfm = dfm.sort_values(['bin_tl', 'bin_ll'])
                                dfstd = dfstd.sort_values(['bin_tl', 'bin_ll'])

                                if export_dfb_2d:
                                    dfm.to_excel(path_results_2d + '/2d-bin_lr_{}bins.xlsx'.format(len(bin_r)))

                                # ---

                                # A. PLOT DIMENSIONAL Z- AND R-DISPLACEMENT BY TIME

                                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.2,
                                                                                              size_y_inches * 1.05),
                                                               gridspec_kw={'height_ratios': [1, 1]})

                                for br, br_lbl, clr, zo in zip(bin_r, d_r_lbls,
                                                               cm.plasma(np.linspace(0.95, 0.15, len(bin_r))),
                                                               [3.1, 3.2, 3.3, 3.4, 3.3, 3.2, 3.2, 3.1, 3.1, 3.1, 3.1]
                                                               ):
                                    dfbr = dfm[dfm['bin_ll'] == br]

                                    if len(dfbr) == 0:
                                        continue

                                    ax1.plot(dfbr.t, dfbr.z_corr + 10, 'o', ms=1,
                                             color=clr, zorder=zo, label=br_lbl)
                                    ax2.plot(dfbr.t, dfbr.drr * microns_per_pixel, 'o', ms=1, lw=0.75,
                                             color=clr, zorder=zo)

                                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1,
                                           markerscale=2, title=r'$r_{bin} \: (\mu m)$')

                                ax2.set_ylabel(r'$\Delta r^{\delta} \: (\mu m)$')
                                # ax2.set_ylim([-4.95, 7.25])
                                # ax2.set_yticks([-3, 0, 3, 6])
                                ax2.set_xlabel(r'$t \: (s)$')

                                plt.tight_layout()
                                plt.savefig(
                                    path_figs_2d + '/2b-bin-frame-rr_lr_plot-dz-dr_num-bins={}_plasma2.svg'.format(len(bin_r)))
                                plt.show()
                                plt.close()

                                # ---

                                # B. PLOT NON-DIMENSIONAL Z- AND R-DISPLACEMENT BY TIME

                                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.2,
                                                                                              size_y_inches * 1.05),
                                                               gridspec_kw={'height_ratios': [1, 1]})

                                for br, clr, zo in zip(bin_r,
                                                       cm.plasma(np.linspace(0.95, 0.15, len(bin_r))),
                                                       [3.1, 3.2, 3.3, 3.4, 3.3, 3.2, 3.2, 3.1, 3.1, 3.1, 3.1]
                                                       ):
                                    dfbr = dfm[dfm['bin_ll'] == br]

                                    if len(dfbr) == 0:
                                        continue

                                    ax1.plot(dfbr.t, (dfbr.z_corr + 10) / (t_membrane * 1e6), 'o', ms=1,
                                             color=clr, zorder=zo, label=np.round(br / r_edge_lr, 2))
                                    ax2.plot(dfbr.t, dfbr.drr * microns_per_pixel / (t_membrane * 1e6), 'o', ms=1, lw=0.75,
                                             color=clr, zorder=zo)

                                # ax1.axhline(y=0, linewidth=0.5, color='black')
                                ax1.set_ylabel(r'$\Delta z^{\delta}/h$')
                                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), ncol=1,
                                           markerscale=2, title=r'$r_{bin}/r$')

                                ax2.set_ylabel(r'$\Delta r^{\delta}/h$')
                                # ax2.set_ylim([-0.2, 0.375])
                                # ax2.set_yticks([-0.1, 0, 0.1, 0.2, 0.3])
                                ax2.set_xlabel(r'$t \: (s)$')

                                plt.tight_layout()
                                plt.savefig(
                                    path_figs_2d + '/2b-bin-frame-rr_lr_plot-norm-dz-dr_num-bins={}_plasma2.svg'.format(
                                        len(bin_r)))
                                plt.show()
                                plt.close()

                                # --

                                # ---

                                # A. PLOT ALL
                                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.2,
                                                                                              size_y_inches * 1.05),
                                                               gridspec_kw={'height_ratios': [1, 1]})

                                ax1.scatter(df.t, df.z_corr + 10, c=df.rr0, s=1, marker='o')
                                ax2.scatter(df.t, df.drr * microns_per_pixel, c=df.rr0, s=1, marker='o')

                                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                                ax2.set_ylabel(r'$\Delta r^{\delta} \: (\mu m)$')
                                ax2.set_xlabel(r'$t \: (s)$')

                                plt.tight_layout()
                                plt.savefig(path_figs_2d + '/2b-frame-rr_lr_plot-dz-dr_plasma2.svg')
                                plt.show()
                                plt.close()

                                # ---

                            # ---

                            # 2. PLOT UPPER LEFT MEMBRANE
                            if plot_ul:

                                export_dfb_2d = True

                                # get dataframe of particles on the lower right membrane
                                df = dfo[dfo.id.isin(pids_ul)]

                                columns_to_bin = ['frame', 'rr0']
                                column_to_count = 'id'
                                bin_frames = df.frame.unique()
                                nd_r_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                                d_r_lbls = [0, 100, 200, 300, 400, 500]
                                bin_r = np.round(np.array(nd_r_bins) * r_edge_ul, 0)
                                bins = [bin_frames, bin_r]
                                round_to_decimals = [0, 1]
                                min_num_bin = 1
                                return_groupby = True

                                dfm, dfstd = bin.bin_generic_2d(df,
                                                                columns_to_bin,
                                                                column_to_count,
                                                                bins,
                                                                round_to_decimals,
                                                                min_num_bin,
                                                                return_groupby
                                                                )

                                dfm = dfm.sort_values(['bin_tl', 'bin_ll'])
                                dfstd = dfstd.sort_values(['bin_tl', 'bin_ll'])

                                if export_dfb_2d:
                                    dfm.to_excel(path_results_2d + '/2d-bin_ul_{}bins.xlsx'.format(len(bin_r)))

                                # ---

                                # A. PLOT DIMENSIONAL Z- AND R-DISPLACEMENT BY TIME

                                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                                               figsize=(size_x_inches * 1.2, size_y_inches * 1),
                                                               gridspec_kw={'height_ratios': [1, 1]})

                                for br, br_lbl, clr, zo in zip(bin_r, d_r_lbls,
                                                               cm.plasma(np.linspace(0.95, 0.15, len(bin_r))),
                                                               [3.1, 3.2, 3.3, 3.4, 3.2, 3.2, 3.1, 3.1]
                                                               ):
                                    dfbr = dfm[dfm['bin_ll'] == br]

                                    if len(dfbr) == 0:
                                        continue

                                    ax1.plot(dfbr.t, dfbr.z_corr + 10, 'o', ms=1,
                                             color=clr, zorder=zo, label=br_lbl)
                                    ax2.plot(dfbr.t, dfbr.drr * microns_per_pixel, 'o', ms=1,
                                             color=clr, zorder=zo)

                                ax1.set_ylabel(r'$\Delta z^{\delta} \: (\mu m)$')
                                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), markerscale=2,
                                           title=r'$r_{bin} \: (\mu m)$')

                                ax2.set_ylabel(r'$\Delta r^{\delta} \: (\mu m)$')
                                # ax2.set_ylim([-2.2, 4.2])
                                # ax2.set_yticks([-2, 0, 2, 4])
                                ax2.set_xlabel(r'$t \: (s)$')

                                plt.tight_layout()
                                plt.savefig(
                                    path_figs_2d + '/2b-bin-frame-rr_ul_plot-dz-dr_num-bins={}_plasma2.svg'.format(len(bin_r)))
                                plt.show()
                                plt.close()

                                # ---

                                # B. PLOT NON-DIMENSIONAL Z- AND R-DISPLACEMENT BY TIME

                                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                                               figsize=(size_x_inches * 1.2, size_y_inches * 1),
                                                               gridspec_kw={'height_ratios': [1, 1]})

                                for br, clr, zo in zip(bin_r,
                                                       cm.plasma(np.linspace(0.95, 0.15, len(bin_r))),
                                                       [3.1, 3.2, 3.3, 3.4, 3.2, 3.2, 3.1, 3.1]
                                                       ):
                                    dfbr = dfm[dfm['bin_ll'] == br]

                                    if len(dfbr) == 0:
                                        continue

                                    ax1.plot(dfbr.t, (dfbr.z_corr + 10) / (t_membrane * 1e6), 'o', ms=1,
                                             color=clr, zorder=zo, label=np.round(br / r_edge_ul, 1))
                                    ax2.plot(dfbr.t, dfbr.drr * microns_per_pixel / (t_membrane * 1e6), 'o', ms=1,
                                             color=clr, zorder=zo)

                                ax1.set_ylabel(r'$\Delta z^{\delta}/h$')
                                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.05), markerscale=2, title=r'$r_{bin}/r$')

                                ax2.set_ylabel(r'$\Delta r^{\delta}/h$')
                                # ax2.set_ylim([-0.16, 0.22])
                                # ax2.set_yticks([-0.1, 0, 0.1, 0.2])
                                ax2.set_xlabel(r'$t \: (s)$')

                                plt.tight_layout()
                                plt.savefig(
                                    path_figs_2d + '/2b-bin-frame-rr_ul_plot-norm-dz-dr_num-bins={}_plasma2.svg'.format(
                                        len(bin_r)))
                                plt.show()
                                plt.close()

                                # ---

                            # ---

                            # 1. PLOT 2D surface map
                            if plot_quiver:
                                import numpy.ma as ma

                                # frames of interest
                                # frois = np.arange(40, 200)
                                frois = np.array([109, 112, 114, 116,
                                                  132, 134, 136, 138, 140, 142,
                                                  161, 163, 166, 168, 170, 172, 174,
                                                  180, 182, 185, 188, 192, 196, 199]
                                                 )  # + frame_offset

                                for froi in frois:
                                    dfr = dfo[dfo['frame'] == froi]

                                    x = dfr.xxm.to_numpy()  # * microns_per_pixel
                                    y = dfr.yym.to_numpy()  # * microns_per_pixel
                                    dxm = dfr.dxm.to_numpy() * microns_per_pixel
                                    dym = dfr.dym.to_numpy() * microns_per_pixel * -1
                                    drm = dfr.drm.to_numpy() * microns_per_pixel

                                    rx = dfr.rx.to_numpy()  # * microns_per_pixel
                                    ry = dfr.ry.to_numpy()  # * microns_per_pixel
                                    drr = dfr.drr.to_numpy() * microns_per_pixel

                                    X = np.linspace(0, 512, 512 + 1)
                                    Y = np.linspace(0, 512, 512 + 1)
                                    X, Y = np.meshgrid(X, Y)
                                    # img = X.copy()
                                    # img += boundary.draw_boundary_circle_perimeter(img, xc=xc_lr, yc=yc_lr, r=r_edge_lr)

                                    U = np.zeros_like(X)
                                    V = np.zeros_like(Y)
                                    UV = np.zeros_like(X)
                                    for px, py, pdx, pdy, pdr in zip(x, y, dxm, dym, drm):
                                        U[int(np.round(py, 0)), int(np.round(px, 0))] = pdx
                                        V[int(np.round(py, 0)), int(np.round(px, 0))] = pdy
                                        UV[int(np.round(py, 0)), int(np.round(px, 0))] = pdr

                                    # mask zeros
                                    mUV = ma.masked_equal(UV, 0)
                                    mU = ma.array(U, mask=mUV.mask)
                                    mV = ma.array(V, mask=mUV.mask)

                                    fig, ax = plt.subplots()
                                    q = ax.quiver(X, Y, mU, mV, angles='xy', scale_units='xy', scale=1 / microns_per_pixel)
                                    ax.quiverkey(q, X=0.85, Y=1.05, U=10, label=r'$10 \mu m$', labelpos='E', labelsep=0.05)
                                    ax.scatter(x, y, s=1, color='red', alpha=0.1)

                                    ax.set_aspect('equal', adjustable='box')
                                    ax.set_xlabel('x (pixels)')
                                    ax.set_ylabel('y (pixels)')
                                    ax.set_title('Frame = {}'.format(froi))
                                    ax.invert_yaxis()
                                    plt.tight_layout()
                                    plt.savefig(path_figs_quiver + '/quiver_frame{}.png'.format(froi), dpi=300)
                                    # plt.show()
                                    plt.close()

                            # -

                        # ---

                        # --- COMPARE FRAMES (i.e., RADIAL DISPLACEMENT AS A FUNCTION OF AXIAL DISPLACEMENT)

                        compare_frames = False
                        if compare_frames is True and time_zone == 'dynamic':

                            # --- PART 1: GET FRAMES WHERE DEFLECTION IS WITHIN DZ-BINS

                            # ---

                            path_figs_compare_frames = path_figs + '/compare-frames'
                            if not os.path.exists(path_figs_compare_frames):
                                os.makedirs(path_figs_compare_frames)

                            # ---

                            # ---

                            # CHOOSE SPECIFIC FRAMES
                            groupby_frame = True
                            if groupby_frame:

                                # fitting functions
                                def sine_decay(x, A, f, b):
                                    """
                                    :param A: amplitude
                                    :param f: frequency - the number of cycles per second; bounds = (0, 1)
                                    :param b: decay rate
                                    """
                                    return A * np.sin(2 * np.pi * f * x) * np.exp(-x * b) * (1 - x ** 12)


                                # 1. PLOT FRAMES WITH POSITIVE DEFLECTION
                                plot_pos = True
                                if plot_pos:

                                    # get particles on the lower right membrane
                                    df = dfo[dfo.id.isin(pids_lr)]

                                    # OPTIONAL - filters
                                    # df = df[df['rr'] < r_edge_lr * .975]
                                    # df = df[df['drm'] * microns_per_pixel / (t_membrane * 1e6) > -0.01]

                                    # ---

                                    # A. GET ALL PARTICLES IN ALL FRAMES OF INTEREST AND ORGANIZE INTO A SINGLE DATAFRAME PER DZ-BIN

                                    # frames of interest
                                    toises = [
                                        [1, 1.1]]  # [[1.9, 2.45], [3.2, 3.8], [4.25, 4.6], [5.4, 6.0], [6.6, 7.2], [7.7, 8.2]]

                                    for tois in toises:
                                        pass
                                        # dftois = df[(df['t'] > tois[0]) & (df['t'] < tois[1])]
                                        # frois = dftois.frame.unique()

                                        frois = [43, 45, 47,
                                                 52]  # [185, 188, 192, 196] # [101, 104, 107, 112]  # [161, 163, 166, 170]  #
                                        dfdzs = []
                                        for froi in frois:
                                            dfdzs.append(df[df['frame'] == froi])

                                        # B. PLOT NON-DIMENSIONAL Z-R-DISPLACEMENT BY NON-DIMENSIONAL RADIAL POSITION
                                        r_edge_extend = 1.00625  # 1.025 = 20 microns
                                        r_edge_f = r_edge_lr * r_edge_extend
                                        r_edge_outer_microns = np.round((r_edge_extend - 1) * r_edge_lr * microns_per_pixel, 1)
                                        r_num_bins = 12

                                        column_to_bin = 'rr0'
                                        column_to_count = 'id'
                                        bin_r = np.round(np.linspace(0, r_edge_f, r_num_bins), 0)
                                        round_to_decimal = 1
                                        return_groupby = True

                                        fit_curve = True
                                        n_points = 150
                                        r_fit = np.linspace(0, 1, n_points)
                                        r_fit_extend = np.linspace(0, r_edge_extend, n_points)

                                        # --- plot

                                        # multi plot
                                        # n_plots = len(frois)
                                        # fig, axes = plt.subplots(nrows=n_plots, sharex=True, figsize=(size_x_inches * 1.15, size_y_inches * n_plots / 2))

                                        # plot on same figure
                                        fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches))
                                        markers = ['o', 's', 'd', '*', 'p', 'D', '+', 'x', '^', 's', 'd', '*', 'p', 'D', '+',
                                                   'x']

                                        for dfr, froi, clr1, clr2, mrk in zip(dfdzs, frois,
                                                                              cm.plasma(np.linspace(0.95, 0.15, len(frois))),
                                                                              cm.viridis(np.linspace(0.1, 0.95, len(frois))),
                                                                              markers):
                                            dfr = dfr.sort_values('rr0')

                                            # plot group #1 - all particles "raw" positions
                                            plot_all = True
                                            if plot_all:
                                                x = dfr.rr.to_numpy() / r_edge_f
                                                y = dfr.drr.to_numpy() * microns_per_pixel / (t_membrane * 1e6)
                                                ax.scatter(x, y, s=1,
                                                           marker=mrk, color=clr1, alpha=0.5,
                                                           label='{} (n={})'.format(froi, len(dfr)))

                                                if fit_curve:
                                                    guess = [1, 0.5, 1]
                                                    bounds = ([1.3, 0.8, 0], [2, 5, 5])  # , bounds=bounds
                                                    try:
                                                        popt, pcov = curve_fit(sine_decay, x, y,
                                                                               p0=guess, xtol=1.49012e-07, maxfev=1000)
                                                        y_fit = sine_decay(r_fit, *popt)
                                                        ax.plot(r_fit_extend, y_fit, color=clr1, linewidth=0.75)

                                                    except RuntimeError:
                                                        pass

                                                # ---

                                            # ---

                                            # plot group #2 - groupby positions
                                            plot_grouped = True
                                            if plot_grouped:
                                                dfm, dfstd = bin.bin_generic(dfr,
                                                                             column_to_bin,
                                                                             column_to_count,
                                                                             bin_r,
                                                                             round_to_decimal,
                                                                             return_groupby,
                                                                             )

                                                # dfm = dfm[dfm.rr / r_edge_f > 0.25]

                                                x = dfm.rr.to_numpy() / r_edge_f
                                                y = dfm.drr.to_numpy() * microns_per_pixel / (t_membrane * 1e6)
                                                ax.scatter(x, y, s=10, marker=mrk, color=clr2, alpha=1, zorder=3.5,
                                                           label='{} s'.format(np.round(froi / frame_rate, 2), r_num_bins))

                                                if fit_curve:
                                                    guess = [1.0, 0.5, 2]
                                                    bounds = ([1.3, 0.8, 0], [2, 5, 5])  # , bounds=bounds
                                                    try:
                                                        popt, pcov = curve_fit(sine_decay, x, y,
                                                                               p0=guess, xtol=1.49012e-06, maxfev=1200)
                                                        y_fit = sine_decay(r_fit, *popt)
                                                        ax.plot(r_fit_extend, y_fit,
                                                                color=clr2, linewidth=0.75, alpha=0.75, zorder=3.2)
                                                        """label='A={}'.format(np.round(popt[0], 4)) + '\n' +
                                                              r'$f$' + '={}'.format(np.round(popt[1], 4)) + '\n' +
                                                              r'$\beta$' + '={}'.format(np.round(popt[2], 2))
                                                        )"""

                                                    except RuntimeError:
                                                        pass

                                                # -

                                            ax.set_ylabel(r'$\Delta r / h$')
                                            if mrk == 'o':
                                                ax.legend(title=r'$time \: (s)$',
                                                          markerscale=1.5, labelspacing=0.35, handletextpad=0.15,
                                                          borderaxespad=0.25)
                                            else:
                                                ax.legend(title=r'$t \: (s)$',
                                                          markerscale=1.5, labelspacing=0.35, handletextpad=0.15,
                                                          borderaxespad=0.25)

                                        # -

                                        ax.set_xlabel(r'$r / a$')
                                        ax.set_xlim([-0.025, r_edge_extend + 0.025])
                                        plt.tight_layout()
                                        plt.savefig(path_figs_compare_frames +
                                                    '/by-frame-{}_pos_lr_norm-drm_by_norm-r_{}bins_r-extend={}microns'
                                                    '.svg'.format(froi, r_num_bins, r_edge_outer_microns))
                                        plt.show()
                                        plt.close()

                                        # ---

                                    # ---

                                # ---

                                # 2. PLOT SELECT FRAMES: NORMALIZED AXIAL AND RADIAL DISPLACEMENT
                                plot_dz_dr = True
                                if plot_dz_dr:

                                    # axial deflections for each frame
                                    df_rzs = pd.read_excel(
                                        path_results + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(tid))

                                    # initialize plate theory expression
                                    fND_lr = fNonDimensionalNonlinearSphericalUniformLoad(
                                        r=r_edge_lr * microns_per_pixel * 1e-6,
                                        h=t_membrane,
                                        youngs_modulus=E_silpuran,
                                        poisson=poisson)

                                    # get particles on the lower right membrane
                                    df = dfo[dfo.id.isin(pids_lr)]

                                    # ---

                                    # A. GET ALL PARTICLES IN ALL FRAMES OF INTEREST AND ORGANIZE INTO A SINGLE DATAFRAME PER DZ-BIN

                                    # frames of interest
                                    # frois = [185, 188, 192, 196]  # [101, 104, 107, 112]  # [161, 163, 166, 170]  # [43, 45, 47, 52]  #
                                    dfdrzs = []
                                    dfdzs = []
                                    for froi in frois:
                                        dfdrzs.append(df_rzs[df_rzs['frame'] == froi])
                                        dfdzs.append(df[df['frame'] == froi])

                                    # B. PLOT NON-DIMENSIONAL Z-R-DISPLACEMENT BY NON-DIMENSIONAL RADIAL POSITION
                                    r_edge_extend = 1.00625  # 1.025 = 20 microns
                                    r_edge_f = r_edge_lr * r_edge_extend
                                    r_edge_outer_microns = np.round((r_edge_extend - 1) * r_edge_lr * microns_per_pixel, 1)
                                    r_num_bins = 12

                                    column_to_bin = 'rr0'
                                    column_to_count = 'id'
                                    bin_r = np.round(np.linspace(0, r_edge_f, r_num_bins), 0)
                                    round_to_decimal = 1
                                    return_groupby = True

                                    fit_curve = True
                                    n_points = 150
                                    r_fit = np.linspace(0, 1, n_points)
                                    r_fit_extend = np.linspace(0, r_edge_extend, n_points)

                                    # --- plot

                                    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                                                   figsize=(size_x_inches * 1, size_y_inches * 1.25),
                                                                   gridspec_kw={'height_ratios': [1, 1]})
                                    markers = ['s', 'o', 'd', '*', 'p', 'D', '+', 'x', '^', 's', 'd', '*', 'p', 'D', '+', 'x']
                                    ss = 15

                                    for dfr, dfz, froi, clr1, clr2, clrmod, mrk in zip(dfdzs, dfdrzs, frois,
                                                                                       cm.plasma(
                                                                                           np.linspace(0.95, 0.15, len(frois))),
                                                                                       cm.magma(
                                                                                           np.linspace(0.1, 0.875, len(frois))),
                                                                                       [0.95, 0.85, 1.25, 1.4],
                                                                                       markers):
                                        dfr = dfr.sort_values('rr0')

                                        # ---

                                        # group by # 1 (removing focal plane bias errors)
                                        z_offset = -10
                                        dfr_fpb = dfr[dfr['z_corr'] < z_offset].copy()

                                        dfm_fpb, dfstd_fpb = bin.bin_generic(dfr_fpb,
                                                                             column_to_bin,
                                                                             column_to_count,
                                                                             bin_r,
                                                                             round_to_decimal,
                                                                             return_groupby,
                                                                             )

                                        # plot #1 - axial displacement

                                        # raw
                                        xr = dfr_fpb.rr.to_numpy() / r_edge_f
                                        yr = (dfr_fpb.z_corr.to_numpy() - z_offset) / (t_membrane * 1e6)
                                        # ax1.scatter(xr, yr, s=2, marker=mrk, color=clr2, alpha=0.25)

                                        # fit
                                        d_p0 = dfz.iloc[0]['fit_lr_pressure']
                                        d_n0 = dfz.iloc[0]['fit_lr_pretension']
                                        nd_P, nd_k = fND_lr.non_dimensionalize_p_k(d_p0, d_n0)
                                        yrf = fND_lr.nd_nonlinear_clamped_plate_p_k(r_fit, nd_P, nd_k)
                                        p1, = ax1.plot(r_fit, yrf, zorder=3.2,
                                                       color=lighten_color(clr2, clrmod), linewidth=0.75)

                                        x = dfm_fpb.rr.to_numpy() / r_edge_f
                                        y = (dfm_fpb.z_corr.to_numpy() - z_offset) / (t_membrane * 1e6)
                                        ax1.scatter(x, y, zorder=3.5,
                                                    s=ss, marker=mrk, color=clr2, label=np.round(froi / frame_rate, 2))

                                        # ---

                                        # group by #2 (including all points)
                                        dfm, dfstd = bin.bin_generic(dfr,
                                                                     column_to_bin,
                                                                     column_to_count,
                                                                     bin_r,
                                                                     round_to_decimal,
                                                                     return_groupby,
                                                                     )

                                        # plot #2 - radial displacement
                                        # dfm = dfm[dfm.rr / r_edge_f > 0.25]

                                        x = dfm.rr.to_numpy() / r_edge_f
                                        y = dfm.drr.to_numpy() * microns_per_pixel / (t_membrane * 1e6)
                                        ax2.scatter(x, y, zorder=3.5,
                                                    s=ss, marker=mrk, color=clr2)

                                        if fit_curve:
                                            guess = [1.0, 0.5, 2]
                                            try:
                                                popt, pcov = curve_fit(sine_decay, x, y, p0=guess, xtol=1.49012e-06,
                                                                       maxfev=1200)
                                                y_fit = sine_decay(r_fit, *popt)
                                                ax2.plot(r_fit_extend, y_fit, zorder=3.2,
                                                         color=lighten_color(clr2, clrmod), linewidth=0.75)

                                            except RuntimeError:
                                                pass

                                    # -

                                    ax1.set_ylabel(r'$\Delta z / h$')
                                    # ax1.set_ylim(top=7.99)
                                    ax1.legend(title=r'$t \: (s)$',
                                               markerscale=1, labelspacing=0.3, handletextpad=0.1,
                                               borderaxespad=0.2)  # loc='upper right',

                                    ax2.set_xlabel(r'$r / a$')
                                    ax2.set_xlim([-0.025, r_edge_extend + 0.025])
                                    ax2.set_ylabel(r'$\Delta r / h$')

                                    plt.tight_layout()
                                    plt.savefig(path_figs_compare_frames +
                                                '/by-frame-{}_lr_norm-drz-drm_by_norm-r_{}bins_r-extend={}microns3'
                                                '.svg'.format(froi, r_num_bins, r_edge_outer_microns))
                                    plt.show()
                                    plt.close()

                                    # ---

                                # ---

                            # ---

                    # ---

                # ---

                # ---

            # ---

            # --------------------------------------------------------------------------------------------------------------
            # --- COLLECTION UNCERTAINTY + ERROR ANALYSIS

            if analyze_uncertainty:

                dz_id = tid
                path_results = base_dir + '/results/dz{}'.format(dz_id)

                path_results_uncertainty = join(path_results, 'uncertainty')
                if not os.path.exists(path_results_uncertainty):
                    os.makedirs(path_results_uncertainty)

                # read
                fp = '/id{}_z_data_and_fit.xlsx'.format(dz_id)
                df = pd.read_excel(path_results + fp)
                dfo = df.copy()

                # -

                # rmse-z by z
                plot_rmse_by_z = True
                if plot_rmse_by_z:

                    save_figs = True

                    # create necessary columns
                    df['z_true'] = df['z_fit']
                    df['cm'] = 1.0

                    # compute mean
                    dfm = bin.bin_local_rmse_z(df, column_to_bin='z_true', bins=1, min_cm=0.5, z_range=None,
                                               round_to_decimal=2,
                                               df_ground_truth=None, dropna=True, error_column=None, include_xy=False)
                    dfm.to_excel(path_results_uncertainty + '/mean-per-particle_rmse.xlsx')

                    # -

                    # bin
                    z_bins = np.arange(-120, 121, 20)  # 20
                    dfb = bin.bin_local_rmse_z(df, column_to_bin='z_true', bins=z_bins, min_cm=0.5, z_range=None,
                                               round_to_decimal=2,
                                               df_ground_truth=None, dropna=True, error_column=None, include_xy=False)

                    # export
                    dfb.to_excel(path_results_uncertainty + '/local-per-particle_rmse-z_{}bins.xlsx'.format(len(z_bins)))

                    # ---

                    # setup
                    ms = 4
                    ylim_rmse = [0, 10.125]
                    ylim_error = [-28.5, 28.5]

                    # ---

                    # plot
                    fig, ax = plt.subplots()

                    ax.plot(dfb.index, dfb.rmse_z, '-o', ms=ms)
                    ax.fill_between([-depth_of_focus / 2, depth_of_focus / 2], ylim_rmse[0], ylim_rmse[1],
                                    color='red', ec='none', alpha=0.1, label='D.o.F.')

                    ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                    ax.set_ylim(ylim_rmse)
                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.legend()
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_results_uncertainty + '/rmse_z_by_z_with_DoF.svg')
                    plt.show()
                    plt.close()

                    # ---

                    # plot
                    fig, ax = plt.subplots()

                    ax.plot(df.z_fit, df.error, 'o', ms=ms / 5, alpha=0.4)
                    ax.fill_between([-depth_of_focus / 2, depth_of_focus / 2], ylim_error[0], ylim_error[1],
                                    color='red', ec='none', alpha=0.1, label='D.o.F.')

                    ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                    ax.set_ylim(ylim_error)
                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.legend()
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_results_uncertainty + '/error_z_by_z_with_DoF.svg')
                    plt.show()
                    plt.close()

                    # ---

                    # ---

                    # bin
                    df_exclude_dof = df[(df['z_true'] < -depth_of_focus / 2) | (df['z_true'] > depth_of_focus / 2)]
                    dfbex = bin.bin_local_rmse_z(df_exclude_dof, column_to_bin='z_true', bins=z_bins, min_cm=0.5, z_range=None,
                                                 round_to_decimal=2,
                                                 df_ground_truth=None, dropna=True, error_column=None, include_xy=False)

                    # ---

                    # setup
                    ms = 4
                    ylim_rmse = [0, 5.125]
                    ylim_error = [-21.5, 21.5]

                    # ---

                    # plot
                    fig, ax = plt.subplots()

                    ax.plot(dfbex.index, dfbex.rmse_z, '-o', ms=ms)
                    ax.fill_between([-depth_of_focus / 2, depth_of_focus / 2], ylim_rmse[0], ylim_rmse[1],
                                    color='red', ec='none', alpha=0.1, label='D.o.F.')

                    ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                    ax.set_ylim(ylim_rmse)
                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.legend()
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_results_uncertainty + '/rmse_z_by_z_exclude_DoF.svg')
                    plt.show()
                    plt.close()

                    # ---

                    # plot
                    fig, ax = plt.subplots()

                    ax.plot(df_exclude_dof.z_fit, df_exclude_dof.error, 'o', ms=ms / 5, alpha=0.4)
                    ax.fill_between([-depth_of_focus / 2, depth_of_focus / 2], ylim_error[0], ylim_error[1],
                                    color='red', ec='none', alpha=0.1, label='D.o.F.')

                    ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
                    ax.set_ylim(ylim_error)
                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.legend()
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_results_uncertainty + '/error_z_by_z_exclude_DoF.svg')
                    plt.show()
                    plt.close()

                # ---

                # rmse-z by r
                plot_rmse_by_r = True
                if plot_rmse_by_r:

                    save_figs = True

                    # create necessary columns
                    df = dfo.copy()
                    df['z_true'] = df['z_fit']
                    df['cm'] = 1.0

                    # -

                    # bin
                    r_bins = 6
                    r_bins = np.linspace(0.0, 1.0, r_bins)

                    # lower right membrane
                    dfmb = df[df['memb_id'] == 1]
                    dfmb['nd_r'] = dfmb.r / (r_edge_lr * microns_per_pixel)
                    dfb1 = bin.bin_local_rmse_z(dfmb, column_to_bin='nd_r', bins=r_bins, round_to_decimal=2)

                    # upper left membrane
                    dfmb = df[df['memb_id'] == 2]
                    dfmb['nd_r'] = dfmb.r / (r_edge_ul * microns_per_pixel)
                    dfb2 = bin.bin_local_rmse_z(dfmb, column_to_bin='nd_r', bins=r_bins, round_to_decimal=2)

                    # plot
                    ms = 5
                    fig, ax = plt.subplots()

                    ax.plot(dfb1.index, dfb1.rmse_z, '-o', ms=ms, label=800)
                    ax.plot(dfb2.index, dfb2.rmse_z, '-o', ms=ms, label=500)

                    ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
                    ax.set_xlabel(r'$r/a$')
                    ax.set_xlim([-0.03, 1.03])
                    ax.legend(title=r'$r \: (\mu m)$')
                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_results_uncertainty + '/norm-rmse_z_by_r.svg')
                    plt.show()
                    plt.close()

                    # ---

                # ---

                # compare all: kernel density estimation
                plot_all_kde = True
                if plot_all_kde:
                    from sklearn.neighbors import KernelDensity

                    for y_dir in ['pos', 'neg']:

                        if y_dir == 'pos':
                            dfd = df[df['z_fit'] > depth_of_focus / 2]
                            dfd = dfd.dropna(subset=['z_fit', 'error'])
                            x = dfd.z_fit.to_numpy()
                            y = dfd.error.to_numpy()

                            ylim_error = [-12, 12]
                            binwidth_x, bandwidth_x = depth_of_focus / 2, depth_of_focus / 1.5
                            binwidth_y, bandwidth_y = 1.5, 2.5

                            # histogram
                            xlim_low = (int(np.min(x) / binwidth_x) - 1) * binwidth_x  # + binwidth_x * 1
                            xlim_high = (int(np.max(x) / binwidth_x) + 1) * binwidth_x  # - binwidth_x * 2
                            xbins = np.arange(xlim_low, xlim_high + binwidth_x, binwidth_x)
                            ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # - binwidth_y
                            ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y + binwidth_y
                            ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)

                            # kernel density estimation
                            x_plot = np.linspace(np.min(x) + binwidth_x * 0, np.max(x) + binwidth_x * 1, 500)
                            y_plot = np.linspace(np.min(y) - binwidth_y * 2, np.max(y) + binwidth_y * 2, 500)

                        elif y_dir == 'neg':
                            dfd = df[df['z_fit'] < -depth_of_focus / 2]
                            dfd = dfd.dropna(subset=['z_fit', 'error'])
                            x = dfd.z_fit.to_numpy()
                            y = dfd.error.to_numpy()

                            ylim_error = [-12, 12]
                            binwidth_x, bandwidth_x = depth_of_focus / 2, depth_of_focus / 1.5
                            binwidth_y, bandwidth_y = 1.5, 2.5

                            # histogram
                            xlim_low = (int(np.min(x) / binwidth_x) - 1) * binwidth_x  # + binwidth_x * 1
                            xlim_high = (int(np.max(x) / binwidth_x) + 1) * binwidth_x  # - binwidth_x * 2
                            xbins = np.arange(xlim_low, xlim_high + binwidth_x, binwidth_x)
                            ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # - binwidth_y
                            ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y + binwidth_y
                            ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)

                            # kernel density estimation
                            x_plot = np.linspace(np.min(x) + binwidth_x * 1, np.max(x) + binwidth_x * 0, 500)
                            y_plot = np.linspace(np.min(y) - binwidth_y * 2, np.max(y) + binwidth_y * 2, 500)

                        # setup
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

                        ax.set_ylim(ylim_error)

                        # ---

                        # histogram
                        nx, binsx, patchesx = ax_histx.hist(x, bins=xbins, zorder=2.5, color='gray')
                        ny, binsy, patchesy = ax_histy.hist(y, bins=ybins, orientation='horizontal', color='gray',
                                                            zorder=2.5)

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
                        plt.savefig(path_results_uncertainty + '/kde-error-z_by_z-{}.svg'.format(y_dir))
                        plt.show()

                # ---

                # relative number of particles measured by z
                analyze_relative_number = True
                if analyze_relative_number:

                    by_membrane = True
                    if by_membrane:
                        df = pd.read_excel(path_results + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(dz_id))

                        ms = 2

                        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

                        ax1.plot(df.rz_lr, df.fit_lr_rmse, 'o', ms=ms, label='LR')
                        ax1.plot(df.rz_ul, df.fit_ul_rmse, 'o', ms=ms, label='UL')
                        ax2.plot(df.rz_lr, df.fit_lr_percent_meas * 100, 'o', ms=ms)
                        ax2.plot(df.rz_ul, df.fit_ul_percent_meas * 100, 'o', ms=ms)

                        ax1.set_ylabel(r'$\sigma_{z}^{i} \: (\mu m)$')
                        ax1.legend()
                        ax2.set_ylabel(r'$\phi^{i} \: (\%)$')
                        ax2.set_xlabel(r'$\Delta z_{max} \: (\mu m)$')
                        plt.tight_layout()
                        plt.savefig(path_results_uncertainty + '/rmsez-and-percent_by_z-max_raw.svg')
                        plt.show()
                        plt.close()

                    # ---

                    by_all = True
                    if by_all:
                        df = pd.read_excel(
                            path_results + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest_vertical.xlsx'.format(dz_id))
                        df = df[df.memb_id.isin([1, 2])]

                        # -

                        # create necessary columns
                        column_to_bin = 'rz'
                        column_to_count = 'frame'
                        bins = 25
                        round_to_decimal = 1
                        return_groupby = True

                        # bins = 1 to get average
                        dfmm, dfmstd = bin.bin_generic(df, column_to_bin, column_to_count, 1, round_to_decimal, return_groupby)
                        dfmm.to_excel(path_results_uncertainty + '/mean-surface-reconstruction-rmse_both-lr-ul.xlsx')

                        # -

                        # local binning
                        dfm, dfstd = bin.bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

                        # plot
                        ms = 4
                        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

                        ax1.plot(dfm.bin, dfm.fit_percent_meas * 100, '-o', ms=ms)
                        ax2.plot(dfm.bin, dfm.fit_rmse, '-o', ms=ms)

                        ax1.set_ylabel(r'$\phi^{i} \: (\%)$')
                        ax2.set_xlabel(r'$\Delta z_{max}^{i} \: (\mu m)$')
                        ax2.set_ylabel(r'$\sigma_{z}^{i} \: (\mu m)$')
                        plt.tight_layout()
                        plt.savefig(path_results_uncertainty + '/bin_rmsez-and-nef-percent_by_z-max_both-lr-ul.svg')
                        plt.show()
                        plt.close()

                        # ---

            # ---

            # ----------------------------------------------------------------------------------------------------------------------
            # --- EVALUATE MEMBRANE FITS: AXIAL DEFLECTION, THETA, PRESSURE

            if analyze_fit_plate_theory:

                dz_id = tid
                path_results = base_dir + '/results/dz{}'.format(dz_id)

                path_results_fit_plate = join(path_results, 'fit-plate-theory')
                if not os.path.exists(path_results_fit_plate):
                    os.makedirs(path_results_fit_plate)

                # ---

                # READ TWO OPTIONS: --- HORIZONTAL DATA (COMPARE COLUMNS), OR, VERTICAL DATA (COMPARE ROWS) ---
                read_data = ['horizontal', 'vertical']

                # ---

                if 'horizontal' in read_data:

                    df = pd.read_excel(path_results + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(dz_id))

                    tid = dz_id
                    df['tid'] = tid

                    # calculate: mean, std, max, min of all columns
                    calc_mean_stats = True
                    if calc_mean_stats:
                        dfg_mean = df.groupby('tid').mean()
                        dfg_std = df.groupby('tid').std()
                        dfg_max = df.groupby('tid').max()
                        dfg_min = df.groupby('tid').min()

                        dfm = pd.concat([dfg_mean, dfg_std, dfg_max, dfg_min]).reset_index()
                        dfm = dfm.rename(index={0: "mean", 1: "std", 2: "max", 3: "min"})
                        dfm.to_excel(path_results_fit_plate + '/id{}_fit-plate-theory_mean-std-max-min.xlsx'.format(tid))

                    # ---

                    # plot membranes relative to each other
                    plot_relative = True
                    if plot_relative:
                        df = df[df['time'] > tid_1_start_time]

                        # calculate deflections ratios
                        ms = 3
                        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
                        ax1.plot(df.time, df.rz_lr, 'o', ms=ms, label=r'$lr$')
                        ax1.plot(df.time, df.rz_ul, 'o', ms=ms, label=r'$ul$')
                        ax1.set_ylabel(r'$\Delta z \: (\mu m)$')
                        ax1.legend()

                        ax2.plot(df.time, df.rz_lr / df.rz_ul, 'o', ms=ms / 2)
                        ax2.set_ylabel(r'$\Delta z_{l.r.} / \Delta z_{u.l.}$')
                        ax2.set_ylim([0.5, 5.5])
                        ax2.set_xlabel(r'$t \: (s)$')
                        plt.tight_layout()
                        plt.savefig(path_results_fit_plate + '/id{}_fit-plate-theory_ratio-of-rz_lr-ul.svg'.format(tid))
                        plt.show()

                    # ---

                if 'vertical' in read_data:

                    df = pd.read_excel(
                        path_results + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest_vertical.xlsx'.format(dz_id))
                    tid = dz_id
                    df['tid'] = tid

                    # calculate: mean, std, max, min of all columns
                    calc_weighted_average = True
                    if calc_weighted_average:
                        # 1. weighted average
                        res_weighted = np.average(df[['fit_rmse', 'fit_percent_meas']], axis=0, weights=df.f_num_pids)
                        res_sum = np.sum(df[['i_num_pids', 'f_num_pids']], axis=0)
                        res = np.hstack([res_weighted, res_sum])
                        dfwa = pd.DataFrame(res,
                                            index=['fit_rmse_wa', 'fit_percent_meas_wa', 'sum_i_num_pids', 'sum_f_num_pids'],
                                            columns=['quantity'])
                        dfwa.to_excel(path_results_fit_plate + '/id{}_fit-plate-theory_weighted-average.xlsx'.format(tid))

            # ---

            # ----------------------------------------------------------------------------------------------------------------------
            # --- CALCULATE NON-DIMENSIONAL TERMS

            if calculate_non_dimensional:

                # setup file paths
                dz_id = tid
                path_results = base_dir + '/results/dz{}'.format(dz_id)

                path_figs_compare_frames = join(path_results, 'compare-frames')
                if not os.path.exists(path_figs_compare_frames):
                    os.makedirs(path_figs_compare_frames)

                # read dataframe
                df = pd.read_excel(path_results + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(dz_id))
                df = df[['frame', 'time', 'rz_lr', 'fit_lr_pressure', 'fit_lr_pretension']]

                # initialize model
                fND_lr = fNonDimensionalNonlinearSphericalUniformLoad(r=r_edge_lr * microns_per_pixel * 1e-6,
                                                                      h=t_membrane,
                                                                      youngs_modulus=E_silpuran,
                                                                      poisson=poisson)

                nd_P, nd_k = fND_lr.non_dimensionalize_p_k(d_p0=df.fit_lr_pressure.to_numpy(),
                                                           d_n0=df.fit_lr_pretension.to_numpy()
                                                           )

                df['nd_p'] = nd_P
                df['nd_k'] = nd_k

                # export
                # df.to_excel(path_results + '/id1_non-dimensional_values.xlsx')

                # ---

                # frames of interest
                froises = [[116, 119, 122, 125], [185, 188, 192, 196], [101, 104, 107, 112], [96, 98, 100]]  #

                for frois in froises:

                    plot_slope_angle = True
                    if plot_slope_angle:

                        clrs = cm.magma(np.linspace(0.1, 0.875, len(frois)))
                        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 0.85, size_y_inches * 1.25))

                        hdls = []
                        for fr, clr in zip(frois, clrs):
                            dfr = df[df['frame'] == fr]
                            dz0 = np.abs(dfr.iloc[0].rz_lr)
                            nd_P = np.abs(dfr.iloc[0].nd_p)
                            nd_k = dfr.iloc[0].nd_k
                            nd_r = np.linspace(0, 1, 100)

                            # calculate loads
                            print(fND_lr.calculate_radial_loads(nd_k))

                            nd_z = fND_lr.nd_nonlinear_clamped_plate_p_k(nd_r, nd_P, nd_k)
                            nd_z_plate = fND_lr.nd_linear_z_plate(nd_r, nd_P)
                            nd_z_memb = fND_lr.nd_linear_z_membrane(nd_r, nd_P, nd_k)

                            nd_theta = fND_lr.nd_nonlinear_theta(nd_r, nd_P, nd_k)
                            nd_theta_plate = fND_lr.C1 * -0.25 * nd_P * nd_r * (1 - nd_r ** 2)
                            nd_theta_memb = fND_lr.C1 * -2 * nd_P * nd_r / 100 ** 2

                            p1, = ax1.plot(nd_r, nd_z / nd_z.max(), linewidth=0.75, label=np.round(fr / frame_rate, 2))
                            hdls.append(p1)
                            p2, = ax2.plot(nd_r, nd_theta / nd_z.max(), linewidth=0.75)

                        p3, = ax1.plot(nd_r, nd_z_plate / nd_z_plate.max(), zorder=4,
                                       linestyle='--', linewidth=0.75, color='black', label='Plate')
                        p4, = ax1.plot(nd_r, nd_z_memb / nd_z_memb.max(), zorder=4,
                                       linestyle='dotted', linewidth=0.75, color='black', label='Membrane')

                        ax2.plot(nd_r, nd_theta_plate / np.max(np.abs(nd_theta_plate)) * 1.5, linestyle='--', linewidth=0.75,
                                 color='black')
                        ax2.plot(nd_r, nd_theta_memb / np.max(np.abs(nd_theta_memb)) * 2, linestyle='dotted', linewidth=0.75,
                                 color='black')

                        ax1.set_ylabel(r'$\Delta z / \Delta z_{max}$')
                        # two legends
                        l1 = ax1.legend(handles=hdls, loc='lower left', title=r'$t \: (s)$',
                                        labelspacing=0.45, handletextpad=0.35, handlelength=1.5, borderaxespad=0.45)
                        ax1.add_artist(l1)
                        ax1.legend(handles=[p3, p4], loc='upper right', title='Model',
                                   labelspacing=0.45, handletextpad=0.35, handlelength=1.5, borderaxespad=0.45)

                        ax2.set_ylabel(r'$\theta / \Delta z_{max}$')
                        ax2.set_xlabel(r'$r/a$')

                        plt.tight_layout()
                        plt.savefig(
                            path_figs_compare_frames + '/compare_def_to_plate-and-memb_frames{}-{}.svg'.format(frois[0], frois[1]))
                        fig.show()

                    # ---

                    plot_curvature = True
                    if plot_curvature:

                        clrs = cm.magma(np.linspace(0.1, 0.875, len(frois)))
                        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 0.85, size_y_inches * 1.25))

                        hdls = []
                        for fr, clr in zip(frois, clrs):
                            dfr = df[df['frame'] == fr]
                            dz0 = np.abs(dfr.iloc[0].rz_lr)
                            nd_P = np.abs(dfr.iloc[0].nd_p)
                            nd_k = dfr.iloc[0].nd_k
                            print(nd_k)
                            nd_r = np.linspace(0., 1, 100)

                            nd_z = fND_lr.nd_nonlinear_clamped_plate_p_k(nd_r, nd_P, nd_k)
                            nd_curvature = fND_lr.nd_nonlinear_curvature_lva(nd_r, nd_P, nd_k)

                            p1, = ax1.plot(nd_r, nd_z / nd_z.max(), linewidth=0.75, label=np.round(fr / frame_rate, 2))
                            p2, = ax2.plot(nd_r[10:], nd_curvature[10:] / nd_P, linewidth=0.75)

                        ax1.set_ylabel(r'$\Delta z / \Delta z_{max}$')
                        ax1.legend(loc='lower left', title=r'$t \: (s)$', labelspacing=0.45, handletextpad=0.35, handlelength=1.5,
                                   borderaxespad=0.45)
                        ax2.set_ylabel(r'$\Psi / P$')
                        ax2.set_xlabel(r'$r/a$')

                        plt.tight_layout()
                        plt.savefig(path_figs_compare_frames + '/nd_curvature_frames{}-{}.svg'.format(frois[0], frois[1]))
                        fig.show()

            # ---


# ----------------------------------------------------------------------------------------------------------------------
# --- BOUNDARY PARTICLE DISPLACEMENT
"""
NOTE, the below function doesn't look very useful. 
"""

analyze_boundary_particles = False
if analyze_boundary_particles:

    # setup file paths
    dz_id = 1
    tid = dz_id
    path_results = base_dir + '/results/dz{}'.format(dz_id)

    path_results_bd = join(path_results, 'boundary-particles')
    if not os.path.exists(path_results_bd):
        os.makedirs(path_results_bd)

    # ---

    # read test coords dataframe
    df_original = pd.read_excel(path_results + '/test_coords_id{}_corrected.xlsx'.format(tid), index_col=0)

    # read particle ID's per membrane
    df_memb_pids = pd.read_excel(path_results + '/df_pids_per_membrane.xlsx', index_col=0)
    pids_bd = df_memb_pids.loc['boundary', :].dropna().values

    # ---

    # processing: "static" == before start time; "dynamic" == after start time
    tid_1_start_time = 0.225
    dfi = df_original[df_original['t'] <= tid_1_start_time]
    dff = df_original[df_original['t'] > tid_1_start_time]

    # ---

    # for loop

    for df, time_zone in zip([dff], ['dynamic', 'static']):
        df = df[~df.id.isin([39, 71])]
        dfo = df.copy()

        # get particles in dataframe
        pids = sorted(df.id.unique())

        # --- PRECISION

        analyze_precision = True
        if analyze_precision:

            # modifiers
            poly_deg = 12
            save_figs = False
            show_figs = False

            # ---

            # file paths
            path_results_precision = join(path_results_bd, time_zone + '-precision')
            if not os.path.exists(path_results_precision):
                os.makedirs(path_results_precision)

            path_results_precision_figs = join(path_results_precision, 'figs')
            if not os.path.exists(path_results_precision_figs):
                os.makedirs(path_results_precision_figs)

            # setup
            column_to_fit = 't'
            precision_columns = ['z_corr', 'drm']
            scale_columns_units = [1, microns_per_pixel]
            precision_data = []
            precision_data_columns = ['tid', 'id', 'precision_z', 'precision_drm_microns']

            # ---

            for pid in pids_bd:

                # get dataframe
                dfpid = df[df['id'] == pid]

                if len(dfpid) < 35:
                    print("pid {} discarded b/c length = {} < 35".format(pid, len(dfpid)))
                    continue

                # list of results
                pid_precision = [tid, pid]

                for pc, scale_units in zip(precision_columns, scale_columns_units):
                    x = dfpid[column_to_fit].to_numpy()
                    y = dfpid[pc].to_numpy() * scale_units

                    # fit polynomial
                    pc_precision = analyze.evaluate_precision_from_polyfit(x,
                                                                           y,
                                                                           poly_deg,
                                                                           )
                    # store this precisions
                    pid_precision.extend([pc_precision])

                    # plot
                    if save_figs or show_figs:

                        # ----------------------------------------------------------------------------------------------
                        # NOTE - this is duplicate of the above function; only here to get polynomial coefficients
                        # Could modify function but not worth the time right now
                        pcoeff, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg=poly_deg, full=True)
                        pf = np.poly1d(pcoeff)
                        y_model = pf(x)
                        y_residuals = y_model - y
                        y_precision = np.mean(np.std(y_residuals))
                        y_rmse = np.sqrt(np.mean(y_residuals ** 2))
                        # ----------------------------------------------------------------------------------------------

                        # resample fit space
                        y_fit = np.linspace(x.min(), x.max(), len(y) * 10)

                        # plot
                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches / 2))

                        ax.scatter(x, y, s=2, label=r'$p_{ID}$' + ': {}'.format(pid))
                        ax.plot(y_fit, pf(y_fit),
                                linestyle='--', color='black', alpha=0.5,
                                label=r'$\overline{\sigma}_{y}=$' + '{}'.format(np.round(y_precision, 2)))

                        ax.set_xlabel(r'$x \: (s)$')
                        ax.set_ylabel(r'$y \: (\mu m)$')
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(
                                path_results_precision_figs +
                                '/id{}_pid{}_precision-{}_by_{}_equals_{}.png'.format(tid,
                                                                                      pid,
                                                                                      pc,
                                                                                      column_to_fit,
                                                                                      np.round(y_precision, 2)),
                            )
                        elif show_figs:
                            plt.show()
                        plt.close()

                    # ---

                # store all precisions
                precision_data.append(pid_precision)

            # ---

            # export per-particle precision
            dfp = pd.DataFrame(np.array(precision_data), columns=precision_data_columns)
            dfp.to_excel(path_results_precision + '/precision_boundary-particles-and-frames.xlsx', index=False)

            # export mean precision
            dfpg = dfp.groupby('tid').mean()
            dfpg.to_excel(path_results_precision + '/mean-precision_boundary-particles-and-frames.xlsx',
                          index_label='tid')

            # ---

        # ---

        # --- DISPLACEMENT

        analyze_displacement = True
        if analyze_displacement:

            # modifiers
            save_figs = True
            show_figs = False

            # ---

            # file paths
            path_results_precision = join(path_results_bd, time_zone + '-displacement')
            if not os.path.exists(path_results_precision):
                os.makedirs(path_results_precision)

            path_results_precision_figs = join(path_results_precision, 'figs')
            if not os.path.exists(path_results_precision_figs):
                os.makedirs(path_results_precision_figs)

            # setup
            column_to_fit = 't'
            precision_columns = ['z_corr', 'drm']
            scale_columns_units = [1, microns_per_pixel]
            precision_data = []
            precision_data_columns = ['tid', 'id',
                                      'z_mean', 'z_std', 'z_A', 'z_freq', 'z_phase',
                                      'r_mean', 'r_std', 'r_A', 'r_freq', 'r_phase',
                                      ]

            # ---

            # A. ANALYZE MEAN DISPLACEMENT OF ALL PARTICLES

            dfbdg = df[df['id'].isin(pids_bd)]
            dfbdg = dfbdg[dfbdg['z_corr'] < 0]
            dfbdg = dfbdg.groupby(column_to_fit).mean().reset_index()
            tm = dfbdg[column_to_fit].to_numpy()
            zm = dfbdg.z_corr.to_numpy() * scale_columns_units[0]
            rm = dfbdg.drm.to_numpy() * scale_columns_units[1]

            dict_resz = functions.fit_sin(tt=tm, yy=zm)
            Az = dict_resz['amp']
            fz = dict_resz['freq']
            periodz = dict_resz['period']
            fit_funcz = dict_resz['fitfunc']
            sine_funcz = dict_resz['sinfunc']
            poptz = dict_resz['popt']

            dict_resr = functions.fit_sin(tt=tm, yy=rm)
            Ar = dict_resr['amp']
            fr = dict_resr['freq']
            periodr = dict_resr['period']
            fit_funcr = dict_resr['fitfunc']
            sine_funcr = dict_resr['sinfunc']
            poptr = dict_resr['popt']

            xm_fit = np.linspace(tm.min(), tm.max(), len(tm) * 5)
            zm_fit = sine_funcz(xm_fit, *poptz)
            rm_fit = sine_funcr(xm_fit, *poptr)

            if save_figs or show_figs:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                ax1.scatter(tm, zm, s=1)
                ax1.plot(xm_fit, zm_fit)
                ax2.scatter(tm, rm, s=1)
                ax2.plot(xm_fit, rm_fit)
                ax1.set_ylabel(r'$z_{corr} \: (\mu m)$')
                ax2.set_ylabel(r'$dr_{m} \: (\mu m)$')
                ax2.set_xlabel(r'$t \: (s)$')
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_results_precision_figs + '/mean-boundary-particles_z-r_by_t.svg')
                elif show_figs:
                    plt.show()
                plt.close()

            # ---

            for pid in pids_bd:

                # get dataframe
                dfpid = df[df['id'] == pid]

                # FILTER! - note, we only filter here b/c we want to analyze radial displacement of boundary particles
                dfpid = dfpid[dfpid['z_corr'] < 0]

                if len(dfpid) < 35:
                    print("pid {} discarded b/c length = {} < 35".format(pid, len(dfpid)))
                    continue

                # list of results
                pid_precision = [tid, pid]

                for pc, scale_units in zip(precision_columns, scale_columns_units):
                    x = dfpid[column_to_fit].to_numpy()
                    y = dfpid[pc].to_numpy() * scale_units

                    # analysis #1 - mean +/- std
                    y_m = np.mean(y)
                    y_std = np.std(y)

                    # ---

                    # analysis #2 - fit function
                    dict_res = functions.fit_sin(tt=x, yy=y)
                    A = dict_res['amp']
                    f = dict_res['freq']
                    period = dict_res['period']
                    fit_func = dict_res['fitfunc']
                    sine_func = dict_res['sinfunc']
                    popt = dict_res['popt']

                    # store results
                    pid_precision.extend([y_m, y_std, A, f, period])

                    # plot
                    if pid in [0, 1, 6, 29, 39, 41, 44, 47, 85]:  # # pid in [8]:  # save_figs or show_figs:

                        # ----------------------------------------------------------------------------------------------

                        # format figures
                        xlbl = r'$t \: (s)$'

                        # resample particle
                        x_fit = np.linspace(x.min(), x.max(), len(y) * 5)
                        y_fit = sine_func(x_fit, *popt)

                        # resample mean
                        if pc == 'z_corr':
                            ym_fit = sine_funcz(x_fit, *poptz)
                            ylbl = r'$\Delta z \: (\mu m)$'
                        elif pc == 'drm':
                            ym_fit = sine_funcr(x_fit, *poptr)
                            ylbl = r'$\Delta r \: (\mu m)$'
                        else:
                            raise ValueError("pc not understood.")

                        # plot
                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches / 1.5))

                        ax.scatter(x, y, s=2, label=r'$p_{ID}$' + ': {}'.format(pid))
                        ax.plot(x_fit, y_fit,
                                linestyle='--', color='black', alpha=0.5,
                                label='A={}, f={}, period={}'.format(np.round(A, 2),
                                                                     np.round(f, 2),
                                                                     np.round(period, 2))
                                )

                        ax.set_xlabel(xlbl)
                        ax.set_ylabel(ylbl)
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(
                                path_results_precision_figs +
                                '/id{}_pid{}_disp-{}_by_{}_std={}.svg'.format(tid,
                                                                              pid,
                                                                              pc,
                                                                              column_to_fit,
                                                                              np.round(y_std, 2)),
                            )
                        elif show_figs:
                            plt.show()
                        plt.close()

                        # ---

                        # PLOT AGAINST MEAN

                        # plot
                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches / 2))

                        ax.scatter(x, y - y[0], s=1, marker='.', alpha=1, label=r'$p_{i}=$' + ' {}'.format(int(pid)))
                        ax.plot(x_fit, y_fit - y[0],
                                '-k', linewidth=0.85, alpha=0.85, label=r'$F_{p_{i}}$')
                        ax.plot(x_fit, ym_fit - ym_fit[0],
                                '--r', linewidth=0.85, alpha=0.85, label=r'$F_{\overline{p_{i, N}}}$')

                        ax.set_xlabel(xlbl)
                        ax.set_ylabel(ylbl)
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(
                                path_results_precision_figs +
                                '/compare-mean-to-id{}_pid{}_disp-{}_by_{}.svg'.format(tid,
                                                                                       pid,
                                                                                       pc,
                                                                                       column_to_fit,
                                                                                       ),
                            )
                        elif show_figs:
                            plt.show()
                        plt.close()

                    # ---

                # store all precisions
                precision_data.append(pid_precision)

            # ---

            # export per-particle precision
            dfp = pd.DataFrame(np.array(precision_data), columns=precision_data_columns)
            dfp.to_excel(path_results_precision + '/displacement_boundary-particles-and-frames.xlsx', index=False)

            # export mean precision
            dfpg = dfp.groupby('tid').mean()
            dfpg.to_excel(path_results_precision + '/mean-displacement_boundary-particles-and-frames.xlsx',
                          index_label='tid')

            # ---

        # ---

    # ---

# ---


# ----------------------------------------------------------------------------------------------------------------------
# --- PLAYGROUND FOR PLOTTING MEMBRANE-DEFLECTION PROFILES


plot_playground = False
if plot_playground:

    # --- OLD FUNCTIONS
    # fitting functions
    # a, b, c = 0.5, 2, 0.2
    def efunc(x, aa, bb, phi):
        return (0.5 * np.cos(x + phi) + 2 * np.sin(x + phi)) * (1 - (x / np.pi) ** 12) * x * np.exp(
            -x / aa) * bb


    def sine_sq_decay(x, A, f, b):
        """
        :param A: amplitude
        :param f: frequency - the number of cycles per second; bounds = (0, 1)
        :param b: decay rate
        """
        # np.sin(a * np.pi * x ** 1.5) * np.exp(-x * j) * (1 - x ** 6)
        return A * np.sin(2 * np.pi * f * x ** 2) * np.exp(-x * b) * (1 - x ** 12)


    def cfunc(xx, aa, bb):
        return (xx - xx ** 2) * (aa ** 6 - xx ** 6) * xx * bb


    # plot random function

    fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches * 1.25))
    a = 4
    b = 0.925
    c = 1.5
    x = np.linspace(0, 1)
    y = x * (1 - x) * (b - x) * 2 * x ** 0.25
    y12 = (x - x ** 2) * (b ** 1.2 - x ** 1.2) * x * 2
    y14 = (x - x ** 2) * (b ** 1.4 - x ** 1.4) * c
    y2 = (x - x ** 2) * (b ** 2 - x ** 2) * x * 2
    y3 = (x - x ** 2) * (b ** 3 - x ** 3) * x * 2
    y4 = (x - x ** 2) * (b ** 4 - x ** 4) * x * 2
    y8 = (x - x ** 2) * (b ** 8 - x ** 8) * x * 2

    """xf = 1 * np.pi
    phi = np.pi / 4
    x = np.linspace(0, xf, 100)
    a, b, c = 0.5, 2, 0.2
    for i in np.linspace(0.45, 0.55, 5):
        for j, lss in zip([1.25, 1.66, 1.9], ['-', '--', '-.']):
            phi = np.pi / i
            # y = a * (np.sin(x) + np.sin(x + phi)) * (1 - (x / xf) ** 1) * x * np.exp(-x)
            # y = (a * np.cos(x + phi) + np.cos(x) + b * np.sin(x + phi) + np.sin(x)) * (1 - (x / xf) ** 12) * x * np.exp(-x)
            # y = (a * np.cos(x + phi) + b * np.sin(x + phi)) * (1 - (x / xf) ** 12) * x * np.exp(-x / j) * c
            ax.plot(x / xf, y, label=np.round(np.pi / i, 3), linestyle=lss)"""

    x = np.linspace(0, 1, 100)

    for a in np.linspace(0.95, 2, 5):
        for j, lss in zip([1, 1.3, 1.8], ['-', '--', '-.']):
            y = np.sin(a * np.pi * x ** 1.5) * np.exp(-x * j) * (1 - x ** 6)
            # y = np.sin(a * np.pi * x ** 1.2) * np.exp(-x * j) * (1 - x ** 12)
            ax.plot(x, y, linestyle=lss, label=np.round(a, 3))

    # ax.plot(x, y, label='y')
    # ax.plot(x, y12, label='y12')
    # ax.plot(x, y2, label='y2')
    # ax.plot(x, y4, label='y4')
    # ax.plot(x, y8, label='y8')

    # ax.plot(x, b - x, label='b-x')
    # ax.plot(x, x - x ** 0.25, label=r'$x-x^{1/4}$')
    # ax.plot(x, x - x ** 0.5, label=r'$x-x^{1/2}$')
    # ax.plot(x, x - x ** 2.5, label=r'$x-x^{2.5}$')
    # ax.plot(x, x - x ** 3, label=r'$x-x^3$')
    # ax.plot(x, (x - x ** 4) * x ** 0.25, label=r'$x-x^4$')
    # ax.plot(x, (x - x ** 6) * x ** 0.25, label=r'$x-x^6$')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    raise ValueError()

    # key parameters
    Pc = 0.3675

    # initialize
    r_fit_lr = np.linspace(0, r_edge_lr * microns_per_pixel)
    P0 = np.linspace(0, Pc * 8)
    fsphere_lr = fSphericalUniformLoad(r=r_edge_lr * microns_per_pixel * 1e-6, h=t_membrane, youngs_modulus=E_silpuran)

    # get membrane deflection profile at critical pressure (transition pressure)
    z_fit_lr_Pc = fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit_lr * 1e-6, Pc) * 1e6

    # get max deflection for array of pressure values
    z_max_lr_Pc = fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(0, P0) * 1e6

    # ---

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(size_x_inches, size_y_inches * 1.25))
    ax1.plot(r_fit_lr, z_fit_lr_Pc, label=r'$P_{c}=$' + ' {} Pa'.format(Pc))
    ax1.set_xlabel(r'$r \: (\mu m)$')
    ax1.set_ylabel(r'$z \: (\mu m)$')
    ax1.legend()

    ax2.plot(P0, z_max_lr_Pc, label=r'$z(r=0)$')
    ax2.set_xlabel(r'$P \: (Pa)$')
    ax2.set_ylabel(r'$z_{max} \: (\mu m)$')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# ---


# ----------------------------------------------------------------------------------------------------------------------
# --- OLD-POST-ANALYSIS

# plot only the larger membrane
only_lower_right = False
if only_lower_right:

    # plot frames of interest
    per_frame_of_interest = False
    all_frame_of_interest = False
    fit_spherical_uniform_load = False

    # plot trajectories
    plot_all_frames = False

    # plot every "ok" particle
    plot_each_particle = False

    # plot only the "best" particles
    analyze_best_particles = False

    # plot only the "best" particles and calculate their fit error
    analyze_fit_error = True
    analyze_trajectory_precision = True

    # assess the in-plane (xy) displacement via cross-correlation tracking
    analyze_template_displacement = True

    # ---

    # read
    df = pd.read_excel(path_results + '/test_coords_id{}_corrected.xlsx'.format(tid))

    # format plots
    xlim = [-30, 830]
    xticks = [0, 200, 400, 600, 800]

    # --------------------------------------------------------------------------------------------------------------
    # 5. GET ONLY PARTICLES ON THE LOWER RIGHT MEMBRANE AREA

    # particle ID's on the lower right membrane
    """NOTE: there is a script to do this but, for simplicity, we do it manually here."""
    particle_ids_lr = [43, 44, 47, 49, 51, 52, 53, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 70, 76, 77, 79,
                       82,
                       83, 84, 88, 89]
    df = df[df.id.isin(particle_ids_lr)]

    # --------------------------------------------------------------------------------------------------------------
    # 6. CALCULATE RADIAL POSITION ON THE LOWER RIGHT MEMBRANE

    # NOTE: this should be corrected to 'rr' which is the radial coordinate for each membrane
    #   --> right now, we choose to overwrite the 'r' which is with respect to the image center.
    df['r'] = functions.calculate_radius_at_xy(df.x, df.y, xc=xc_lr, yc=yc_lr)

    # --------------------------------------------------------------------------------------------------------------
    # 7. PLOT MEMBRANE DEFLECTION (Z ~ f(R)) AT SPECIFIC FRAMES

    if per_frame_of_interest:
        for fr in frames_of_interest:
            dfr = df[df['frame'] == fr]

            fig, ax = plt.subplots()
            ax.plot(dfr.r * microns_per_pixel, dfr.z_corr, 'o', label=fr)
            ax.set_xlabel(r'$r \: (\mu m)$')
            ax.set_xlim(xlim)
            ax.set_xticks(xticks)
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.legend(title=r'$Frame$')
            plt.tight_layout()
            if save_figs:
                plt.savefig(path_figs + '/id{}_lr_membrane_deflection_z_by_r_frame{}.png'.format(tid, fr))
            elif show_figs:
                plt.show()
            plt.close()

    if all_frame_of_interest:
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))
        for fr in frames_of_interest:
            dfr = df[df['frame'] == fr]
            ax.plot(dfr.r * microns_per_pixel, dfr.z_corr, 'o', ms=3, label=fr)
        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$Frame$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(path_figs + '/id{}_lr_membrane_deflection_z_by_r_all-frame-of-interest.png'.format(tid))
        elif show_figs:
            plt.show()
        plt.close()

    if fit_spherical_uniform_load:

        # SILPURAN material properties
        known_radius = 800e-6
        known_thickness = t_membrane
        know_youngs_modulus = E_silpuran
        known_poisson = 0.5

        # instantiate class
        fsphere = functions.fSphericalUniformLoad(r=known_poisson,
                                                  h=known_thickness,
                                                  youngs_modulus=know_youngs_modulus,
                                                  poisson=known_poisson,
                                                  )

        # plot + fit
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))

        for fr in frames_of_interest:
            dfr = df[df['frame'] == fr]
            dfr = dfr.sort_values('r')

            # curve_fit
            unknown_z_offset = 0  # NOTE: this used to be +20; not sure why
            print("Unkown z offset used to be 20!")
            popt, pcov = curve_fit(fsphere.spherical_uniformly_loaded_simply_supported_plate_r_p,
                                   dfr.r * microns_per_pixel * 1e-6,
                                   (dfr.z_corr + unknown_z_offset) * 1e-6,
                                   )

            # resample
            r_fit = np.linspace(np.min(xticks), np.max(xticks)) * 1e-6
            z_fit = fsphere.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit, *popt)

            # plot data
            p1, = ax.plot(dfr.r * microns_per_pixel, dfr.z_corr,
                          'o',
                          ms=3,
                          label='{}: {} Pa'.format(fr, np.round(popt[0], 3))
                          )

            # plot fit
            ax.plot(r_fit * 1e6, z_fit * 1e6,
                    linestyle='--',
                    color=lighten_color(p1.get_color(), amount=1.25),
                    alpha=0.75)

        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$Frame$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(
                path_figs + '/id{}_fit_lr_membrane_deflection_z_by_r_all-frame-of-interest.png'.format(tid))
        elif show_figs:
            plt.show()
        plt.close()

    # --------------------------------------------------------------------------------------------------------------
    # 8. PLOT MEMBRANE DEFLECTION (Z ~ f(TIME)) FOR ALL FRAMES

    if plot_all_frames:
        fig, ax = plt.subplots()
        ax.scatter(df.frame / frame_rate, df.z_corr, c=df.id, s=1)
        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(path_figs + '/id{}_lr_membrane_deflection_z_by_t.png'.format(tid))
        elif show_figs:
            plt.show()
        plt.close()

    # --------------------------------------------------------------------------------------------------------------
    # 9. PLOT PARTICLE DISPLACEMENT (Z ~ f(TIME, PID)) FOR EACH PARTICLE

    if plot_each_particle:

        for pid in particle_ids_lr:
            dfpid = df[df['id'] == pid]

            fig, ax = plt.subplots()
            ax.scatter(dfpid.frame / frame_rate, dfpid.z_corr, s=2, label=pid)
            ax.set_xlabel(r'$t \: (s)$')
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.legend(title=r'$p_{ID}$')
            plt.tight_layout()
            if save_figs:
                plt.savefig(path_figs + '/id{}_lr_membrane_deflection_z_by_t_pid{}.png'.format(tid, pid))
            if show_figs:
                plt.show()
            plt.close()

    # --------------------------------------------------------------------------------------------------------------
    # 10. PLOT "BEST" PARTICLE DISPLACEMENT (Z ~ f(TIME, PID)) FOR EACH PARTICLE

    # hand-selected particle ID's (lowest focal plane bias errors)
    particle_ids_lr_best = [47, 49, 52, 58, 59, 62, 64, 65, 67, 68, 76, 89]
    df = df[df.id.isin(particle_ids_lr_best)]

    # filter to start of deflection
    dfst = df[df['t'] > 3.66]  # start_time
    # dfst = dfst[dfst['t'] < 7.6]

    if analyze_best_particles:

        # plot all trajectories
        fig, ax = plt.subplots()
        for pid in particle_ids_lr_best:
            dfpid = dfst[dfst['id'] == pid]
            dfpid = dfpid.sort_values('t')

            ax.plot(dfpid.t, dfpid.z_corr, '-o', ms=1)

        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(path_figs + '/id{}_lr_membrane_deflection_z_by_t_best-pids.png'.format(tid))
        elif show_figs:
            plt.show()
        plt.close()

        # ---

        # scatter all trajectories
        fig, ax = plt.subplots()
        ax.scatter(dfst.t, dfst.z_corr, c=dfst.id, s=2)
        ax.set_xlabel(r'$t \: (s)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(path_figs + '/lr_membrane_deflection_z_by_t_best-pids_scatter.png'.format(tid))
        elif show_figs:
            plt.show()
        plt.close()

    # fit and plot each trajectory

    if analyze_fit_error:

        for pid in particle_ids_lr_best:
            dfpid = dfst[dfst['id'] == pid]

            # fit polynomial
            pcoeff, residuals, rank, singular_values, rcond = np.polyfit(dfpid.t, dfpid.z_corr, deg=12,
                                                                         full=True)
            pf = np.poly1d(pcoeff)

            # error assessment
            z_model = pf(dfpid.t)
            z_error = z_model - dfpid.z_corr.to_numpy()
            z_precision = np.mean(np.std(z_error))
            z_rmse = np.sqrt(np.sum(z_error ** 2) / len(dfpid))

            # resample
            z_fit = np.linspace(dfpid.t.min(), dfpid.t.max(), len(dfpid) * 10)

            fig, ax = plt.subplots()

            ax.scatter(dfpid.t, dfpid.z_corr, s=2, label=r'$p_{ID}$' + ': {}'.format(pid))
            ax.plot(z_fit, pf(z_fit),
                    linestyle='--', color='black', alpha=0.5,
                    label=r'$\overline{\sigma}_{z}=$' + '{}'.format(np.round(z_rmse, 2)) +
                          '\n' +
                          r'$\overline{\sigma}^p_{z}=$' + '{}'.format(np.round(z_precision, 2)))

            ax.set_xlabel(r'$t \: (s)$')
            ax.set_ylabel(r'$z \: (\mu m)$')
            # ax.legend(loc='upper left', markerscale=0.66, borderpad=0.2, labelspacing=0.25, handletextpad=0.4, borderaxespad=0.25)
            ax.legend(loc='upper right')
            plt.tight_layout()
            if save_figs:
                plt.savefig(
                    path_figs + '/id{}_fit-error_lr_memb-def_z_by_t_residual{}_pid{}.png'.format(tid,
                                                                                                 np.round(
                                                                                                     residuals[
                                                                                                         0],
                                                                                                     -1),
                                                                                                 pid))
            elif show_figs:
                plt.show()
            plt.close()

    # --------------------------------------------------------------------------------------------------------------
    # 11. PLOT "BEST" PARTICLE TEMPLATE DISPLACEMENT (X, Y, Z ~ f(TIME, PID)) FOR EACH PARTICLE

    if analyze_template_displacement:

        df = df[df['z_corr'] > 0]

        # scatter all trajectories
        fig, ax = plt.subplots()

        for pid in particle_ids_lr_best:
            dfpid = df[df['id'] == pid]
            dfpid = dfpid.sort_values('t')

            dfpid['r_drm_exaggerated'] = dfpid['r'] / 10 + dfpid['drm']

            # plot
            ax.plot(dfpid.r_drm_exaggerated * microns_per_pixel, dfpid.z_corr, '-o', ms=0.5, linewidth=0.5)

        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(
                path_figs + '/id{}_lr_membrane_deflection_z_by_r_by_t_best-pids_scatter_r-10X-reduced.png'.format(
                    tid))
        elif show_figs:
            plt.show()
        plt.close()

        # scatter all trajectories - fit quartic sliding function
        exag = 10

        fig, ax = plt.subplots()

        for pid in particle_ids_lr_best:
            dfpid = df[df['id'] == pid]
            dfpid = dfpid.sort_values('t')

            dfpid['r_drm_exaggerated'] = dfpid['r'] + dfpid['drm'] * exag

            # fit quartic sliding function
            popt, pcov = curve_fit(functions.quartic_slide, dfpid.z_corr, dfpid.r_drm_exaggerated)

            # resample
            z_fit = np.linspace(dfpid.z_corr.min(), dfpid.z_corr.max())
            r_drm_fit = functions.quartic_slide(z_fit, *popt)

            p1, = ax.plot(dfpid.r_drm_exaggerated * microns_per_pixel, dfpid.z_corr, 'o', ms=1, alpha=0.125)
            ax.plot(r_drm_fit * microns_per_pixel, z_fit, color=lighten_color(p1.get_color(), amount=1.25))

        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(path_figs +
                        '/id{}_lr_membrane_deflection_z_by_r_by_t_best-pids_scatter_exaggerated-{}X-fit.png'.format(
                            tid, exag))
        elif show_figs:
            plt.show()
        plt.close()

        # scatter all trajectories - fit quartic sliding function + fit envelope of maximum deflection
        fig, ax = plt.subplots()

        max_z_corrs = []
        min_z_corrs = []

        for pid in particle_ids_lr_best:
            dfpid = df[df['id'] == pid]
            dfpid = dfpid.sort_values('t').reset_index()

            dfpid['rm'] = np.sqrt(dfpid.xm ** 2 + dfpid.ym ** 2)
            dfpid['drm'] = dfpid['rm'] - dfpid.iloc[0].rm
            dfpid['r_drm'] = dfpid['r'] + dfpid['drm']
            dfpid['r_drm_exaggerated'] = dfpid['r'] + dfpid['drm'] * 1

            # append max values
            max_z_corrs.append(
                [dfpid.iloc[dfpid.z_corr.idxmax()].r_drm * microns_per_pixel, dfpid.z_corr.max()])
            min_z_corrs.append(
                [dfpid.iloc[dfpid.z_corr.idxmin()].r_drm * microns_per_pixel, dfpid.z_corr.min()])

            # fit quadratic
            popt, pcov = curve_fit(functions.quartic_slide, dfpid.z_corr, dfpid.r_drm_exaggerated)

            # resample
            z_fit = np.linspace(dfpid.z_corr.min(), dfpid.z_corr.max())
            r_drm_fit = functions.quartic_slide(z_fit, *popt)

            p1, = ax.plot(dfpid.r_drm_exaggerated * microns_per_pixel, dfpid.z_corr, 'o', ms=1, alpha=0.06125)
            ax.plot(r_drm_fit * microns_per_pixel, z_fit,
                    linewidth=0.5, color=lighten_color(p1.get_color(), amount=1.25))

        # structure the max/min deflection envelope
        max_z_corrs = np.array(max_z_corrs) * 1e-6
        min_z_corrs = np.array(min_z_corrs) * 1e-6

        # fit plate theory
        fsphere = fSphericalUniformLoad(r=800e-6, h=t_membrane, youngs_modulus=E_silpuran)
        r_fit = np.linspace(0, 800e-6)

        # max
        popt_max, pcov_max = curve_fit(fsphere.spherical_uniformly_loaded_simply_supported_plate_r_p,
                                       max_z_corrs[:, 0],
                                       max_z_corrs[:, 1],
                                       )
        z_fit_max = fsphere.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit, *popt_max)

        # min
        popt_min, pcov_min = curve_fit(fsphere.spherical_uniformly_loaded_simply_supported_plate_r_p,
                                       min_z_corrs[:, 0],
                                       min_z_corrs[:, 1],
                                       )
        z_fit_min = fsphere.spherical_uniformly_loaded_simply_supported_plate_r_p(r_fit, *popt_min)

        ax.plot(r_fit * 1e6, z_fit_max * 1e6,
                color='black', linestyle='--', linewidth=0.5, alpha=0.25,
                label=r'$z_{max}(r,P=$' + '{} '.format(np.round(popt_max[0], 1)) + r'$ Pa)$')
        ax.plot(r_fit * 1e6, z_fit_min * 1e6,
                color='black', linestyle='-.', linewidth=0.5, alpha=0.25,
                label=r'$z_{min}(r,P=$' + '{} '.format(np.round(popt_min[0], 1)) + r'$ Pa)$')

        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend(markerscale=0.5)
        plt.tight_layout()
        if save_figs:
            plt.savefig(
                path_figs + '/id{}_lr_membrane_deflection_z_by_r_by_t_best-pids_scatter_fit-max-min-envelope.png'.format(
                    tid))
        elif show_figs:
            plt.show()
        plt.close()

# ---

print("Analysis completed without errors.")