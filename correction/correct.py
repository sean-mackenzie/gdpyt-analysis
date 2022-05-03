# gdpyt-analysis: correct
"""
Description:
    This .py file holds methods for determining and correcting particle distributions.

Corrections:
    1. Initial particle distribution:
        a) angled (e.g. particles on tilted glass slide)
        b) curved (e.g. particles on a curved surface; cylindrical or spherical)
    2. Distortions:
        a) spherical (e.g. particles in-focus position is distorted by spherical aberrations)
    3. Arbitrary to In-focus Units Conversion:
        a) Converting calibration image z-coordinates (arbitrary) to distance wrt in-focus plane (in-focus).
    4. Optics
        a) calculate effective magnification
            1. Using known particle diameter and pixel size, calculate pixels-per-particle.
        b) calculate effective focal length
            1. Using pixels-per-particle at different z-coordinates.

Methods:
    1. Find in-focus z by interpolating peak intensity (a) per-particle, or (b) per-collection.
    2. Re-assign particles' z-coord and true z-coord based on in-focus z.

Plotting:
    1. Peak intensity (z)

"""

# Note on adjusting subplots spacing
"""
Using tight_layout():
    plt.tight_layout(pad: float (fraction of font size), h_pad, w_pad: float (defaults to pad_inches)))

Manually adjusting subplot spacing:
    Call Signature: 
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    Suggested Defaults:
        left  = 0.125  # the left side of the subplots of the figure
        right = 0.9    # the right side of the subplots of the figure
        bottom = 0.1   # the bottom of the subplots of the figure
        top = 0.9      # the top of the subplots of the figure
        wspace = 0.2   # the amount of width reserved for blank space between subplots
        hspace = 0.2   # the amount of height reserved for white space between subplots
"""

# imports
from os.path import join, exists
from os import makedirs

import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.neighbors import NearestNeighbors

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import analyze
from utils import io, details, fit, bin, plotting, modify, boundary, functions
from utils.functions import calculate_z_of_3d_plane
import filter
from tracking import plotting as trackplot

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# setup figures
scale_fig_dim = [1, 1]
scale_fig_dim_legend_outside = [1.3, 1]
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# scripts


def inspect_calibration_surface(df, param_zf, microns_per_pixel):
    """
    To run:
    dict_fit_plane, fig_xy, fig_xyz, fig_plane = correct.inspect_calibration_surface(df, param_zf, microns_per_pixel)

    :param df:
    :param param_zf:
    :param microns_per_pixel:
    :return:
    """

    if 'x' not in df.columns:
        raise ValueError('Run function merge_calib_pid_defocus_and_correction_coords(path_calib_coords) to merge x-y.')

    # 1. Plot initial distribution in x and y
    dict_fit_plane = fit_in_focus_plane(df, param_zf, microns_per_pixel)

    fig_xy, ax = plotting.scatter_z_by_xy(df=df, z_params=param_zf)

    fig_xyz = plotting.scatter_xy_color_z(df, param_z=param_zf)

    fig_plane = plotting.plot_fitted_plane_and_points(df, dict_fit_plane)

    return dict_fit_plane, fig_xy, fig_xyz, fig_plane


def fit_in_focus_plane(df, param_zf, microns_per_pixel):

    if not len(df) == len(df.id.unique()):
        df = df.groupby('id').mean().reset_index()

    # fitting stats:
    num_locations = len(df)
    x_span = df.x.max() - df.x.min()
    y_span = df.y.max() - df.y.min()
    num_density = num_locations / (x_span * y_span)
    zf_mean_of_points = df[param_zf].mean()
    zf_std_of_points = df[param_zf].std()

    # fit plane (x, y, z units: pixels)
    points_pixels = np.stack((df.x, df.y, df[param_zf])).T
    px_pixels, py_pixels, pz_pixels, popt_pixels = fit.fit_3d_plane(points_pixels)
    d, normal = popt_pixels[3], popt_pixels[4]

    # calculate fit error
    fit_results = functions.calculate_z_of_3d_plane(df.x, df.y, popt=popt_pixels)
    rmse, r_squared = fit.calculate_fit_error(fit_results, data_fit_to=df[param_zf].to_numpy())

    # fit plane (x, y, z units: microns) to calculate tilt angle
    points_microns = np.stack((df.x * microns_per_pixel, df.y * microns_per_pixel, df[param_zf])).T
    px_microns, py_microns, pz_microns, popt_microns = fit.fit_3d_plane(points_microns)
    tilt_x = np.rad2deg(np.arctan((pz_microns[0, 1] - pz_microns[0, 0]) / (px_microns[0, 1] - px_microns[0, 0])))
    tilt_y = np.rad2deg(np.arctan((pz_microns[1, 0] - pz_microns[0, 0]) / (py_microns[1, 0] - py_microns[0, 0])))

    # calculate zf at image center: (x = 256, y = 256)
    zf_mean_image_center = functions.calculate_z_of_3d_plane(256, 256, popt=popt_pixels)

    dict_fit_plane = {'z_f': param_zf,
                      'z_f_fit_plane_image_center': zf_mean_image_center,
                      'z_f_mean_points': zf_mean_of_points,
                      'z_f_std_points': zf_std_of_points,
                      'rmse': rmse, 'r_squared': r_squared,
                      'tilt_x_degrees': tilt_x, 'tilt_y_degrees': tilt_y,
                      'num_locations': num_locations,
                      'num_density_pixels': num_density, 'num_density_microns': num_density / microns_per_pixel**2,
                      'x_span_pixels': x_span, 'x_span_microns': x_span * microns_per_pixel,
                      'y_span_pixels': y_span, 'y_span_microns': y_span * microns_per_pixel,
                      'popt_pixels': popt_pixels,
                      'px': px_pixels, 'py': py_pixels, 'pz': pz_pixels,
                      'd': d, 'normal': normal,
                      }

    return dict_fit_plane


def perform_correction(io_dict, mask_dict, exp_dict, compute_all=False):

    # ------------------------
    # setup: io_dict
    path_calib_coords = io_dict['path_calib_coords']
    calib_sort_strings = io_dict['calib_sort_string']
    filetype = io_dict['filetype']
    save_plots = io_dict['save_plots']
    save_path = io_dict['save_path']

    path_figs_all_in_focus_correction = join(save_path, 'all_intensity_fits')
    if not exists(path_figs_all_in_focus_correction):
        makedirs(path_figs_all_in_focus_correction)

    path_figs_boundary_in_focus_correction = join(save_path, 'boundary_intensity_fits')
    if not exists(path_figs_boundary_in_focus_correction):
        makedirs(path_figs_boundary_in_focus_correction)

    export_path = join(save_path, 'data')
    if not exists(export_path):
        makedirs(export_path)

    export_boundary_points_in_focus = join(export_path, 'boundary_points_in_focus.xlsx')
    export_interior_points_in_focus = join(export_path, 'interior_points_in_focus.xlsx')
    export_all_points_in_focus = join(export_path, 'all_points_in_focus.xlsx')

    path_figs = join(save_path, 'figs')
    if not exists(path_figs):
        makedirs(path_figs)
    path_figs_in_focus_fit = path_figs

    path_corrections = join(save_path, 'corrections')
    if not exists(path_corrections):
        makedirs(path_corrections)

    save_corrections = join(path_corrections, 'per_particle_corrections.xlsx')
    save_correction_params = join(path_corrections, 'correction_params.xlsx')

    path_results = join(save_path, 'results')
    if not exists(path_results):
        makedirs(path_results)

    save_results = join(path_results, 'correction_results.xlsx')

    # ------------------------
    # setup: mask_dict
    mask_boundary = mask_dict['mask_boundary']

    # ------------------------
    # setup: exp_dict
    microns_per_pixel = exp_dict['microns_per_pixel']
    calibration_direction = exp_dict['calibration_direction']

    # ------------------------
    # setup: results_dict
    results_dict = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Use already computed results (if available) to skip re-computing every iteration

    if compute_all is False:
        if exists(export_boundary_points_in_focus):
            cficts_boundary_df = pd.read_excel(export_boundary_points_in_focus)
            z_f_mean_boundary = cficts_boundary_df.z_f_calc.mean()
            boundary_pids = cficts_boundary_df.id.unique()

        if exists(export_interior_points_in_focus):
            cficts_interior_df = pd.read_excel(export_interior_points_in_focus)
            interior_pids = cficts_interior_df.id.unique()

        if exists(export_all_points_in_focus):
            cficts_orig_df = pd.read_excel(export_all_points_in_focus)
            z_f_mean_orig = cficts_orig_df.z_f_calc.mean()

    else:
        # --------------------------------------------------------------------------------------------------------------
        # Compute the in-focus z-coordinates for all particles
        """
        NOTES:
            * This process currently uses "cficts" (a dict. of dataframes), whereas, above uses dataframes.
                --> This descrepancy should be solved.
        """

        # set up some toggles
        perform_correction = False  # CAREFUL HERE: this actually corrects the particles' coordinates
        per_particle_correction = True  # CAREFUL HERE: correct per-particle or, if false, per-collection.
        plot_average_z_per_particle = True
        include_quartiles = 3

        # --------------------------------------------------------------------------------------------------------------
        # read files
        """
        ----- FOLLOWING THE ABOVE NOTE -----
        The following should read dataframes and not dictionaries of dataframes.
            --> Because correction happens at the per-dataframe level; specifically, for single calibration dataframes.
        """

        # calib correction coords
        cficts = io.read_files('df', path_calib_coords, calib_sort_strings, filetype, startswith=calib_sort_strings[0])

        # read details
        cficts_details = details.parse_filename_to_details(path_calib_coords, calib_sort_strings, filetype,
                                                           startswith=calib_sort_strings[0])
        cficts_details = details.read_dficts_coords_to_details(cficts, cficts_details, calib=True)

        # --------------------------------------------------------------------------------------------------------------
        # calculate in-focus z-coordinates for all particles

        cficts_orig = cficts.copy()
        cficts_orig, z_f_mean_orig = calc_calib_in_focus_z(cficts_orig,
                                                            dficts=None,
                                                            perform_correction=perform_correction,
                                                            per_particle_correction=per_particle_correction,
                                                            only_quartiles=include_quartiles,
                                                            show_z_focus_plot=False,
                                                            show_particle_plots=False,
                                                            plot_average_z_per_particle=plot_average_z_per_particle,
                                                            save_plots=save_plots,
                                                            save_path=path_figs_all_in_focus_correction,
                                                            num_particle_plots=1,
                                                            round_to_decimals=4)

        cficts_orig_df = cficts_orig[1.0]
        z_f_mean_orig = z_f_mean_orig[1.0]
        # --------------------------------------------------------------------------------------------------------------
        # Apply boundary mask to calibration particles

        cficts_df = cficts[1.0]
        boundary_pids, interior_pids = boundary.get_boundary_particles(mask_boundary, cficts_df, return_interior_particles=True)

        cficts_boundary = cficts.copy()
        cficts_boundary, z_f_mean_boundary = calc_calib_in_focus_z(cficts_boundary,
                                                                    dficts=None,
                                                                    only_particle_ids=boundary_pids,
                                                                    perform_correction=perform_correction,
                                                                    per_particle_correction=per_particle_correction,
                                                                    only_quartiles=include_quartiles,
                                                                    show_z_focus_plot=False,
                                                                    show_particle_plots=False,
                                                                    plot_average_z_per_particle=plot_average_z_per_particle,
                                                                    save_plots=save_plots,
                                                                    save_path=path_figs_boundary_in_focus_correction,
                                                                    num_particle_plots=1,
                                                                    round_to_decimals=4)

        cficts_boundary_df = cficts_boundary[1.0]
        z_f_mean_boundary = z_f_mean_boundary[1.0]
        # --------------------------------------------------------------------------------------------------------------
        # Export particle collections to Excel

        # store boundary points dataframe
        calib_image_nearest_focus = int(np.round(z_f_mean_boundary[1.0], 0))
        cficts_boundary_df = cficts_boundary_df[cficts_boundary_df['frame'] == calib_image_nearest_focus]
        cficts_boundary_df.to_excel(excel_writer=export_boundary_points_in_focus, index=True, index_label=True)

        # store boundary points dataframe
        cficts_interior_df = cficts_orig_df[cficts_orig_df['frame'] == calib_image_nearest_focus].copy()
        cficts_interior_df = cficts_interior_df[cficts_interior_df['id'].isin(interior_pids)]
        cficts_interior_df.to_excel(excel_writer=export_interior_points_in_focus, index=True, index_label=True)

        # store all points dataframe where boundary points approx. in focus
        cficts_orig_df = cficts_orig_df[cficts_orig_df['frame'] == calib_image_nearest_focus]
        cficts_orig_df.to_excel(excel_writer=export_all_points_in_focus, index=True, index_label=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Fit a 3D plane to the boundary particles to get true zero-plane (UNITS: MICRONS)

    # units: (x, y units: pixels; z units: microns)
    xy_units = 'microns'
    z_units = 'microns'

    # fit plane (x, y, z units: microns)
    #       This is used for calculating a tilt angle
    points_microns = np.stack((cficts_boundary_df.x * microns_per_pixel,
                               cficts_boundary_df.y * microns_per_pixel,
                               cficts_boundary_df.z_f_calc)).T
    px_microns, py_microns, pz_microns, popt_microns = fit.fit_3d(points_microns, fit_function='plane')
    pz_microns = np.round(pz_microns, 8)

    # tilt angle (degrees)
    tilt_x = np.rad2deg(np.arctan((pz_microns[0, 1] - pz_microns[0, 0]) / (px_microns[0, 1] - px_microns[0, 0])))
    tilt_y = np.rad2deg(np.arctan((pz_microns[1, 0] - pz_microns[0, 0]) / (py_microns[1, 0] - py_microns[0, 0])))
    print("x-tilt = {} degrees".format(np.round(tilt_x, 3)))
    print("y-tilt = {} degrees".format(np.round(tilt_y, 3)))

    # ------------------------------------------------------------------------------------------------------------------
    # Fit a 3D plane to the boundary particles to get true zero-plane (UNITS: PIXELS)

    # units: (x, y units: pixels; z units: microns)
    xy_units = 'pixels'
    z_units = 'microns'

    # fit plane
    #       This is used for converting x-y pixel locations into z-coords for correction.
    points = np.stack((cficts_boundary_df.x, cficts_boundary_df.y, cficts_boundary_df.z_f_calc)).T
    px, py, pz, popt = fit.fit_3d(points, fit_function='plane')
    pz = np.round(pz, 8)
    d, normal = popt[3], popt[4]

    # ------------------------------------------------------------------------------------------------------------------
    # Plot the fitted 3D plane

    if save_plots:
        # plot boundary points + fitted plane
        fig = plt.figure(figsize=(6.5, 5))
        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(cficts_boundary_df.x, cficts_boundary_df.y, cficts_boundary_df.z_f_calc,
                            c=cficts_boundary_df.z_f_calc, label=r'$z_{in-focus}$')
            ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue', label=r'$fit_{3D plane}$')
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])
        plt.suptitle(r"$0 = n_x x + n_y y + n_z z - d$" + "= {}x + {}y + {}z - {} \n"
                                                          "(x, y: pixels; z: microns)".format(np.round(normal[0], 3),
                                                                                            np.round(normal[1], 3),
                                                                                            np.round(normal[2], 3),
                                                                                            np.round(d, 3)),
                     y=0.875)
        plt.subplots_adjust(hspace=-0.1, wspace=0.15)
        plt.savefig(join(path_figs_in_focus_fit, 'calib_boundaries_in-focus-z_fit_3d.png'))
        plt.close()

        # plot all points + fitted plane
        fig = plt.figure(figsize=(6.5, 5))
        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(cficts_orig_df.x, cficts_orig_df.y, cficts_orig_df.z_f_calc, c=cficts_orig_df.z_f_calc, s=1)
            ax.scatter(cficts_boundary_df.x, cficts_boundary_df.y, cficts_boundary_df.z_f_calc, color='red', s=2, marker='d', alpha=0.5)
            ax.plot_surface(px, py, pz, alpha=0.4, color='red')
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])
        plt.suptitle(r"$0 = n_x x + n_y y + n_z z - d$" + "= {}x + {}y + {}z - {} \n"
                                                          "(x, y: pixels; z: microns)".format(np.round(normal[0], 3),
                                                                                            np.round(normal[1], 3),
                                                                                            np.round(normal[2], 3),
                                                                                            np.round(d, 3)),
                     y=0.875)
        plt.subplots_adjust(hspace=-0.1, wspace=0.15)
        plt.savefig(join(path_figs_in_focus_fit, 'calib_all_in-focus-z_fit_3d.png'))
        plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Plot the corrected boundary points + fitted 3D plane
    z_corr_tilt_name = 'z_corr_tilt'
    dfc = correct_z_by_fit_function(cficts_boundary_df,
                                    fit_func=functions.calculate_z_of_3d_plane,
                                    popt=popt,
                                    x_param='x',
                                    y_param='y',
                                    z_param='z_f_calc',
                                    z_corr_name=z_corr_tilt_name)

    if save_plots:

        # plot 3D scatter: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
        fig = plt.figure(figsize=(6.5, 5))
        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            ax.scatter(cficts_boundary_df.x, cficts_boundary_df.y, cficts_boundary_df.z, color='black', s=2, marker='.', label='z')
            ax.scatter(cficts_boundary_df.x, cficts_boundary_df.y, cficts_boundary_df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
            ax.scatter(dfc.x, dfc.y, dfc[z_corr_tilt_name], color='red', s=4, marker='d', label=r'$z_{tilt corrected}$')
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                ax.legend(loc='upper left', bbox_to_anchor=(0.85, 0.85))
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])
        plt.suptitle('Tilt correction demonstration on boundary particles \n (not used)', y=0.875)
        plt.subplots_adjust(hspace=-0.1, wspace=0.15)
        plt.savefig(join(path_figs_in_focus_fit, 'demo_tilt_correction_on_boundary_in-focus_3d.png'))

        # plot 2D scatter along x-y axes: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
        x_line = np.linspace(cficts_boundary_df.x.min(), cficts_boundary_df.x.max())
        y_line = np.linspace(cficts_boundary_df.y.min(), cficts_boundary_df.y.max())
        zx_line = functions.calculate_z_of_3d_plane(x=x_line, y=np.zeros_like(x_line), popt=popt)
        zy_line = functions.calculate_z_of_3d_plane(x=np.zeros_like(y_line), y=y_line, popt=popt)
        zz_line = functions.calculate_z_of_3d_plane(x=x_line, y=y_line, popt=popt)

        fig, [ax1, ax2] = plt.subplots(nrows=2, figsize=(3.5, 3.3))
        ax1.scatter(cficts_boundary_df.x, cficts_boundary_df.z, color='black', s=2, marker='.', label='z')
        ax1.scatter(cficts_boundary_df.x, cficts_boundary_df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
        ax1.scatter(dfc.x, dfc[z_corr_tilt_name], color='red', s=4, marker='d', label=r'$z_{corrected}$')
        ax1.plot(x_line, zx_line, color='gray', alpha=0.25, linestyle='-.', label=r'$z_{plane, x}$')
        ax1.plot(x_line, zz_line, color='gray', alpha=0.5, linestyle='--', label=r'$z_{plane, xy}$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('z')
        ax1.set_title('Tilt correction demonstration on boundary particles \n (not used)', fontsize=6)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1))
        ax2.scatter(cficts_boundary_df.y, cficts_boundary_df.z, color='black', s=2, marker='.')
        ax2.scatter(cficts_boundary_df.y, cficts_boundary_df.z_f_calc, color='cornflowerblue', s=2)
        ax2.scatter(dfc.y, dfc[z_corr_tilt_name], color='red', s=4, marker='d')
        ax2.plot(y_line, zy_line, color='gray', alpha=0.25, linestyle=':', label=r'$z_{plane, y}$')
        ax2.plot(y_line, zz_line, color='gray', alpha=0.5, linestyle='--')
        ax2.set_xlabel('y')
        ax2.set_ylabel('z')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1))
        plt.tight_layout(h_pad=0.125)
        plt.savefig(join(path_figs_in_focus_fit, 'demo_tilt_correction_on_boundary_in-focus_2d_xy.png'))

    # ------------------------------------------------------------------------------------------------------------------
    # Calculate the per-particle correction for all particles
    dfc = correct_z_by_fit_function(cficts_orig_df,
                                    fit_func=functions.calculate_z_of_3d_plane,
                                    popt=popt,
                                    x_param='x',
                                    y_param='y',
                                    z_param='z_f_calc',
                                    z_corr_name=z_corr_tilt_name)

    if save_plots:

        # plot 3D scatter: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
        fig = plt.figure(figsize=(6.5, 5))
        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            ax.scatter(cficts_orig_df.x, cficts_orig_df.y, cficts_orig_df.z, color='black', s=2, marker='.', label='z')
            ax.scatter(cficts_orig_df.x, cficts_orig_df.y, cficts_orig_df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
            ax.scatter(dfc.x, dfc.y, dfc[z_corr_tilt_name], color='red', s=4, marker='d', label=r'$z_{corrected}$')
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                ax.legend(loc='upper left', bbox_to_anchor=(0.85, 0.85))
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])
        plt.suptitle('Tilt correction demonstration on all particles \n (not used)', y=0.875)
        plt.subplots_adjust(hspace=-0.1, wspace=0.15)
        plt.savefig(join(path_figs_in_focus_fit, 'demo_tilt_correction_on_all_in-focus_3d.png'))

        # plot 2D scatter along x-y axes: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
        x_line = np.linspace(cficts_orig_df.x.min(), cficts_orig_df.x.max())
        y_line = np.linspace(cficts_orig_df.y.min(), cficts_orig_df.y.max())
        zx_line = functions.calculate_z_of_3d_plane(x=x_line, y=np.zeros_like(x_line), popt=popt)
        zy_line = functions.calculate_z_of_3d_plane(x=np.zeros_like(y_line), y=y_line, popt=popt)
        zz_line = functions.calculate_z_of_3d_plane(x=x_line, y=y_line, popt=popt)

        fig, [ax1, ax2] = plt.subplots(nrows=2, figsize=(3.5, 3.3))
        ax1.scatter(cficts_orig_df.x, cficts_orig_df.z, color='black', s=2, marker='.', label='z')
        ax1.scatter(cficts_orig_df.x, cficts_orig_df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
        ax1.scatter(dfc.x, dfc[z_corr_tilt_name], color='red', s=4, marker='d', label=r'$z_{corrected}$')
        ax1.plot(x_line, zx_line, color='gray', alpha=0.25, linestyle='-.', label=r'$z_{plane, x}$')
        ax1.plot(x_line, zz_line, color='gray', alpha=0.5, linestyle='--', label=r'$z_{plane, xy}$')
        ax1.set_xlabel('x')
        ax1.set_ylabel('z')
        ax1.set_title('Tilt correction demonstration on all particles \n (not used)', fontsize=6)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1))
        ax2.scatter(cficts_orig_df.y, cficts_orig_df.z, color='black', s=2, marker='.')
        ax2.scatter(cficts_orig_df.y, cficts_orig_df.z_f_calc, color='cornflowerblue', s=2)
        ax2.scatter(dfc.y, dfc[z_corr_tilt_name], color='red', s=4, marker='d')
        ax2.plot(y_line, zy_line, color='gray', alpha=0.25, linestyle=':', label=r'$z_{plane, y}$')
        ax2.plot(y_line, zz_line, color='gray', alpha=0.5, linestyle='--')
        ax2.set_xlabel('y')
        ax2.set_ylabel('z')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1))
        plt.tight_layout(h_pad=0.125)
        plt.savefig(join(path_figs_in_focus_fit, 'demo_tilt_correction_on_all_in-focus_2d_xy.png'))

        # plot 3D scatter with colorbar: z_corr (corrected) == the "true" z
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(dfc.x, dfc.y, dfc[z_corr_tilt_name], c=dfc[z_corr_tilt_name], s=4)
        ax.set_xlabel(r'$x \: (pix)$')
        ax.set_ylabel(r'$y \: (pix)$')
        ax.set_zlabel(r'$z \: (\mu m)$')
        ax.view_init(5, 45)
        ax.set_title("In-focus z-coordinates after tilt correction \n (note: tilt correction is not used)", fontsize=6)
        plt.colorbar(sc, shrink=0.5)
        plt.tight_layout()
        plt.savefig(join(path_figs_in_focus_fit, 'demo_tilt_correction_on_in-focus_profile_3d.png'))

    # ------------------------------------------------------------------------------------------------------------------
    # correct z such that z=0 is the in-focus position
    z_corr_in_focus_name = 'z_corr_in_focus'
    dfc = cficts_orig_df.copy()
    dfc[z_corr_in_focus_name] = dfc['z'] - dfc['z_f_calc']

    if save_plots:
        # plot 2D scatter along x-y axes: z
        fig, [ax1, ax2] = plt.subplots(nrows=2)
        ax1.scatter(dfc.x, dfc[z_corr_in_focus_name], c=dfc[z_corr_in_focus_name], s=3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('z')
        ax1.set_title('z-coordinates after in-focus correction \n (note: in-focus correction is used)', fontsize=6)
        ax2.scatter(dfc.y, dfc[z_corr_in_focus_name], c=dfc[z_corr_in_focus_name], s=3)
        ax2.set_xlabel('y')
        ax2.set_ylabel('z')
        plt.tight_layout()
        plt.savefig(join(path_figs_in_focus_fit, 'calib_all_in-focus-z_2d_xy_correction_z_offset.png'))

        # plot 3D scatter with colorbar: z_corr (corrected) == the "true" z
        fig = plt.figure(figsize=(6.5, 5))
        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(dfc.x, dfc.y, dfc[z_corr_in_focus_name], c=dfc[z_corr_in_focus_name], s=4)
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])
        plt.suptitle('z-coordinates after in-focus correction \n (note: in-focus correction is used)', y=0.875)
        plt.subplots_adjust(hspace=-0.1, wspace=0.15)
        plt.savefig(join(path_figs_in_focus_fit, 'in-focus_correction_3d.png'))

    # ------------------------------------------------------------------------------------------------------------------
    # Export to Excel

    # export per-particle corrections to Excel
    df_per_particle_corrections = dfc[['id', 'x', 'y', 'z_f_calc']]
    df_per_particle_corrections.to_excel(save_corrections)

    # export correction parameters to dictionary and save as Excel
    correction_dict = package_correction_params(correction_type='in-focus',
                                                function='calculate_z_of_3d_plane',
                                                popt=popt,
                                                save_path=save_correction_params,
                                                xy_units=xy_units,
                                                z_units=z_units)

    # ------------------------------------------------------------------------------------------------------------------

    results_dict.update({
        'z_correction_name': 'z_f_calc',
        'z_f_mean_orig': z_f_mean_orig,
        'z_f_mean_boundary': z_f_mean_boundary,
        'boundary_pids': boundary_pids,
        'interior_pids': interior_pids,
        'x-tilt': tilt_x,
        'y-tilt': tilt_y,
    })

    df_results = pd.DataFrame.from_dict(data=results_dict, orient='index')
    df_results.to_excel(save_results)

    results_dict.update({
        'correction_dict': correction_dict,
        'per_particle_corrections': df_per_particle_corrections,
    })

    return results_dict

# ----------------------------------------------------------------------------------------------------------------------


def correct_z_by_fit_function(df, fit_func, popt, x_param='x', y_param='y', z_param='z', z_corr_name='z_corr_tilt',
                              mean_z_at_zero=False):
    """
    Correct the z-coordinates in dataframe "df" using a fitting function with x, y inputs and z outputs.

    NOTE: z_param (or the z-value that is to be corrected) should usually be the in-focus z-coord (z_f_calc)

    :param df:
    :param fit_func:
    :param popt:
    :param x_param:
    :param y_param:
    :param z_param:
    :param z_corr_name:
    :return:
    """
    dfc = df.copy()

    z_on_plane = fit_func(dfc[x_param], dfc[y_param], popt)

    if mean_z_at_zero:
        dfc[z_corr_name] = dfc[z_param] - z_on_plane
    else:
        z_orig = dfc[z_param]
        z_intercept = fit_func(x=0, y=0, popt=popt)
        dfc[z_corr_name] = z_intercept + z_orig - z_on_plane

    # print("mean z corrected = {}".format(dfc[z_corr_name].mean()))

    return dfc


def correct_z_by_xy_surface(df, fit_func, fit_params, fit_var='z'):
    df['z_cal_surf'] = fit_func(np.array([df.x.to_numpy(), df.y.to_numpy()]), *fit_params)
    df['z_corr'] = df[fit_var] - df.z_cal_surf
    return df


def correct_z_by_spline(df, bispl, param_z):
    df['z_cal_surf'] = bispl.ev(df.x, df.y)
    df['z_corr'] = df[param_z] - df.z_cal_surf
    return df


def correct_z_by_plane_tilt(dfcal, dftest, param_zf='z_f', param_z='z', param_z_true='z_true',
                            popt_calib=None, params_correct=None):
    """

    :param dfcal: calibration dataframe with z_in-focus coordinate.
    :param dftest: dataframe to correct OR == None which adds 'z_plane' to dfcal and enables correction of a column.
    :param param_zf: column in dfcal with z_in-focus coordinate.
    :param param_z: column in dftest to correct; if None, a ValueError is raised.
    :param param_z_true: column in dftest to correct; if None, no column is corrected.
    :param popt_calib: (optional) fitted plane parameters
    :param params_correct: (optional) additional columns to correct using 'z_plane'
    :return:
    """

    # --- prepare inputs

    # dfcal: 1 id == 1 (x, y)
    if len(dfcal) > len(dfcal.id.unique()):
        dfcal = dfcal.groupby('id').mean().reset_index()

    # dftest:
    if dftest is None:
        dftest = dfcal.copy()

    if param_z is None:
        raise ValueError('param_z cannot be None. Use a different function if you just want to calculate z_plane.')

    # fit plane on calibration in-focus (x, y, z units: pixels)
    if popt_calib is None:
        points_pixels = np.stack((dfcal.x, dfcal.y, dfcal[param_zf])).T
        px_pixels, py_pixels, pz_pixels, popt_calib = fit.fit_3d_plane(points_pixels)

    # calculate z-position on plane
    dftest['z_plane'] = functions.calculate_z_of_3d_plane(dftest.x, dftest.y, popt=popt_calib)

    # calculate corrected z-position
    dftest['z_corr'] = dftest[param_z] - dftest['z_plane']

    if param_z_true is not None:
        dftest['z_true_corr'] = dftest[param_z_true] - dftest['z_plane']

    # calculate corrected z-position of other params
    if params_correct is not None:
        for pc in params_correct:
            dftest[pc + '_corr'] = dftest[pc] - dftest['z_plane']

    return dftest


def correct_z_by_in_focus_z(dft, dfc, column_zf='zf_from_peak_int'):
    """

    :param dft: standard test_coords dataframe (id, frame, z_true, z, error, etc)
    :param dfc: should be 'calib_spct_pid_defocus_stats' dataframe.
    :param column_zf: column to read 'zf' from 'dfc'.
    :return:
    """

    # create arrays of unique_identifier and value to map
    calibration_pids = dfc.id.values
    calibration_pids_zf = dfc[column_zf].values

    # create a new column in dft to map values to
    dft['zf'] = dft['id']
    dft = dft[dft['id'].isin(calibration_pids)]

    # create the mapping dictionary
    mapping_dict = {calibration_pids[i]: calibration_pids_zf[i] for i in range(len(calibration_pids))}

    # insert the mapped values
    dft.loc[:, 'zf'] = dft.loc[:, 'zf'].map(mapping_dict)

    dft['z_true'] = dft.z_true - dft.zf
    dft['z'] = dft.z - dft.zf
    dft['zfo'] = dft.zf
    dft['zf'] = dft.zf - dft.zf


def correct_from_mapping(df_to_correct, df_particle_corrections, z_correction_name='z_f_calc', z_params='z'):
    if df_to_correct is not None:
        raise ValueError('Function no longer supported. Needs to be updated. Use correct_nonuniform_particle_ids as'
                         'a template.')
    else:
        dft = df_to_correct
    return dft


def correct_nonuniform_particle_ids(baseline, coords, threshold=5, dropna=True, save_path=False, save_id=None):
    """
    Read 'baseline' and 'coords' dataframes and uniformize 'coords' particle ID's to match baseline's particle ID's.

    To run:
    coords_orig, mapping_dict, cids_not_mapped = correct.correct_nonuniform_particle_ids(baseline, coords, threshold=5, dropna=True, save_path=False, save_id=None)

    :param baseline: per-particle-correction dataframe (needs to have one (x, y) for one particle ID.
    :param coords: dataframe of identical particle distribution but particle identifications.
    :param baseline_padding:
    :param coords_padding:
    :param threshold:
    :return:
    """

    if len(baseline) > len(baseline.id.unique()):
        baseline = baseline.groupby('id').mean().reset_index()

    # plot before and after particle locations
    fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 2, size_y_inches))

    ax1.set_title('Before')
    ax1.scatter(coords.x, coords.y, s=5, color='blue')
    ax1.scatter(baseline.x, baseline.y, s=4, marker='x', color='tab:red')

    # copy so as not to change original
    baseline = baseline.copy()
    coords_orig = coords.copy()
    coords = coords.copy()

    # group coords by id
    coords = coords.groupby('id', sort=True).mean()  # sort by id
    coords_ids = coords.index.values.tolist()

    # coords
    coords_xy = coords[['x', 'y']].values.tolist()  # convert to list because ordering is important

    # baseline
    baseline = baseline.sort_values('id')
    baseline = baseline.set_index('id')
    baseline_ids = baseline.index.values.tolist()
    baseline_xy = baseline[['x', 'y']]

    # NearestNeighbors
    nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_xy.values)
    distances, indices = nneigh.kneighbors(np.array(coords_xy))

    mapping_dict = {}
    cids_not_mapped = []
    for distance, idx, cid in zip(distances, indices, coords_ids):
        if distance < threshold:
            mapping_dict.update({cid: baseline_xy.index.values[idx.squeeze()]})
        else:
            cids_not_mapped.append([cid, distance, idx])

    # warn if multiple cids are assigned the same baseline ID
    if len(list(mapping_dict.values())) != len(set(list(mapping_dict.values()))):
        print("Duplicated mapped particle ID's. Need to resolve.")

    # map ID's
    coords_orig['id'] = coords_orig['id'].map(mapping_dict)

    # drop NaNs
    if dropna:
        coords_orig = coords_orig.dropna()

    # save figures
    export_orig = coords_orig[['id', 'x', 'y', 'z']]
    export_orig = export_orig.groupby('id').mean()
    export_orig = export_orig.rename({'x': 'x_test', 'y': 'y_test'})

    ax2.set_title('After')
    ax2.scatter(export_orig.x, export_orig.y, s=5, color='blue', label='Test Coords')
    ax2.scatter(baseline.x, baseline.y, s=4, marker='x', color='tab:red', label='Calibration Coords')

    ax1.set_xlabel('x (pix)')
    ax1.set_ylabel('y (pix)')
    ax2.set_xlabel('x (pix)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(join(save_path, 'before_after_padding_shift_id{}.png'.format(save_id)))
        plt.close()
    else:
        plt.show()

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))
    ax.scatter(export_orig.x, export_orig.y, color='tab:blue', s=50, alpha=0.5, label='Matched Test ID')
    ax.scatter(baseline.x, baseline.y, color='red', marker='.', s=4, label='Original Calib. ID')
    ax.set_ylabel('y (pix)')
    ax.set_xlabel('x (pix)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(join(save_path, 'overlay_scatter_matched_ids_id{}.png'.format(save_id)))
        plt.close()
    else:
        plt.show()

    # export validation dataframe
    if save_path:
        export_validation = pd.concat([export_orig, baseline], axis=1, join='inner', sort=True)
        export_validation.to_excel(join(save_path, 'uniformize_ids_validation_id{}.xlsx'.format(save_id)))
        coords_orig.to_excel(join(save_path, 'test_coords_uniformize-ids_{}.xlsx'.format(save_id)), index=False)

    return coords_orig, mapping_dict, cids_not_mapped


def package_correction_params(correction_type, function, popt, save_path=None, xy_units=None, z_units=None):

    a, b, c, d, normal = popt[0], popt[1], popt[2], popt[3], popt[4]

    correction_dict = {
        'correction_type': correction_type,
        'function': function,
        'xy_units': xy_units,
        'z_units': z_units,
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'nx': normal[0],
        'ny': normal[1],
        'nz': normal[2],
    }

    if save_path is not None:
        df = pd.DataFrame.from_dict(data=correction_dict, orient='index')
        df.to_excel(save_path)

    return correction_dict


def read_correction_params(path, parse_popt=False):
    df_correct = pd.read_excel(path, index_col=0)
    dict_tier = df_correct.to_dict(orient='dict')
    correction_dict = dict_tier[0]

    if parse_popt:
        a = correction_dict['a']
        b = correction_dict['b']
        c = correction_dict['c']
        d = correction_dict['d']
        nx = correction_dict['nx']
        ny = correction_dict['ny']
        nz = correction_dict['nz']
        normal = np.array([nx, ny, nz])
        popt = np.array([a, b, c, d, normal], dtype='object')
        correction_dict.update({'popt': popt})

    return correction_dict


# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# ----------------------------------------------- -------------------- -------------------------------------------------


# ----------------------------------------------- DEPRECATED FUNCTIONS -------------------------------------------------


def calc_calib_in_focus_z(cficts, dficts=None, only_particle_ids=None,
                          perform_correction=False, per_particle_correction=False,
                          only_quartiles=2, num_particle_plots=1, round_to_decimals=3, fit_widths=None,
                          plot_average_z_per_particle=True,
                          show_particle_plots=False, show_z_focus_plot=True, save_plots=False, save_path=None):
    """
    Method:
        This function calculates the "in-focus" z-coord. for each particle and collection average.
    Returns:
        (minimum) additional columns for per-particle in-focus z-coord (z_f_calc) and per-collection (z_img_f_calc)
        (additional) actually performs the correction ONLY ON "cficts"

    KEY NOTES:
        1. using "dficts", the calibration correction is only performed on particle ID's in dficts. Thus, dficts should
        only be used when the calibration correction will directly lead into analyzing dficts. Otherwise, it's best to
        analyze all of the particles in cficts because this is a generalizable calibration/correction.

    :param cficts:
    :param dficts: pass in dficts to only calibrate particles ID's that are in dficts.
    :param only_particle_ids: if you want to perform the in-focus calibration using only certain particle ID's.
    :param fit_widths:
    :param perform_correction: toggle to return corrected or original coordinates.
    :param per_particle_correction: toggle to perform per-particle-correction on all or some datasets in list.
    :param plot_average_z_per_particle: toggle to plot average or all z-coords per particle.
    :param only_quartiles:
    :param show_particle_plots:
    :param show_z_focus_plot:
    :param save_plots:
    :param save_path:
    :param num_particle_plots:
    :param round_to_decimals:
    :return:
    """

    # widths of peaks to fit parabola across particle intensity
    if fit_widths is None:
        fit_widths = [1, 3, 5]

    # standard deviation quartiles to accept for mean in-focus z-coord.
    if only_quartiles:
        std_filter = only_quartiles
    else:
        std_filter = 2

    # apply per-particle-correction to all or some datasets in list
    if not isinstance(per_particle_correction, list):
        per_particle_correction = [per_particle_correction] * len(cficts.keys())
    ppc = iter(per_particle_correction)

    # "dfc" is a new dictionary that will replace "cficts" because dictionaries are immutable.
    dfc = {}
    z_f_means = {}

    for name, df in cficts.items():

        per_particle_correction = next(ppc)

        df = df.copy()
        df['z_f_calc'] = 0

        # if particle ID's are available, use only these for calibration correction
        if only_particle_ids:
            pids = only_particle_ids
            boolean_series = df.id.isin(pids)
            df_excluded = df[~boolean_series]
            df = df[boolean_series]
            pids = df.id.unique()
        elif dficts:
            # if dficts (meta-assessment) is available, get only these particle ID's for calibration correction
            pids = dficts[name].id.unique()
            boolean_series = df.id.isin(pids)
            df = df[boolean_series]
            pids = df.id.unique()
        else:
            pids = df.id.unique()

        plot_counter = 0
        for pid in pids:

            # get dataframe for only particle 'pid'
            dfpid = df[df['id'] == pid].copy()
            dfpid = dfpid.reset_index()

            # find maximum intensity value to fit parabola onto
            z_max_int = dfpid['peak_int'].idxmax()

            fit_zs = []
            fit_peaks = []
            for fit_width in fit_widths:
                # fit a parabola for each width across the particle's intensity profile centered at its peak
                # and append the fitted maximum (in-focus z-coord) to a list to average

                # adjust z_max_int to enable parabola fit width
                if len(dfpid['peak_int']) - 1 - z_max_int < np.ceil(fit_width / 2):
                    z_max_int_orig = z_max_int
                    z_max_int = int(np.floor(len(dfpid['peak_int']) - 1 - fit_width / 2))
                elif z_max_int < np.floor(fit_width / 2):
                    z_max_int_orig = z_max_int
                    z_max_int = int(np.ceil(fit_width / 2))

                z_maxs = dfpid.iloc[z_max_int - fit_width:z_max_int + fit_width + 1].z
                peak_int_maxs = dfpid.iloc[z_max_int - fit_width:z_max_int + fit_width + 1].peak_int

                popt, pcov, fit_func = fit.fit(z_maxs, peak_int_maxs, fit_function=None)

                z_fits = np.linspace(z_maxs.min(), z_maxs.max(), 100)
                peak_int_fits = fit_func(z_fits, popt[0], popt[1], popt[2])

                z_infocus_fit = z_fits[np.argmax(peak_int_fits)]
                fit_zs.append(z_infocus_fit)
                fit_peaks.append(np.max(peak_int_fits))

                # OPTIONAL: plot the fitted profiles and maximums
                if fit_width == fit_widths[-1]:
                    if any([save_plots, show_particle_plots]) and plot_counter < num_particle_plots:
                        fig, ax = plt.subplots()
                        # ax.axvline(np.mean(fit_zs), ymin=0.4, ymax=0.925, color='black', linewidth=0.5, linestyle='--', alpha=0.5, label=r'$z_{f}$')
                        ax.axvline(np.mean(fit_zs), ymin=0, ymax=0.05, color='tab:red', linewidth=0.5, alpha=1)
                        ax.plot(z_fits, peak_int_fits, color='black', label='fit')
                        ax.scatter(z_maxs, peak_int_maxs, s=5, color='tab:blue', label='data')
                        ax.scatter(fit_zs, fit_peaks, s=5, color='tab:red', label='peaks')
                        ax.set_xlabel(r'$z_{c}\: (A.U.)$')
                        ax.set_ylabel(r'$I\: (A.U.)$')
                        ax.grid(alpha=0.125)
                        ax.set_title('ID{}, '.format(name) +
                                     r'$p_{ID} (x, y)=$' + 'pID{}, ({}, {})'.format(pid,
                                                                                    int(dfpid.x.to_numpy()[0]),
                                                                                    int(dfpid.y.to_numpy()[0])
                                                                                    ))
                        ax.legend(loc='lower center')
                        plt.tight_layout()
                        if save_plots:
                            plt.savefig(join(save_path, 'id{}_fit_peak_intensity_pid{}.png'.format(name, pid)))
                        if show_particle_plots:
                            plt.show()
                        plt.close(fig)
                        plot_counter += 1

            # calculate mean(fit_zs) only on NON-OUTLIER values
            fit_zs = np.array(fit_zs)
            fit_peaks = np.array(fit_peaks)
            if not all_equal(fit_zs):
                boolean_outliers = np.abs(fit_zs - np.mean(fit_zs)) < 1 * np.std(fit_zs)
                fit_zs_non_outliers = fit_zs[boolean_outliers]
                fit_peaks_non_outliers = fit_peaks[boolean_outliers]
                z_fitted = np.round(np.mean(fit_zs_non_outliers), round_to_decimals + 1)
            else:
                z_fitted = fit_zs[0]
                fit_zs_non_outliers = None
                fit_peaks_non_outliers = None

            # PER-PARTICLE IN-FOCUS CALCULATIONS AND CORRECTIONS ARE PERFORMED HERE!
            # "z_f_calc" is the "per-particle" in-focus z-coordinate

            # Replace z_f_calc (which was initialized == 0) with calculated z_f_calc (average of fitted maximums)
            df['z_f_calc'] = df['z_f_calc'].where(df['id'] != pid, z_fitted, axis='index')

            # if we want to actually adjust the particle's coordinates on a "per-particle" basis, we do that here
            if perform_correction & per_particle_correction:
                df['z'] = df['z'].where(df['id'] != pid, df['z'] - z_fitted, axis='index')

        # PER-COLLECTION IN-FOCUS CALCULATIONS ARE PERFORMED HERE!
        # NOTE: No corrections are performed here. This is largely just for printing the results to the console.
        z_f_calcs = df.groupby('id').mean().z_f_calc.to_numpy()
        print('{} particles assessed for calibration correction.'.format(len(z_f_calcs)))
        z_f_max = np.round(np.max(z_f_calcs), 1)
        z_f_min = np.round(np.min(z_f_calcs), 1)

        # calculate mean(z_f_calc) only on NON-OUTLIER values
        if len(z_f_calcs) > 1:
            z_f0_mean = np.mean(z_f_calcs)
            z_f0_limits = np.std(z_f_calcs) * std_filter
            boolean_z_f_outliers = np.abs(z_f_calcs - np.mean(z_f_calcs)) < z_f0_limits
            z_f_calcs_non_outliers = z_f_calcs[boolean_z_f_outliers]
            z_f_mean = np.round(np.mean(z_f_calcs_non_outliers), round_to_decimals)
            z_f_limits = np.round(np.std(z_f_calcs_non_outliers), 2) * std_filter
            print('{}/{} particles z_f = {} +/- {} um.'.format(len(z_f_calcs_non_outliers), len(z_f_calcs),
                                                               z_f_mean, np.round(z_f_limits, 2)))
        else:
            z_f0_mean = z_f_calcs[0]
            z_f0_limits = None
            z_f_mean = z_f_calcs[0]
            z_f_limits = None

        # "z_img_f_calc" is the "per-collection" in-focus z-coordinate which is the average of all the per-particle z's
        df['z_img_f_calc'] = z_f_mean

        # OPTIONAL: show or save z(x) and z(y) figures
        if show_z_focus_plot or save_plots:
            # average z-coord. per particle ID
            if plot_average_z_per_particle:
                dfplot = df.groupby('id').mean()
                color = dfplot.index
            else:
                dfplot = df.copy()
                color = dfplot.id

            fig, [ax1, ax2] = plt.subplots(nrows=2)

            # plot x
            ax1.axhline(y=z_f0_mean, color='darkred', linestyle='--', linewidth=1, alpha=0.5, label=r'$z_{f0}$')
            if z_f0_limits:
                ax1.axhline(y=z_f0_mean + z_f0_limits, color='darkred', linestyle='--', linewidth=0.5, alpha=0.25, label=r'$z_{f0,\: lim}$')
                ax1.axhline(y=z_f0_mean - z_f0_limits, color='darkred', linestyle='--', linewidth=0.5, alpha=0.25)
            ax1.axhline(y=z_f_mean, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label=r'$z_{f}$')
            if z_f_limits:
                ax1.axhline(y=z_f_mean + z_f_limits, color='darkgreen', linestyle='--', linewidth=0.5, alpha=0.25, label=r'$z_{f,\: lim}$')
                ax1.axhline(y=z_f_mean - z_f_limits, color='darkgreen', linestyle='--', linewidth=0.5, alpha=0.25)
            ax1.scatter(dfplot.x, dfplot.z_f_calc, c=color, s=2, label=r'$z_{raw}(p_{i})$')
            ax1.set_xlabel('x (pixels)')
            ax1.set_ylabel(r'z $(\mu m)$')
            ax1.grid(alpha=0.125)
            ax1.legend(loc='upper left', fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0))
            ax1.set_title('ID{}: '.format(name) + r'($z_{min}, z_{f}, z_{max})=$' +
                          ' ({}, {}, {})'.format(z_f_min, np.round(z_f_mean, 2), z_f_max) + r' $\mu m$')

            # plot y
            ax2.axhline(y=z_f0_mean, color='darkred', linestyle='--', linewidth=1, alpha=0.5)
            if z_f0_limits:
                ax2.axhline(y=z_f0_mean + z_f0_limits, color='darkred', linestyle='--', linewidth=0.5, alpha=0.25)
                ax2.axhline(y=z_f0_mean - z_f0_limits, color='darkred', linestyle='--', linewidth=0.5, alpha=0.25)
            ax2.axhline(y=z_f_mean, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5)
            if z_f_limits:
                ax2.axhline(y=z_f_mean + z_f_limits, color='darkgreen', linestyle='--', linewidth=0.5, alpha=0.25)
                ax2.axhline(y=z_f_mean - z_f_limits, color='darkgreen', linestyle='--', linewidth=0.5, alpha=0.25)
            ax2.scatter(dfplot.y, dfplot.z_f_calc, c=color, s=2)
            ax2.set_xlabel('y (pixels)')
            ax2.set_ylabel(r'z $(\mu m)$')
            ax2.grid(alpha=0.125)
            plt.tight_layout()
            if save_plots:
                plt.savefig(join(save_path, '{}_in-focus-z-xy-scatter.png'.format(name)))
            if show_z_focus_plot:
                plt.show()
            plt.close(fig)

        # OPTIONAL: show or save z(x_i, y_i) and z(x_f, y_f) figures
        if show_z_focus_plot or save_plots:

            # get "original" in-focus z-coords of particles
            calib_image_nearest_focus = int(np.round(z_f_mean, 0))
            df_orig = df[df['frame'] == calib_image_nearest_focus]

            fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=(6, 3))

            # plot "original" in-focus z-coords (particles as they were imaged during acquisition)
            ax1.scatter(df_orig.x, df_orig.y, c=df_orig.z, s=7)
            ax1.set_xlabel('x (pixels)')
            ax1.set_ylabel('y (pixels)')
            ax1.set_title('Original z-coord = {} '.format(calib_image_nearest_focus) + r'$\mu m$')
            # plot in-focus z-coords (particles as "in-focus" from post-processing above)
            sc = ax2.scatter(df_orig.x, df_orig.y, c=df_orig.z_f_calc, s=7)
            ax2.set_xlabel('x (pixels)')
            ax2.set_title('In-focus z-coords = {} '.format(np.round(z_f_mean, 2)) + r'$\mu m$' + ' (avg)')
            fig.colorbar(sc, ax=ax2)
            # plot excluded particles as highly transparent
            if only_particle_ids:
                df_excluded_orig = df_excluded[df_excluded['frame'] == calib_image_nearest_focus]
                ax1.scatter(df_excluded_orig.x, df_excluded_orig.y, s=4, marker='.', color='gray', alpha=0.5)
                ax2.scatter(df_excluded_orig.x, df_excluded_orig.y, s=4, marker='.', color='gray', alpha=0.5)

            plt.tight_layout()
            if save_plots:
                plt.savefig(join(save_path, '{}_in-focus-z-2d-scatter.png'.format(name)))
            if show_z_focus_plot:
                plt.show()
            plt.close(fig)


        # PER-COLLECTION IN-FOCUS CORRECTIONS ARE PERFORMED HERE!
        # NOTE: the actual corrections are performed here.
        # NOTE: this should only be used if "per-particle" corrections cannot be made for some reason.
        if perform_correction is True and per_particle_correction is False:
            df['z'] = df['z'] - df['z_img_f_calc']

        # update "dfc" to replace "cficts" with corrected particle coordinates
        # AND / OR
        # additional columns in the dataframe which denote per-particle and per-collection "in-focus" z-coords.
        dfc.update({name: df})
        z_f_means.update({name: z_f_mean})

    return dfc, z_f_means


def correct_ficts_z_by_in_focus_z(cficts, dficts, per_particle_correction=False):
    """
    Use this correction method ONLY WHEN the TRUE distribution of particles is known very accurately.
    --> The only example of this is when particles are positioned on a flat glass slide.
        --> Even in this special case, this correction method should be identical to the more generalizable tilt correct
    :param cficts:
    :param dficts:
    :param per_particle_correction:
    :return:
    """

    if not isinstance(per_particle_correction, list):
        per_particle_correction = [per_particle_correction] * len(cficts.keys())
    ppc = iter(per_particle_correction)

    dcficts = {}

    for name, df in cficts.items():

        per_particle_correction = next(ppc)

        dft = dficts[name]

        # Note on the application of this script to plotting half calibration depths
        """
        This should only be used when aligning plots of a full calibration stack and half calibration stack.
        if name - 10 in dficts.keys():
            dfht_true = True
            dfht = dficts[name - 10]
        else:
            dfht_true = False
        """
        dfht_true = False
        dfht = None

        z_img_f_calc = df['z_img_f_calc'].unique()

        if per_particle_correction:
            pids = df.id.unique()
            for pid in pids:
                dfpid = df[df['id'] == pid].copy()
                z_f_calc = dfpid['z_f_calc'].mean()
                dft['z'] = dft['z'].where(dft['id'] != pid, dft['z'] - z_f_calc, axis='index')
                dft['z_true'] = dft['z_true'].where(dft['id'] != pid, dft['z_true'] - z_f_calc, axis='index')

                if dfht_true:
                    dfht['z'] = dfht['z'].where(dfht['id'] != pid, dfht['z'] - z_f_calc, axis='index')
                    dfht['z_true'] = dfht['z_true'].where(dfht['id'] != pid, dfht['z_true'] - z_f_calc, axis='index')

        else:
            dft['z'] = dft['z'] - z_img_f_calc
            dft['z_true'] = dft['z_true'] - z_img_f_calc

            if dfht_true:
                dfht['z'] = dfht['z'] - z_img_f_calc
                dfht['z_true'] = dfht['z_true'] - z_img_f_calc

        if dfht_true:
            dcficts.update({name - 10: dfht})

        dcficts.update({name: dft})

    return dcficts


def correct_nonuniform_particle_ids_with_padding(baseline, coords, baseline_padding, coords_padding, threshold=5,
                                                 dropna=True, save_path=None, save_id=None):
    """
    Read 'baseline' and 'coords' dataframes and uniformize 'coords' particle ID's to match baseline's particle ID's.

    :param baseline: per-particle-correction dataframe
    :param coords: dataframe of identical particle distribution but different padding or particle identifications.
    :param baseline_padding:
    :param coords_padding:
    :param threshold:
    :return:
    """
    # plot before and after particle locations
    fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches*2, size_y_inches))

    ax1.set_title('Before Padding Shift')
    ax1.scatter(coords.x, coords.y, s=5, color='blue')
    ax1.scatter(baseline.x, baseline.y, s=4, marker='x', color='tab:red')

    # fix x, y coordinates to account for differences in image padding during GDPyT analysis
    padding_diff = baseline_padding - coords_padding
    coords['x'] = coords['x'] + padding_diff
    coords['y'] = coords['y'] + padding_diff

    # copy so as not to change original
    baseline = baseline.copy()
    coords_orig = coords
    coords = coords.copy()

    # group coords by id
    coords = coords.groupby('id', sort=True).mean()  # sort by id
    coords_ids = coords.index.values.tolist()

    # coords
    coords_xy = coords[['x', 'y']].values.tolist()  # convert to list because ordering is important

    # baseline
    baseline = baseline.sort_values('id')
    baseline = baseline.set_index('id')
    baseline_ids = baseline.index.values.tolist()
    baseline_xy = baseline[['x', 'y']]

    # NearestNeighbors
    nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_xy.values)
    distances, indices = nneigh.kneighbors(np.array(coords_xy))

    mapping_dict = {}
    cids_not_mapped = []
    for distance, idx, cid in zip(distances, indices, coords_ids):
        if distance < threshold:
            mapping_dict.update({cid: baseline_xy.index.values[idx.squeeze()]})
        else:
            cids_not_mapped.append([cid, distance, idx])

    # warn if multiple cids are assigned the same baseline ID
    if len(list(mapping_dict.values())) != len(set(list(mapping_dict.values()))):
        logger.warning("Duplicated mapped particle ID's. Need to resolve.")

    # map ID's
    coords_orig['id'] = coords_orig['id'].map(mapping_dict)

    # drop NaNs
    if dropna:
        coords_orig = coords_orig.dropna()

    # save figures
    export_orig = coords_orig[['id', 'x', 'y', 'z']]
    export_orig = export_orig.groupby('id').mean()

    ax2.set_title('After Padding Shift: ({}, {}) pixels'.format(padding_diff, padding_diff))
    ax2.scatter(export_orig.x, export_orig.y, s=5, color='blue', label='Test Coords')
    ax2.scatter(baseline.x, baseline.y, s=4, marker='x', color='tab:red', label='Calibration Coords')

    ax1.set_xlabel('x (pix)')
    ax1.set_ylabel('y (pix)')
    ax2.set_xlabel('x (pix)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(join(save_path, 'before_after_padding_shift_id{}.png'.format(save_id)))
        plt.close()
    else:
        plt.show()

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches*1.25, size_y_inches))
    ax.scatter(export_orig.x, export_orig.y, c=export_orig.index, s=10, alpha=0.25, label='Matched Test ID')
    ax.scatter(baseline.x, baseline.y, c=baseline.index, marker='x', s=4, label='Original Calib. ID')
    ax.set_ylabel('y (pix)')
    ax.set_xlabel('x (pix)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(join(save_path, 'overlay_scatter_matched_ids_id{}.png'.format(save_id)))
        plt.close()
    else:
        plt.show()

    # export validation dataframe
    if save_path:
        export_validation = pd.concat([export_orig, baseline], axis=1, join='inner', sort=True)
        export_validation.to_excel(join(save_path, 'uniformize_ids_validation_id{}.xlsx'.format(save_id)))

    return coords_orig, mapping_dict, cids_not_mapped