# plot topographic map from membrane coordinates (xc, yc, r) and deflection (rz)

import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import CloughTocher2DInterpolator

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

import filter
import analyze
from correction import correct
from utils import fit, functions, bin, io, plotting, modify, plot_collections, boundary
from utils.plotting import lighten_color, set_3d_axes_equal
from utils.functions import fSphericalUniformLoad

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

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. SETUP

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/topographic-mapping'

path_data = join(base_dir, 'data')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# file name
fn = 'test_coords_id1_corrected.xlsx'

# optics
method = 'idpt'
mag_eff = 5.0
microns_per_pixel = 3.2
frame_rate = 24.444
start_time = 1.25

# ----------------------------------------------------------------------------------------------------------------------
# 2. TOPOGRAPHIC MAPPING

# topographic mapping --- PART 1

plot_topographic_map = False
if plot_topographic_map:

    # ----------------------------------------------------------------------------------------------------------------------
    # 2. READ X, Y, Z COORDS

    # 2.1 perform tilt correction
    perform_tilt_correction = False

    if perform_tilt_correction:

        # setup file paths
        fp_test_coords = join(path_data, 'test_coords_id1_dynamic_neg_first.xlsx')
        fp_tilt_corr = join(path_data, 'calib-coords_tilt_from-field-curvature-corrected.xlsx')

        # 2.1. READ FIELD-CURVATURE CORRECTED FITTED PLANE FOR TILT CORRECTION

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
        # 2.2. PERFORM TILT-CORRECTION

        # read test coords dataframe
        dft = pd.read_excel(fp_test_coords)

        # correct plane tilt
        dft = correct.correct_z_by_plane_tilt(dfcal=None,
                                              dftest=dft,
                                              param_zf='none',
                                              param_z='z',
                                              param_z_true='none',
                                              popt_calib=popt_calib,
                                              params_correct=None,
                                              dict_fit_plane=None,
                                              )

        # 2.3 EXPORT CORRECTED COORDS
        dft.to_excel(join(path_data, 'test_coords_id1_dynamic_neg_first_tilt-corrected.xlsx'))

        # ---

    else:
        # read tilt-corrected dataframe
        dft = pd.read_excel(join(path_data, fn), index_col=0)

    # ---

    # add a 'time' column from frame_rate
    dft['time'] = dft['frame'] / frame_rate

    # ---

    # 2.2 get particles on each boundary
    interior_pids = io.read_txt_file_to_list(join(path_data, 'additional/boundary/interior_pids.txt'), data_type='int')

    # split interior pids into (1) lower right membrane and (2) upper left membrane
    df_interior_pids = dft[dft['id'].isin(interior_pids)]

    dflr = df_interior_pids[df_interior_pids['y'] > 220]
    dful = df_interior_pids[df_interior_pids['y'] < 220]

    # get mean z-coordinate of interior particles prior to 'start time'
    z_i_mean_lr = dflr[dflr['time'] < start_time].z_corr.mean()
    z_i_mean_ul = dful[dful['time'] < start_time].z_corr.mean()
    z_i_mean = np.mean([z_i_mean_lr, z_i_mean_ul])

    # add z-offset so all particles are approximately at z=0 initially
    # dft['z_corr'] = dft['z_corr'] - z_i_mean
    # dflr['z_corr'] = dflr['z_corr'] - z_i_mean_lr
    # dful['z_corr'] = dful['z_corr'] - z_i_mean_ul

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. TOPOGRAPHIC MAPPING

# topographic mapping --- PART 2

if plot_topographic_map:
    # ------------------------------------------------------------------------------------------------------------------
    # 1. SETUP

    # mechanics
    h = 20e-6
    youngs_modulus = 5e6

    # x-y coordinates of membranes (in pixels)
    xc_lr, yc_lr, r_edge_lr = 423 * microns_per_pixel, 502 * microns_per_pixel, 250 * microns_per_pixel
    xc_ul, yc_ul, r_edge_ul = 169 * microns_per_pixel, 35 * microns_per_pixel, 157 * microns_per_pixel

    # ------------------------------------------------------------------------------------------------------------------
    # 2. READ FILES

    # read dataframe
    df = pd.read_excel(join(path_data, 'id1_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'))

    # ------------------------------------------------------------------------------------------------------------------
    # 3. DATA PRE-PROCESSING

    # remove rows where membranes deflect opposite directions
    df['lrp_ulp_norm'] = df['fit_lr_pressure'] / df['fit_lr_pressure'].abs() * \
                         df['fit_ul_pressure'] / df['fit_ul_pressure'].abs()
    df = df[df['lrp_ulp_norm'] > 0]

    # remove rows with negative deflection
    df = df[df['fit_lr_pressure'] > 0]

    # remove rows where smaller membrane deflects more
    df = df[df['rz_lr'] > df['rz_ul']]

    # add a 'time' column from frame_rate
    df['time'] = df['frame'] / frame_rate

    # ------------------------------------------------------------------------------------------------------------------
    # 3. CALCULATE DEFLECTION PROFILE

    # get constants: membrane radii
    r_lr = df.iloc[0]['r_lr']
    r_ul = df.iloc[0]['r_ul']

    # create radial space vectors
    r_fit_lr = np.linspace(0, r_lr)
    r_fit_ul = np.linspace(0, r_ul)

    # get list of: radii, pressure(frame)
    arr = df[['frame', 'time', 'fit_lr_pressure', 'fit_ul_pressure']].to_numpy()

    # instantiate plate theory class
    fsphere_lr = fSphericalUniformLoad(r=r_lr * 1e-6, h=h, youngs_modulus=youngs_modulus)
    fsphere_ul = fSphericalUniformLoad(r=r_ul * 1e-6, h=h, youngs_modulus=youngs_modulus)

    # iterate through frames
    for fr, t, lrp, ulp in arr:

        # fit plate theory
        rz_lr = fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(r=r_fit_lr * 1e-6, P=lrp) * 1e6
        rz_ul = fsphere_ul.spherical_uniformly_loaded_simply_supported_plate_r_p(r=r_fit_ul * 1e-6, P=ulp) * 1e6

        # get x, y, z coordinates of particles in this frame
        dft_fr = dft[dft['frame'] == fr]
        xp, yp, zp = dft_fr.x * microns_per_pixel, dft_fr.y * microns_per_pixel, dft_fr.z_corr

        # add 'r' param in reference to each membrane
        dflr['rr'] = functions.calculate_radius_at_xy(dflr.x * microns_per_pixel, dflr.y * microns_per_pixel,
                                                      xc=xc_lr, yc=yc_lr)
        dful['rr'] = functions.calculate_radius_at_xy(dful.x * microns_per_pixel, dful.y * microns_per_pixel,
                                                      xc=xc_ul, yc=yc_ul)

        # --------------------------------------------------------------------------------------------------------------
        # 4. PLOT

        # setup

        # modifiers
        save_figs = True
        show_figs = False

        # ---

        # plot radial deflection profile
        plot_radial_profile = True
        if plot_radial_profile:

            path_figs_radial = path_figs + '/radial'
            if not os.path.exists(path_figs_radial):
                os.makedirs(path_figs_radial)

            dflr_fr = dflr[dflr['frame'] == fr]
            dful_fr = dful[dful['frame'] == fr]

            fig, ax = plt.subplots()
            ax.plot(r_fit_lr, rz_lr + z_i_mean_lr, label='lr')
            ax.plot(r_fit_ul, rz_ul + z_i_mean_ul, label='ul')

            ax.scatter(dflr_fr.rr, dflr_fr.z_corr, color=sciblue, s=2)
            ax.scatter(dful_fr.rr, dful_fr.z_corr, color=scigreen, s=2)

            ax.set_xlabel('r')
            ax.set_ylabel('z')
            ax.legend()
            if save_figs:
                plt.savefig(path_figs_radial + '/plot_z_by_r_frame{}.png'.format(fr))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # plot topographic map
        plot_topography = False
        if plot_topography:

            # setup
            cmap = plt.cm.viridis  # full range: plt.cm.coolwarm
            vmin, vmax = -10, 155  # full range: -175, 155; positive range: -10, 155
            alpha = 0.75
            elev, azim = 20, 30
            z_axis_scale = 0.015
            zticks_full_range = [-150, -75, 0, 75, 150]
            zticks_pos_range = [0, 50, 100, 150]

            # figure
            fig = plt.figure(figsize=(size_x_inches * 1.2, size_y_inches * 1.2))
            ax = fig.add_subplot(projection='3d')

            # Create the mesh in polar coordinates and compute corresponding Z.
            p = np.linspace(0, 2 * np.pi, 50)
            R_lr, P_lr = np.meshgrid(r_fit_lr, p)
            R_ul, P_ul = np.meshgrid(r_fit_ul, p)

            # Compute corresponding Z
            Z_lr = fsphere_lr.spherical_uniformly_loaded_simply_supported_plate_r_p(r=R_lr * 1e-6, P=lrp) * 1e6
            Z_ul = fsphere_ul.spherical_uniformly_loaded_simply_supported_plate_r_p(r=R_ul * 1e-6, P=ulp) * 1e6

            # Express the mesh in the cartesian system.
            X_lr, Y_lr = R_lr * np.cos(P_lr), R_lr * np.sin(P_lr)
            X_ul, Y_ul = R_ul * np.cos(P_ul), R_ul * np.sin(P_ul)

            # shift x-y coordinates to membrane centers
            X_lr, Y_lr = X_lr + xc_lr, Y_lr + yc_lr
            X_ul, Y_ul = X_ul + xc_ul, Y_ul + yc_ul

            # plot the surface
            surf = ax.plot_surface(X_lr, Y_lr, Z_lr + z_i_mean_lr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
            ax.plot_surface(X_ul, Y_ul, Z_ul + z_i_mean_ul, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

            # scatter plot points
            ax.scatter(xp, yp, zp, color='black', marker='.', s=2)

            # plot the radii centers
            # ax.scatter(xc_lr, yc_lr, color='gray', marker='o', s=1)
            # ax.scatter(xc_ul, yc_ul, color='gray', marker='D', s=1)

            # plot field-of-view box
            xfov = np.array([0, 512, 512, 0, 0]) * microns_per_pixel
            yfov = np.array([0, 0, 512, 512, 0]) * microns_per_pixel
            zfov = np.ones_like(xfov) * np.mean([z_i_mean_lr, z_i_mean_ul])
            ax.plot(xfov, yfov, zfov, color='black', alpha=0.125)

            # set_3d_axes_equal(ax, z_axis_scale=z_axis_scale)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim([-50, 2000])
            ax.set_ylim([-100, 2300])
            ax.set_zlim3d([vmin, vmax])
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
            # ax.zaxis.set_ticks(zticks_full_range, label=zticks_full_range, labelpad=-50)
            ax.zaxis.set_ticks(zticks_pos_range, label=zticks_pos_range, labelpad=-50)

            # fig.colorbar(surf, ax=ax, aspect=25, shrink=0.5, location='left', pad=0.15, panchor=(0, 0.75))
            # ax.dist = 8.15

            if save_figs:
                plt.savefig(path_figs + '/plot_3d_topography_frame{}.png'.format(fr))
            if show_figs:
                plt.show()
            plt.close()

        j = 1

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 3. AVERAGES

# topographic mapping
analyze_results = True

if analyze_results:
    # read
    df = pd.read_excel(join(path_data, 'id1_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'))

    # export statistics
    export_stats = True

    if export_stats:
        export_data = {'measurement_depth': 270}

        # averages
        mean_rmse_lr = df.fit_lr_rmse.mean()
        mean_rmse_lr_pos = df[df['rz_lr_signed'] > 0].fit_lr_rmse.mean()
        mean_rmse_lr_neg = df[df['rz_lr_signed'] < 0].fit_lr_rmse.mean()

        mean_rmse_ul = df.fit_ul_rmse.mean()
        mean_rmse_ul_pos = df[df['rz_ul_signed'] > 0].fit_ul_rmse.mean()
        mean_rmse_ul_neg = df[df['rz_ul_signed'] < 0].fit_ul_rmse.mean()

        mean_rmse = np.mean([mean_rmse_lr, mean_rmse_ul])
        mean_rmse_pos = np.mean([mean_rmse_lr_pos, mean_rmse_ul_pos])
        mean_rmse_neg = np.mean([mean_rmse_lr_neg, mean_rmse_ul_neg])

        mean_data = {'mean_rmse': mean_rmse,
                     'mean_rmse_pos': mean_rmse_pos,
                     'mean_rmse_neg': mean_rmse_neg,
                     'mean_rmse_lr': mean_rmse_lr,
                     'mean_rmse_ul': mean_rmse_ul,
                     'mean_rmse_lr_pos': mean_rmse_lr_pos,
                     'mean_rmse_ul_pos': mean_rmse_ul_pos,
                     'mean_rmse_lr_neg': mean_rmse_lr_neg,
                     'mean_rmse_ul_neg': mean_rmse_ul_neg,
                     }

        # max's
        max_rz_lr_pos = df[df['rz_lr_signed'] > 0].rz_lr.max()
        max_theta_peak_deg_lr_pos = df[df['rz_lr_signed'] > 0].theta_lr_peak_deg.max()
        max_rz_lr_neg = df[df['rz_lr_signed'] < 0].rz_lr_signed.min()
        max_theta_peak_deg_lr_neg = df[df['rz_lr_signed'] < 0].theta_lr_peak_deg.min()

        max_rz_ul_pos = df[df['rz_ul_signed'] > 0].rz_ul.max()
        max_theta_peak_deg_ul_pos = df[df['rz_ul_signed'] > 0].theta_ul_peak_deg.max()
        max_rz_ul_neg = df[df['rz_ul_signed'] < 0].rz_ul_signed.min()
        max_theta_peak_deg_ul_neg = df[df['rz_ul_signed'] < 0].theta_ul_peak_deg.min()

        max_data = {'max_rz_lr_pos': max_rz_lr_pos,
                    'max_theta_peak_deg_lr_pos': max_theta_peak_deg_lr_pos,
                    'max_rz_lr_neg': max_rz_lr_neg,
                    'max_theta_peak_deg_lr_neg': max_theta_peak_deg_lr_neg,
                    'max_rz_ul_pos': max_rz_ul_pos,
                    'max_theta_peak_deg_ul_pos': max_theta_peak_deg_ul_pos,
                    'max_rz_ul_neg': max_rz_ul_neg,
                    'max_theta_peak_deg_ul_neg': max_theta_peak_deg_ul_neg,
                    }

        # ---

        dicts = [mean_data, max_data]
        for d in dicts:
            export_data.update(d)

        export_df = pd.DataFrame.from_dict(data=export_data, orient='index')
        export_df = export_df.rename(columns={0: 'mean'})
        export_df['normalized'] = export_df['mean'] / export_df.loc['measurement_depth', 'mean']
        export_df.to_excel(path_results + '/export_data.xlsx')

        # ---

    # ---

    # plot

    path_figs_by_z = path_figs + '/by_z'
    if not os.path.exists(path_figs_by_z):
        os.makedirs(path_figs_by_z)

    # setup
    ms = 3

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(df.rz_lr, df.fit_lr_rmse, 'o', ms=ms, label='lr')
    ax1.set_xlabel(r'$z \: (\mu m)$')
    ax1.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax1.set_ylim([0, 10])
    ax1.legend()

    ax2.plot(df.rz_ul, df.fit_ul_rmse, 'o', ms=ms, color=scired, label='ul')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax2.set_ylim([0, 10])
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(path_figs_by_z + '/rmse-z_by_abs-z.svg')
    plt.show()
    plt.close()

    # ---

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(df.rz_lr_signed, df.fit_lr_rmse, 'o', ms=ms, label='lr')
    ax1.set_xlabel(r'$z \: (\mu m)$')
    ax1.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax1.set_ylim([0, 10])
    ax1.legend()

    ax2.plot(df.rz_ul_signed, df.fit_ul_rmse, 'o', ms=ms, color=scired, label='ul')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax2.set_ylim([0, 10])
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(path_figs_by_z + '/rmse-z_by_signed-z.svg')
    plt.show()
    plt.close()

    j = 1