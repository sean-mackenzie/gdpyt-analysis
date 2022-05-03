# imports

import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, Akima1DInterpolator
from scipy.stats import norm
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version

from utils import bin, fit, functions, io, plotting
from utils.plotting import lighten_color
from correction import correct
import analyze

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import collections, colors, transforms

# formatting
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'
fontP = FontProperties()
fontP.set_size('medium')

plt.style.use(['science', 'ieee', 'std-colors'])
# plt.style.use(['science', 'scatter'])
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# -------------------------------------- SPCT STATS ANALYSIS FUNCTIONS -------------------------------------------------


def plot_spct_stats(base_dir):
    # modifiers
    calc_plane_angle = True
    spct_image_stats = True
    position_3d_by_frame = True
    precision = True
    per_mindx_id = True
    per_percent_dx_diameter_id = True
    sampling_frequency = True

    # export results & show, save plots
    export_results = True
    save_figs = False
    show_figs = False

    # filepaths
    path_calib_coords = join(base_dir, 'coords/calib-coords')
    path_results = join(base_dir, 'results')
    path_figs = join(base_dir, 'figs')

    # --- experimental details
    method = 'spct'

    # read calibration coords
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method=method)

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

    # dictionaries
    dict_spct_pid_plane_angle = None
    dict_penetrance = None
    dict_spct_stats_bin_z = None
    dict_1d_static_precision_id = None
    dict_spct_stats_sampling_frequency = None

    # ------------------------------------------------- FIT PLANE ------------------------------------------------------

    if calc_plane_angle:

        # --------------------------------------- FIT PLANE (MICRONS) --------------------------------------------------

        # Fit a 3D plane to the boundary particles to get true zero-plane (UNITS: MICRONS)
        if dfcpid is not None:
            if 'x' in dfcpid.columns:
                dfcg = dfcpid
                param_zf = 'zf_from_peak_int'
                df_for_plane_angle = 'dfcpid_xy'
        elif dfc is not None:
            dfcg = dfc.groupby('id').mean()
            param_zf = 'z_f'
            df_for_plane_angle = 'dfc_correction_coords'
        else:
            raise ValueError('Either dfpid_xy (pid_defocus) or dfc (correction coords) must be available.')

        # fit plane (x, y, z units: microns)
        #       This is used for calculating a tilt angle
        points_microns = np.stack((dfcg.x * microns_per_pixel,
                                   dfcg.y * microns_per_pixel,
                                   dfcg[param_zf])).T
        px_microns, py_microns, pz_microns, popt_microns = fit.fit_3d_plane(points_microns)

        # tilt angle (degrees)
        tilt_x = np.rad2deg(np.arctan((pz_microns[0, 1] - pz_microns[0, 0]) / (px_microns[0, 1] - px_microns[0, 0])))
        tilt_y = np.rad2deg(np.arctan((pz_microns[1, 0] - pz_microns[0, 0]) / (py_microns[1, 0] - py_microns[0, 0])))
        print("x-tilt = {} degrees".format(np.round(tilt_x, 3)))
        print("y-tilt = {} degrees".format(np.round(tilt_y, 3)))

        dict_spct_pid_plane_angle = {'param_zf': param_zf,
                                     'df_used_for_plane_angle': df_for_plane_angle,
                                     'x_tilt_degrees': tilt_x,
                                     'y_tilt_degrees': tilt_y,
                                     }

    # ---------------------------------------------- SPCT IMAGE STATS --------------------------------------------------

    if spct_image_stats:

        df = dfcstats.dropna()
        cleaned_up = False

        if cleaned_up:

            dfg = df.groupby('frame').count()
            fig, ax = plt.subplots()
            ax.plot(dfg.index, dfg.z_corr)
            plt.show()

            # check if there is mean displacement of the entire particle group
            df = df[(df['z_corr'] > -50) & (df['z_corr'] < 50)]
            dfcounts = df.groupby('id').count().reset_index()
            df = df[df['id'].isin(dfcounts[dfcounts['id'] == dfcounts.id.max()].id.unique())]
            dfg = df.groupby('frame').mean().reset_index()

            fig, ax = plt.subplots()
            axr = ax.twinx()
            ax.plot(dfg.z_corr, dfg.x, label='x')
            axr.plot(dfg.z_corr, dfg.y, label='y')
            plt.show()

            # ----------------- SPCT (SINGLE IDs, Z)

            # 3D scatter plot of apparent lateral position as a function of axial position
            inspect_pids = np.random.choice(df.id.unique(), 10, replace=False)

            df = df[(df['z_corr'] > -50) & (df['z_corr'] < 50)]

            for pid in inspect_pids:
                dfpid = df[df['id'] == pid]

                # correct gaussian centers
                dfpid['gauss_xc_corr'] = df['gauss_xc'] - dfpid['x'] + (dfpid['y'] - dfpid.iloc[0].y)
                dfpid['gauss_yc_corr'] = dfpid['gauss_yc'] - dfpid['y'] + (dfpid['x'] - dfpid.iloc[0].x)

                dfpid = dfpid[['id', 'z', 'z_corr', 'x', 'y',
                               'gauss_xc', 'gauss_yc', 'gauss_xc_corr', 'gauss_yc_corr']]

                fig = plt.figure(figsize=(size_x_inches / 1.5, size_y_inches * 2))
                ax = Axes3D(fig, box_aspect=(1.0, 1.0, 3.0))

                # 3D scatter
                ax.scatter(dfpid.gauss_xc_corr, dfpid.gauss_yc_corr, dfpid.z_corr, s=3, marker='o')

                # 2D scatter on bottom of plot
                ax.plot(dfpid.gauss_xc_corr, dfpid.gauss_yc_corr, np.ones_like(dfpid.gauss_xc_corr.values) * -19.95,
                        '-o', ms=0.5, color='gray')

                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlim3d(bottom=-20, top=20)
                zticks = [-20, -10, 0, 10, 20]
                ax.set_zticks(zticks)
                ax.set_zticklabels(zticks)
                ax.xaxis._axinfo["grid"].update({"linewidth": 0.0125})
                ax.yaxis._axinfo["grid"].update({"linewidth": 0.0125})
                ax.zaxis._axinfo["grid"].update({"linewidth": 0.25})
                plt.grid(visible=False)
                plt.grid(visible=True, which='major')
                # ax.view_init(elev, azim)
                plt.savefig(path_figs + '/pid{}_x{}y{}_xcg-ycg-zcorr_3d.png'.format(pid,
                                                                                    np.round(dfpid.x.mean(), 0),
                                                                                    np.round(dfpid.y.mean(), 0)
                                                                                    )
                            )
                plt.show()
                plt.close()

            # plot single particles
            xparam = 'z_corr'
            plot_columns = ['gauss_xc_corr', 'gauss_yc_corr', 'gauss_sigma_x_y']
            particle_ids = 44

            plot_spct_stats_id_by_param(df, xparam, plot_columns, particle_ids, save_figs, path_figs, show_figs)

            # ----------------------------------------------- SPCT (ID, Z) -------------------------------------------------

            columns_to_bin = ['x', 'y']
            bins = [5, 11]
            plot_columns = ['gauss_xc_corr', 'gauss_yc_corr', 'gauss_sigma_x_y']
            column_to_plot_along = 'z_corr'
            round_to_decimals = [1, 1]

            # ----------------- SPCT (IDs sliced along X, Y, Z)
            for low_level_bins_to_plot in range(bins[1]):
                plot_spct_stats_compare_ids_by_along_param(df,
                                                           columns_to_bin,
                                                           bins,
                                                           low_level_bins_to_plot,
                                                           plot_columns,
                                                           column_to_plot_along,
                                                           round_to_decimals,
                                                           save_figs, path_figs, show_figs)

            # ----------------- SPCT (ID PAIRS, X, Z)

            xparam = 'z_corr'

            # filter by x-y deviation
            dfg = df.groupby('id').std().reset_index()
            dfg['xy_std'] = np.mean([dfg.x, dfg.y])
            passing_ids = [p for p in particle_ids if p not in dfg[dfg['xy_std'] > 1.35].id.unique()]
            df = df[df.id.isin(passing_ids)]

            # compare particles on opposite sides of x-axis
            compare_param = 'x'
            x_left_pids = df.sort_values('x', ascending=True).id.unique()[0:5]
            x_right_pids = df.sort_values('x', ascending=False).id.unique()[0:5]

            # plot particle pairs
            particle_ids = [[i, j] for i, j in zip(x_left_pids, x_right_pids)]

            plot_spct_stats_compare_ids_by_param(df, xparam, compare_param, plot_columns, particle_ids,
                                                 save_figs, path_figs, show_figs)

            # ----------------- SPCT (ID PAIRS, Y, Z)

            # compare particles on opposite sides of x-axis
            compare_param = 'y'
            y_top_pids = df.sort_values('y', ascending=True).id.unique()[0:3]
            y_bottom_pids = df.sort_values('y', ascending=False).id.unique()[0:3]

            # plot particle pairs
            particle_ids = [[i, j] for i, j in zip(y_top_pids, y_bottom_pids)]

            plot_spct_stats_compare_ids_by_param(df, xparam, compare_param, plot_columns, particle_ids,
                                                 save_figs, path_figs, show_figs)

        # ------------------------------------------- SPCT PENETRANCE --------------------------------------------------
        """
        penetrance = evaluation of particles in >80% of max possible frames.
        
            * max possible frames = the number of frames for the maximally identified particles.
            
                ** This is necessary because for some datasets, there will be many images where zero particles are 
                identified. Thus the penetrance here might == 0; rendering this evaluation not extremely useful.
        
        """
        dfcounts = df.groupby('id').count().reset_index()
        max_num_frames = dfcounts.z_corr.max()

        df_penetrance = dfcounts[dfcounts['z_corr'] > max_num_frames * 0.8]

        penetrance_num_frames = max_num_frames * 0.8
        penetrance_num_pids = len(df_penetrance.id.unique())

        dict_penetrance = {'max_idd_num_frames': max_num_frames,
                           'penetrance_num_frames': penetrance_num_frames,
                           'penetrance_num_pids': penetrance_num_pids,
                           }

        # --------------------------------------------- SPCT (Z, R) ----------------------------------------------------

        columns_to_bin = ['z_corr', 'r']
        plot_columns = ['gauss_sigma_x_y']
        column_to_count = 'id'
        bin_z = [-20, -10, 0, 10, 20]
        bin_r = 4
        min_num_bin = 10

        plot_spct_stats_bin_2d(df,
                               columns_to_bin,
                               column_to_count,
                               bins=[bin_z, bin_r],
                               round_to_decimals=[0, 0],
                               min_num_bin=min_num_bin,
                               save_figs=save_figs,
                               path_figs=path_figs,
                               show_figs=show_figs,
                               export_results=export_results,
                               path_results=path_results,
                               plot_columns=plot_columns
                               )

        # ----------------------------------------------- SPCT (Z, X) --------------------------------------------------

        columns_to_bin = ['z_corr', 'x']
        bin_x = 4

        plot_spct_stats_bin_2d(df,
                               columns_to_bin,
                               column_to_count,
                               bins=[bin_z, bin_x],
                               round_to_decimals=[0, 0],
                               min_num_bin=min_num_bin,
                               save_figs=save_figs,
                               path_figs=path_figs,
                               show_figs=show_figs,
                               export_results=export_results,
                               path_results=path_results,
                               plot_columns=plot_columns
                               )

        # ----------------------------------------------- SPCT (Z, Y) --------------------------------------------------

        columns_to_bin = ['z_corr', 'y']
        bin_y = 4

        plot_spct_stats_bin_2d(df,
                               columns_to_bin,
                               column_to_count,
                               bins=[bin_z, bin_y],
                               round_to_decimals=[0, 0],
                               min_num_bin=min_num_bin,
                               save_figs=save_figs,
                               path_figs=path_figs,
                               show_figs=show_figs,
                               export_results=export_results,
                               path_results=path_results,
                               plot_columns=plot_columns
                               )

        # ----------------------------------------------- SPCT (Z) -----------------------------------------------------

        column_to_bin = 'z_corr'
        column_to_count = 'id'
        bins = num_frames
        round_to_decimal = 4

        dict_spct_stats_bin_z = plot_spct_stats_bin_z(df,
                                                      column_to_bin,
                                                      column_to_count,
                                                      bins,
                                                      round_to_decimal,
                                                      save_figs,
                                                      path_figs,
                                                      show_figs,
                                                      export_results=export_results,
                                                      path_results=path_results,
                                                      )

        # ----------------------------------------------- SPCT (ID) ----------------------------------------------------
        column_to_count = 'frame'

        plot_spct_stats_bin_id(df, column_to_count, num_pids, save_figs, path_figs, show_figs,
                               export_results=export_results, path_results=path_results)

    # ------------------------------------------ PRECISION (Z, R, ID) --------------------------------------------------

    if position_3d_by_frame:

        df = dfcstats
        particle_ids_to_inspect = [21, 27, 29, 44, 45, 65]

        for pid in particle_ids_to_inspect:
            dfpid = df[df['id'] == pid]

            fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))

            ax1.plot(dfpid.frame, dfpid.gauss_xc, label='Gaussian')
            ax1.plot(dfpid.frame, dfpid.x, label='Centroid')
            ax1.set_ylabel('x')

            ax2.plot(dfpid.frame, dfpid.gauss_xc)
            ax2.plot(dfpid.frame, dfpid.x)
            ax2.set_ylabel('y')
            ax2.set_xlabel('frame')
            plt.show()

    # ------------------------------------------ PRECISION (Z, R, ID) --------------------------------------------------

    if precision:

        df = dfcstats

        if 'gauss_rc' not in df.columns:
            df['gauss_rc'] = np.sqrt((df['gauss_xc'] - image_dimensions[0] / 2) ** 2 +
                                     (df['gauss_yc'] - image_dimensions[1] / 2) ** 2
                                     )

        bin_z, round_z = int(np.round(measurement_depth / 5, 0)), 4
        bin_r, round_r = 4, 0

        # ------------------------------------------ PRECISION (Z, R, ID) ----------------------------------------------
        columns_to_bin = ['z_corr', 'r']
        precision_columns = ['x', 'gauss_xc', 'y', 'gauss_yc', 'gauss_rc']
        bins = [bin_z // 2, bin_r]
        round_to_decimals = [round_z, round_r]

        bin_plot_spct_stats_3d_static_precision_z_r_id(df, columns_to_bin, precision_columns, bins, round_to_decimals,
                                                       export_results, path_results, save_figs, path_figs, show_figs)

        # ------------------------------------------ PRECISION (Z, ID) -------------------------------------------------
        column_to_bin = 'z_corr'
        precision_columns = ['x', 'gauss_xc', 'y', 'gauss_yc', 'gauss_rc']
        bins = bin_z
        round_to_decimal = round_z

        bin_plot_spct_stats_2d_static_precision_z_id(df, column_to_bin, precision_columns, bins, round_to_decimal,
                                                     export_results, path_results, save_figs, path_figs, show_figs)

        # ------------------------------------------ PRECISION (ID) ----------------------------------------------------

        precision_columns = ['x', 'gauss_xc', 'y', 'gauss_yc', 'gauss_rc']
        bins = particle_ids
        bin_count_threshold = num_frames / 5

        dict_1d_static_precision_id = bin_plot_spct_stats_1d_static_precision_id(df,
                                                                                 precision_columns,
                                                                                 bins,
                                                                                 export_results,
                                                                                 path_results,
                                                                                 bin_count_threshold,
                                                                                 save_figs,
                                                                                 path_figs,
                                                                                 show_figs)

    # ---------------------------------------- PRECISION (MIN DX, ID) --------------------------------------------------

    if per_mindx_id:
        df = dfcstats

        column_to_bin = 'min_dx'
        precision_columns = ['x', 'gauss_xc', 'y', 'gauss_yc']
        bin_mdx = 5
        round_mdx = 4
        bin_count_threshold = 20

        bin_plot_spct_stats_2d_static_precision_mindx_id(df, column_to_bin, precision_columns, bin_mdx, round_mdx,
                                                         bin_count_threshold,
                                                         export_results, path_results, save_figs, path_figs, show_figs)

    # ----------------------------------- PRECISION (PERCENT DIAMETER OVERLAP, ID) -------------------------------------

    if per_percent_dx_diameter_id:
        df = dfcstats

        column_to_bin = 'percent_dx_diameter'
        precision_columns = ['x', 'gauss_xc', 'y', 'gauss_yc']
        bin_pdo = 4
        round_pdo = 4
        bin_count_threshold = 20
        pdo_threshold = -3

        bin_plot_spct_stats_2d_static_precision_pdo_id(df, column_to_bin, precision_columns, bin_pdo, round_pdo,
                                                       pdo_threshold, bin_count_threshold,
                                                       export_results, path_results, save_figs, path_figs, show_figs)

    # ---------------------------------------- SAMPLING FREQUENCY (DX) -------------------------------------------------

    if sampling_frequency:

        df = dfcstats

        # ---------------------------------------- SAMPLING AVERAGE ----------------------------------------------------

        # emitter density
        emitter_density = num_pids / area

        # lateral sampling frequency
        dfid = df.groupby('id').mean()

        mean_mean_points_spacing = dfid.num_dx.mean()
        mean_mean_lateral_spacing = dfid.mean_dx.mean() * microns_per_pixel
        mean_min_lateral_spacing = dfid.min_dx.mean() * microns_per_pixel

        nyquist_mean_sampling = 2 * mean_mean_lateral_spacing
        nyquist_min_sampling = 2 * mean_min_lateral_spacing

        if 'contour_diameter' in df.columns:
            zf_contour_diameter = df[(df['z_corr'] < 1) & (
                    df['z_corr'] > -1)].contour_diameter.mean() * microns_per_pixel
            zmin_contour_diameter = df[df['frame'] == df.frame.min()].gauss_diameter.mean() * microns_per_pixel
            zmax_contour_diameter = df[df['frame'] == df.frame.max()].gauss_diameter.mean() * microns_per_pixel

            nyquist_sampling_min_no_contour_overlap = 2 * zf_contour_diameter
            nyquist_sampling_max_no_contour_overlap = 2 * np.max([zmin_contour_diameter, zmax_contour_diameter])

            k_zmin = zf_contour_diameter / zmin_contour_diameter
            k_zmax = zf_contour_diameter / zmax_contour_diameter

            dict_contour = {'zf_contour_diameter_microns': zf_contour_diameter,
                            'zmin_contour_diameter_microns': zmin_contour_diameter,
                            'k_contour_zmin': k_zmin,
                            'zmax_contour_diameter_microns': zmax_contour_diameter,
                            'k_contour_zmax': k_zmax,
                            'nyquist_sampling_min_no_contour_overlap_microns': nyquist_sampling_min_no_contour_overlap,
                            'nyquist_sampling_max_no_contour_overlap_microns': nyquist_sampling_max_no_contour_overlap,
                            }
        else:
            dict_contour = None

        if 'gauss_diameter' in df.columns:
            zf_gauss_diameter = df[(df['z_corr'] < 1) & (df['z_corr'] > -1)].gauss_diameter.mean() * microns_per_pixel
            zmin_gauss_diameter = df[df['frame'] == df.frame.min()].gauss_diameter.mean() * microns_per_pixel
            zmax_gauss_diameter = df[df['frame'] == df.frame.max()].gauss_diameter.mean() * microns_per_pixel

            nyquist_sampling_min_no_overlap = 2 * zf_gauss_diameter
            nyquist_sampling_max_no_overlap = 2 * np.max([zmin_gauss_diameter, zmax_gauss_diameter])

            k_zmin = zf_gauss_diameter / zmin_gauss_diameter
            k_zmax = zf_gauss_diameter / zmax_gauss_diameter

            dict_gauss = {'zf_gauss_diameter_microns': zf_gauss_diameter,
                          'zmin_gauss_diameter_microns': zmin_gauss_diameter,
                          'k_gauss_zmin': k_zmin,
                          'zmax_gauss_diameter_microns': zmax_gauss_diameter,
                          'k_gauss_zmax': k_zmax,
                          'nyquist_sampling_min_no_overlap_microns': nyquist_sampling_min_no_overlap,
                          'nyquist_sampling_max_no_overlap_microns': nyquist_sampling_max_no_overlap,
                          }
        else:
            dict_gauss = None

        # package results to update overview dictionary
        dict_spct_stats_sampling_frequency = {'emitter_density_microns_squared': emitter_density,
                                              'num_points_spacing': mean_mean_points_spacing,
                                              'mean_mean_lateral_spacing_microns': mean_mean_lateral_spacing,
                                              'mean_min_lateral_spacing_microns': mean_min_lateral_spacing,
                                              'nyquist_sampling_mean_dx_microns': nyquist_mean_sampling,
                                              'nyquist_sampling_min_dx_microns': nyquist_min_sampling,
                                              }

        if dict_contour is not None:
            dict_spct_stats_sampling_frequency.update(dict_contour)

        if dict_gauss is not None:
            dict_spct_stats_sampling_frequency.update(dict_gauss)

        # ---------------------------------------- SAMPLING FREQUENCY (Z) ----------------------------------------------

        export_results = True
        save_figs = True
        show_figs = False
        column_to_bin = 'z_corr'
        bins = num_frames

        bin_plot_spct_stats_sampling_frequency_z_id(df, column_to_bin, bins, area, microns_per_pixel,
                                                    export_results, path_results, save_figs, path_figs, show_figs)

    # ----------------------------------------- PACKAGE AND EXPORT RESULTS ---------------------------------------------

    spct_results = {'method': method,
                    'mag_eff': mag_eff,
                    'microns_per_pixel': microns_per_pixel,
                    'area_microns_squared': area,
                    'num_particles': num_pids,
                    'num_frames': num_frames,
                    'measurement_depth': measurement_depth,
                    'zf': zf,
                    'c1': c1,
                    'c2': c2,
                    }

    if dict_spct_pid_plane_angle is not None:
        spct_results.update(dict_spct_pid_plane_angle)

    if dict_penetrance is not None:
        spct_results.update(dict_penetrance)

    if dict_spct_stats_bin_z is not None:
        spct_results.update(dict_spct_stats_bin_z)

    if dict_1d_static_precision_id is not None:
        spct_results.update(dict_1d_static_precision_id)

    if dict_spct_stats_sampling_frequency is not None:
        spct_results.update(dict_spct_stats_sampling_frequency)

    df_spct_results = pd.DataFrame.from_dict(spct_results, orient='index', columns=['value'])

    # export
    df_spct_results.to_excel(path_results + '/spct-stats-overview-results.xlsx')


# ------------------------------------------ META ANALYSIS FUNCTIONS ---------------------------------------------------


def plot_meta_assessment(base_dir, method, min_cm, min_percent_layers, microns_per_pixel, path_calib_spct_pop=None,
                         save_figs=True, show_figs=False):
    if method == 'idpt' and path_calib_spct_pop is None:
        raise ValueError('Must specifiy path_calib_spct_pop for IDPT analyses.')

    # filepaths
    path_test_coords = join(base_dir, 'coords/test-coords')
    path_calib_coords = join(base_dir, 'coords/calib-coords')
    path_similarity = join(base_dir, 'similarity')
    path_results = join(base_dir, 'results')
    path_figs = join(base_dir, 'figs')

    # --- --- META ASSESSMENT

    # calibration coords
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method=method)

    if 'x' not in dfcpid.columns:
        print('Running correct.merge_calib_pid_defocus_and_correction_coords(path_calib_coords) to merge x-y.')
        dfcpid = correct.merge_calib_pid_defocus_and_correction_coords(path_calib_coords=path_calib_coords,
                                                                       method=method,
                                                                       dfs=[dfc, dfcpid])

    # inspect initial distribution of in-focus particle positions
    fig, ax = plotting.scatter_z_by_xy(df=dfcpid, z_params=['zf_from_peak_int', 'zf_from_nsv'])
    fig.savefig(path_figs + '/zf_scatter_xy_int-and-nsv.png')
    dict_fit_plane, fig_xy, fig_xyz, fig_plane = correct.inspect_calibration_surface(df=dfcpid,
                                                                                     param_zf='zf_from_nsv',
                                                                                     microns_per_pixel=microns_per_pixel)
    fig_xy.savefig(path_figs + '/zf_scatter_xy.png')
    plt.close(fig_xy)
    fig_xyz.savefig(path_figs + '/zf_scatter_xyz.png')
    plt.close(fig_xyz)
    fig_plane.savefig(path_figs + '/zf_fit-3d-plane.png')
    plt.close(fig_plane)

    # read diameter paramaters
    if path_calib_spct_pop is not None:
        mag_eff, zf, c1, c2 = io.read_pop_gauss_diameter_properties(path_calib_spct_pop)
    else:
        mag_eff, zf, c1, c2 = io.read_pop_gauss_diameter_properties(dfcpop)

    # test coords
    dft = io.read_test_coords(path_test_coords)

    # correct test coords
    dft = correct.correct_z_by_plane_tilt(dfcpid,
                                          dft,
                                          param_zf='zf_from_nsv',
                                          param_z='z',
                                          param_z_true='z_true')

    # --- CALIBRATION CURVE
    fig, ax = plt.subplots()
    ax.scatter(dft.z_true_corr, dft.z_corr, s=0.5, alpha=0.25)
    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    plt.tight_layout()
    if save_figs:
        plt.savefig(path_figs + '/calibration_curve_corrected.png')
    if show_figs:
        plt.show()
    plt.close()

    # --- RMSE
    # bin
    dfrmse = bin.bin_local_rmse_z(dft, column_to_bin='z_true_corr', bins=25, min_cm=min_cm, z_range=None,
                                  round_to_decimal=4, df_ground_truth=None, dropna=True, error_column='error')

    dfrmse_mean = bin.bin_local_rmse_z(dft, column_to_bin='z_true_corr', bins=1, min_cm=min_cm, z_range=None,
                                       round_to_decimal=4, df_ground_truth=None, dropna=True, error_column='error')

    # export
    dfrmse.to_excel(path_results + '/meta_rmse-z_binned.xlsx')
    dfrmse_mean.to_excel(path_results + '/meta_rmse-z_mean.xlsx')

    # plot
    fig, ax = plt.subplots()
    ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    plt.tight_layout()
    if save_figs:
        plt.savefig(path_figs + '/meta_rmse-z_corrected.png')
    if show_figs:
        plt.show()
    plt.close()

    # --- --- --- PARTICLE IMAGE SIMILARITY

    # --- CALIBRATION STACK SIMILARITY
    # read
    dfs, dfsf, dfsm, dfas, dfcs = io.read_similarity(path_similarity)

    # plot
    if dfsf is not None:
        fig, ax = plotting.plot_calib_stack_self_similarity(dfsf, min_percent_layers=min_percent_layers)
        ax.set_xlabel(r'$z_{calib.} \: (\mu m)$')
        ax.set_ylabel(r'$\overline{S}_{(i, i+1)}$')
        plt.tight_layout()
        plt.savefig(path_figs + '/calib_self-similarity-forward.png')
        plt.show()

    if dfsm is not None:
        fig, ax = plotting.plot_calib_stack_self_similarity(dfsm, min_percent_layers=min_percent_layers)
        ax.set_xlabel(r'$z_{calib.} \: (\mu m)$')
        ax.set_ylabel(r'$\overline{S}_{(i-1, i, i+1)}$')
        plt.tight_layout()
        plt.savefig(path_figs + '/calib_self-similarity-middle.png')
        plt.show()

    if dfas is not None:
        fig, ax = plotting.plot_particle_to_particle_similarity(dfcs, min_particles_per_frame=10)
        ax.set_xlabel(r'$z_{calib.} \: (\mu m)$')
        ax.set_ylabel(r'$\overline{S}_{i}(p_{i}, p_{N})$')
        plt.tight_layout()
        plt.savefig(path_figs + '/calib_per-frame_particle-to-particle-similarity.png')
        plt.show()

    # --- --- INTRINSIC ABERRATIONS ASSESSMENT
    # --- RAW
    # evaluate
    dict_ia = analyze.evaluate_intrinsic_aberrations(dfs,
                                                     z_f=zf,
                                                     min_cm=min_cm,
                                                     param_z_true='z_true',
                                                     param_z_cm='z_cm')

    dict_ia = analyze.fit_intrinsic_aberrations(dict_ia)
    io.export_dict_intrinsic_aberrations(dict_ia, path_results, unique_id='raw')

    # plot
    fig, ax = plotting.plot_intrinsic_aberrations(dict_ia, cubic=True, quartic=True)
    ax.set_xlabel(r'$z_{raw} \: (\mu m)$')
    ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
    ax.grid(alpha=0.125)
    ax.legend(['Data', 'Cubic', 'Quartic'])
    plt.tight_layout()
    plt.savefig(path_figs + '/intrinsic-aberrations_raw.png')
    plt.show()

    # --- CORRECTED
    # evaluate
    dfs_corr = correct.correct_z_by_plane_tilt(dfcal=dfcpid,
                                               dftest=dfs,
                                               param_zf='zf_from_nsv',
                                               param_z='z_est',
                                               param_z_true='z_true',
                                               params_correct=['z_cm'])

    dict_iac = analyze.evaluate_intrinsic_aberrations(dfs_corr,
                                                      z_f=0,
                                                      min_cm=min_cm,
                                                      param_z_true='z_true_corr',
                                                      param_z_cm='z_cm_corr')

    dict_iac = analyze.fit_intrinsic_aberrations(dict_iac)
    io.export_dict_intrinsic_aberrations(dict_iac, path_results, unique_id='corrected')

    # plot
    fig, ax = plotting.plot_intrinsic_aberrations(dict_iac, cubic=True, quartic=True)
    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
    ax.grid(alpha=0.125)
    ax.legend(['Data', 'Cubic', 'Quartic'])
    plt.tight_layout()
    plt.savefig(path_figs + '/intrinsic-aberrations_corrected.png')
    plt.show()


# ------------------------------- TEST RIGID DISPLACEMENT ANALYSIS FUNCTIONS -------------------------------------------


def plot_rigid_displacement_test(test_coords_path, spct_stats_path=None):
    # details
    microns_per_pixel = 0.8
    area = (512 * microns_per_pixel) ** 2

    # read dataframe
    df = pd.read_excel(test_coords_path)

    # get number of particles
    pids = df.id.unique()
    num_pids = len(pids)

    # compute the radial distance
    if 'r' not in df.columns:
        df['r'] = np.sqrt((256 - df.x) ** 2 + (256 - df.y) ** 2)

    # ------------------------------------------- PRECISION (R, Z) ---------------------------------------------------
    precision = True

    if precision:

        # ----------------------------------------- PRECISION (R, ID) --------------------------------------------------

        # ------------------------------------------- PRECISION (ID) ---------------------------------------------------

        # precision @ z
        per_id = True

        if per_id:
            pos = ['z', 'r', 'x', 'y']

            dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                                column_to_bin='id',
                                                                precision_columns=pos,
                                                                bins=pids,
                                                                round_to_decimal=0)

            # plot bin(id)
            count_column = 'counts'
            plot_columns = ['z']
            xparams = ['r_m', 'x_m', 'y_m']

            # axial localization precision: z(id, r)
            for xparam in xparams:
                for pc in plot_columns:
                    dfp_id = dfp_id.sort_values(xparam)

                    # measure statistical significance of correlation
                    measurement = dfp_id[pc].to_numpy()
                    dependency = dfp_id[xparam].to_numpy()
                    pearson_r, pearson_pval = np.round(pearsonr(measurement, dependency), 3)
                    spearman_r, spearman_pval = np.round(spearmanr(measurement, dependency), 3)

                    fig, ax = plt.subplots()

                    ax.plot(dfp_id[xparam], dfp_id[pc], '-o')

                    # fit line
                    popt, pcov, ffunc = fit.fit(dfp_id[xparam], dfp_id[pc], fit_function=functions.line)
                    ax.plot(dfp_id[xparam], ffunc(dfp_id[xparam], *popt), linestyle='--', color='black',
                            label='Fit: d{}/d{} = {} \n'
                                  'Pearson(r, p): {}, {}\n'
                                  'Spearman(r, p): {}, {}'.format(pc,
                                                                  xparam,
                                                                  np.format_float_scientific(popt[0],
                                                                                             precision=2,
                                                                                             exp_digits=2),
                                                                  pearson_r, pearson_pval,
                                                                  spearman_r, spearman_pval,
                                                                  )
                            )

                    ax.set_xlabel(xparam)
                    ax.set_ylabel('{} precision'.format(pc))
                    ax.legend()

                    axr = ax.twinx()
                    axr.plot(dfp_id[xparam], dfp_id[count_column], '-s', markersize=2, alpha=0.25)
                    axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
                    axr.set_ylim([0, int(np.round(dfp_id[count_column].max() + 6, -1))])

                    plt.tight_layout()
                    plt.show()

    # ------------------------------------------------- FIT PLANE ------------------------------------------------------
    fit_plane = False

    if fit_plane and spct_stats_path:
        # group test dataframe by particle id
        dft = df.groupby('id').mean()

        # get mean z from test
        z_test_mean = dft.z.mean()

        # read calibration dataframe to fit plane for in-focus image
        dfc = pd.read_excel(spct_stats_path)

        # --------------------------------------- FIT PLANE TO CALIBRATION (PIXELS) ------------------------------------

        # get z in-focs
        zf_nearest = dfc.iloc[np.argmin(np.abs(dfc['z_true'] - z_test_mean))].z_true
        dff = dfc[dfc['z_true'] == zf_nearest]

        # fit plane on calibration in-focus (x, y, z units: pixels)
        points_pixels = np.stack((dff.x,
                                  dff.y,
                                  dff.z_corr)).T
        px_pixels, py_pixels, pz_microns, popt_calib = fit.fit_3d(points_pixels, fit_function='plane')

        # --------------------------------------- FIT PLANE TO TEST (PIXELS) ------------------------------------

        # fit plane to test (x, y, z units: pixels)
        points_pixels = np.stack((dft.x,
                                  dft.y,
                                  dft.z)).T
        px_pixels, py_pixels, pz_microns, popt_test = fit.fit_3d(points_pixels, fit_function='plane')

        # -------------------------------- CALCULATE PLANE POSITIONS FOR TEST PARTICLES --------------------------------

        # calculate z on fitted 3D plane for all particle locations

        # in-focus (from calibration)
        dft['z_plane_f'] = functions.calculate_z_of_3d_plane(dft.x, dft.y, popt=popt_calib)

        # for test
        dft['z_plane_t'] = functions.calculate_z_of_3d_plane(dft.x, dft.y, popt=popt_test)

        # calculate error
        dft['z_error_t'] = dft['z_plane_t'] - dft['z']

        # ------------------------------------- PLOT DEVIATIONS FROM FITTED PLANE --------------------------------------

        # scatter plot
        fig, ax = plt.subplots()

        sc = ax.scatter(dft.x, dft.y, c=dft.z_error_t)
        plt.colorbar(sc)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.tight_layout()
        plt.show()

        # heat map 1
        # move x, y, z series to numpy arrays
        x = dft.x.to_numpy()
        y = dft.y.to_numpy()
        z = dft.z_error_t.to_numpy()

        # get spatial coordinate extents
        xspace = np.max(x) - np.min(x)
        yspace = np.max(y) - np.min(y)
        zspace = np.max(z) - np.min(z)

        # Create grid values first.
        xi = np.linspace(np.min(x), np.max(x), 250)
        yi = np.linspace(np.min(y), np.max(y), 250)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

        X, Y = np.meshgrid(xi, yi)

        fig, ax = plt.subplots()
        sc = ax.pcolormesh(X, Y, zi, shading='auto', cmap="RdBu_r")
        ax.scatter(x, y, color='black', label=r"$p_{i}$")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.set_title(r'$\epsilon_{z,plane}$')
        ax.set_xlabel(r'$x \: (pix)$')
        ax.set_xlim([0, 512])
        ax.set_ylabel(r'$y \: (pix)$')
        ax.set_ylim([0, 512])
        ax.legend()
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------- DISTORTIONS (R, Z) ---------------------------------------------------
    distortions = False

    if distortions:
        # -------------------------------------- DISTORTIONS (R, Z) ----------------------------------------------------

        # deviations from fitted plane
        pass


# ------------------------------------------ TEST ANALYSIS FUNCTIONS ---------------------------------------------------


def plot_test_coords(df, evaluate):
    """
    Inputs:
        * A filtered, corrected dataframe.
            Note: This means metrics like 'percent_measured' should have already been calculated.

    Core Functions:
        A. Precision
            1. mean lateral and axial precision.
            2. z-dependent lateral and axial precision.
        B. Error
            1. mean lateral and axial r.m.s. error.
            2. z-dependent lateral and axial r.m.s. error.
        C. Sampling Frequency
            1. number density
            2. mean particle to particle spacing
            3. Nyquist sampling frequency

    Optional Functions:
        Opt.A. Precision
            1. r-dependence
        Opt.B. Error
            1. r-dependence

    :param df:
    :param evaluate:
    :return:
    """

    # ----------------------------------------------- PRECISION ANALYSIS -----------------------------------------------
    """
    Precision Analysis Module:
        1. Mean lateral precision
            a. measure lateral precision by particle ID at each discrete z-step (requires n >= 3 samples).
            b. weighted-average (1.a) to get mean lateral precision.
        2. Z-dependent lateral precision
            b. weighted-average for each discrete z-step.
        3. Mean axial precision
            a. measure axial precision by particle ID at each discrete z-step (requires n >= 3 samples).
            b. weighted-average (3.a) to get mean axial precision.
        4. Z-dependent axial precision
            a. weighted-average (3.a) for each discrete z-step.
    """


# ------------------------------------------------ ERROR ANALYSIS ------------------------------------------------------


def plot_error_analysis(dft, path_figs, path_results):
    # setup error
    error_threshold = 7.5
    bins_r = 5
    bins_z = 33

    # plot
    xlim = [-57.5, 62.5]
    xyticks = [-50, -25, 0, 25, 50]
    yerr_lims = [-7.5, 7.5]
    yerr_ticks = [-5, 0, 5]

    # basic
    image_dimensions = (512, 512)

    # figs
    fig_1, fig_2, fig_3, fig_4, fig_5 = True, True, True, True, True

    # test-coords without (mostly) focal plane bias errors
    df_error = dft[dft['error'].abs() < error_threshold]
    df_error = df_error.sort_values('z_true')

    # --- PLOTTING

    # FIGURE 1: all errors by z_true & fit quadratic
    if fig_1:
        fig, ax = plt.subplots()

        # data
        ax.scatter(df_error.z_true, df_error.error, s=0.125, marker='.', label='Data')

        # fit quadratic
        popt, pcov = curve_fit(functions.quadratic, df_error.z_true, df_error.error)
        ax.plot(df_error.z_true, functions.quadratic(df_error.z_true, *popt), linestyle='--', color='black',
                label='Fit')

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim(xlim)
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        ax.set_title('Fit: {}'.format(np.round(popt[0], 4)) + r'$x^2$' +
                     ' + {}'.format(np.round(popt[1], 4)) + r'$x$' +
                     ' + {}'.format(np.round(popt[2], 4))
                     )
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(path_figs + '/all_normalized-z-errors_by_z_and_fit-quadratic_errlim{}.png'.format(error_threshold))
        plt.show()

    # ---

    # FIGURE 2: bin(r) - all errors by z_true & fit quadratic
    if fig_2:
        if 'r' not in df_error.columns:
            df_error['r'] = np.sqrt(
                (df_error.x - image_dimensions[0] / 2) ** 2 +
                (df_error.y - image_dimensions[1] / 2) ** 2
            )

        # bin(r)
        column_to_bin = 'r'
        column_to_count = 'id'
        round_to_decimal = 1
        return_groupby = False

        dfb = bin.bin_generic(df_error,
                              column_to_bin,
                              column_to_count,
                              bins_r,
                              round_to_decimal,
                              return_groupby
                              )

        dfb = dfb.sort_values('bin')

        # plot
        fig, ax = plt.subplots(nrows=bins_r, sharex=True, figsize=(size_x_inches, size_y_inches * bins_r / 2.4))
        popts_r = []
        lbls_r = []

        for i, bin_r in enumerate(dfb.bin.unique()):
            dfbr = dfb[dfb['bin'] == bin_r]
            dfbr = dfbr.sort_values('z_true')

            # formatting
            bin_id = int(np.round(bin_r, 0))
            lbl_bin = r'$r_{bin}=$' + '{}'.format(bin_id)
            lbls_r.append(bin_id)

            # data
            sc, = ax[i].plot(dfbr.z_true, dfbr.error,
                             marker='o',
                             ms=0.5,
                             linestyle='',
                             color=sci_color_cycle[i],
                             alpha=0.5,
                             label=lbl_bin)

            # fit quadratic
            popt, pcov = curve_fit(functions.quadratic, dfbr.z_true, dfbr.error)
            popts_r.append(popt)
            lbl_quadratic = r'Fit: {}'.format(np.round(popt[0], 4)) + r'$x^2$' + \
                            r' + {}'.format(np.round(popt[1], 4)) + r'$x$' + \
                            r' + {}'.format(np.round(popt[2], 4))

            ax[i].plot(dfbr.z_true, functions.quadratic(dfbr.z_true, *popt),
                       linestyle='--',
                       color=lighten_color(sc.get_color(), amount=1.25),
                       )

            # label each figure
            ax[i].set_ylim(yerr_lims)
            ax[i].set_yticks(ticks=yerr_ticks, labels=yerr_ticks)
            ax[i].legend(loc='upper right')

        ax[bins_r - 1].set_xlabel(r'$z_{true} \: (\mu m)$')
        ax[bins_r - 1].set_xlim(xlim)
        ax[bins_r - 1].set_xticks(ticks=xyticks, labels=xyticks)
        ax[int(np.floor(bins_r / 2))].set_ylabel(r'$\epsilon_{z} \: (\mu m)$')

        plt.savefig(
            path_figs + '/bin-r_normalized-z-errors_by_z_and_fit-quadratic_errlim{}.png'.format(error_threshold))
        plt.tight_layout()
        plt.show()
        plt.close()

    # ---

    # FIGURE 3: plot only fit quadratics
    if fig_3:

        fit_z_true = np.linspace(df_error.z_true.min(), df_error.z_true.max(), 100)

        # plot
        fig, ax = plt.subplots()

        for i, popt in enumerate(popts_r):
            ax.plot(fit_z_true, functions.quadratic(fit_z_true, *popt),
                    linestyle='-',
                    color=sci_color_cycle[i],
                    label=lbls_r[i])

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim(xlim)
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        ax.set_ylim(yerr_lims)
        ax.set_yticks(ticks=yerr_ticks, labels=yerr_ticks)
        ax.legend(loc='upper right')

        plt.savefig(
            path_figs + '/bin-r_normalized-z-errors_by_z_fit-quadratic_errlim{}.png'.format(error_threshold))
        plt.tight_layout()
        plt.show()
        plt.close()

    # ---

    # FIGURE 4: bin(r).groupby(r).mean() - all errors by z_true & fit quadratic
    if fig_4:
        return_groupby = True
        dfm, dfstd = bin.bin_generic(df_error,
                                     column_to_bin,
                                     column_to_count,
                                     bins_r,
                                     round_to_decimal,
                                     return_groupby)

        dfm = dfm.sort_values('bin')
        dfstd = dfstd.sort_values('bin')

        fig, ax = plt.subplots()
        ax.errorbar(dfm.bin, dfm.error, yerr=dfstd.error, fmt='-o', elinewidth=1, capsize=2)
        ax.set_xlabel(r'$r_{bin} \: (\mu m)$')
        ax.set_xlim([-10, 350])
        ax.set_xticks(ticks=[0, 100, 200, 300])
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        plt.savefig(
            path_figs + '/bin-r_normalized-z-errors_by_z_mean+std_errlim{}.png'.format(error_threshold))
        plt.tight_layout()
        plt.show()
        plt.close()

        # export
        dfm.to_excel(path_results + '/bin-r_mean-z-errors_errlim{}.xlsx'.format(error_threshold))
        dfstd.to_excel(path_results + '/bin-r_std-z-errors_errlim{}.xlsx'.format(error_threshold))

    # ---

    # FIGURE 4: bin(r, z).groupby(r).mean() - all errors by z_true & fit quadratic
    # bin(r)
    plot_fit = False

    if fig_5:
        columns_to_bin = ['r', 'z_true']
        column_to_count = 'id'
        bins = [bins_r, bins_z]
        round_to_decimals = [1, 3]
        min_num_bin = 10
        return_groupby = True

        dfm, dfstd = bin.bin_generic_2d(df_error,
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

        # plot
        fig, ax = plt.subplots()

        for i, bin_r in enumerate(dfm.bin_tl.unique()):
            dfbr = dfm[dfm['bin_tl'] == bin_r]
            dfbr_std = dfstd[dfstd['bin_tl'] == bin_r]

            # scatter: mean +/- std
            ax.errorbar(dfbr.bin_ll, dfbr.error, yerr=dfbr_std.error,
                        fmt='-o', ms=2, elinewidth=0.5, capsize=1, label=int(np.round(bin_r, 0)))

            # plot: fit
            if plot_fit:
                fit_bin_ll = np.linspace(dfbr.bin_ll.min(), dfbr.bin_ll.max())
                ax.plot(fit_bin_ll, functions.quadratic(fit_bin_ll, *popts_r[i]),
                        linestyle='--',
                        color=lighten_color(sci_color_cycle[i], amount=1.25),
                        )

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim(xlim)
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        ax.set_ylim(yerr_lims)
        ax.set_yticks(ticks=yerr_ticks, labels=yerr_ticks)
        ax.legend(loc='upper right', title=r'$r_{bin}$',
                  borderpad=0.2, handletextpad=0.6, borderaxespad=0.25, markerscale=0.75)

        plt.savefig(
            path_figs + '/bin-r-z_normalized-z-errors_by_z_and_fit-quadratic_errlim{}.png'.format(error_threshold))
        plt.tight_layout()
        plt.show()
        plt.close()


# ---------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------


def plot_spct_stats_bin_z(df, column_to_bin, column_to_count, bins, round_to_decimal, save_figs, path_figs, show_figs,
                          export_results, path_results):
    if save_figs:
        path_save_figs = path_figs + '/spct-stats_bin-z'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

    dfm, dfstd = bin.bin_generic(df,
                                 column_to_bin=column_to_bin,
                                 column_to_count=column_to_count,
                                 bins=bins,
                                 round_to_decimal=round_to_decimal,
                                 return_groupby=True)

    # export results
    if export_results:

        path_save_results = path_results + '/spct-stats_bin-z'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        dfm.to_excel(path_save_results + '/spct-stats_bin-z_dfm.xlsx')
        dfstd.to_excel(path_save_results + '/spct-stats_bin-z_dfstd.xlsx')

    # plot
    count_column = 'count_{}'.format(column_to_count)

    plot_columns = ['peak_int', 'snr', 'nsv', 'nsv_signal',
                    'solidity', 'thinness_ratio', 'gauss_diameter', 'gauss_dia_x_y', 'gauss_sigma_x_y',
                    'min_dx', 'mean_dxo', 'num_dxo', 'percent_dx_diameter']

    for pc in plot_columns:
        fig, ax = plt.subplots()
        ax.errorbar(dfm.bin, dfm[pc], yerr=dfstd[pc], fmt='o', linewidth=2, capsize=6, color='tab:blue',
                    alpha=0.25, label=r'$\mu + \sigma$')
        ax.plot(dfm.bin, dfm[pc], color='tab:blue')
        ax.set_xlabel(r'$z_{corr}$')
        ax.set_ylabel(pc)
        ax.legend()

        axr = ax.twinx()
        axr.plot(dfm.bin, dfm[count_column], '-o', markersize=2, color='gray', alpha=0.125)
        axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
        axr.set_ylim([0, int(np.round(dfm[count_column].max() + 6, -1))])

        plt.tight_layout()
        if save_figs:
            plt.savefig(path_save_figs + '/spct-stats_bin-z_{}.png'.format(pc))
        if show_figs:
            plt.show()
        plt.close()

    # package results to update overview dictionary
    dict_spct_stats_bin_z = {'zmin_snr': dfm.iloc[0].snr,
                             'zmax_snr': dfm.iloc[-1].snr,
                             'zmin_peak_int': dfm.iloc[0].peak_int,
                             'zmax_peak_int': dfm.iloc[-1].peak_int,
                             }

    return dict_spct_stats_bin_z


def plot_spct_stats_bin_id(df, column_to_count, num_pids, save_figs, path_figs, show_figs, export_results,
                           path_results):
    if save_figs:
        path_save_figs = path_figs + '/spct-stats_bin-id'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

    dfm, dfstd = bin.bin_generic(df,
                                 column_to_bin='id',
                                 column_to_count=column_to_count,
                                 bins=num_pids,
                                 round_to_decimal=0,
                                 return_groupby=True)

    # export results
    if export_results:

        path_save_results = path_results + '/spct-stats_bin-id'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        dfm.to_excel(path_save_results + '/spct-stats_bin-id_dfm.xlsx')
        dfstd.to_excel(path_save_results + '/spct-stats_bin-id_dfstd.xlsx')

    count_column = 'count_{}'.format(column_to_count)
    plot_columns = ['peak_int', 'snr', 'nsv', 'nsv_signal',
                    'solidity', 'thinness_ratio', 'gauss_diameter', 'gauss_dia_x_y', 'gauss_sigma_x_y',
                    'min_dx', 'mean_dxo', 'num_dxo', 'percent_dx_diameter']

    for pc in plot_columns:
        fig, ax = plt.subplots()
        ax.errorbar(dfm.bin, dfm[pc], yerr=dfstd[pc], fmt='o', linewidth=2, capsize=6, color='tab:blue',
                    alpha=0.25,
                    label=r'$\mu + \sigma$')
        ax.plot(dfm.bin, dfm[pc], color='tab:blue')
        ax.set_xlabel(r'$p_{ID}$')
        ax.set_ylabel(pc)
        ax.legend()

        axr = ax.twinx()
        axr.plot(dfm.bin, dfm[count_column], '-o', markersize=2, color='gray', alpha=0.125)
        axr.set_ylabel(r'$N_{frames} \: (\#)$', color='gray')
        axr.set_ylim([0, int(np.round(dfm[count_column].max() + 6, -1))])

        plt.tight_layout()
        if save_figs:
            plt.savefig(path_save_figs + '/spct-stats_bin-id_{}.png'.format(pc))
        if show_figs:
            plt.show()
        plt.close()


def plot_spct_stats_bin_2d(df, columns_to_bin, column_to_count, bins, round_to_decimals, min_num_bin,
                           save_figs, path_figs, show_figs, export_results, path_results, plot_columns):
    column_to_bin_top_level = columns_to_bin[0]
    column_to_bin_low_level = columns_to_bin[1]

    id_string = 'spct-stats_bin-{}-{}'.format(column_to_bin_top_level, column_to_bin_low_level)

    if save_figs:
        path_save_figs = path_figs + '/' + id_string
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

    dfm, dfstd = bin.bin_generic_2d(df,
                                    columns_to_bin,
                                    column_to_count,
                                    bins,
                                    round_to_decimals,
                                    min_num_bin,
                                    return_groupby=True,
                                    )

    # export results
    if export_results:

        path_save_results = path_results + '/' + id_string
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        dfm.to_excel(path_save_results + '/{}_dfm.xlsx'.format(id_string))
        dfstd.to_excel(path_save_results + '/{}_dfstd.xlsx'.format(id_string))

    # plot
    count_column = 'count_{}'.format(column_to_count)

    for pc in plot_columns:
        fig, [ax, axr] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.25, size_y_inches * 1.25))

        for i, bin_tl in enumerate(dfm.bin_tl.unique()):
            dfbm_tl = dfm[dfm['bin_tl'] == bin_tl]
            dfbstd_tl = dfstd[dfstd['bin_tl'] == bin_tl]

            sx = np.mean(dfbm_tl.bin_ll.diff()) / 50 * (i - np.floor(len(dfm.bin_tl.unique()) / 2))

            ax.errorbar(dfbm_tl.bin_ll + sx, dfbm_tl[pc], yerr=dfbstd_tl[pc],
                        fmt='-o', ms=3, linewidth=1, capsize=2, alpha=0.75, label=np.round(bin_tl, 1))

            axr.plot(dfbm_tl.bin_ll + sx, dfbm_tl[count_column], '-d', markersize=2)

        ax.set_ylabel(pc + r' $\: (\mu \pm \sigma)$')
        if pc in ['gauss_dia_x_y', 'gauss_sigma_x_y']:
            ax.set_ylim([0.7, 1.3])
        ax.legend(title=column_to_bin_top_level, loc='upper left', bbox_to_anchor=(1, 1))

        axr.set_xlabel(column_to_bin_low_level)
        axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
        axr.set_ylim([0, int(np.round(dfm[count_column].max() + 6, -1))])

        plt.tight_layout()
        if save_figs:
            plt.savefig(path_save_figs + '/{}_{}.png'.format(id_string, pc))
        if show_figs:
            plt.show()
        plt.close()


def bin_plot_spct_stats_3d_static_precision_z_r_id(df, columns_to_bin, precision_columns, bins, round_to_decimals,
                                                   export_results, path_results, save_figs, path_figs, show_figs):
    df_bin_z_r_id, df_bin_z_r = analyze.evaluate_3d_static_precision(df,
                                                                     columns_to_bin=columns_to_bin,
                                                                     precision_columns=precision_columns,
                                                                     bins=bins,
                                                                     round_to_decimals=round_to_decimals)

    # export results
    if export_results:
        path_save_results = path_results + '/spct-stats_3d-precision_z-r-id'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        df_bin_z_r_id.to_excel(path_save_results + '/spct-stats_bin-id-r-z.xlsx')
        df_bin_z_r.to_excel(path_save_results + '/spct-stats_bin-id-r-z_weighted-average.xlsx')

    # save and/or show plots
    if save_figs or show_figs:

        path_save_figs = path_figs + '/spct-stats_3d-precision_z-r-id'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

        # plot bin(z, r)
        count_column = 'counts'
        plot_columns = precision_columns
        xparam = columns_to_bin[0]
        pparams = columns_to_bin[1]
        line_plots = np.unique(df_bin_z_r[pparams].to_numpy())

        for pc in plot_columns:
            fig, ax = plt.subplots()
            axr = ax.twinx()

            for lpt in line_plots:
                dfpt = df_bin_z_r[df_bin_z_r[pparams] == lpt]

                ax.plot(dfpt[xparam], dfpt[pc], '-o', markersize=3, label=np.round(lpt, 1))

                axr.plot(dfpt[xparam], dfpt[count_column], '-s', markersize=2, alpha=0.25)

            ax.set_xlabel(xparam)
            ax.set_ylabel('{} precision'.format(pc))
            ax.legend(title=pparams)

            axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
            axr.set_ylim([0, int(np.round(df_bin_z_r[count_column].max() + 6, -1))])

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_save_figs + '/spct-stats_bin-z-r-id_{}.png'.format(pc))
            if show_figs:
                plt.show()
            plt.close()


def bin_plot_spct_stats_2d_static_precision_z_id(df, column_to_bin, precision_columns, bins, round_to_decimal,
                                                 export_results, path_results, save_figs, path_figs, show_figs):
    df_bin_z_id, df_bin_z = analyze.evaluate_2d_static_precision(df,
                                                                 column_to_bin=column_to_bin,
                                                                 precision_columns=precision_columns,
                                                                 bins=bins,
                                                                 round_to_decimal=round_to_decimal)

    # export results
    if export_results:

        path_save_results = path_results + '/spct-stats_2d-precision_z-id'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        df_bin_z_id.to_excel(path_save_results + '/spct-stats_bin-id-z.xlsx')
        df_bin_z.to_excel(path_save_results + '/spct-stats_bin-id-z_weighted-average.xlsx')

    # save and/or show plots
    if save_figs or show_figs:

        path_save_figs = path_figs + '/spct-stats_2d-precision_z-id'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

        # plot bin(z, id)
        count_column = 'counts'
        plot_columns = precision_columns
        xparam = column_to_bin

        for pc in plot_columns:
            fig, ax = plt.subplots()

            ax.plot(df_bin_z[xparam], df_bin_z[pc], '-o', markersize=3)

            ax.set_xlabel(xparam)
            ax.set_ylabel('{} precision'.format(pc))

            axr = ax.twinx()
            axr.plot(df_bin_z[xparam], df_bin_z[count_column], '-s', markersize=2, alpha=0.25)
            axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
            axr.set_ylim([0, int(np.round(df_bin_z[count_column].max() + 6, -1))])

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_save_figs + '/spct-stats_bin-z-id_{}.png'.format(pc))
            if show_figs:
                plt.show()
            plt.close()


def bin_plot_spct_stats_2d_static_precision_mindx_id(df, column_to_bin, precision_columns, bins, round_to_decimals,
                                                     bin_count_threshold,
                                                     export_results, path_results, save_figs, path_figs, show_figs):
    df_bin_mindx_id, df_bin_mindx = analyze.evaluate_2d_static_precision(df,
                                                                         column_to_bin=column_to_bin,
                                                                         precision_columns=precision_columns,
                                                                         bins=bins,
                                                                         round_to_decimal=round_to_decimals)

    # export results
    if export_results:

        path_save_results = path_results + '/spct-stats_2d-precision_mindx-id'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        df_bin_mindx_id.to_excel(path_save_results + '/spct-stats_bin-mindx-id.xlsx')
        df_bin_mindx.to_excel(path_save_results + '/spct-stats_bin-mindx-id_weighted-average.xlsx')

    # save and/or show plots
    if save_figs or show_figs:

        path_save_figs = path_figs + '/spct-stats_2d-precision_mindx-id'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

        count_column = 'counts'
        if bin_count_threshold is not None:
            df_bin_mindx = df_bin_mindx[df_bin_mindx[count_column] > bin_count_threshold]

        for pc in precision_columns:
            fig, ax = plt.subplots()

            ax.plot(df_bin_mindx[column_to_bin], df_bin_mindx[pc], '-o', markersize=3)

            ax.set_xlabel(column_to_bin)
            ax.set_ylabel('{} precision'.format(pc))

            axr = ax.twinx()
            axr.plot(df_bin_mindx[column_to_bin], df_bin_mindx[count_column], '-s', markersize=2, alpha=0.25)
            axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
            axr.set_ylim([0, int(np.round(df_bin_mindx[count_column].max() + 6, -1))])

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_save_figs + '/spct-stats_bin-mindx-id_{}.png'.format(pc))
            if show_figs:
                plt.show()
            plt.close()


def bin_plot_spct_stats_2d_static_precision_pdo_id(df, column_to_bin, precision_columns, bins, round_to_decimal,
                                                   pdo_threshold, bin_count_threshold,
                                                   export_results, path_results, save_figs, path_figs, show_figs):
    if pdo_threshold is not None:
        df[column_to_bin] = df[column_to_bin].where(df[column_to_bin] > pdo_threshold, pdo_threshold)

    df_bin_pdxo_id, df_bin_pdxo = analyze.evaluate_2d_static_precision(df,
                                                                       column_to_bin=column_to_bin,
                                                                       precision_columns=precision_columns,
                                                                       bins=bins,
                                                                       round_to_decimal=round_to_decimal)

    # export results
    if export_results:

        path_save_results = path_results + '/spct-stats_2d-precision_pdo-id'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        df_bin_pdxo_id.to_excel(path_save_results + '/spct-stats_bin-pdo-id.xlsx')
        df_bin_pdxo.to_excel(path_save_results + '/spct-stats_bin-pdo-id_weighted-average.xlsx')

    # save and/or show plots
    if save_figs or show_figs:

        path_save_figs = path_figs + '/spct-stats_2d-precision_pdo-id'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

        count_column = 'counts'
        if bin_count_threshold is not None:
            df_bin_pdxo = df_bin_pdxo[df_bin_pdxo[count_column] > bin_count_threshold]

        # plot bin(percent dx diameter, id)
        for pc in precision_columns:
            fig, ax = plt.subplots()

            ax.plot(df_bin_pdxo[column_to_bin], df_bin_pdxo[pc], '-o', markersize=3)

            ax.set_xlabel(column_to_bin)
            ax.set_ylabel('{} precision'.format(pc))

            axr = ax.twinx()
            axr.plot(df_bin_pdxo[column_to_bin], df_bin_pdxo[count_column], '-s', markersize=2, alpha=0.25)
            axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
            axr.set_ylim([0, int(np.round(df_bin_pdxo[count_column].max() + 6, -1))])

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_save_figs + '/spct-stats_bin-pdo-id_{}.png'.format(pc))
            if show_figs:
                plt.show()
            plt.close()


def bin_plot_spct_stats_1d_static_precision_id(df, precision_columns, bins, export_results, path_results,
                                               bin_count_threshold, save_figs, path_figs, show_figs):
    xparam = 'id'
    round_to_decimal = 0

    dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                        column_to_bin=xparam,
                                                        precision_columns=precision_columns,
                                                        bins=bins,
                                                        round_to_decimal=round_to_decimal)

    # export results
    if export_results:

        path_save_results = path_results + '/spct-stats_1d-precision_id'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        dfp_id.to_excel(path_save_results + '/spct-stats_bin-id.xlsx')
        dfpm.to_excel(path_save_results + '/spct-stats_bin-id_weighted-average.xlsx')

    # save and/or show plots
    if save_figs or show_figs:

        path_save_figs = path_figs + '/spct-stats_1d-precision_id'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

        count_column = 'counts'
        if bin_count_threshold is not None:
            dfp_id = dfp_id[dfp_id[count_column] > bin_count_threshold]

        for pc in precision_columns:
            fig, ax = plt.subplots()

            ax.plot(dfp_id[xparam], dfp_id[pc], '-o', markersize=3)

            ax.set_xlabel(xparam)
            ax.set_ylabel('{} precision'.format(pc))

            axr = ax.twinx()
            axr.plot(dfp_id[xparam], dfp_id[count_column], '-s', markersize=2, alpha=0.25)
            axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
            axr.set_ylim([0, int(np.round(dfp_id[count_column].max() + 6, -1))])

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_save_figs + '/spct-stats_bin-id_{}.png'.format(pc))
            if show_figs:
                plt.show()
            plt.close()

    # package precision to update overview dictionary
    dict_1d_static_precision_id = {'precision_x': dfpm.loc['x'],
                                   'precision_gauss_xc': dfpm.loc['gauss_xc'],
                                   'precision_y': dfpm.loc['y'],
                                   'precision_gauss_yc': dfpm.loc['gauss_yc'],
                                   'precision_gauss_rc': dfpm.loc['gauss_rc'],
                                   }

    return dict_1d_static_precision_id


def bin_plot_spct_stats_sampling_frequency_z_id(df, column_to_bin, bins, area, microns_per_pixel,
                                                export_results, path_results, save_figs, path_figs, show_figs):
    dfzm, dfzstd = bin.bin_generic(df,
                                   column_to_bin=column_to_bin,
                                   column_to_count='id',
                                   bins=bins,
                                   round_to_decimal=4,
                                   return_groupby=True)

    # emitter density
    dfzm['emitter_density'] = dfzm.count_id / area

    # lateral sampling frequency
    dfzm['nyquist_mean_dx'] = 2 * dfzm.mean_dx * microns_per_pixel
    dfzm['nyquist_min_dx'] = 2 * dfzm.min_dx * microns_per_pixel

    # minimum lateral Nyquist sampling
    if 'contour_diameter' in dfzm.columns:
        dfzm['nyquist_min_no_contour_overlap'] = 2 * dfzm.contour_diameter * microns_per_pixel

    if 'gauss_diameter' in dfzm.columns:
        dfzm['nyquist_min_no_overlap'] = 2 * dfzm.gauss_diameter * microns_per_pixel

    # export results
    if export_results:

        path_save_results = path_results + '/spct-stats_sampling_frequency_z-id'
        if not os.path.exists(path_save_results):
            os.makedirs(path_save_results)

        dfzm.to_excel(path_save_results + '/spct-stats_sampling_frequency_z-id_mean.xlsx')
        dfzstd.to_excel(path_save_results + '/spct-stats_sampling_frequency_z-id_std.xlsx')

    # save and/or show plots
    if save_figs or show_figs:

        path_save_figs = path_figs + '/spct-stats_sampling_frequency_z-id'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

    # plot: emitter density (z)
    fig, ax = plt.subplots()
    ax.plot(dfzm.bin, dfzm.emitter_density, '-o', markersize=3)
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'emitter density $(\#/\mu m^2)$')

    plt.tight_layout()
    if save_figs:
        plt.savefig(path_save_figs + '/spct-stats_emitter-density.png')
    if show_figs:
        plt.show()
    plt.close()

    # plot: Nyquist sampling (z)
    fig, ax = plt.subplots()

    ax.plot(dfzm.bin, dfzm.nyquist_mean_dx, '-o', markersize=3, label=r'$\overline {\delta x}$')
    ax.plot(dfzm.bin, dfzm.nyquist_min_dx, '-o', markersize=3, label=r'$\delta x_{min}$')

    ax.plot(dfzm.bin, dfzm.nyquist_min_no_overlap, '-o', markersize=3,
            label=r'$\delta x_{min, N.O.}^{Gaussian}$')

    ax.plot(dfzm.bin, dfzm.nyquist_min_no_contour_overlap, '-o', markersize=3,
            label=r'$\delta x_{min, N.O.}^{Contour}$')

    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.set_ylabel(r'$f_{Nyquist} \: (\mu m)$')
    ax.legend()

    plt.tight_layout()
    if save_figs:
        plt.savefig(path_save_figs + '/spct-stats_nyquist-sampling.png')
    if show_figs:
        plt.show()
    plt.close()


def plot_spct_stats_id_by_param(df, xparam, plot_columns, particle_ids, save_figs, path_figs, show_figs):
    # save and/or show plots
    if save_figs:

        path_save_figs = path_figs + '/spct-stats_id_by_param'
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

    df = df.copy().dropna()
    all_pids = df.id.unique()

    if particle_ids == 'all':
        particle_ids = all_pids
    elif (isinstance(particle_ids, (list, np.ndarray))):
        pass
    elif (isinstance(particle_ids, float)):
        particle_ids = np.random.choice(all_pids, int(particle_ids), replace=False)
    elif particle_ids is None:
        particle_ids = np.random.choice(all_pids, 5, replace=False)
    elif (isinstance(particle_ids, int)):
        particle_ids = [particle_ids]

    # filter out particle that translate too much
    dfg = df.groupby('id').std().reset_index()
    dfg['xy_std'] = np.mean([dfg.x, dfg.y])

    particle_ids = [p for p in particle_ids if p not in dfg[dfg['xy_std'] > 1.35].id.unique()]

    for pid in particle_ids:

        dfpid = df[df['id'] == pid]

        # correct gaussian centers
        if 'gauss_xc_corr' in plot_columns:
            dfpid['gauss_xc_corr'] = df['gauss_xc'] - dfpid['x'] + (dfpid['y'] - dfpid.iloc[0].y)
        if 'gauss_yc_corr' in plot_columns:
            dfpid['gauss_yc_corr'] = dfpid['gauss_yc'] - dfpid['y'] + (dfpid['x'] - dfpid.iloc[0].x)

        path_save_figs_pid = path_save_figs + '/pid{}_x{}_y{}'.format(pid,
                                                                      int(np.round(dfpid.x.mean(), 0)),
                                                                      int(np.round(dfpid.y.mean(), 0))
                                                                      )
        if not os.path.exists(path_save_figs_pid):
            os.makedirs(path_save_figs_pid)

        for pc in plot_columns:

            fig, ax = plt.subplots()

            if isinstance(pc, list):
                for ppc in pc:
                    ax.plot(dfpid[xparam], dfpid[ppc], '-o', markersize=3, label=ppc)
                ax.legend()

            else:
                ax.plot(dfpid[xparam], dfpid[pc], '-o', markersize=3)

            ax.set_xlabel(xparam)
            ax.set_ylabel(pc)
            ax.set_title(r'$p_{ID}$' + '{} (x={}, y={})'.format(pid,
                                                                np.round(dfpid.x.mean(), 1),
                                                                np.round(dfpid.y.mean(), 1)
                                                                )
                         )

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_save_figs_pid + '/spct-stats_pid{}_plot-{}_by-{}.png'.format(pid, pid, pc, xparam))
            if show_figs:
                plt.show()
            plt.close(fig)


def plot_spct_stats_compare_ids_by_param(df, xparam, compare_param, plot_columns, particle_ids,
                                         save_figs, path_figs, show_figs):
    # save and/or show plots
    if save_figs:
        path_save_figs = path_figs + '/spct-stats_compare-id_by_param/compare-{}'.format(compare_param)
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

    # get number of axes
    num_axes = len(plot_columns)

    # make axes labels
    axes_labels = []
    for pc in plot_columns:

        if pc == 'gauss_xc_corr':
            pc = r'$x_{c} \: (G)$'
        elif pc == 'gauss_yc_corr':
            pc = r'$y_{c} \: (G)$'
        elif pc == 'gauss_sigma_x_y':
            pc = r'$\sigma_{x} / \sigma_{y} \: (G)$'

        axes_labels.append(pc)

    # make sure particle ids is a list of lists
    if not isinstance(particle_ids[0], list):
        particle_ids = [particle_ids]

    # copy the dataframe
    df = df.copy().dropna()

    for pair_ids in particle_ids:

        save_this_fig = True
        fig, ax = plt.subplots(nrows=num_axes, sharex=True, figsize=(size_x_inches,
                                                                     size_y_inches * (num_axes - 1) / 1.5))

        save_strings = []
        for pid in pair_ids:

            dfpid = df[df['id'] == pid]

            # create save string
            save_strings.append([int(np.round(dfpid.x.mean(), 0)), int(np.round(dfpid.y.mean(), 0))])

            # correct gaussian centers
            if 'gauss_xc_corr' in plot_columns:
                dfpid['gauss_xc_corr'] = df['gauss_xc'] - dfpid['x'] + (dfpid['y'] - dfpid.iloc[0].y)
            if 'gauss_yc_corr' in plot_columns:
                dfpid['gauss_yc_corr'] = dfpid['gauss_yc'] - dfpid['y'] + (dfpid['x'] - dfpid.iloc[0].x)

            for i, pc in enumerate(plot_columns):
                ax[i].plot(dfpid[xparam], dfpid[pc], '-o', markersize=3)

                # modifying for not saving bad plots
                if dfpid[pc].diff().max() > 1:
                    save_this_fig = False

        # axes labels
        ax[num_axes - 1].set_xlabel(xparam)
        for i, lbl in enumerate(axes_labels):
            ax[i].set_ylabel(lbl)

        plt.tight_layout()

        if save_figs and save_this_fig:
            plt.savefig(path_save_figs + '/pids_{}-x{}y{}_{}-x{}y{}_plot_by-{}.png'.format(pair_ids[0],
                                                                                           save_strings[0][0],
                                                                                           save_strings[0][1],
                                                                                           pair_ids[1],
                                                                                           save_strings[1][0],
                                                                                           save_strings[1][1],
                                                                                           xparam,
                                                                                           )
                        )

        if show_figs:
            plt.show()

        plt.close(fig)


def plot_spct_stats_compare_ids_by_along_param(df, columns_to_bin, bins, low_level_bins_to_plot,
                                               plot_columns, column_to_plot_along, round_to_decimals,
                                               save_figs, path_figs, show_figs):
    if isinstance(plot_columns, str):
        plot_columns = [plot_columns]

    # save and/or show plots
    if save_figs:
        path_save_figs = path_figs + '/spct-stats_compare-along'

        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)

    # get number of axes
    num_axes = len(plot_columns)
    markersize = 1.5

    # make axes labels
    axes_labels = []
    for pc in plot_columns:
        if pc == 'gauss_xc':
            pc = r'$x_{c}$'
        elif pc == 'gauss_yc':
            pc = r'$y_{c}$'
        elif pc == 'gauss_sigma_x_y':
            pc = r'$\sigma_{x} / \sigma_{y}$'
        axes_labels.append(pc)

    # get particle ID's by binning
    df = bin.bin_generic_2d(df,
                            columns_to_bin=columns_to_bin,
                            column_to_count='id',
                            bins=bins,
                            round_to_decimals=round_to_decimals,
                            min_num_bin=1,
                            return_groupby=False,
                            )

    # # columns_to_bin, bins, bins_to_plot, plot_columns, column_to_plot_along,
    top_level_bins_to_plot = df.sort_values('bin_tl').bin_tl.unique()
    low_level_bin_to_plot = df.sort_values('bin_ll').bin_ll.unique()[low_level_bins_to_plot]
    df = df.loc[df['bin_ll'] == low_level_bin_to_plot]

    # plot
    fig, ax = plt.subplots(nrows=num_axes, sharex=True, figsize=(size_x_inches * 1.2,
                                                                 size_y_inches * (num_axes - 1) / 1.25)
                           )
    legend_handles = []

    for bntl in top_level_bins_to_plot:

        # get the dataframe of this bin only
        bin_pids = df[df['bin_tl'] == bntl].id.unique()

        # sometimes there won't be a particle in a bin so you have to skip
        if len(bin_pids) < 1:
            continue

        # get a random particle id in this bin
        pid = np.random.choice(bin_pids, 1)[0]

        # get the dataframe for pid only
        dfpid = df[df['id'] == pid].reset_index()

        # filter 1: remove particles with segmentation errors
        if dfpid.gauss_xc.diff().abs().max() > 0.5:
            continue
        if dfpid.gauss_yc.diff().abs().max() > 0.5:
            continue

        # add a column for gaussian location displacement
        if 'gauss_dxc' in plot_columns:
            dfpid['gauss_dxc'] = dfpid['gauss_xc'] - dfpid.iloc[dfpid.z_corr.abs().idxmin()].gauss_xc
        if 'gauss_dyc' in plot_columns:
            dfpid['gauss_dyc'] = dfpid['gauss_yc'] - dfpid.iloc[dfpid.z_corr.abs().idxmin()].gauss_yc

        add_to_legend = True
        for i, pc in enumerate(plot_columns):

            ax[i].plot(dfpid[column_to_plot_along], dfpid[pc], '-o', markersize=markersize)

            if add_to_legend is True:
                legend_handles.append("{}: {}, {}".format(pid,
                                                          np.round(dfpid.x.mean(), 1),
                                                          np.round(dfpid.y.mean(), 1)))
                add_to_legend = False

    # axes labels
    ax[num_axes - 1].set_xlabel(column_to_plot_along)
    for i, lbl in enumerate(axes_labels):
        ax[i].set_ylabel(lbl)

    ax[int(np.floor(num_axes // 2))].legend(legend_handles,
                                            loc='upper left',
                                            bbox_to_anchor=(1, 1),
                                            title=r'$p_{ID}: x, y$'
                                            )

    plt.tight_layout()
    if save_figs:
        plt.savefig(path_save_figs + '/slice-{}_along-{}={}_plot-{}.png'.format(columns_to_bin[0],
                                                                                columns_to_bin[1],
                                                                                low_level_bin_to_plot,
                                                                                column_to_plot_along,
                                                                                ))
    if show_figs:
        plt.show()

    plt.close(fig)