# test bin, analyze, and plot functions
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, CloughTocher2DInterpolator

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

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# TEST COORDS (FINAL)
"""
IDPT:
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
            FINAL-04.21-22_IDPT_1um-calib_5um-test'
           
SPCT:
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
           'FINAL-04.25.22_SPCT_1um-calib_5um-test'
"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
           'FINAL-04.25.22_SPCT_1um-calib_5um-test'
method = 'spct'

path_test_coords = join(base_dir, 'coords/test-coords')
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_calib_spct_pop = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/coords/calib-coords/calib_spct_pop_defocus_stats.xlsx'
path_calib_spct_stats = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/coords/calib-coords/calib_spct_stats_.xlsx'
path_test_spct_pop = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-04.12.22-SPCT-5umStep-meta/coords/calib-coords/calib_spct_pop_defocus_stats_11.06.21_z-micrometer-v2_5umMS__sim-sym.xlsx'
path_test_calib_coords = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-04.12.22-SPCT-5umStep-meta/coords/calib-coords/calib_correction_coords_11.06.21_z-micrometer-v2_5umMS__sim-sym.xlsx'
path_similarity = join(base_dir, 'similarity')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# true particle in-focus positions in 5 um test images (from FIJI)
path_true_particle_locations = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/' \
                               'analyses/shared-results/fiji-particle-locations/fiji_in-focus_particle-locations.xlsx'

# ----------------------------------------------------------------------------------------------------------------------
# 0. SETUP PROCESS CONTROLS

# experimental
mag_eff = 10.0
microns_per_pixel = 1.6
area_pixels = 512 ** 2
img_xc, img_yc = 256, 256
area_microns = (512 * microns_per_pixel) ** 2

# processing
true_num_particles_per_frame = 92
z_range = [-55, 55]
measurement_depth = z_range[1] - z_range[0]
num_frames_per_step = 3
filter_barnkob = measurement_depth / 10
min_cm = 0.5
min_percent_layers = 0.5
remove_ids = None

# ----------------------------------------------------------------------------------------------------------------------
# 1. READ FILES

# read test coords (where z is based on 1-micron steps and z_true is 5-micron steps every 3 frames)
analyze_test = True
if analyze_test:

    # ------------------------------------------------------------------------------------------------------------------
    # 2. PERFORM CORRECTION

    perform_correction = False
    plot_calib_plane = False
    plot_test_plane = False

    if perform_correction:
        measure_focal_plane_curvature = False
        correction_method = 'correct_test'

        # calibration coords z in-focus
        mag_eff_c, zf_c, c1_c, c2_c = io.read_pop_gauss_diameter_properties(path_calib_spct_pop)
        mag_eff_t, zf_t, c1_t, c2_t = io.read_pop_gauss_diameter_properties(path_test_spct_pop)
        dz_f_test_to_calibration = zf_t - zf_c

        if measure_focal_plane_curvature:

            # 1. read calib coords
            dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)

            param_zf = 'zf_from_peak_int'
            kx = 2
            ky = 2

            dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
                correct.fit_plane_correct_plane_fit_spline(dfcal=dfcpid,
                                                           param_zf=param_zf,
                                                           microns_per_pixel=microns_per_pixel,
                                                           img_xc=img_xc,
                                                           img_yc=img_yc,
                                                           kx=kx,
                                                           ky=ky,
                                                           path_figs=path_figs)

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
            dfcstats_field_curvature_tilt_corrected.to_excel(path_results +
                                                             '/calib_spct_stats_field-curvature-and-tilt-corrected.xlsx',
                                                             index=False)

        else:
            # deprecated fit calibration surface
            """
            # read calibration coords (where z is based on 5-micron steps) and fit plane
            dfc_five_micron_steps = pd.read_excel(path_test_calib_coords)
            dfc5g = dfc_five_micron_steps.groupby('id').mean().reset_index()
            dictc_fit_plane = correct.fit_in_focus_plane(df=dfc5g, param_zf='z_f', microns_per_pixel=microns_per_pixel,
                                                         img_xc=img_xc, img_yc=img_yc)
            popt_c5g = dictc_fit_plane['popt_pixels']

            if plot_calib_plane:
                fig = plotting.plot_fitted_plane_and_points(df=dfc5g, dict_fit_plane=dictc_fit_plane)
                plt.savefig(path_figs + '/______fit-plane_raw.png')
                plt.show()
                plt.close()

                dfict_fit_plane = pd.DataFrame.from_dict(dictc_fit_plane, orient='index', columns=['value'])
                dfict_fit_plane.to_excel(path_figs + '/_____fit-plane_raw.xlsx')
            """

        # read test coords (where z is based on 1-micron steps and z_true is 5-micron steps every 3 frames)
        dft = io.read_test_coords(path_test_coords)

        # plot particle trajectories to verify
        """dft_ids = dft.id.unique()
        num_pids = len(dft_ids)
        pids_per_fig = 5
        num_figs = np.ceil(num_pids / pids_per_fig)
        for i in range(int(num_figs)):
            pid_list = dft_ids[i * pids_per_fig: (i + 1) * pids_per_fig]

            fig, ax = plt.subplots()
            for pid in pid_list:
                dfpid = dft[dft['id'] == pid]
                ax.plot(dfpid.frame, dfpid.z, '-o', ms=1, label=pid)
            ax.set_xlabel('frame')
            ax.set_ylabel('z')
            ax.legend()
            plt.tight_layout()
            plt.show()
        """

        if correction_method == 'correct_test':

            path_test_coords_corrected = path_test_coords + '/custom-correction'
            if not os.path.exists(path_test_coords_corrected):
                os.makedirs(path_test_coords_corrected)

            # calculate the corrected values
            dft['z_true_test'] = (dft['z_true'] - dft['z_true'] % 3) / 3 * 5
            dft['z_true_corr'] = dft['z_true_test'] - 68.6519
            dft['z_corr'] = dft['z'] - 49.9176 - 5
            dft['error_corr'] = dft['z_corr'] - dft['z_true_corr']

            # replace the old values
            dft['z_true'] = dft['z_true_corr']
            dft['z'] = dft['z_corr']
            dft['error'] = dft['error_corr']

            # drop the extra columns
            dft = dft.drop(columns=['z_true_test', 'z_true_corr', 'z_corr', 'error_corr'])

            dft.to_excel(path_test_coords_corrected)

    else:
        dft = io.read_test_coords(path_test_coords)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. SETUP MODIFIERS

    theory_diam_path = path_calib_spct_pop
    dft['z_corr'] = dft['z']

    # compute the radial distance
    if 'r' not in dft.columns:
        dft['r'] = np.sqrt((256 - dft.x) ** 2 + (256 - dft.y) ** 2)

    # ------------------------------------------------------------------------------------------------------------------
    # 4. STORE RAW STATS (AFTER FILTERING RANGE)

    # filter range so test matches calibration range
    dft = dft[(dft['z_true'] > z_range[0]) & (dft['z_true'] < z_range[1])]

    i_num_rows = len(dft)
    i_num_pids = len(dft.id.unique())

    # get filtering stats
    num_rows_cm_filter = len(dft[dft['cm'] < min_cm])
    num_rows_barnkob_filter = len(dft[dft['error'].abs() > filter_barnkob])

    print("{} out of {} rows ({}%) below min_cm filter: {}".format(num_rows_cm_filter,
                                                                    i_num_rows,
                                                                    np.round(num_rows_cm_filter / i_num_rows * 100, 1),
                                                                    min_cm)
          )

    print("{} out of {} rows ({}%) above error filter: {}".format(num_rows_barnkob_filter,
                                                                   i_num_rows,
                                                                   np.round(num_rows_barnkob_filter / i_num_rows * 100,
                                                                            1),
                                                                   filter_barnkob)
          )

    raise ValueError()
    # Barnkob filter
    # dft = dft[dft['error'].abs() < filter_barnkob]
    # print("Barnkob filtering!")

    # ------------------------------------------------------------------------------------------------------------------
    # 5. SPLIT DATAFRAME INTO BINS

    true_zs = dft.z_true.unique()
    dft['bin'] = np.round(dft['z_true'], 0).astype(int)
    dzs = dft.bin.unique()
    num_dz_steps = len(dzs)

    dfs = []
    names = []
    initial_stats = []

    for i, dz in enumerate(dzs):
        dftz = dft[dft['bin'] == dz]
        dfs.append(dftz)
        names.append(i)
        initial_stats.append([i, dz, len(dftz), len(dftz.id.unique())])

# ----------------------------------------------------------------------------------------------------------------------
# 6. PRECISION SWEEP
anything_precision = False
if anything_precision:
    # Evaluate percent measure as a function of precision (in order to find an adequate precision limit)
    analyze_percent_measure_by_precision = False
    export_results = False
    read_and_plot_percent_measure_by_precision = False

    if analyze_percent_measure_by_precision:

        data_percent_measure_by_precision = []

        for df, name, dz in zip(dfs, names, dzs):

            # precision @ z

            xparam = 'id'
            pos = ['z_corr']
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
                remove_ids = dfp_id[dfp_id['z_corr'] > filt_p].id.unique()

                ip_num_rows = len(df)
                ip_num_pids = len(dfp_id.id.unique())

                dffp = df.copy()
                dffp = dffp[~dffp.id.isin(remove_ids)]

                # --- calculate percent overlap
                dffpo = analyze.calculate_particle_to_particle_spacing(test_coords_path=dffp,
                                                                       theoretical_diameter_params_path=theory_diam_path,
                                                                       mag_eff=mag_eff,
                                                                       z_param='z_corr',
                                                                       zf_at_zero=True,
                                                                       max_n_neighbors=5,
                                                                       true_coords_path=None,
                                                                       maximum_allowable_diameter=None)

                # sampling frequency results
                num_rows = len(dffp)
                num_pids = len(dffp.id.unique())
                percent_rows = num_rows / ip_num_rows
                percent_pids = num_pids / ip_num_pids
                particle_density = num_pids / area_pixels
                emitter_density_microns = num_pids / area_microns
                mean_mean_dx = dffpo.mean_dx.mean()
                mean_min_dx = dffpo.min_dx.mean()
                mean_num_dxo = dffpo.num_dxo.mean()
                data_percent_measure_by_precision.append([name, dz, filt_p, num_rows, num_pids,
                                                          percent_rows, percent_pids,
                                                          particle_density, emitter_density_microns,
                                                          mean_mean_dx, mean_min_dx, mean_num_dxo])

        # evaluate
        data_percent_measure_by_precision = np.array(data_percent_measure_by_precision)
        cols = ['name', 'dz', 'fprecision', 'num_rows', 'num_pids', 'percent_rows', 'percent_pids', 'p_density_pixels',
                'p_density_microns', 'mean_dx', 'min_dx', 'num_dxo']
        dfpmp = pd.DataFrame(data_percent_measure_by_precision, columns=cols)

        # export results
        if export_results:
            dfpmp.to_excel(path_results + '/percent_measure_by_precision.xlsx', index=False)

            dfpmpg = dfpmp.groupby('fprecision').mean().reset_index()
            dfpmpg.to_excel(path_results + '/percent_measure_by_precision_mean.xlsx', index=False)

        if read_and_plot_percent_measure_by_precision:
            dfpmpg = pd.read_excel(path_results + '/percent_measure_by_precision_mean.xlsx')

            for pc in ['percent_rows', 'percent_pids', 'p_density_microns', 'mean_dx', 'min_dx', 'num_dxo']:
                fig, ax = plt.subplots()
                ax.plot(dfpmpg.fprecision, dfpmpg[pc], '-o')
                ax.set_xlabel(r'z-precision $(\mu m)$')
                ax.set_ylabel(pc)
                plt.tight_layout()
                plt.savefig(path_figs + '/meas-{}_by_precision_filter.png'.format(pc))
                plt.show()

    # ----------------------------------------------------------------------------------------------------------------------
    # 7. EVALUATE DISPLACEMENT PRECISION (ID)

    analyze_precision_per_id = False
    analyze_match_location_precision = False
    export_results = False
    save_plots, show_plots = False, False
    save_plots_collection, show_plots_collection = False, False

    filter_precision = 50

    if analyze_precision_per_id:

        if analyze_match_location_precision:

            # precision (z, ID)

            precision_per_id_results = []
            remove_ids = []
            dfp_ids = []

            for df, name, dz in zip(dfs, names, dzs):

                # precision @ z

                xparam = 'id'
                pos = ['z_corr', 'xm', 'ym']
                particle_ids = df[xparam].unique()
                count_column = 'counts'

                dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                                    column_to_bin=xparam,
                                                                    precision_columns=pos,
                                                                    bins=particle_ids,
                                                                    round_to_decimal=0)

                # store raw results
                dz_i_num_rows = len(df)
                dz_i_num_pids = len(dfp_id)

                # filter out particles with precision > filter_precision
                remove_ids.append([name, dz, dfp_id[dfp_id['z_corr'] > filter_precision].id.unique()])
                dfp_id = dfp_id[dfp_id['z_corr'] < filter_precision]

                # store results
                dz_f_num_rows = dfp_id.counts.sum()
                dz_f_num_pids = len(dfp_id)
                dz_percent_rows = dz_f_num_rows / dz_i_num_rows * 100
                dz_percent_pids = dz_f_num_pids / dz_i_num_pids * 100
                dz_mean_dz = dfp_id.z_corr_m.mean()
                dz_precision = dfp_id.z_corr.mean()

                dz_mean_xm = dfp_id.xm_m.mean()
                dz_mean_ym = dfp_id.ym_m.mean()
                dz_precision_xm = dfp_id.xm.mean()
                dz_precision_ym = dfp_id.ym.mean()

                precision_per_id_results.append([name, dz, dz_mean_dz, dz_precision, dz_precision_xm, dz_precision_ym,
                                                 dz_mean_xm, dz_mean_ym, dz_percent_rows, dz_percent_pids,
                                                 dz_i_num_rows, dz_i_num_pids, dz_f_num_rows, dz_f_num_pids])

                dfp_id['name'] = name
                dfp_id['dz'] = dz
                dfp_ids.append(dfp_id)

                # plot bin(id)
                if save_plots or show_plots:
                    for pc in pos:
                        fig, ax = plt.subplots()

                        ax.plot(dfp_id[xparam], dfp_id[pc], '-o')

                        ax.set_xlabel(xparam)
                        ax.set_ylabel('{} precision'.format(pc))

                        axr = ax.twinx()
                        axr.plot(dfp_id[xparam], dfp_id[count_column], '-s', markersize=2, alpha=0.25)
                        axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
                        axr.set_ylim([0, int(np.round(dfp_id[count_column].max() + 6, -1))])
                        plt.tight_layout()
                        if save_plots:
                            plt.savefig(path_figs + '/{}-precision-by-id_{}_dz{}.png'.format(pc, name, dz))
                        if show_plots:
                            plt.show()
                        plt.close()

            dfp_dz = pd.DataFrame(np.array(precision_per_id_results), columns=['name', 'dz',
                                                                               'm_dz', 'm_z_precision',
                                                                               'm_xm_precision', 'm_ym_precision',
                                                                               'm_xm', 'm_ym',
                                                                               'percent_rows', 'percent_pids',
                                                                               'i_rows', 'i_pids', 'f_rows', 'f_pids'])

            dfp_ids = pd.concat(dfp_ids, ignore_index=True)

            # calculate mean precision for all
            dfp_mean = dfp_ids.copy()
            dfp_mean['bin'] = 1
            dfp_mean = dfp_mean.groupby('bin').mean()

            if export_results:
                filter_precision_this_loop = True
                if filter_precision_this_loop:
                    dfp_mean.to_excel(
                        path_results + '/match-location-precision_mean-all_fprecision={}.xlsx'.format(filter_precision),
                        index=True)
                    dfp_dz.to_excel(
                        path_results + '/match-location-precision_by_dz_fprecision={}.xlsx'.format(filter_precision),
                        index=False)
                    dfp_ids.to_excel(
                        path_results + '/match-location-precision_by_dz-id_fprecision={}.xlsx'.format(filter_precision),
                        index=False)
                else:
                    dfp_mean.to_excel(path_results + '/match-location-precision_mean-all_no-fprecision.xlsx',
                                      index=True)
                    dfp_dz.to_excel(path_results + '/match-location-precision_by_dz_no-fprecision.xlsx', index=False)
                    dfp_ids.to_excel(path_results + '/match-location-precision_by_dz-id_no-fprecision.xlsx',
                                     index=False)

        else:

            precision_per_id_results = []
            remove_ids = []
            dfp_ids = []

            for df, name, dz in zip(dfs, names, dzs):

                # precision @ z

                xparam = 'id'
                pos = ['z_corr', 'x', 'y']
                particle_ids = df[xparam].unique()
                count_column = 'counts'

                dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                                    column_to_bin=xparam,
                                                                    precision_columns=pos,
                                                                    bins=particle_ids,
                                                                    round_to_decimal=0)

                # store raw results
                dz_i_num_rows = len(df)
                dz_i_num_pids = len(dfp_id)

                # filter out particles with precision > filter_precision
                remove_ids.append([name, dz, dfp_id[dfp_id['z_corr'] > filter_precision].id.unique()])
                dfp_id = dfp_id[dfp_id['z_corr'] < filter_precision]

                # store results
                dz_f_num_rows = dfp_id.counts.sum()
                dz_f_num_pids = len(dfp_id)
                dz_percent_rows = dz_f_num_rows / dz_i_num_rows * 100
                dz_percent_pids = dz_f_num_pids / dz_i_num_pids * 100
                dz_mean_dz = dfp_id.z_corr_m.mean()
                dz_precision = dfp_id.z_corr.mean()

                dz_precision_x = dfp_id.x.mean()
                dz_precision_y = dfp_id.y.mean()
                dz_mean_x = dfp_id.x_m.mean()
                dz_mean_y = dfp_id.y_m.mean()

                precision_per_id_results.append([name, dz, dz_mean_dz, dz_precision, dz_precision_x, dz_precision_y,
                                                 dz_mean_x, dz_mean_y, dz_percent_rows, dz_percent_pids,
                                                 dz_i_num_rows, dz_i_num_pids, dz_f_num_rows, dz_f_num_pids])

                dfp_id['name'] = name
                dfp_id['dz'] = dz
                dfp_ids.append(dfp_id)

                # plot bin(id)
                if save_plots or show_plots:
                    for pc in pos:
                        fig, ax = plt.subplots()

                        ax.plot(dfp_id[xparam], dfp_id[pc], '-o')

                        ax.set_xlabel(xparam)
                        ax.set_ylabel('{} precision'.format(pc))

                        axr = ax.twinx()
                        axr.plot(dfp_id[xparam], dfp_id[count_column], '-s', markersize=2, alpha=0.25)
                        axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
                        axr.set_ylim([0, int(np.round(dfp_id[count_column].max() + 6, -1))])
                        plt.tight_layout()
                        if save_plots:
                            plt.savefig(path_figs + '/z-precision-by-id_{}_dz{}.png'.format(name, dz))
                        if show_plots:
                            plt.show()
                        plt.close()

            dfp_dz = pd.DataFrame(np.array(precision_per_id_results), columns=['name', 'dz', 'm_dz', 'm_z_precision',
                                                                               'm_x_precision', 'm_y_precision',
                                                                               'm_x', 'm_y',
                                                                               'percent_rows', 'percent_pids',
                                                                               'i_rows', 'i_pids', 'f_rows', 'f_pids'])

            dfp_ids = pd.concat(dfp_ids, ignore_index=True)

            if export_results:
                dfp_dz.to_excel(path_results + '/precision_by_dz_fprecision={}.xlsx'.format(filter_precision),
                                index=False)
                dfp_ids.to_excel(path_results + '/precision_by_dz-id_fprecision={}.xlsx'.format(filter_precision),
                                 index=False)

                dfp_mean = dfp_ids[dfp_ids['counts'] >= 3].mean()
                dfp_mean.to_excel(path_results + '/precision_mean_fprecision={}.xlsx'.format(filter_precision),
                                  index=False)

        if save_plots_collection or show_plots_collection:

            if analyze_match_location_precision:
                x_param = 'dz'
                y_params = ['m_xm_precision', 'm_ym_precision']
                lbls = [r'$\sigma_{p} \: (\mu m)$', r'$\phi_{ID} \: (\%)$']

                fig, ax = plt.subplots()

                ax.plot(dfp_dz[x_param], dfp_dz[y_params[0]], '-o', label='x')
                ax.plot(dfp_dz[x_param], dfp_dz[y_params[1]], '-s', label='y')
                ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                ax.set_ylabel(lbls[0])
                ax.legend()

                plt.tight_layout()
                if save_plots_collection:
                    plt.savefig(path_figs + '/match-location_x-y-precision-by-dz.png')
                if show_plots_collection:
                    plt.show()
                plt.close()

            else:

                x_param = 'dz'
                y_params = ['m_y_precision', 'm_x_precision', 'm_z_precision', 'percent_pids']
                lbls = [r'$\sigma_{y} \: (\mu m)$', r'$\sigma_{x} \: (\mu m)$',
                        r'$\sigma_{z} \: (\mu m)$', r'$\phi_{ID} \: (\%)$']

                fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(size_x_inches,
                                                                                        size_y_inches * 1.5))

                ax1.plot(dfp_dz[x_param], dfp_dz[y_params[0]], '-o', ms=2)
                ax1.set_ylabel(lbls[0])

                ax2.plot(dfp_dz[x_param], dfp_dz[y_params[1]], '-o', ms=2)
                ax2.set_ylabel(lbls[1])

                ax3.plot(dfp_dz[x_param], dfp_dz[y_params[2]], '-o', ms=2)
                ax3.set_ylabel(lbls[2])

                ax4.plot(dfp_dz[x_param], dfp_dz[y_params[3]], '-o', ms=2)
                ax4.set_ylabel(lbls[3])
                ax4.set_xlabel(r'$\Delta z \: (\mu m)$')

                plt.tight_layout()
                if save_plots_collection:
                    plt.savefig(path_figs + '/z-precision-by-dz.png')
                if show_plots_collection:
                    plt.show()
                plt.close()

    else:
        remove_ids = None

# ----------------------------------------------------------------------------------------------------------------------
# 8. EVALUATE RMSE Z
analyze_rmse = True
export_results = False
save_plots = False
show_plots = False

if analyze_rmse:

    # filter out bad particle ids and barnkob filter
    dffs_errors = []
    dffs = []
    dfilt = []
    for df, name, dz in zip(dfs, names, dzs):

        # initial rows and pids
        i_pid_num_rows = true_num_particles_per_frame * num_frames_per_step  # 246  # len(df)
        i_pid_num_pids = true_num_particles_per_frame  # 82  # len(df.id.unique())

        # filter
        if remove_ids:
            exclude_ids = remove_ids[name][2]
            df = df[~df.id.isin(exclude_ids)]

        # final rows and pids
        f_pid_num_rows = len(df)
        f_pid_num_pids = len(df.id.unique())

        # barnkob filter
        df_error = df[df['error'].abs() > filter_barnkob]
        df = df[df['error'].abs() < filter_barnkob]
        f_barnkob_num_rows = len(df)
        f_barnkob_num_pids = len(df.id.unique())

        # calculate stats
        dz_percent_pids = f_barnkob_num_pids / i_pid_num_pids * 100
        dz_percent_rows = f_barnkob_num_rows / i_pid_num_rows * 100

        dffs_errors.append(df_error)
        dffs.append(df)
        dfilt.append([name, dz, dz_percent_pids, i_pid_num_pids, f_pid_num_pids, f_barnkob_num_pids,
                      dz_percent_rows, i_pid_num_rows, f_pid_num_rows, f_barnkob_num_rows])

    # focal plane bias errors
    dffp_errors = pd.concat(dffs_errors, ignore_index=True)
    dffp_errors.to_excel(path_results + '/test_coords_focal-plane-bias-errors.xlsx', index=False)

    # filtered and corrected test coords
    dffp = pd.concat(dffs, ignore_index=True)
    dfilt = pd.DataFrame(np.array(dfilt), columns=['name', 'dz', 'dz_percent_pids', 'i_pid_num_pids', 'f_pid_num_pids',
                                                   'f_barnkob_num_pids', 'dz_percent_rows', 'i_pid_num_rows',
                                                   'f_pid_num_rows', 'f_barnkob_num_rows'])
    dfilt = dfilt.set_index('dz')

    # store stats
    f_num_rows = len(dffp)
    f_num_pids = len(dffp.id.unique())

    # rmse
    dffp = dffp.rename(columns={'bin': 'binn'})

    # filter to remove focal plane bias errors
    # dffp = dffp[(dffp['binn'] < -6) | (dffp['binn'] > 6)]

    # export final + filtered test coords
    if export_results:
        dffp.to_excel(path_results + '/test_coords_spct-corrected-and-filtered.xlsx', index=False)

    # bin generic
    dfrmsem, dfrmsestd = bin.bin_generic(dffp,
                                         column_to_bin='binn',
                                         column_to_count='id',
                                         bins=dzs,
                                         round_to_decimal=0,
                                         return_groupby=True)

    dfrmse_mean = bin.bin_local_rmse_z(dffp, column_to_bin='binn', bins=1, min_cm=min_cm,
                                       z_range=None, round_to_decimal=0, df_ground_truth=None, dropna=True,
                                       error_column='error')

    dfrmse_bin = bin.bin_local_rmse_z(dffp, column_to_bin='binn', bins=dzs, min_cm=min_cm,
                                      z_range=None, round_to_decimal=0, df_ground_truth=None, dropna=True,
                                      error_column='error')

    # join rmse and filter dataframes
    dfrmse = pd.concat([dfrmse_bin, dfilt], axis=1, join='inner')

    # export
    if export_results:
        dfrmse.to_excel(path_results + '/rmse-z_binned.xlsx')
        dfrmse_mean.to_excel(path_results + '/rmse-z_mean.xlsx')

    # --- plot
    if save_plots or show_plots:

        # fit line
        popt, pcov = curve_fit(functions.line, dfrmse.z_true, dfrmse.z)
        z_fit = np.linspace(dfrmse.z_true.min(), dfrmse.z_true.max())

        rmse_fit_line = np.sqrt(np.sum((functions.line(dfrmse.z_true, *popt) - dfrmse.z) ** 2) / len(dfrmse.z))
        print("{} = r.m.s. error of the mean z per z-step from fitted line".format(rmse_fit_line))

        xylim = 62.5
        xyticks = [-50, -25, 0, 25, 50]

        # close all figs
        plt.close('all')

        # binned calibration curve with std-z errorbars (microns) + fit line
        if len(dfrmse) == len(dfrmsestd):
            fig, ax = plt.subplots()
            ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmsestd.z, fmt='o', ms=3, elinewidth=0.5, capsize=1,
                        color=sciblue,
                        label=r'$\overline{z} \pm \sigma$')  #
            ax.plot(z_fit, functions.line(z_fit, *popt), linestyle='--', linewidth=1.5, color='black', alpha=0.25,
                    label=r'$dz/dz_{true} = $' + ' {}'.format(np.round(popt[0], 3)))
            ax.plot(dffp_errors.z_true, dffp_errors.z, 'x', ms=3, color='tab:red', label=r'$\epsilon_{z} > h/10$')  #
            ax.set_xlabel(r'$z_{true} \: (\mu m)$')
            ax.set_xlim([-xylim, xylim])
            ax.set_xticks(ticks=xyticks, labels=xyticks)
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.set_ylim([-xylim, xylim])
            ax.set_yticks(ticks=xyticks, labels=xyticks)
            ax.legend(loc='lower right', handletextpad=0.25, borderaxespad=0.3)
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs +
                            '/calibration_curve_z+std-errobars_fit_line_a{}_b{}_slope-label-blk.png'.format(
                                np.round(popt[0],
                                         3),
                                np.round(popt[1],
                                         3))
                            )
            if show_plots:
                plt.show()
            plt.close()

        # rmse-z: microns
        fig, ax = plt.subplots()
        ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/rmse-z_microns.png')
        if show_plots:
            plt.show()
        plt.close()

        # rmse-z: normalized
        fig, ax = plt.subplots()
        ax.plot(dfrmse.index, dfrmse.rmse_z / measurement_depth, '-o')
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\sigma_{z}/h$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/rmse-z_normalized.png')
        if show_plots:
            plt.show()
        plt.close()

        # rmse-z + percent rows
        fig, ax = plt.subplots()
        ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')

        axr = ax.twinx()
        axr.plot(dfrmse.index, dfrmse.dz_percent_rows, '-s', ms=2, color='gray', alpha=0.5)
        axr.set_ylabel(r'$\phi_{ID} \: (\%)$', color='gray')
        axr.set_ylim([-5, 105])
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/rmse-z_and_percent_measure.png')
        if show_plots:
            plt.show()
        plt.close()

        # binned calibration curve with rmse-z errorbars (microns)
        fig, ax = plt.subplots()
        ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmse.rmse_z / 2, fmt='o', ms=1, elinewidth=0.5, capsize=1,
                    label=r'$\overline{z} \pm \sigma_{z}/2$')
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        ax.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/rmse-z_microns_errorbars-are-rmse.png')
        if show_plots:
            plt.show()
        plt.close()

        # binned calibration curve with rmse-z errorbars (microns) + fit line
        fig, ax = plt.subplots()
        ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmse.rmse_z / 2, fmt='o', ms=1.5, elinewidth=0.75, capsize=1,
                    label=r'$\overline{z} \pm \sigma_{z}/2$')
        ax.plot(z_fit, functions.line(z_fit, *popt), linestyle='--', linewidth=1, color='black', alpha=0.25,
                label=r'$dz/dz_{true} = $' + ' {}'.format(np.round(popt[0], 3)))
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        ax.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(
                path_figs + '/rmse-z_microns_errorbars-are-rmse_fit_line_a{}_b{}.png'.format(np.round(popt[0], 3),
                                                                                             np.round(popt[1], 3)))
        if show_plots:
            plt.show()
        plt.close()

        # all points calibration curve (filtered)
        fig, ax = plt.subplots()
        ax.scatter(dffp.z_true, dffp.z, s=3)
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/calibration_curve_filtered.png')
        if show_plots:
            plt.show()
        plt.close()

        # all points calibration curve (raw)
        fig, ax = plt.subplots()
        ax.scatter(dft.z_true, dft.z, s=3)
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/calibration_curve_raw.png')
        if show_plots:
            plt.show()
        plt.close()

        # all points calibration curve (filtered + errors)
        fig, ax = plt.subplots()
        ax.plot(dffp.z_true, dffp.z, linestyle='', marker="o", ms=3,
                label=r'Valid%'.format(np.round(len(dffp) / (len(dffp) + len(dffp_errors)) * 100, 1)))
        ax.plot(dffp_errors.z_true, dffp_errors.z, linestyle='', marker=r'$\times$', ms=3, mfc='white',
                fillstyle='none', color='tab:red', label=r'$\epsilon_{z} > h/10$')
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        # ax.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/calibration_curve_filtered_and_outliers_no-label.png')
        if show_plots:
            plt.show()
        plt.close()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 8. PLOT POSITION AND PRECISION
plot_position_and_precision = False

if plot_position_and_precision:

    # compute relative position to best focus
    compute_relative = False
    xy_cols = 'SPCT'

    if compute_relative:
        dft_rel = []
        for pid in dft.id.unique():
            dfpid = dft[dft['id'] == pid]

            if xy_cols == 'IDPT':
                dfpid_focus = dfpid[dfpid['bin'] == -4][['xm', 'ym']].mean()
                dfpid['x_rel'] = dfpid['xm'] - dfpid_focus['xm']
                dfpid['y_rel'] = dfpid['ym'] - dfpid_focus['ym']
            elif xy_cols == 'SPCT':
                dfpid_focus = dfpid[dfpid['bin'] == -4][['x', 'y']].mean()
                dfpid['x_rel'] = dfpid['x'] - dfpid_focus['x']
                dfpid['y_rel'] = dfpid['y'] - dfpid_focus['y']
            else:
                raise ValueError("Need to specifically define 'IDPT' or 'SPCT' to be careful.")

            dft_rel.append(dfpid)

        dft_rel = pd.concat(dft_rel)
        dft_rel.to_excel(path_results + '/dft_relative_x_y_positions.xlsx')

    # ---

    # read
    dfr = pd.read_excel(path_results + '/dft_relative_x_y_positions.xlsx')

    # setup
    xylim = 62.5
    xyticks = [-50, -25, 0, 25, 50]


    # plot x-y position relative to best focus

    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 1
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')


    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure()

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(dfr.x_rel, dfr.y_rel, ax, ax_histx, ax_histy)

    plt.show()

    # ---

    #
    fig, ax = plt.subplots()
    ax.plot(dfr.x_rel, dfr.y_rel, 'o', ms=1)
    ax.set_xlabel(r'$\epsilon_{x} \: (pix.)$')
    # ax.set_xlim([-2.5, 2.5])
    ax.set_ylabel(r'$\epsilon_{y} \: (pix.)$')
    # ax.set_ylim([-2.5, 2.5])
    plt.tight_layout()
    plt.show()

    # plot x-y position relative to best focus: by z_true
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(dfr.z_true, dfr.x_rel, 'o', ms=1)
    ax1.set_ylabel(r'$\epsilon_{x} \: (pix.)$')
    # ax1.set_ylim([-2.5, 2.5])

    ax2.plot(dfr.z_true, dfr.y_rel, 'o', ms=1)
    ax2.set_ylabel(r'$\epsilon_{y} \: (pix.)$')
    # ax2.set_ylim([-2.5, 2.5])

    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.set_xlim([-xylim, xylim])
    ax2.set_xticks(ticks=xyticks, labels=xyticks)
    plt.tight_layout()
    plt.show()

    # ---

    # plot x-y position relative to best focus: by r
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(dfr.r, dfr.x_rel, 'o', ms=1)
    ax1.set_ylabel(r'$\epsilon_{x} \: (pix.)$')
    # ax1.set_ylim([-2.5, 2.5])

    ax2.plot(dfr.r, dfr.y_rel, 'o', ms=1)
    ax2.set_ylabel(r'$\epsilon_{y} \: (pix.)$')
    # ax2.set_ylim([-2.5, 2.5])

    ax2.set_xlabel(r'$r \: (pix.)$')
    plt.tight_layout()
    plt.show()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#           EVALUATE DEPENDENCIES


# ----------------------------------------------------------------------------------------------------------------------
# 1. R-DEPENDENCE
analyze_r_dependence = False

if analyze_r_dependence:
    dfr = pd.read_excel(path_results + '/test_coords_spct-corrected-and-filtered.xlsx')
    dfr = dfr.drop(columns=['binn'])

    plot_collections.plot_error_analysis(dfr, path_results)

# ----------------------------------------------------------------------------------------------------------------------
# --- --- PERCENT DIAMETER OVERLAP
export_percent_diameter_overlap = False
plot_percent_diameter_overlap = False

if plot_percent_diameter_overlap or export_percent_diameter_overlap:

    # read test coords
    dft = pd.read_excel(path_results + '/test_coords_spct-corrected-and-filtered.xlsx')
    dft = dft.drop(columns=['binn'])

    # --- setup

    # --- read each percent diameter overlap dataframe (if available)
    calculate_percent_overlap = False
    param_diameter = 'contour_diameter'
    max_n_neighbors = 5

    # create directories for files
    path_overlap = path_results + '/percent-overlap'
    if not os.path.exists(path_overlap):
        os.makedirs(path_overlap)

    if calculate_percent_overlap:

        if param_diameter == 'contour_diameter':
            popt_contour = analyze.fit_contour_diameter(path_calib_spct_stats, fit_z_dist=40, show_plot=False)
        elif param_diameter == 'gauss_diameter':
            popt_contour = None
        else:
            raise ValueError('Parameter for percent diameter overlap is not understood.')

        # calculate overlap
        dfo = analyze.calculate_particle_to_particle_spacing(
            test_coords_path=dft,
            theoretical_diameter_params_path=path_calib_spct_pop,
            mag_eff=None,
            z_param='z_true',
            zf_at_zero=True,
            zf_param=None,
            max_n_neighbors=max_n_neighbors,
            true_coords_path=path_true_particle_locations,
            maximum_allowable_diameter=None,
            popt_contour=popt_contour,
            param_percent_diameter_overlap=param_diameter
        )

        # save to excel
        if export_percent_diameter_overlap:
            dfo.to_excel(path_overlap + '/{}_percent_overlap_{}.xlsx'.format(method, param_diameter),
                         index=False)

    else:
        dfo = pd.read_excel(path_overlap + '/{}_percent_overlap_{}.xlsx'.format(method, param_diameter))

    # --- --- EVALUATE RMSE Z

    # --- setup
    plot_all_errors = False
    plot_spacing_dependent_rmse = False
    export_min_dx = False
    export_precision = True
    plot_min_dx = False
    save_plots = True
    show_plots = False

    # binning
    bin_z = np.arange(-50, 60, 25)  # [-35, -25, -12.5, 12.5, 25, 35]
    bin_pdo = np.linspace(0.125, 2.125, 25)  # [-2.5, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8]
    bin_min_dx = np.arange(10, 80, 10)  # [10, 20, 30, 40, 50]
    num_bins = 6
    min_num_per_bin = 50
    round_z = 3
    round_pdo = 2
    round_min_dx = 1
    ylim_rmse = [-0.0625, 5.25]

    # filters
    error_limits = [4.5]  # [None, 5, None, 5]  # 5
    depth_of_focuss = [None]  # [None, None, 7.5, 7.5]  # 7.5
    max_overlap = 1.999

    for error_limit, depth_of_focus in zip(error_limits, depth_of_focuss):

        # apply filters
        if error_limit is not None:
            dfo = dfo[dfo['error'].abs() < error_limit]
        if depth_of_focus is not None:
            dfo = dfo[(dfo['z_true'] < -depth_of_focus) | (dfo['z_true'] > depth_of_focus)]
        if max_overlap is not None:
            dfo = dfo[dfo['percent_dx_diameter'] < max_overlap]
            # dfo['percent_dx_diameter'] = dfo['percent_dx_diameter'].where(dfo['percent_dx_diameter'] < max_overlap, max_overlap)

        # create directories for files
        path_pdo = path_overlap + \
                   '/percent-overlap_{}/max-pdo-{}_error-limit-{}_exclude-dof-{}_min-num-{}'.format(param_diameter,
                                                                                                    max_overlap,
                                                                                                    error_limit,
                                                                                                    depth_of_focus,
                                                                                                    min_num_per_bin,
                                                                                                    )
        if not os.path.exists(path_pdo):
            os.makedirs(path_pdo)

        # --- --- EVALUATE DATA

        # evaluate precision by min_dx and pdo
        plot_precision = True
        if plot_precision:

            # precision columns
            if 'IDPT' in base_dir:
                if 'rm' not in dfo.columns:
                    dfo['rm'] = np.sqrt(dfo.xm ** 2 + dfo.ym ** 2)
                precision_columns = ['xm', 'ym', 'rm']
            elif 'SPCT' in base_dir:
                precision_columns = ['x', 'y', 'r']
            else:
                raise ValueError("Dataset not understood for precision column selection.")
            # min dx
            column_to_bin = 'min_dx'
            plot_collections.bin_plot_spct_stats_2d_static_precision_mindx_id(
                dfo, column_to_bin, precision_columns, bin_min_dx, round_min_dx, min_num_per_bin,
                export_results=export_precision, path_results=path_pdo,
                save_figs=save_plots, path_figs=path_pdo, show_figs=show_plots,
            )

            # percent diameter overlap
            column_to_bin = 'percent_dx_diameter'
            bin_pdo = np.round(np.linspace(0.125, 1.875, 11), 3)
            pdo_threshold = None
            plot_collections.bin_plot_spct_stats_2d_static_precision_pdo_id(
                dfo, column_to_bin, precision_columns, bin_pdo, round_pdo, pdo_threshold, min_num_per_bin,
                export_results=export_precision, path_results=path_pdo,
                save_figs=save_plots, path_figs=path_pdo, show_figs=show_plots,
            )

        raise ValueError()

        # scatter plots of errors
        if plot_all_errors:

            # similarity
            fig, ax = plt.subplots()
            sc = ax.scatter(dfo.cm, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.set_xlabel(r'$c_{m}$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_error_by_cm.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # percent diameter
            fig, ax = plt.subplots()
            sc = ax.scatter(dfo.percent_dx_diameter, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.set_xlabel(r'$\varphi \: $(\%)')
            ax.set_xscale('log')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_error_by_percent_dx_diameter.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # min dx
            fig, ax = plt.subplots()
            sc = ax.scatter(dfo.min_dx, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.set_xlabel(r'$\delta x_{min} \: $ (pixels)')
            ax.set_xscale('log')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_error_by_min_dx.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # mean dx
            fig, ax = plt.subplots()
            sc = ax.scatter(dfo.mean_dx, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.set_xlabel(r'$\overline {\delta x}_{n=5} \: (pix.)$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_error_by_mean_dx.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # mean dxo
            fig, ax = plt.subplots()
            sc = ax.scatter(dfo.mean_dxo, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.set_xlabel(r'$\overline {\delta x_{o}} \: (pix.)$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_error_by_mean_dxo.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # num dxo
            fig, ax = plt.subplots()
            sc = ax.scatter(dfo.num_dxo, dfo.error.abs(), c=dfo.z_true, cmap='RdBu', s=0.5)
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.set_xlabel(r'$N_{o} \: (\#)$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_error_by_num_dxo.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # group by - num dxo
            dfogm = dfo.groupby('num_dxo').mean().reset_index()
            dfogstd = dfo.groupby('num_dxo').std().reset_index()

            fig, ax = plt.subplots()
            ax.errorbar(dfogm.num_dxo, dfogm.error.abs(), yerr=dfogstd.error, fmt='o', ms=2, elinewidth=1, capsize=2)
            ax.plot(dfogm.num_dxo, dfogm.error.abs())

            axr = ax.twinx()
            axr.errorbar(dfogm.num_dxo, dfogm.cm, yerr=dfogstd.cm, fmt='o', ms=1, elinewidth=0.75, capsize=1.5,
                         color='gray', alpha=0.25)
            axr.plot(dfogm.num_dxo, dfogm.cm, color='gray', alpha=0.125)

            ax.set_ylabel(r'$|\epsilon_{z}| \: (\mu m)$')
            ax.set_xlabel(r'$N_{o} \: (\#)$')
            axr.set_ylabel(r'$c_{m}$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_error-cm_grouped-by_num_dxo.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

        # spacing dependent uncertainty
        if plot_spacing_dependent_rmse:

            columns_to_bin = ['mean_dx', 'min_dx', 'mean_dxo', 'num_dxo', 'percent_dx_diameter']
            column_labels = [r'$\overline {\delta x}_{n=5} \: (pix.)$',
                             r'$\delta x_{min} \: (pix.)$',
                             r'$\overline {\delta x_{o}} \: (pix.)$',
                             r'$N_{o} \: (\#)$',
                             r'$\tilde{\varphi} \: (\%)$',
                             ]

            for col, col_lbl in zip(columns_to_bin, column_labels):

                if col == 'num_dxo':
                    num_bins = np.arange(max_n_neighbors + 1)
                elif col == 'percent_dx_diameter':
                    num_bins = bin_pdo

                dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin=col, bins=num_bins, min_cm=min_cm,
                                            z_range=None, round_to_decimal=3, df_ground_truth=None)

                fig, ax = plt.subplots()
                ax.plot(dfob.index, dfob.rmse_z, '-o')
                ax.set_xlabel(col_lbl)
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                ax.set_ylim(ylim_rmse)
                axr = ax.twinx()
                axr.plot(dfob.index, dfob.num_bind, '-d', ms=3, color='gray', alpha=0.125)
                axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
                axr.set_ylim([0, dfob.num_bind.max() * 1.1])
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_pdo + '/{}_binned_rmsez_by_{}.png'.format(method, col))
                if show_plots:
                    plt.show()
                plt.close()

        # --- --- EVALUATE RMSE Z

        # binning
        columns_to_bin = ['z_true', 'percent_dx_diameter']

        dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfo,
                                                       columns_to_bin=columns_to_bin,
                                                       bins=[bin_z, bin_pdo],
                                                       round_to_decimals=[round_z, round_pdo],
                                                       min_cm=min_cm,
                                                       equal_bins=[False, False])

        # --- --- PLOT RMSE Z

        if plot_percent_diameter_overlap:

            # Plot rmse z + number of particles binned as a function of percent diameter overlap for different z bins

            # linear y-scale
            fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.35, size_y_inches * 1.5))
            for name, df in dfbicts.items():
                df_nums = df[df['num_bind'] > min_num_per_bin]
                ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax.grid(alpha=0.25)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
            ax2.set_xlabel(r'$\tilde{\varphi} \: (\%)$')
            ax2.set_ylabel(r'$N_{p}$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_rmsez_num-binned_pdo.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # log y-scale
            fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.35, size_y_inches * 1.5))
            for name, df in dfbicts.items():
                df_nums = df[df['num_bind'] > min_num_per_bin]
                ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax.set_yscale('log')
            ax.grid(alpha=0.25)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
            ax2.set_xlabel(r'$\tilde{\varphi} \: (\%)$')
            ax2.set_ylabel(r'$N_{p}$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_rmsez_num-binned_pdo_log-y.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

        # --- --- EXPORT RMSE Z TO EXCEL
        if export_percent_diameter_overlap:
            dfstack = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
            dfstack.to_excel(path_pdo + '/{}_binned_rmsez_by_z_pdo.xlsx'.format(method), index=False)

        # --- --- PLOT OTHER METRICS

        # --- calculate the local rmse_z uncertainty

        # bin by percent diameter overlap
        if plot_percent_diameter_overlap:
            dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin='percent_dx_diameter', bins=bin_pdo, min_cm=min_cm,
                                        z_range=None, round_to_decimal=round_z, df_ground_truth=None)

            fig, ax = plt.subplots()
            ax.plot(dfob.index, dfob.rmse_z, '-o')
            ax.set_xlabel(r'$\varphi \: (\%)$')
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax.set_ylim(ylim_rmse)
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_binned_rmsez_by_pdo.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

            # bin by z
            dfobz = bin.bin_local_rmse_z(df=dfo, column_to_bin='z_true', bins=25, min_cm=min_cm, z_range=None,
                                         round_to_decimal=round_z, df_ground_truth=None)

            fig, ax = plt.subplots()
            ax.plot(dfobz.index, dfobz.rmse_z, '-o')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_pdo + '/{}_binned_rmsez_by_z.png'.format(method))
            if show_plots:
                plt.show()
            plt.close()

        # ------------------------------------------------------------------------------------------------------------------
        # --- --- MINIMUM PARTICLE TO PARTICLE SPACING (MIN_DX)

        if export_min_dx or plot_min_dx:

            # read percent overlap dataframe
            dfo = pd.read_excel(path_overlap + '/{}_percent_overlap_{}.xlsx'.format(method, param_diameter))

            # create directories for files
            path_min_dx = path_results + '/min_dx/error-limit-{}_exclude-dof-{}_min-num-{}'.format(error_limit,
                                                                                                   depth_of_focus,
                                                                                                   min_num_per_bin)
            if not os.path.exists(path_min_dx):
                os.makedirs(path_min_dx)

            # binning
            columns_to_bin = ['z_true', 'min_dx']

            # apply filters
            if error_limit is not None:
                dfo = dfo[dfo['error'].abs() < error_limit]
            if depth_of_focus is not None:
                dfo = dfo[(dfo['z_true'] < -depth_of_focus) | (dfo['z_true'] > depth_of_focus)]

            # compute rmse-z

            dfbicts = analyze.evaluate_2d_bin_local_rmse_z(df=dfo,
                                                           columns_to_bin=columns_to_bin,
                                                           bins=[bin_z, bin_min_dx],
                                                           round_to_decimals=[round_z, round_min_dx],
                                                           min_cm=min_cm,
                                                           equal_bins=[False, False])

            # --- --- PLOT RMSE Z

            if plot_min_dx:

                # Plot rmse z + number of particles binned as a function of percent diameter overlap for different z bins

                # linear y-scale
                fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.35, size_y_inches * 1.5))

                for name, df in dfbicts.items():
                    df_nums = df[df['num_bind'] > min_num_per_bin]

                    ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                    ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                ax.grid(alpha=0.25)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                ax2.set_xlabel(r'$\delta x_{min} \: (pix.)$')
                ax2.set_ylabel(r'$N_{p}$')
                plt.tight_layout()
                plt.savefig(path_min_dx + '/{}_rmsez_num-binned_min_dx.png'.format(method))
                if show_plots:
                    plt.show()
                plt.close()

                # log y-scale
                fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.35, size_y_inches * 1.5))

                for name, df in dfbicts.items():
                    df_nums = df[df['num_bind'] > min_num_per_bin]

                    ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
                    ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                ax.set_yscale('log')
                ax.grid(alpha=0.25)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
                ax2.set_xlabel(r'$\delta x_{min} \: (pix.)$')
                ax2.set_ylabel(r'$N_{p}$')
                plt.tight_layout()
                plt.savefig(path_min_dx + '/{}_rmsez_num-binned_min_dx_log-y.png'.format(method))
                if show_plots:
                    plt.show()
                plt.close()

            # --- --- EXPORT RMSE Z TO EXCEL
            if export_min_dx:
                dfstack = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
                dfstack.to_excel(path_min_dx + '/{}_binned_rmsez_by_z_min_dx.xlsx'.format(method), index=False)

            # --- --- PLOT OTHER METRICS

            # --- calculate the local rmse_z uncertainty

            if plot_min_dx:

                # bin by min dx
                dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin='min_dx', bins=bin_min_dx, min_cm=min_cm,
                                            z_range=None, round_to_decimal=round_z, df_ground_truth=None)

                fig, ax = plt.subplots()
                ax.plot(dfob.index, dfob.rmse_z, '-o')
                ax.set_xlabel(r'$\delta x_{min} \: (pix.)$')
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(path_min_dx + '/{}_binned_rmsez_by_min_dx.png'.format(method))
                if show_plots:
                    plt.show()
                plt.close()

                # bin by z
                dfobz = bin.bin_local_rmse_z(df=dfo, column_to_bin='z_true', bins=40, min_cm=min_cm, z_range=None,
                                             round_to_decimal=round_z, df_ground_truth=None)

                fig, ax = plt.subplots()
                ax.plot(dfobz.index, dfobz.rmse_z, '-o')
                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                plt.tight_layout()
                plt.savefig(path_min_dx + '/{}_binned_rmsez_by_z.png'.format(method))
                if show_plots:
                    plt.show()
                plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# 10. META ASSESSMENT
perform_meta_assessment = False
if perform_meta_assessment:
    dft = dft[dft['error'].abs() < filter_barnkob]

    # --- CALIBRATION CURVE
    fig, ax = plt.subplots()
    ax.scatter(dft.z_true, dft.z, s=1)
    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.grid()
    plt.tight_layout()
    plt.savefig(path_figs + '/calibration_curve_corrected.png')
    plt.show()

    # --- RMSE
    # bin
    dfrmse = bin.bin_local_rmse_z(dft, column_to_bin='z_true', bins=25, min_cm=min_cm, z_range=None,
                                  round_to_decimal=4, df_ground_truth=None, dropna=True, error_column='error')

    dfrmse_mean = bin.bin_local_rmse_z(dft, column_to_bin='z_true', bins=1, min_cm=min_cm, z_range=None,
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
    plt.savefig(path_figs + '/meta_rmse-z_corrected.png')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 11. INTRINSIC ABERRATIONS ASSESSMENT
perform_intrinsic_aberrations_assessment = False
perform_test_coord_mapping = False
plot_cm_gradient = False

if perform_intrinsic_aberrations_assessment:

    unique_id = 'correct-on-test-coords'

    # create directories for files
    path_ia = path_results + '/intrinsic-aberrations/{}'.format(unique_id)

    if not os.path.exists(path_ia):
        os.makedirs(path_ia)

    # correct particle similarity curves z-coords using test coords
    if perform_test_coord_mapping:
        # read
        dfs, dfsf, dfsm, dfas, dfcs = io.read_similarity(path_similarity)

        # perform mapping
        dfs = modify.map_values_on_frame_id(dfs, dft)

        # export
        dfs.to_excel(path_ia + '/particle_similarity_curves_{}_{}.xlsx'.format(method, unique_id), index=False)

        # plot to confirm coordinate change
        fig, [ax1, ax2] = plt.subplots(nrows=2)
        ax1.scatter(dfs.z_true_raw, dfs.z_est_raw, s=0.5)
        ax1.set_xlabel(r'$z_{raw, calib-coords}$')
        ax1.grid()
        ax2.scatter(dfs.z_true, dfs.z_est, s=0.5)
        ax2.set_xlabel(r'$z_{corr, test-coords}$')
        ax2.grid()
        plt.tight_layout()
        plt.savefig(path_ia + '/particle_similarity_curves_{}_{}.png'.format(method, unique_id))
        plt.show()
    elif os.path.exists(path_ia + '/ia_values_{}.xlsx'.format(unique_id)):
        dfs = pd.read_excel(path_ia + '/ia_values_{}.xlsx'.format(unique_id), index_col=0)
        dfs_fits = pd.read_excel(path_ia + '/ia_fits_{}.xlsx'.format(unique_id), index_col=0)
    else:
        dfs = pd.read_excel(path_ia + '/particle_similarity_curves_{}_{}.xlsx'.format(method, unique_id))

    # --- EVALUATE

    zlim = 50
    xylim = 57.5  # 62.5
    xyticks = [-50, -25, 0, 25, 50]

    if 'zs' in dfs.columns:

        # filter range so test matches calibration range
        dfs = dfs[(dfs['zs'] > -zlim) & (dfs['zs'] < zlim)]
        dfs_fits = dfs_fits[(dfs_fits['zfit'] > -zlim) & (dfs_fits['zfit'] < zlim)]

        # scatter plot

        fig, ax = plt.subplots()

        ax.scatter(dfs.zs, dfs.cms, s=1, alpha=0.0625, label='data')
        ax.plot(dfs_fits.zfit, dfs_fits.cmfit_cubic, color='black', linewidth=0.5, alpha=0.5, label='cubic')

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
        ax.set_ylim([-0.075, 0.075])
        ax.grid(alpha=0.125)
        ax.legend(['Data', 'Cubic', 'Quartic'])

        plt.tight_layout()
        plt.savefig(path_ia + '/intrinsic-aberrations_{}.png'.format(unique_id))
        plt.show()

        # groupby z_true to plot errorbars

        dfsg_mean = dfs.groupby('zs').mean().reset_index()
        dfsg_std = dfs.groupby('zs').std().reset_index()

        fig, ax = plt.subplots()

        # plot data
        ax.errorbar(dfsg_mean.zs, dfsg_mean.cms, yerr=dfsg_std.cms, fmt='o', ms=2, elinewidth=1, capsize=2)

        # plot fit
        # ax.plot(dfs_fits.zfit, dfs_fits.cmfit_cubic, linestyle='--', linewidth=1, color='black', alpha=0.25, label=r'$~f(z^3)$')

        # plot horizontal and vertical guide lines
        ax.axhline(y=0, linestyle='-', linewidth=0.25, color='gray', alpha=0.25)
        ax.axvline(x=0, linestyle='-', linewidth=0.25, color='gray', alpha=0.25)

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
        ax.set_ylim([-0.0375, 0.0375])
        ax.grid(alpha=0.125)
        # ax.legend()

        plt.tight_layout()
        plt.savefig(path_ia + '/intrinsic-aberrations_{}_errorbars_no-fit_guidelines.png'.format(unique_id))
        plt.show()

    else:
        # filter range so test matches calibration range
        dfs = dfs[(dfs['z_true'] > -50) & (dfs['z_true'] < 50)]

        dict_ia = analyze.evaluate_intrinsic_aberrations(dfs,
                                                         z_f=zf_c,
                                                         min_cm=min_cm,
                                                         param_z_true='z_true',
                                                         param_z_cm='z_cm',
                                                         shift_z_by_z_f=False,
                                                         )

        dict_ia = analyze.fit_intrinsic_aberrations(dict_ia)
        io.export_dict_intrinsic_aberrations(dict_ia, path_ia, unique_id=unique_id)

        # plot
        fig, ax = plotting.plot_intrinsic_aberrations(dict_ia, cubic=True, quartic=False)

        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
        ax.set_ylim([-0.075, 0.075])
        ax.grid(alpha=0.125)
        ax.legend(['Data', 'Cubic', 'Quartic'])

        plt.tight_layout()
        plt.savefig(path_ia + '/intrinsic-aberrations_{}.png'.format(unique_id))
        plt.show()

    # --- EVALUATE CM GRADIENT
    if plot_cm_gradient:
        dfdcmdz = analyze.evaluate_cm_gradient(dfs)

        # plot all
        fig, ax = plt.subplots()
        ax.scatter(dfdcmdz.z_true, dfdcmdz.dcmdz, c=dfdcmdz.id, s=0.5)
        ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
        ax.set_ylabel(r'$d c_{m} / dz$')
        plt.tight_layout()
        plt.savefig(path_ia + '/cm_gradient_all_{}.png'.format(unique_id))
        plt.show()

        # --- group by
        dfg = dfdcmdz.groupby('z_true').mean().reset_index()

        # fit polynomial
        x = dfg.z_true.to_numpy()
        y = dfg.dcmdz.to_numpy()
        p6 = np.poly1d(np.polyfit(x, y, 6))
        xf = np.linspace(x.min(), x.max(), 250)

        fig, ax = plt.subplots()
        ax.scatter(dfg.z_true, dfg.dcmdz)
        ax.plot(xf, p6(xf), color='black')
        ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
        ax.set_ylabel(r'$d c_{m} / dz$')
        plt.tight_layout()
        plt.savefig(path_ia + '/cm_gradient_groupby-fit-z_true_{}.png'.format(unique_id))
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 12. SIMILARITY GRADIENT ASSESSMENT

analyze_similarity_gradient = False

if analyze_similarity_gradient:
    # forward self-similarity
    fpt = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
          'results-04.11.22-IDPT-5um-micrometer-test/similarity/calib_stacks_forward_self-similarity_idpt.xlsx'
    dft = pd.read_excel(fpt)
    z_f = 49.8

    dfcm = analyze.evaluate_self_similarity_gradient(dft,
                                                     z_f,
                                                     dcm_threshold=0.015,
                                                     path_figs=path_figs,
                                                     z_range=50)

    # bin both dfcm and dfp_ids along bin
    dfcm['dcm_abs'] = dfcm['dcm'].abs()
    dfgid = dfcm.groupby('id').mean().reset_index()
    dfgid = dfgid.sort_values('dcm_abs', ascending=False)
    dfgid = dfgid.reset_index(drop=True)

    dfpg = dfp_id.groupby('id').mean().reset_index()
    dfpg = dfpg.sort_values('z_corr', ascending=True)
    dfpg = dfpg.reset_index(drop=True)

    # combine precision and self-similarity dataframes
    dfgid = dfgid.set_index('id')
    dfpg = dfpg.set_index('id')

    dfcmb = pd.concat([dfgid, dfpg], axis=1, join='inner')

    dfcmb = dfcmb.sort_values('dcm_abs', ascending=False)
    dfcmb = dfcmb.reset_index(drop=True)

    # plot
    fig, ax = plt.subplots()
    ax.plot(dfcmb.dcm_abs, dfcmb.z_corr, 'o')
    ax.set_xlabel(r'$dcm/dz$')
    ax.set_ylabel(r'$z-precision (\mu m)$')
    plt.tight_layout()
    plt.savefig(path_figs + '/similarity-gradient_precision_relationship.png')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 13. SPCT STATS

analyze_spct_stats = False

if analyze_spct_stats:
    # setup file paths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
               'results-04.24.22_SPCT_meta_gauss-xyc-is-absolute'

    # read
    plot_collections.plot_spct_stats(base_dir)

# ----------------------------------------------------------------------------------------------------------------------
# 14. COMPARE POST-PROCESSED RMSE-Z FOR IDPT AND SPCT

compare_idpt_and_spct_binned = False

if compare_idpt_and_spct_binned:
    """
    Other Colors:
    Light Blue: #7BC8F6
    Paler Blue: #0343DF
    Azure: #069AF3
    Dark Green: #054907  
    """
    sciblue = '#0C5DA5'
    scigreen = '#00B945'

    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/'
    path_figs = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/discussion/figs'
    path_idpt_rmse_bins = base_dir + 'analyses/FINAL-04.21-22_IDPT_1um-calib_5um-test/results/rmse-z_binned.xlsx'
    path_spct_rmse_bins = base_dir + 'analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/results/rmse-z_binned.xlsx'

    save_plots = False
    show_plots = True

    dfi = pd.read_excel(path_idpt_rmse_bins, index_col=0)
    dfs = pd.read_excel(path_spct_rmse_bins, index_col=0)

    # setup figures
    xylim = 62.5
    xyticks = [-50, -25, 0, 25, 50]
    ms = 3
    sciblue_mod = 0.85
    scigreen_mod = 1.25

    # plot precision and rmse: x, y, z
    plot_precision_and_rmse = True
    if plot_precision_and_rmse:
        path_idpt_precision = base_dir + 'analyses/FINAL-04.21-22_IDPT_1um-calib_5um-test/results/' \
                                         'match-location-precision_by_dz_fprecision=50.xlsx'
        path_spct_precision = base_dir + 'analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/results/' \
                                         'precision_by_dz_fprecision=50.xlsx'

        dfi = dfi.reset_index()
        dfs = dfs.reset_index()
        dfi = dfi.rename(columns={'index': 'dz'})
        dfs = dfs.rename(columns={'index': 'dz'})

        dfip = pd.read_excel(path_idpt_precision)
        dfsp = pd.read_excel(path_spct_precision)

        # add column for combined x-y precision
        dfip['m_xy_precision'] = np.hypot(dfip['m_xm_precision'], dfip['m_ym_precision'])
        dfsp['m_xy_precision'] = np.hypot(dfsp['m_x_precision'], dfsp['m_y_precision'])

        # setup
        px = 'dz'
        ms = 3

        # plot rmse-x-y-z by z-true
        plot_xyz_by_ztrue = False
        if plot_xyz_by_ztrue:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(size_x_inches * 1.125, size_y_inches))

            ax1.plot(dfip[px], dfip.m_xy_precision, '-o', ms=ms, color=sciblue, label='IDPT')
            ax1.plot(dfsp[px], dfsp.m_xy_precision, '-o', ms=ms, color=scigreen, label='SPCT')
            ax1.plot(dfip[px], dfip.m_xy_precision, '-o', ms=ms, color=sciblue)
            ax1.set_ylabel(r'$\nu_{xy}^{\delta} \: (pix.)$')
            ax1.set_ylim(top=0.4)
            ax1.set_yticks([0, 0.2, 0.4])
            ax1.set_xticks([-50, 0, 50], [])
            ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax1.tick_params(axis='y', which='minor', left=False, right=False)
            ax1.legend(loc='upper left', markerscale=0.75, borderpad=0.2, labelspacing=0.25, handletextpad=0.4,
                       borderaxespad=0.25)

            ax2.plot(dfs[px], dfs.dz_percent_rows, '-o', ms=ms, color=scigreen)
            ax2.plot(dfi[px], dfi.dz_percent_rows, '-o', ms=ms, color=sciblue)
            ax2.set_ylabel(r'$\phi \: (\%)$')
            ax2.set_ylim([-2.5, 102.5])
            ax2.set_yticks([0, 50, 100])
            ax2.set_xticks([-50, 0, 50], [])
            ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax2.tick_params(axis='y', which='minor', left=False, right=False)

            ax3.plot(dfsp[px], dfsp.m_z_precision, '-o', ms=ms, color=scigreen)
            ax3.plot(dfip[px], dfip.m_z_precision, '-o', ms=ms, color=sciblue)
            ax3.set_ylabel(r'$\nu_{z}^{\delta} \: (\mu m)$')
            ax3.set_ylim(top=0.8)
            ax3.set_yticks([0, 0.4, 0.8])
            ax3.set_xlabel(r'$z \: (\mu m)$')
            ax3.set_xticks([-50, 0, 50])
            ax3.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax3.tick_params(axis='y', which='minor', left=False, right=False)

            ax4.plot(dfs[px], dfs.rmse_z, '-o', ms=ms, color=scigreen)
            ax4.plot(dfi[px], dfi.rmse_z, '-o', ms=ms, color=sciblue)
            ax4.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
            ax4.set_ylim(top=8)
            ax4.set_yticks([0, 4, 8])
            ax4.set_xlabel(r'$z \: (\mu m)$')
            ax4.set_xticks([-50, 0, 50])
            ax4.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax4.tick_params(axis='y', which='minor', left=False, right=False)

            # ax.set_ylim(top=3.75)
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.175, wspace=0.5)  # adjust space between axes
            if save_plots:
                plt.savefig(path_figs + '/compare_rmse-x-y-z_and_percent-meas_by_z-true_wide.svg')
            if show_plots:
                plt.show()
            plt.close()

        # ---

        # histogram precision-x-y-z
        histogram_xyz = False
        if histogram_xyz:
            # data
            i_xy = dfip.m_xy_precision.to_numpy()
            i_z = dfip.m_z_precision.to_numpy()
            s_xy = dfsp.m_xy_precision.to_numpy()
            s_z = dfsp.m_z_precision.to_numpy()

            # bins
            bins_xy = np.linspace(0, 0.3, 7)
            bins_z = np.linspace(0, 0.7, 8)
            bins_z[-1] = 0.75
            print(bins_z)

            # histogram
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False)
            n, bins, patches = ax1.hist(s_xy, bins=bins_xy, density=False, facecolor=scigreen, alpha=0.75, label='SPCT')
            n, bins, patches = ax1.hist(i_xy, bins=bins_xy, density=False, facecolor=sciblue, alpha=0.75, label='IDPT')
            ax1.set_xlabel(r'$\sigma_{xy}$')
            ax1.set_ylabel('Counts')
            ax1.legend()

            n, bins, patches = ax2.hist(s_z, bins=bins_z, density=False, facecolor=scigreen, alpha=0.75)
            n, bins, patches = ax2.hist(i_z, bins=bins_z, density=False, facecolor=sciblue, alpha=0.75)

            ax2.set_xlabel(r'$\sigma_{z}$')
            ax2.set_ylabel('Counts')
            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs + '/compare_precision-x-y-z_histogram.svg')
            if show_plots:
                plt.show()
            plt.close()

        # ---

    # ---

    # plot rmse z and percent measure
    plot_old = False
    if plot_old:

        # Figure 1. dz by z

        # calculate difference between z-steps
        dfidz = dfi.diff()
        dfsdz = dfs.diff()

        # calculate measured z-step mean and stdev
        idz_mean = dfidz.z.mean()
        idz_std = dfidz.z.std()
        sdz_mean = dfsdz.z.mean()
        sdz_std = dfsdz.z.std()

        # label
        lbl_i = 'IDPT: {}'.format(np.round(idz_mean, 2)) + r'$\pm$' + '{}'.format(np.round(idz_std, 3))
        lbl_s = 'SPCT: {}'.format(np.round(sdz_mean, 2)) + r'$\pm$' + '{}'.format(np.round(sdz_std, 3))

        fig, ax = plt.subplots()
        ax.errorbar(dfidz.index, dfidz.z, yerr=dfidz.rmse_z, label=lbl_i, fmt='-o', markersize=ms, color=sciblue,
                    elinewidth=1, ecolor=lighten_color(sciblue, sciblue_mod), capsize=2)
        ax.errorbar(dfsdz.index, dfsdz.z, yerr=dfsdz.rmse_z, label=lbl_s, fmt='-o', markersize=ms, color=scigreen,
                    elinewidth=1, ecolor=lighten_color(scigreen, scigreen_mod), capsize=2)
        ax.axhline(5.0, linewidth=0.5, linestyle='--', color='black')  # , label=r'$\Delta z_{true}$'
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\Delta z + \sigma_{z} \: (\mu m)$')
        ax.set_ylim([-0.25, 10.25])
        ax.set_yticks(ticks=[0, 2.5, 5, 7.5, 10])
        ax.legend(title=r'$\overline{\Delta z} \pm \sigma_{z} \: (\mu m)$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/idpt-and-spct_dz_by_z.png')
        if show_plots:
            plt.show()
        plt.close()

        # Figure 2. cm, percent rows by z

        # number of focal plane bias errors
        dfi['fpb_errors'] = dfi.f_pid_num_rows - dfi.f_barnkob_num_rows
        dfs['fpb_errors'] = dfs.f_pid_num_rows - dfs.f_barnkob_num_rows

        fig, [axr, ax] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25),
                                      gridspec_kw={'height_ratios': [1, 2]})

        # cm
        axr.plot(dfi.index, dfi.cm, '-', ms=ms / 2, color=sciblue, label='IDPT')
        axr.plot(dfs.index, dfs.cm, '-', ms=ms / 2, color=scigreen, label='SPCT')
        axr.set_ylabel(r'$\overline{c_{m}}$')
        axr.set_ylim([0.45, 1.05])
        axr.set_yticks(ticks=[0.5, 0.75, 1.0])

        axrr = axr.twinx()
        axrr.plot(dfi.index, dfi.fpb_errors, '-s', ms=ms / 3, color=lighten_color(sciblue, sciblue_mod))
        axrr.plot(dfs.index, dfs.fpb_errors, '-s', ms=ms / 3, color=lighten_color(scigreen, scigreen_mod))
        axrr.set_ylabel(r'$\epsilon_{f.p.b.} \: (\#)$')
        axrr.set_ylim([-10, 150])
        axrr.set_yticks(ticks=[0, 50, 100])  # 50, 60, 70, 80, 90, 100

        # num particles ID'd
        ax.plot(dfi.index, dfi.rmse_z, '-o', ms=ms, color=sciblue, label='IDPT')
        ax.plot(dfs.index, dfs.rmse_z, '-o', ms=ms, color=scigreen, label='SPCT')
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        ax.set_ylim([-0.75, 7.5])
        ax.set_yticks(ticks=[0, 2, 4, 6])
        # ax.legend(loc='lower left')

        axxr = ax.twinx()
        axxr.plot(dfi.index, dfi.dz_percent_rows, '--', color=lighten_color(sciblue, sciblue_mod))
        axxr.plot(dfs.index, dfs.dz_percent_rows, '--', color=lighten_color(scigreen, scigreen_mod))
        axxr.set_ylabel(r'$\phi \: (\%)$')
        axxr.set_ylim([-2.5, 102.5])
        axxr.set_yticks(ticks=[0, 25, 50, 75, 100])

        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/idpt-and-spct_cm_percent_by_z.png')
        if show_plots:
            plt.show()
        plt.close()

        # rmse-z + percent rows
        fig, [axr, ax] = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]})

        ax.scatter(dfi.index, dfi.rmse_z, marker='o', s=ms * 1.5, label='IDPT')
        ax.plot(dfi.index, dfi.rmse_z, '-o', ms=ms)
        ax.scatter(dfs.index, dfs.rmse_z, marker='s', s=ms * 1.5, label='SPCT')
        ax.plot(dfs.index, dfs.rmse_z, '-s', ms=ms)
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        ax.set_ylim([-0.25, 6.75])
        ax.set_yticks(ticks=[0, 2, 4, 6])
        ax.legend(loc='upper left', handletextpad=0.2)  # , borderaxespad=0.25
        # ax.grid(alpha=0.125, which='minor', axis='x')

        axr.plot(dfi.index, dfi.dz_percent_rows, '-o', ms=ms / 1.25)
        axr.plot(dfs.index, dfs.dz_percent_rows, '-s', ms=ms / 1.25)
        axr.set_ylabel(r'$\phi \: (\%)$')
        axr.set_ylim([-10, 110])
        axr.set_yticks(ticks=[0, 50, 100])
        # axr.grid(alpha=0.125, which='minor', axis='x')

        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/idpt-and-spct_rmse-z_and_percent_measure.png')
        if show_plots:
            plt.show()
        plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# 15. COMPARE RAW TEST COORDS FOR IDPT AND SPCT

compare_idpt_and_spct_raw = False

if compare_idpt_and_spct_raw:
    # ------------------------------------------------------------------------------------------------------------------
    # 1. filepaths and read dataframes
    base_dir_idpt = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                    'FINAL-04.21-22_IDPT_1um-calib_5um-test'
    base_dir_spct = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                    'FINAL-04.25.22_SPCT_1um-calib_5um-test'
    path_figs = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/discussion/figs'

    # dfi_raw = io.read_test_coords(base_dir_idpt + '/coords/test-coords')
    dfi_f = pd.read_excel(base_dir_idpt + '/results/test_coords_spct-corrected-and-filtered.xlsx')
    # dfi_rmse = pd.read_excel(base_dir_idpt + '/results/rmse-z_binned.xlsx')

    # dfs_raw = io.read_test_coords(base_dir_spct + '/coords/test-coords')
    dfs_f = pd.read_excel(base_dir_spct + '/results/test_coords_spct-corrected-and-filtered.xlsx')
    # dfs_rmse = pd.read_excel(base_dir_spct + '/results/rmse-z_binned.xlsx')

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. process data

    # --- calculate relative percent measured (phi_ID'd)
    """
    dfi_rmse['rel_meas'] = dfi_rmse['f_barnkob_num_rows'] / dfi_rmse['f_pid_num_rows'] * 100
    dfs_rmse['rel_meas'] = dfs_rmse['f_barnkob_num_rows'] / dfs_rmse['f_pid_num_rows'] * 100

    # --- fit line to raw test coords and calculate r.m.s. error of fit

    # 0. shared
    z_fit = np.linspace(dfi_f.z_true.min(), dfi_f.z_true.max())

    # 1. fit line to IDPT raw coords
    popt_i, pcov_i = curve_fit(functions.line, dfi_f.z_true, dfi_f.z)
    rmse_fit_line = np.sqrt(np.sum((functions.line(dfi_f.z_true, *popt_i) - dfi_f.z) ** 2) / len(dfi_f.z))
    print("{} = IDPT; z r.m.s. error from fitted line".format(rmse_fit_line))

    # 2. fit line to SPCT raw coords
    popt_s, pcov_s = curve_fit(functions.line, dfs_f.z_true, dfs_f.z)
    rmse_fit_line = np.sqrt(np.sum((functions.line(dfs_f.z_true, *popt_s) - dfs_f.z) ** 2) / len(dfs_f.z))
    print("{} = SPCT; z r.m.s. error from fitted line".format(rmse_fit_line))
    """

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. plot figures

    # --- setup
    save_plots = True
    show_plots = True

    # figures
    sciblue_mod = 1.125
    scigreen_mod = 1.125
    ms = 4
    xylim = 62.5
    xyticks = [-50, -25, 0, 25, 50]
    plt.close('all')

    # ---

    # Figure - plot error by z_true (note: this is NOT the error magnitude. Here, we evaluate systematic errors)
    plot_average_error = True
    if plot_average_error:

        dfi_merrors = dfi_f.groupby('z_true').mean()
        dfi_std = dfi_f.groupby('z_true').std()
        dfs_merrors = dfs_f.groupby('z_true').mean()
        dfs_std = dfs_f.groupby('z_true').std()

        fig, ax = plt.subplots()
        ax.errorbar(dfi_merrors.index, dfi_merrors.error, yerr=dfi_std.error,
                    fmt='o', ms=1, elinewidth=0.5, capsize=2,
                    label='IDPT: {}'.format(np.round(dfi_merrors.error.mean(), 2)) +
                          r'$\pm$' +
                          '{}'.format(np.round(dfi_std.error.mean(), 2)))
        ax.errorbar(dfs_merrors.index, dfs_merrors.error, yerr=dfs_std.error,
                    fmt='o', ms=1, elinewidth=0.5, capsize=2, alpha=0.5,
                    label='SPCT: {}'.format(np.round(dfs_merrors.error.mean(), 2)) +
                          r'$\pm$' +
                          '{}'.format(np.round(dfs_std.error.mean(), 2)))
        ax.errorbar(dfi_merrors.index, dfi_merrors.error, yerr=dfi_std.error,
                    fmt='o', ms=1, elinewidth=0.5, capsize=2, color=sciblue)
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(xyticks)
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
        ax.set_ylim([-2.5, 2.5])
        ax.axhline(y=0, linestyle='--', linewidth=0.5, color='black', alpha=0.75)
        ax.legend(title=r'$\overline{\epsilon_{z}} \pm \sigma $')
        plt.tight_layout()
        if save_plots:
            plt.savefig(join(path_figs, 'compare_errorbars_z_by_z_true_zoom.svg'))
        if show_plots:
            plt.show()
        plt.close()

    raise ValueError()

    # ---

    # Uncertainty map by fitting 3D plane at each dz
    fit_plane_at_each_z = False
    compute_fit_plane = False

    if fit_plane_at_each_z:
        if compute_fit_plane:
            res = []
            dfi_fzs = []
            for bn in dfi_f.binn.unique():
                dfi_fz = dfi_f[dfi_f['binn'] == bn]

                # fit plane (x, y, z units: pixels)
                points_pixels = np.stack((dfi_fz.x, dfi_fz.y, dfi_fz.z)).T
                px_pixels, py_pixels, pz_pixels, popt_pixels = fit.fit_3d_plane(points_pixels)
                d, normal = popt_pixels[3], popt_pixels[4]

                # calculate fit error
                fit_results = functions.calculate_z_of_3d_plane(dfi_fz.x, dfi_fz.y, popt=popt_pixels)
                rmse, r_squared = fit.calculate_fit_error(fit_results, data_fit_to=dfi_fz.z.to_numpy())

                # add z-plane and error wrt z-plane
                dfi_fz['z_plane'] = fit_results
                dfi_fz['z_plane_error'] = dfi_fz['z'] - dfi_fz['z_plane']

                # store results
                res.append([bn, rmse, r_squared])
                dfi_fzs.append(dfi_fz)

            # export
            df_results = pd.DataFrame(np.array(res), columns=['bin', 'rmse', 'r_squared'])
            # df_results.to_excel(base_dir_spct + '/results/quality_of_fit_plane_by_z.xlsx', index=False)

            dfi_fzs = pd.concat(dfi_fzs)
            # dfi_fzs.to_excel(base_dir_spct + '/results/test_coords_spct-corrected-and-filtered-and-plane-error.xlsx', index=False)

        else:
            df = pd.read_excel(base_dir_spct + '/results/test_coords_spct-corrected-and-filtered-and-plane-error.xlsx')

            # take mean and std
            dfm = df.groupby('id').mean()
            dfstd = df.groupby('id').std()

            # get data arrays
            x = dfm.x.to_numpy()
            y = dfm.y.to_numpy()
            z = dfstd.z_plane_error.to_numpy()
            print("mean z precision: {}".format(np.mean(z)))

            plot_3d = False
            if plot_3d:
                xr = (x.min(), x.max())
                yr = (y.min(), y.max())
                X, Y = np.mgrid[yr[0]:yr[1]:32j, xr[0]:xr[1]:32j]
                Z = griddata(np.vstack((x, y)).T, z, (Y, X), method='cubic')  # 'nearest', 'linear', 'cubic'
                plt.pcolormesh(X, Y, Z, shading='gouraud', cmap=cm.coolwarm)  # 'gouraud', 'auto'
                plt.colorbar(label=r'$\epsilon_z^{\delta} \: (\mu m)$')
                plt.xlim([0, 512])
                plt.xlabel(r'$x \: (pix.)$')
                plt.ylim([0, 512])
                plt.ylabel(r'$y \: (pix.)$')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()
                plt.close()
            else:
                xy_lim = [0, 512]

                fig, ax = plt.subplots()
                sc = ax.scatter(x, y, c=z, cmap='coolwarm')

                ax.set_xlim(xy_lim)
                ax.set_ylim(xy_lim)
                plt.colorbar(sc, label=r'$\sigma_z^{i} \: (\mu m)$')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(join(path_figs, 'spct_average-z-precision_dropna.svg'))
                plt.show()
                plt.close()

        # ---

    # ---

    # Figure - 2d uncertainty surface across image
    plot_rmse_z_by_x_y = False
    if plot_rmse_z_by_x_y:

        for df, lbl in zip([dfi_f, dfs_f], ['idpt', 'spct']):

            df['error_abs'] = df['error'].abs()
            df['sq_err'] = df['error'] * df['error']
            dfig_surf = df.groupby('id').mean().reset_index()
            dfig_surf['rmse_z'] = np.sqrt(dfig_surf['sq_err'])
            dfig_counts = df.groupby('id').count().reset_index()

            # filter out pids in too few frames
            min_num_per_id = 12
            exclude_ids = dfig_counts[dfig_counts['frame'] < min_num_per_id].id.values.tolist()
            dfig_surf = dfig_surf[~dfig_surf.id.isin(exclude_ids)]

            # plot value
            pz = 'rmse_z'  # 'abs_error'
            pl = r'$\sigma_z^{i} \: (\mu m)$'  # r'$\epsilon_z^{i} \: (\mu m)$'

            # get data arrays
            x = dfig_surf.x.to_numpy()
            y = dfig_surf.y.to_numpy()
            z = dfig_surf[pz].to_numpy()
            r = dfig_surf.r.to_numpy()

            # scatter plot: z-error
            plot_scatter_error = True
            if plot_scatter_error:
                fig, (ax, axr) = plt.subplots(2, 1, figsize=(size_x_inches * 0.75, size_y_inches * 1.15),
                                              gridspec_kw={'height_ratios': [4, 1]})

                sc = ax.scatter(x, y, c=z, s=10, vmin=0.5, vmax=5, cmap='coolwarm', alpha=1)
                ax.set_ylabel(r'$y \: (\mu m)$', labelpad=-8)
                ax.set_ylim([0, 512])
                ax.set_yticks([0, 512], [0, int(np.ceil(512 * microns_per_pixel))])
                ax.set_xlabel(r'$x \: (\mu m)$', labelpad=-4)  # (pix.)
                ax.set_xlim([0, 512])
                ax.set_xticks([0, 512], [0, int(np.ceil(512 * microns_per_pixel))])

                axr.scatter(r, z, c=z, s=3, vmin=0.5, vmax=5, cmap='coolwarm', alpha=1)
                # axr.set_xlim([])
                axr.set_xticks([100, 200, 300], [int(100 * microns_per_pixel),
                                                 int(200 * microns_per_pixel),
                                                 int(300 * microns_per_pixel)])
                axr.set_xlabel(r'$r \: (\mu m)$')
                axr.set_ylabel(pl)
                axr.set_ylim([0, 6])
                axr.set_yticks([0, 3, 6])
                axr.tick_params(axis='x', which='minor', bottom=False, top=False)
                axr.tick_params(axis='y', which='minor', left=False, right=False)

                # colorbar

                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.tight_layout()
                fig.subplots_adjust(right=0.8)
                cax = plt.axes([0.85, 0.425, 0.025, 0.5])
                plt.colorbar(sc, cax=cax, label=pl, extend='both', aspect=25)

                # plt.tight_layout()
                fig.subplots_adjust(hspace=0.35)  # adjust space between axes

                if save_plots:
                    plt.savefig(join(path_figs, '{}_exp-val_scatter-x-y-r_{}_microns_square.svg'.format(lbl, pz)))
                if show_plots:
                    plt.show()
                plt.close()

            # plot surface
            plot_surf = False
            if plot_surf:
                # get the range of points for the 2D surface space
                num = 125
                xr = (x.min(), x.max())
                yr = (y.min(), y.max())
                X = np.linspace(min(x), max(x), num)
                Y = np.linspace(min(y), max(y), num)
                X, Y = np.meshgrid(X, Y)
                interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
                Z = interp(X, Y)

                # plt.pcolormesh(X, Y, Z, shading='auto', cmap=cm.coolwarm)
                # plt.colorbar(label=r'$\epsilon_z^{\delta} \: (\mu m)$')

                plt.scatter(x, y, c=z, s=10, vmin=-3, vmax=3, cmap='coolwarm', alpha=1)
                plt.colorbar(label=r'$\epsilon_z^{\delta} \: (\mu m)$', extend='both')

                plt.xlim([0, 512])
                plt.xlabel(r'$x \: (pix.)$')
                plt.ylim([0, 512])
                plt.ylabel(r'$y \: (pix.)$')
                plt.gca().invert_yaxis()
                plt.tight_layout()

                if save_plots:
                    plt.savefig(join(path_figs, 'bin-dx-abs-theta_2d-surf-plot-rmse-z_units-pixels.png'))
                if show_plots:
                    plt.show()
                plt.close()

        # ---

    # ---

    # 2.1 Figure - (2 sub figures): 1. measured z by z_true for IDPT and SPCT; 2. rmse-z for IDPT and SPCT
    plot_version_vertical = False
    plot_version_horizontal = False

    if plot_version_vertical:
        # 2.1 Figure - (2 sub figures): 1. measured z by z_true for IDPT and SPCT; 2. rmse-z for IDPT and SPCT

        # calibration curve
        fig = plt.figure(figsize=(size_x_inches, size_y_inches * 2))
        gs = GridSpec(4, 1, figure=fig)  # create a 1x3 grid of axes
        # gsr = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])  # split the right axis into a top and bottom

        ax = fig.add_subplot(gs[0, 0])
        axrr_top = fig.add_subplot(gs[1, 0])
        axrr = fig.add_subplot(gs[2, 0])
        axr = fig.add_subplot(gs[3, 0])

        # ax1:

        # fig, [ax, axr] = plt.subplots(ncols=2, figsize=(size_x_inches * 2, size_y_inches))
        # ax.plot(dfi_raw.z_true, dfi_raw.z, 'o', ms=ms, color=lighten_color(sciblue, sciblue_mod))
        # ax.plot(dfs_raw.z_true, dfs_raw.z, 'o', ms=ms, color=lighten_color(scigreen, scigreen_mod))
        ax.plot(dfi_f.z_true, dfi_f.z, 'o', ms=ms / 2, color=sciblue, label='IDPT')
        ax.plot(dfs_f.z_true, dfs_f.z, 'o', ms=ms / 2, color=scigreen, label='SPCT')
        ax.plot(dfi_f.z_true, dfi_f.z, 'o', ms=ms / 2, color=sciblue)
        # ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=[])
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        # ax.legend(markerscale=2)  # loc='center right', handletextpad=0.25, borderaxespad=0.3

        # percent measure (top)
        pi3, = axrr_top.plot(dfi_rmse.z_true, dfi_rmse.rel_meas, 'd', ms=ms / 1.25)
        ps3, = axrr_top.plot(dfs_rmse.z_true, dfs_rmse.rel_meas, 'd', ms=ms / 1.25)
        axrr_top.set_xlim([-xylim, xylim])
        axrr_top.set_xticks(ticks=xyticks, labels=[])
        axrr_top.set_ylabel(r'$\phi_{ID} \: (\%)$')
        axrr_top.set_ylim(bottom=45, top=105)
        axrr_top.set_yticks([50, 100])

        # true percent measure
        pi2, = axrr.plot(dfi_rmse.z_true, dfi_rmse.dz_percent_rows, 'D', ms=ms / 1.25, linewidth=1)
        ps2, = axrr.plot(dfs_rmse.z_true, dfs_rmse.dz_percent_rows, 'D', ms=ms / 1.25, linewidth=1)
        axrr.set_xlim([-xylim, xylim])
        axrr.set_xticks(ticks=xyticks, labels=[])
        axrr.set_ylabel(r'$\phi \: (\%)$')
        axrr.set_ylim(bottom=5, top=105)
        axrr.set_yticks([50, 100])

        # rmse-z
        pi1, = axr.plot(dfi_rmse.z_true, dfi_rmse.rmse_z, 'o', ms=ms, label='IDPT')
        ps1, = axr.plot(dfs_rmse.z_true, dfs_rmse.rmse_z, 'o', ms=ms, label='SPCT')
        axr.set_xlabel(r'$z_{true} \: (\mu m)$')
        axr.set_xlim([-xylim, xylim])
        axr.set_xticks(ticks=xyticks, labels=xyticks)
        axr.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        axr.set_ylim(bottom=0)

        # calib curve legend with all handles
        l1 = ax.legend([(pi1, pi2, pi3), (ps1, ps2, ps3)], ['IDPT', 'SPCT'], numpoints=1,
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       )

        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/idpt-spct_calib-curve-percent-rmse-z_vertical.svg')
        if show_plots:
            plt.show()
        plt.close()
        j = 1

    # ---

    if plot_version_horizontal:
        # calibration curve
        fig = plt.figure(figsize=(size_x_inches * 3, size_y_inches))
        gs = GridSpec(1, 3, figure=fig)  # create a 1x3 grid of axes
        gsr = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])  # split the right axis into a top and bottom

        ax = fig.add_subplot(gs[0, 0])
        axrr = fig.add_subplot(gsr[1, 0])
        axrr_top = fig.add_subplot(gsr[0, 0])
        axr = fig.add_subplot(gs[0, 2])

        # ax1:

        # fig, [ax, axr] = plt.subplots(ncols=2, figsize=(size_x_inches * 2, size_y_inches))
        # ax.plot(dfi_raw.z_true, dfi_raw.z, 'o', ms=ms, color=lighten_color(sciblue, sciblue_mod))
        # ax.plot(dfs_raw.z_true, dfs_raw.z, 'o', ms=ms, color=lighten_color(scigreen, scigreen_mod))
        ax.plot(dfi_f.z_true, dfi_f.z, 'o', ms=ms / 2, color=sciblue, label='IDPT')
        ax.plot(dfs_f.z_true, dfs_f.z, 'o', ms=ms / 2, color=scigreen, label='SPCT')
        ax.plot(dfi_f.z_true, dfi_f.z, 'o', ms=ms / 2, color=sciblue)
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-xylim, xylim])
        ax.set_xticks(ticks=xyticks, labels=xyticks)
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_ylim([-xylim, xylim])
        ax.set_yticks(ticks=xyticks, labels=xyticks)
        # ax.legend(markerscale=2)  # loc='center right', handletextpad=0.25, borderaxespad=0.3

        # rmse-z
        pi1, = axr.plot(dfi_rmse.z_true, dfi_rmse.rmse_z, 'o', ms=ms * 1.25, label='IDPT')
        ps1, = axr.plot(dfs_rmse.z_true, dfs_rmse.rmse_z, 'o', ms=ms * 1.25, label='SPCT')
        axr.set_xlabel(r'$z_{true} \: (\mu m)$')
        axr.set_xlim([-xylim, xylim])
        axr.set_xticks(ticks=xyticks, labels=xyticks)
        axr.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
        axr.set_ylim(bottom=0)
        # axr.set_yticks(ticks=xyticks, labels=xyticks)
        # axr.legend()

        # true percent measure
        pi2, = axrr.plot(dfi_rmse.z_true, dfi_rmse.dz_percent_rows, 'D', ms=ms / 1.25, linewidth=1)
        ps2, = axrr.plot(dfs_rmse.z_true, dfs_rmse.dz_percent_rows, 'D', ms=ms / 1.25, linewidth=1)
        axrr.set_xlabel(r'$z_{true} \: (\mu m)$')
        axrr.set_xticks(ticks=xyticks, labels=xyticks)
        axrr.set_ylabel(r'$\phi \: (\%)$')
        axrr.set_ylim(bottom=5, top=105)
        axrr.set_yticks([50, 100])

        # percent measure (top)
        pi3, = axrr_top.plot(dfi_rmse.z_true, dfi_rmse.rel_meas, 'd', color=pi2.get_color(), ms=ms / 1.25)
        ps3, = axrr_top.plot(dfs_rmse.z_true, dfs_rmse.rel_meas, 'd', color=ps2.get_color(), ms=ms / 1.25)
        axrr_top.set_xticks(ticks=xyticks, labels=[])
        # axrr_top.tick_params(labelbottom=False)
        axrr_top.set_ylabel(r'$\phi_{ID} \: (\%)$')
        axrr_top.set_ylim(bottom=45, top=105)
        axrr_top.set_yticks([50, 100])

        # percent meas of ID'd legend
        """l2 = axrr.legend([(pi3, ps3)], [r'$\phi_{ID}$'], numpoints=1,
                         handler_map={tuple: HandlerTuple(ndivide=None)},
                         loc='lower center', handletextpad=0.3, borderaxespad=0.35,
                         markerscale=1,
                         )"""

        # calib curve legend with all handles
        l1 = ax.legend([(pi1, pi2, pi3), (ps1, ps2, ps3)], ['IDPT', 'SPCT'], numpoints=1,
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       )

        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/idpt-spct_calib-curve-percent-rmse-z_horizontal.svg')
        if show_plots:
            plt.show()
        plt.close()
        j = 1

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 16. COMPUTE SIMILARITY SWEEP

compute_cm_sweep = False

if compute_cm_sweep:

    path_results_cm_sweep = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/' \
                            'analyses/shared-results/cm-sweep'
    method_ = 'name'

    # compute for each: cm_i, z
    compute_2d_cm_sweep = False

    if compute_2d_cm_sweep:
        cm_sweep = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]
        true_num_per_bin = 6072  # 3 frames * 92 particles * 22 z-steps
        dft_low_error = dft[dft['error'].abs() < filter_barnkob]
        res_cm = []

        # binning
        columns_to_bin = ['']
        for cm_i in cm_sweep:
            dfrmse_cm = bin.bin_local_rmse_z(dft_low_error, column_to_bin='bin', bins=dzs, min_cm=cm_i, z_range=None,
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
    compute_1d_cm_sweep = False

    if compute_1d_cm_sweep:
        cm_sweep = np.linspace(0.5, 0.995, 250)
        true_num_per_bin = 6072  # 3 frames * 92 particles * 22 z-steps
        dft_low_error = dft[dft['error'].abs() < filter_barnkob]
        res_cm = []
        for cm_i in cm_sweep:
            dfrmse_cm = bin.bin_local_rmse_z(dft_low_error, column_to_bin='frame', bins=1, min_cm=cm_i, z_range=None,
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
        dfcm.to_excel(path_results + '/dfrmse_cm-sweep.xlsx', index_label='i')

# ----------------------------------------------------------------------------------------------------------------------
# 16. COMPARE CM SWEEP OF IDPT AND SPCT

compare_cm_sweep = False

if compare_cm_sweep:

    plot_2d_cm_sweep = True
    plot_1d_cm_sweep = False

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

        raise ValueError()

        j = 1

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

        # add column for true number of particles
        dzs = [-54, -49, -44, -39, -34, -29, -24, -19, -14, -9, -4, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51]
        num_dz_steps = len(dzs)
        # true_num_per_bin = true_num_particles_per_frame * num_dz_steps * num_frames_per_step
        # dfi['true_num'] = true_num_per_bin
        # dfi['true_percent_meas'] = dfi['num_meas'] / dfi['true_num'] * 100
        # dfs['true_num'] = true_num_per_bin
        # dfs['true_percent_meas'] = dfs['num_meas'] / dfs['true_num'] * 100

        # processing
        i_idxhalf = np.argmin(np.abs(dfi.true_percent_meas - 50))
        i_half_rmse = dfi.iloc[i_idxhalf].rmse_z
        s_idxhalf = np.argmin(np.abs(dfs.true_percent_meas - 50))
        s_half_rmse = dfs.iloc[s_idxhalf].rmse_z

        # setup figures
        sciblue_mod = 0.85
        scigreen_mod = 1.25
        save_plots = False
        show_plots = True

        # figure 1. rmse-z, phi-ID (cm_input)
        xlim_lefts = [0.475, 0.675]
        lbls = [['IDPT: ' + r'$\sigma_{z}(\phi=50\%)=$' + '{}'.format(np.round(i_half_rmse, 4)),
                 'SPCT: ' + r'$\sigma_{z}(\phi=50\%)=$' + '{}'.format(np.round(s_half_rmse, 4))],
                ['IDPT', 'SPCT']]

        print(lbls)

        fig, ax = plt.subplots()
        ax.plot(dfi.cmi, dfi.true_percent_meas)
        plt.show()

        for xleft, lbl in zip(xlim_lefts, lbls):
            fig, [axr, ax] = plt.subplots(nrows=2, sharex=True)
            ax.plot(dfi.cmi, dfi.rmse_z, color=sciblue)  # , label=lbl[0]
            ax.plot(dfs.cmi, dfs.rmse_z, color=scigreen)  # , label=lbl[1]
            ax.set_xlabel(r'$c_{m,min}$')
            ax.set_xlim(left=xleft)
            ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            # ax.set_ylim(bottom=-0.0625)

            pi1, = axr.plot(dfi.cmi, dfi.true_percent_meas, color=sciblue, linestyle='-')
            ps1, = axr.plot(dfs.cmi, dfs.true_percent_meas, color=scigreen, linestyle='-')
            axr.set_ylabel(r'$\overline{\phi} \: (\%)$')
            axr.set_ylim(top=105)

            axrr = axr.twinx()
            pi2, = axrr.plot(dfi.cmi, dfi.percent_meas, color=sciblue, linestyle='dotted')
            ps2, = axrr.plot(dfs.cmi, dfs.percent_meas, color=scigreen, linestyle='dotted')
            axrr.set_ylabel(r'$\cdots \: \overline{\phi_{ID}} \: (\%)$')

            l = ax.legend([(pi1, pi2), (ps1, ps2)], ['IDPT', 'SPCT'], numpoints=1,
                          handler_map={tuple: HandlerTuple(ndivide=None)},
                          loc='lower left', handletextpad=0.25, borderaxespad=0.3)

            plt.tight_layout()
            if save_plots:
                plt.savefig(path_figs + '/cm-sweep_rmse-z-phi-ID_by_cmi_i{}_dots.png'.format(xleft))
            if show_plots:
                plt.show()
            plt.close()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 17. COMPARE MIN DX AND OVERLAP OF IDPT AND SPCT
compare_mindx_and_overlap = False

if compare_mindx_and_overlap:

    # ------------------------------------------------------------------------------------------------------------------
    # 1. filepaths and read dataframes
    base_dir_idpt = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                    'FINAL-04.21-22_IDPT_1um-calib_5um-test'
    base_dir_spct = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                    'FINAL-04.25.22_SPCT_1um-calib_5um-test'
    path_figs = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/discussion/figs'
    path_min_dx = path_figs + '/min-dx'
    path_pdo = path_figs + '/percent-overlap'

    # read percent overlap dataframe
    param_diameter = 'contour_diameter'
    dfoi = pd.read_excel(base_dir_idpt + '/results/percent-overlap/spct_percent_overlap_{}.xlsx'.format(param_diameter))
    dfos = pd.read_excel(base_dir_spct + '/results/percent-overlap/spct_percent_overlap_{}.xlsx'.format(param_diameter))

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. compile true coords dataframe

    # read FIJI particle coords
    path_fiji = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/shared-results/fiji-particle-locations'
    path_true_coords_overlap = path_fiji + '/fiji_in-focus_percent_overlap_{}.xlsx'.format(param_diameter)

    if os.path.exists(path_true_coords_overlap):
        dfto = pd.read_excel(path_true_coords_overlap)
    else:
        path_results_true_coords = path_fiji + '/fiji_in-focus_corrected-and-filtered.xlsx'

        if os.path.exists(path_results_true_coords):
            df_true = pd.read_excel(path_results_true_coords)
        else:
            df_true = pd.read_excel(path_true_particle_locations)
            frames_and_ztrues = dfoi.sort_values('frame')[['frame', 'z_true']].values.tolist()
            frames_and_ztrues = np.unique(np.array(frames_and_ztrues), axis=0)
            temp = []
            for ft, zt in frames_and_ztrues:
                df_true['z_true'] = zt
                df_true['z'] = zt
                df_true['frame'] = ft
                temp.append(df_true.copy())
            df_true = pd.concat(temp)
            df_true.to_excel(path_results_true_coords, index=True)

        # --- read each percent diameter overlap dataframe (if available)
        max_n_neighbors = 5

        if param_diameter == 'contour_diameter':
            popt_contour = analyze.fit_contour_diameter(path_calib_spct_stats, fit_z_dist=40, show_plot=False)
        elif param_diameter == 'gauss_diameter':
            popt_contour = None
        else:
            raise ValueError('Parameter for percent diameter overlap is not understood.')

        # calculate overlap
        dfo = analyze.calculate_particle_to_particle_spacing(
            test_coords_path=df_true,
            theoretical_diameter_params_path=path_calib_spct_pop,
            mag_eff=None,
            z_param='z_true',
            zf_at_zero=True,
            zf_param=None,
            max_n_neighbors=max_n_neighbors,
            true_coords_path=path_true_particle_locations,
            maximum_allowable_diameter=None,
            popt_contour=popt_contour,
            param_percent_diameter_overlap=param_diameter
        )

        dfo.to_excel(path_true_coords_overlap, index=False)

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 3. plot

    # setup
    save_plots = True
    show_plots = True
    plt.close('all')

    # Figure: plot (1) contour diameter, (2) SPCT mean dx, (3) IDPT mean dx
    plot_nyquist_sampling = False
    if plot_nyquist_sampling:
        z_dia = dfoi.groupby('z_true').mean().reset_index()
        z_dia = z_dia[['z_true', 'contour_diameter', 'mean_dx', 'min_dx']]

        z_i_num = dfoi.groupby('z_true').count().reset_index()
        z_i_num['mdx'] = np.sqrt(512 ** 2 / z_i_num['z'])
        z_i_num['true_percent_meas'] = z_i_num['z'] / (92 * 3)
        z_i_num = z_i_num[['z_true', 'z', 'mdx', 'true_percent_meas']]

        z_s_num = dfos.groupby('z_true').count().reset_index()
        z_s_num['mdx'] = np.sqrt(512 ** 2 / z_s_num['z'])
        z_s_num['true_percent_meas'] = z_s_num['z'] / (92 * 3)
        z_s_num = z_s_num[['z_true', 'z', 'mdx', 'true_percent_meas']]

        # 1. plot: diameter + mean dx
        fig, ax = plt.subplots()
        ax.plot(z_dia.z_true, z_dia.contour_diameter, '--', color=scired, label=r'$d_{e}$')
        ax.plot(z_i_num.z_true, z_i_num.mdx, '-o', color=sciblue, label=r'$\delta x_{IDPT}$')
        ax.plot(z_s_num.z_true, z_s_num.mdx, '-o', color=scigreen, label=r'$\delta x_{SPCT}$')

        plt.show()

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
                        ['true_percent_meas'],
                        ]
        plot_labels = [[r'$c_{m}$'],
                       [r'$\bullet \: d_{e} \: (pix.)$', r'$\diamond \: \overline{\delta x} \: (pix.)$'],
                       [r'$\bullet \: d_{e} \: (pix.)$', r'$\diamond \: \delta x_{min} \: (pix.)$'],
                       [r'$N_{p} \: (\#)$'],
                       [r'$\phi$'],
                       ]

        # binning
        bin_z = np.arange(-50, 60, 25)  # [-35, -25, -12.5, 12.5, 25, 35]
        bin_pdo = np.round(np.linspace(0.125, 2.125, 25),
                           3)  # 19 or 38 # [-2.5, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8]
        bin_min_dx = np.arange(5, 86, 10)  # [10, 20, 30, 40, 50]
        num_bins = 6
        round_z = 3
        round_pdo = 2
        min_num_per_bin = 50

        # filters
        error_limits = [10]  # 5
        dof = [-18, 0]
        depth_of_focuss = [None]  # 7.5
        max_overlap = bin_pdo[-1]  # 1.125

        for error_limit, depth_of_focus in zip(error_limits, depth_of_focuss):

            # apply filters
            if error_limit is not None:
                dfoi = dfoi[dfoi['error'].abs() < error_limit]
                dfos = dfos[dfos['error'].abs() < error_limit]
            if depth_of_focus is not None:
                dfoi = dfoi[(dfoi['z_true'] < -depth_of_focus) | (dfoi['z_true'] > depth_of_focus)]
                dfos = dfos[(dfos['z_true'] < -depth_of_focus) | (dfos['z_true'] > depth_of_focus)]
            if max_overlap is not None:
                dfoi = dfoi[dfoi['percent_dx_diameter'] < max_overlap]
                dfos = dfos[dfos['percent_dx_diameter'] < max_overlap]

            # compute rmse-z; bin by min dx
            dfoib = bin.bin_local_rmse_z(df=dfoi, column_to_bin='min_dx', bins=bin_min_dx, min_cm=min_cm,
                                         z_range=z_range, round_to_decimal=round_z, df_ground_truth=dfto)
            dfosb = bin.bin_local_rmse_z(df=dfos, column_to_bin='min_dx', bins=bin_min_dx, min_cm=min_cm,
                                         z_range=z_range, round_to_decimal=round_z, df_ground_truth=dfto)

            # if true_percent_measure is > 100, make 100
            dfoib['true_percent_meas'] = dfoib['true_percent_meas'].where(dfoib['true_percent_meas'] < 100, 100)
            dfosb['true_percent_meas'] = dfosb['true_percent_meas'].where(dfosb['true_percent_meas'] < 100, 100)

            # remove bins with < min_num_per_bin measurements
            dfoib = dfoib[dfoib['num_meas'] > min_num_per_bin]
            dfosb = dfosb[dfosb['num_meas'] > min_num_per_bin]

            # ---

            # plot: cm, percent measure, rmse-z
            ms = 3
            fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True,
                                                figsize=(size_x_inches * 0.6, size_y_inches * 1.265))
            ax1.plot(dfoib.index, dfoib.cm, '-o', ms=ms, label='IDPT')
            ax1.plot(dfosb.index, dfosb.cm, '-o', ms=ms, label='SPCT')
            ax1.set_ylabel(r'$c_{m}$')
            ax1.set_ylim([0.89, 1.01])
            ax1.set_yticks([0.9, 1.0])
            # ax1.set_xticks([-50, 0, 50], [])
            ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax1.tick_params(axis='y', which='minor', left=False, right=False)
            ax1.legend(markerscale=0.75, handlelength=1, borderpad=0.2, labelspacing=0.25,
                       handletextpad=0.4, borderaxespad=0.25)

            ax2.plot(dfoib.index, dfoib.true_percent_meas, '-o', ms=ms, zorder=2)
            ax2.plot(dfosb.index, dfosb.true_percent_meas, '-o', ms=ms, zorder=2)
            ax2.set_ylabel(r'$\phi \: (\%)$')
            ax2.set_ylim([-5, 105])
            ax2.set_yticks([0, 50, 100])
            # ax1.set_xticks([-50, 0, 50], [])
            ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax2.tick_params(axis='y', which='minor', left=False, right=False)

            ax3.plot(dfoib.index, dfoib.rmse_z, '-o', ms=ms, zorder=2)
            ax3.plot(dfosb.index, dfosb.rmse_z, '-o', ms=ms, zorder=2)
            ax3.set_xlabel(r'$\delta x_{min} \: (pix.)$')
            ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
            ax3.set_ylim([0.875, 4.125])
            ax3.set_yticks([1, 2, 3, 4])
            ax3.set_xlim([-5, 68])
            ax3.set_xticks([0, 20, 40, 60])
            ax3.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax3.tick_params(axis='y', which='minor', left=False, right=False)

            plt.tight_layout()
            if save_plots:
                plt.savefig(path_min_dx + '/idpt-spct_rmsez_by_min_dx_erlim{}dof{}_small.svg'.format(error_limit,
                                                                                                     depth_of_focus,
                                                                                                     )
                            )
            if show_plots:
                plt.show()
            plt.close('all')

            continue

            # plot - multi
            for pc, pl in zip(plot_columns, plot_labels):
                fig, [axr, ax] = plt.subplots(nrows=2, sharex=True)
                ax.plot(dfoib.index, dfoib.rmse_z, '-o', ms=ms, label='IDPT')
                ax.plot(dfosb.index, dfosb.rmse_z, '-o', ms=ms, label='SPCT')
                ax.set_xlabel(r'$\delta x_{min} \: (pix.)$')
                ax.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
                ax.legend()

                pi, = axr.plot(dfoib.index, dfoib[pc[0]], '-o', ms=ms, zorder=2)
                ps, = axr.plot(dfosb.index, dfosb[pc[0]], '-o', ms=ms, zorder=2)
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
                    axrr.plot(dfosb.index, dfosb[pc[1]], '-D',
                              color=lighten_color(ps.get_color(), scigreen_mod),
                              mfc='white',
                              mec=lighten_color(ps.get_color(), scigreen_mod),
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
                    plt.savefig(path_min_dx + '/idpt-spct_rmsez_by_min_dx_erlim{}dof{}_{}.png'.format(error_limit,
                                                                                                      depth_of_focus,
                                                                                                      pc,
                                                                                                      )
                                )
                if show_plots:
                    plt.show()
                plt.close('all')

            # bin by percent diameter overlap
            plot_overlap = False
            if plot_overlap:
                dfoib = bin.bin_local_rmse_z(df=dfoi, column_to_bin='percent_dx_diameter', bins=bin_pdo, min_cm=min_cm,
                                             z_range=z_range, round_to_decimal=round_z, df_ground_truth=dfto)
                dfosb = bin.bin_local_rmse_z(df=dfos, column_to_bin='percent_dx_diameter', bins=bin_pdo, min_cm=min_cm,
                                             z_range=z_range, round_to_decimal=round_z, df_ground_truth=dfto)

                # if true_percent_measure is > 100, make 100
                dfoib['true_percent_meas'] = dfoib['true_percent_meas'].where(dfoib['true_percent_meas'] < 100, 100)
                dfosb['true_percent_meas'] = dfosb['true_percent_meas'].where(dfosb['true_percent_meas'] < 100, 100)

                # remove bins with < min_num_per_bin measurements
                dfoib = dfoib[dfoib['num_meas'] > min_num_per_bin]
                dfosb = dfosb[dfosb['num_meas'] > min_num_per_bin]

                # plot
                for pc, pl in zip(plot_columns, plot_labels):
                    fig, [axr, ax] = plt.subplots(nrows=2, sharex=True)
                    ax.plot(dfoib.index, dfoib.rmse_z, '-o', ms=ms, label='IDPT')
                    ax.plot(dfosb.index, dfosb.rmse_z, '-o', ms=ms, label='SPCT')
                    ax.set_xlabel(r'$\overline{\varphi} \: (\%)$')
                    ax.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
                    # ax.set_ylim(ylim_rmse)
                    ax.legend()

                    pi, = axr.plot(dfoib.index, dfoib[pc[0]], 'o', ms=ms2, zorder=2)
                    ps, = axr.plot(dfosb.index, dfosb[pc[0]], 'o', ms=ms2, zorder=2)
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
                        axrr.plot(dfosb.index, dfosb[pc[1]], 'D',
                                  color=lighten_color(ps.get_color(), scigreen_mod),
                                  mfc='white',
                                  mec=lighten_color(ps.get_color(), scigreen_mod),
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
                        plt.savefig(path_pdo + '/idpt-spct_rmsez_by_pdo_erlim{}dof{}_{}.png'.format(error_limit,
                                                                                                    depth_of_focus,
                                                                                                    pc,
                                                                                                    )
                                    )
                    if show_plots:
                        plt.show()
                    plt.close('all')

            # ---

        # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 17. PLOT SIMPLE OVERLAP STATS
simple_mindx_and_overlap = False

if simple_mindx_and_overlap:

    # ------------------------------------------------------------------------------------------------------------------
    # 1. filepaths and read dataframes
    base_dir_idpt = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                    'FINAL-04.21-22_IDPT_1um-calib_5um-test'
    base_dir_spct = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
                    'FINAL-04.25.22_SPCT_1um-calib_5um-test'
    path_figs = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/discussion/figs'
    path_pdo = path_figs + '/overlap-stats'

    # read percent overlap dataframe
    param_diameter = 'contour_diameter'
    dfoi = pd.read_excel(base_dir_idpt + '/results/percent-overlap/spct_percent_overlap_{}.xlsx'.format(param_diameter))
    dfos = pd.read_excel(base_dir_spct + '/results/percent-overlap/spct_percent_overlap_{}.xlsx'.format(param_diameter))

    # ---

    # ------------------------------------------------------------------------------------------------------------------
    # 2. process data

    # export mean
    for df, lbl in zip([dfoi, dfos], ['idpt', 'spct']):
        df['bin'] = 1
        df = df.groupby('bin').mean()
        df.to_excel(path_pdo + '/{}_mean_percent_overlap_{}.xlsx'.format(lbl, param_diameter))

    # groupby z_true
    dfoig = dfoi.groupby('z_true').mean()
    dfosg = dfos.groupby('z_true').mean()

    # ------------------------------------------------------------------------------------------------------------------
    # 2. plot figs

    # setup
    plt.close('all')
    ms = 2

    # mean dx, min dx by z_true
    fig, ax = plt.subplots()

    pi1, = ax.plot(dfoig.index, dfoig.mean_dx, '-o', ms=ms, label='IDPT')
    ps1, = ax.plot(dfosg.index, dfosg.mean_dx, '-o', ms=ms, label='SPCT')

    pi2, = ax.plot(dfoig.index, dfoig.min_dx, '--s', ms=ms, color=sciblue, )
    ps2, = ax.plot(dfosg.index, dfosg.min_dx, '--s', ms=ms, color=scigreen, )

    pi3, = ax.plot(dfoig.index, dfoig.contour_diameter, 'D', mfc='white', mec=sciblue, ms=ms)
    ps3, = ax.plot(dfosg.index, dfosg.contour_diameter, 'D', mfc='white', mec=scigreen, ms=ms)

    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.set_xlim([-65, 65])
    ax.set_xticks(np.arange(-50, 55, 25))
    ax.set_ylabel(r'$(pix.)$')
    ax.set_ylim(bottom=0)

    # calib curve legend with all handles
    l1 = ax.legend([(pi1, ps1), (pi2, ps2), (pi3, ps3)],
                   [r'$\overline{\delta x} \: (pix.)$',
                    r'$\delta x_{min} \: (pix.)$',
                    r'$d_{e} \: (pix.)$',
                    ],
                   numpoints=1,
                   handler_map={tuple: HandlerTuple(ndivide=None)},
                   )

    plt.tight_layout()
    if save_plots:
        plt.savefig(path_pdo + '/idpt-spct_min-dx_mean-dx_by_z.png')
    if show_plots:
        plt.show()
    plt.close()

print("Analysis completed without errors.")