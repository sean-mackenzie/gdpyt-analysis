# test bin, analyze, and plot functions
import os
from os.path import join
from os import listdir

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import filter
import analyze
from correction import correct
from utils import fit, functions, bin, io, plotting, modify, details

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

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/' \
           'results-04.16.22_10X-idpt-dynamic-pos-first'

path_test_coords = join(base_dir, 'coords/test-coords')
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_calib_spct_pop = join(base_dir, 'coords/calib-coords/spct/calib_spct_pop_defocus_stats.xlsx')
path_similarity = join(base_dir, 'similarity')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# ----------------------------------------------------------------------------------------------------------------------
make_folders = False
if make_folders:
    results_correction = path_results + '/correction'
    figs_correction = path_figs + '/correction'

    results_spct_stats = path_results + '/spct-stats'
    figs_spct_stats = path_figs + '/spct-stats'

    results_raw = path_results + '/raw'
    figs_raw = path_figs + '/raw'

    results_precision = path_results + '/precision'
    figs_precision = path_figs + '/precision'

    results_pdo = path_results + '/pdo'
    figs_pdo = path_figs + '/pdo'

    for path in [results_correction, figs_correction, results_spct_stats, figs_spct_stats,
                 results_raw, figs_raw, results_precision, figs_precision, results_pdo, figs_pdo]:
        if not os.path.exists(path):
            os.makedirs(path)

# ----------------------------------------------------------------------------------------------------------------------
# 1. READ FILES

method = 'idpt'

# calibration coords z in-focus
mag_eff, zf, c1, c2 = io.read_pop_gauss_diameter_properties(path_calib_spct_pop)
microns_per_pixel = 1.6

# ----------------------------------------------------------------------------------------------------------------------
# 2. PERFORM CORRECTION

perform_correction = False
if perform_correction:
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='idpt')

    fig, ax = plotting.scatter_z_by_xy(df=dfcpid, z_params=['zf_from_peak_int', 'zf_from_nsv'])
    fig.savefig(figs_correction + '/scatter_xy_int-and-nsv.png')

    dict_fit_plane, fig_xy, fig_xyz, fig_plane = correct.inspect_calibration_surface(df=dfcpid, param_zf='zf_from_nsv', microns_per_pixel=microns_per_pixel)

    fig_xy.savefig(figs_correction + '/scatter_xy.png')

    fig_xyz.savefig(figs_correction + '/scatter_xyz.png')

    fig_plane.savefig(figs_correction + '/fit-3d-plane.png')



j = 1






# ----------------------------------------------------------------------------------------------------------------------
# 3. SETUP MODIFIERS

# read test coords
dft = io.read_test_coords(path_test_coords)

theory_diam_path = path_calib_spct_pop
mag_eff = mag_eff
dft['z_corr'] = dft['z']

# experimental
microns_per_pixel = 1.6
area_pixels = 512**2
area_microns = (512 * microns_per_pixel)**2

# processing
min_cm = 0.5
min_percent_layers = 0.75

# compute the radial distance
if 'r' not in dft.columns:
    dft['r'] = np.sqrt((256 - dft.x) ** 2 + (256 - dft.y) ** 2)


# ----------------------------------------------------------------------------------------------------------------------
# 4. STORE RAW STATS (AFTER FILTERING RANGE)

# filter range so test matches calibration range
dft = dft[(dft['z_true'] > -50) & (dft['z_true'] < 60)]

i_num_rows = len(dft)
i_num_pids = len(dft.id.unique())


# ----------------------------------------------------------------------------------------------------------------------
# 5. SPLIT DATAFRAME INTO BINS

true_zs = dft.z_true.unique()
dft['bin'] = np.round(dft['z_true'], 0).astype(int)
dzs = dft.bin.unique()

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

# Evaluate percent measure as a function of precision (in order to find an adequate precision limit)
analyze_percent_measure_by_precision = False
export_results = False
read_and_plot_percent_measure_by_precision = False

if analyze_percent_measure_by_precision:

    data_percent_measure_by_precision = []

    for df, name, dz in zip(dfs, names, dzs):

        # precision @ z

        # for t_id, df in zip(['init', 'final'], [dfi, dff]):

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
            num_pids = len(dffp.id.unique())
            emitter_density_microns = num_pids / area_microns
            mean_mean_dx = dffpo.mean_dx.mean()
            mean_min_dx = dffpo.min_dx.mean()
            mean_num_dxo = dffpo.num_dxo.mean()
            data_percent_measure_by_precision.append([name, dz, filt_p, num_pids, emitter_density_microns,
                                                      mean_mean_dx, mean_min_dx, mean_num_dxo])

    # evaluate
    dfpmp = pd.DataFrame(np.array(data_percent_measure_by_precision), columns=['name', 'dz', 'fprecision',
                                                                               'num_pids', 'p_density', 'mean_dx',
                                                                               'min_dx', 'num_dxo'])

    # export results
    if export_results:
        dfpmp.to_excel(path_results + '/percent_measure_by_precision.xlsx', index=False)


if read_and_plot_percent_measure_by_precision:
    dfpmp = pd.read_excel(path_results + '/percent_measure_by_precision.xlsx')

    dfpmpg = dfpmp.groupby('fprecision').mean().reset_index()

    dfpmpg['percent_meas'] = dfpmpg['num_pids'] / i_num_pids

    for pc in ['percent_meas', 'p_density', 'mean_dx', 'min_dx', 'num_dxo']:
        fig, ax = plt.subplots()
        ax.plot(dfpmpg.fprecision, dfpmpg[pc], '-o')
        ax.set_xlabel(r'z-precision $(\mu m)$')
        ax.set_ylabel(pc)
        plt.tight_layout()
        plt.savefig(path_figs + '/meas-{}_by_precision_filter.png'.format(pc))
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 7. EVALUATE DISPLACEMENT PRECISION (ID)

analyze_precision_per_id = True
export_results = False
save_plots, show_plots = False, False
save_plots_collection, show_plots_collection = False, False

filter_precision = 0.5

if analyze_precision_per_id:

    precision_per_id_results = []
    remove_ids = []
    dfp_ids = []

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
        # store raw results
        dz_i_num_rows = len(dfp_id)
        dz_i_num_pids = dfp_id.counts.sum()

        # filter out particles with precision > filter_precision
        remove_ids.append([name, dz, dfp_id[dfp_id['z_corr'] > filter_precision].id.unique()])
        dfp_id = dfp_id[dfp_id['z_corr'] < filter_precision]

        # store results
        dz_f_num_rows = len(dfp_id)
        dz_f_num_pids = dfp_id.counts.sum()
        dz_percent_rows = dz_f_num_rows / dz_i_num_rows * 100
        dz_percent_pids = dz_f_num_pids / dz_i_num_pids * 100
        dz_mean_dz = dfp_id.z_corr_m.mean()
        dz_precision = dfp_id.z_corr.mean()

        precision_per_id_results.append([name, dz, dz_mean_dz, dz_precision, dz_percent_rows, dz_percent_pids,
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

    dfp_dz = pd.DataFrame(np.array(precision_per_id_results), columns=['name', 'dz', 'm_dz', 'm_precision',
                                                                       'percent_rows', 'percent_pids',
                                                                       'i_rows', 'i_pids', 'f_rows', 'f_pids'])
    if export_results:
        dfp_dz.to_excel(path_results + '/precision_by_dz_fprecision={}.xlsx'.format(filter_precision), index=False)

        dfp_ids = pd.concat(dfp_ids, ignore_index=True)
        dfp_ids.to_excel(path_results + '/precision_by_dz-id_fprecision={}.xlsx'.format(filter_precision), index=False)

    if save_plots_collection or show_plots_collection:
        x_param = 'dz'
        y_params = ['m_precision', 'percent_pids']
        lbls = [r'$\sigma_{p} \: (\mu m)$', r'$\phi_{ID} \: (\%)$']

        fig, ax = plt.subplots()
        axr = ax.twinx()

        ax.plot(dfp_dz[x_param], dfp_dz[y_params[0]], '-o')
        ax.set_xlabel(x_param)
        ax.set_ylabel(lbls[0])

        axr.plot(dfp_dz[x_param], dfp_dz[y_params[1]], '-s', markersize=2, alpha=0.25)
        axr.set_ylabel(lbls[1], color='gray')
        axr.set_ylim([-5, 105])

        plt.tight_layout()
        if save_plots_collection:
            plt.savefig(path_figs + '/z-precision-by-dz.png')
        if show_plots_collection:
            plt.show()
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# 8. EVALUATE RMSE Z
analyze_rmse = True
export_results = False
save_plots = False
show_plots = False

measurement_depth = 110
filter_barnkob = 110  # measurement_depth / 10

if analyze_rmse:

    # filter out bad particle ids and barnkob filter
    dffs = []
    dfilt = []
    for df, name, dz in zip(dfs, names, dzs):

        # initial rows and pids
        i_pid_num_rows = 83 * 3  # len(df)
        i_pid_num_pids = 83  # len(df.id.unique())

        # filter
        exclude_ids = remove_ids[name][2]
        df = df[~df.id.isin(exclude_ids)]

        # initial rows and pids
        f_pid_num_rows = len(df)
        f_pid_num_pids = len(df.id.unique())

        # barnkob filter
        df = df[df['error'].abs() < filter_barnkob]
        f_barnkob_num_rows = len(df)
        f_barnkob_num_pids = len(df.id.unique())

        # calculate stats
        dz_percent_pids = f_barnkob_num_pids / i_pid_num_pids * 100
        dz_percent_rows = f_barnkob_num_rows / i_pid_num_rows * 100

        dffs.append(df)
        dfilt.append([name, dz, dz_percent_pids, i_pid_num_pids, f_pid_num_pids, f_barnkob_num_pids,
                      dz_percent_rows, i_pid_num_rows, f_pid_num_rows, f_barnkob_num_rows])

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
    dfrmse_bin = bin.bin_local_rmse_z(dffp, column_to_bin='binn', bins=dzs, min_cm=min_cm,
                                      z_range=None, round_to_decimal=0, df_ground_truth=None, dropna=True,
                                      error_column='error')

    dfrmse_mean = bin.bin_local_rmse_z(dffp, column_to_bin='binn', bins=1, min_cm=min_cm,
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
        # rmse-z: microns
        fig, ax = plt.subplots()
        ax.plot(dfrmse.index, dfrmse.rmse_z, '-o')
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
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
        ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmse.rmse_z, fmt='o', ms=1, elinewidth=0.5, capsize=2,
                    label=r'$\overline{z} \pm \sigma_{z}$')
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/rmse-z_microns.png')
        if show_plots:
            plt.show()
        plt.close()

        # binned calibration curve with rmse-z errorbars (microns) + fit line
        popt, pcov = curve_fit(functions.line, dfrmse.z_true, dfrmse.z)
        z_fit = np.linspace(dfrmse.z_true.min(), dfrmse.z_true.max())

        fig, ax = plt.subplots()
        ax.errorbar(dfrmse.z_true, dfrmse.z, yerr=dfrmse.rmse_z, fmt='o', ms=1, elinewidth=0.5, capsize=2,
                    label=r'$\overline{z} \pm \sigma_{z}$')
        ax.plot(z_fit, functions.line(z_fit, *popt), linestyle='--', color='black', alpha=0.25,
                label=r'$dz/dz_{true} = $' + ' {}'.format(np.round(popt[0], 3)))
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/rmse-z_microns_fit_line_a{}_b{}.png'.format(np.round(popt[0], 3),
                                                                                  np.round(popt[1], 3)))
        if show_plots:
            plt.show()
        plt.close()

        # all points calibration curve (filtered)
        fig, ax = plt.subplots()
        ax.scatter(dffp.z_true, dffp.z, s=1)
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/calibration_curve_filtered.png')
        if show_plots:
            plt.show()
        plt.close()

        # all points calibration curve (raw)
        fig, ax = plt.subplots()
        ax.scatter(dft.z_true, dft.z, s=1)
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_figs + '/calibration_curve_raw.png')
        if show_plots:
            plt.show()
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# 9. EVALUATE DEPENDENCIES

# --- --- PERCENT DIAMETER OVERLAP
export_percent_diameter_overlap = True
plot_percent_diameter_overlap = True
plot_all_errors = True
plot_spacing_dependent_rmse = True
export_min_dx = False
plot_min_dx = True
show_plots = True
save_plots = True

# diameter parameter
param_diameter = 'contour_diameter'

# filters
error_limit = 10
depth_of_focus = 7.5
max_overlap = -3

# binning
bin_z = [-35, -25, -12.5, 12.5, 25, 35]
bin_pdo = [-2.5, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8]
bin_min_dx = [10, 20, 30, 40, 50]
num_bins = 6
min_num_per_bin = 50
round_z = 3
round_pdo = 4
round_min_dx = 1
max_n_neighbors = 5


if plot_percent_diameter_overlap or export_percent_diameter_overlap:

    # --- read each percent diameter overlap dataframe (if available)
    calculate_percent_overlap = False

    # create directories for files
    if not os.path.exists(path_results + '/percent-overlap'):
        os.makedirs(path_results + '/percent-overlap')

    if calculate_percent_overlap:

        if param_diameter == 'contour_diameter':
            popt_contour = analyze.fit_contour_diameter(path_calib_spct_stats)
        elif param_diameter == 'gauss_diameter':
            popt_contour = None
        else:
            raise ValueError('Parameter for percent diameter overlap is not understood.')

        dfo = analyze.calculate_particle_to_particle_spacing(
                        test_coords_path=dft,
                        theoretical_diameter_params_path=path_calib_spct_pop,
                        mag_eff=mag_eff_c,
                        z_param='z_true',
                        zf_at_zero=True,
                        zf_param=None,
                        max_n_neighbors=max_n_neighbors,
                        true_coords_path=None,
                        maximum_allowable_diameter=None,
                        popt_contour=popt_contour,
                        param_percent_diameter_overlap=param_diameter
            )

        # save to excel
        if export_percent_diameter_overlap:
            dfo.to_excel(path_results + '/percent-overlap/{}_percent_overlap_{}.xlsx'.format(method, param_diameter),
                         index=False)

    else:
        dfo = pd.read_excel(path_results + '/percent-overlap/{}_percent_overlap_{}.xlsx'.format(method, param_diameter))

    # --- --- EVALUATE RMSE Z

    # apply filters
    dfo = dfo[dfo['error'].abs() < error_limit]
    dfo = dfo[(dfo['z_true'] < -depth_of_focus) | (dfo['z_true'] > depth_of_focus)]
    # dfo = dfo[(dfo['z_true'] > -depth_of_focus * 5) & (dfo['z_true'] < depth_of_focus * 5)]
    dfo = dfo[dfo['percent_dx_diameter'] > max_overlap]
    # dfo['percent_dx_diameter'] = dfo['percent_dx_diameter'].where(dfo['percent_dx_diameter'] > -10, -10)

    # create directories for files
    path_pdo = path_results + \
               '/percent-overlap_{}/max-pdo-{}_error-limit-{}_exclude-dof-{}_min-num-{}'.format(param_diameter,
                                                                                                max_overlap,
                                                                                                error_limit,
                                                                                                depth_of_focus,
                                                                                                min_num_per_bin,
                                                                                                )
    if not os.path.exists(path_pdo):
        os.makedirs(path_pdo)

    # --- --- EVALUATE DATA

    if plot_all_errors:

        # similarity
        fig, ax = plt.subplots()
        ax.scatter(dfo.cm, dfo.error.abs(), c=dfo.z_true, s=0.5)
        ax.set_ylabel(r'$error \: (\mu m)$')
        ax.set_xlabel(r'$c_{m}$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_error_by_cm.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

        # percent diameter
        fig, ax = plt.subplots()
        ax.scatter(dfo.percent_dx_diameter, dfo.error.abs(), c=dfo.z_true, s=0.5)
        ax.set_ylabel(r'$error \: (\mu m)$')
        ax.set_xlabel(r'$\gamma \: $(\%)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_error_by_percent_dx_diameter.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

        # min dx
        fig, ax = plt.subplots()
        ax.scatter(dfo.min_dx, dfo.error.abs(), c=dfo.z_true, s=0.5)
        ax.set_ylabel(r'$error \: (\mu m)$')
        ax.set_xlabel(r'$\delta x_{min} \: $ (pixels)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_error_by_min_dx.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

        # mean dx
        fig, ax = plt.subplots()
        ax.scatter(dfo.mean_dx, dfo.error.abs(), c=dfo.z_true, s=0.5)
        ax.set_ylabel(r'$error \: (\mu m)$')
        ax.set_xlabel(r'$\overline {\delta x} \: $ (pixels)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_error_by_mean_dx.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

        # mean dxo
        fig, ax = plt.subplots()
        ax.scatter(dfo.mean_dxo, dfo.error.abs(), c=dfo.z_true, s=0.5)
        ax.set_ylabel(r'$error \: (\mu m)$')
        ax.set_xlabel(r'$\overline {\delta x_{o}} \: $ (pixels)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_error_by_mean_dxo.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

        # num dxo
        fig, ax = plt.subplots()
        ax.scatter(dfo.num_dxo, dfo.error.abs(), c=dfo.z_true, s=0.5)
        ax.set_ylabel(r'$error \: (\mu m)$')
        ax.set_xlabel(r'$N_{o} \: $ (\#)')
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

        ax.set_ylabel(r'$error \: (\mu m)$')
        ax.set_xlabel(r'$N_{o} \: $ (\#)')
        axr.set_ylabel(r'$c_{m}$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_error-cm_grouped-by_num_dxo.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

    if plot_spacing_dependent_rmse:

        columns_to_bin = ['mean_dx', 'min_dx', 'mean_dxo', 'num_dxo', 'percent_dx_diameter']

        for col in columns_to_bin:

            if col == 'num_dxo':
                num_bins = np.arange(max_n_neighbors + 1)
            elif col == 'percent_dx_diameter':
                num_bins = bin_pdo

            dfob = bin.bin_local_rmse_z(df=dfo, column_to_bin=col, bins=num_bins, min_cm=min_cm,
                                        z_range=None, round_to_decimal=3, df_ground_truth=None)

            fig, ax = plt.subplots()
            ax.plot(dfob.index, dfob.rmse_z, '-o')
            ax.set_xlabel('{}'.format(col))
            ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
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
        fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches*1.35, size_y_inches*1.5))
        for name, df in dfbicts.items():

            df_nums = df[df['num_bind'] > min_num_per_bin]
            ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
            ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
        ax2.set_xlabel(r'$\gamma \: $(\%)')
        ax2.set_ylabel(r'$N_{p}$')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_rmsez_num-binned_pdo.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

        # log y-scale
        fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches*1.35, size_y_inches*1.5))
        for name, df in dfbicts.items():

            df_nums = df[df['num_bind'] > min_num_per_bin]
            ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
            ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        ax.set_yscale('log')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
        ax2.set_xlabel(r'$\gamma \: $(\%)')
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
        ax.set_xlabel(r'$\gamma \: $(\%)')
        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_binned_rmsez_by_pdo.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

        # bin by z
        dfobz = bin.bin_local_rmse_z(df=dfo, column_to_bin='z_true', bins=bin_z, min_cm=min_cm, z_range=None,
                                       round_to_decimal=round_z, df_ground_truth=None)

        fig, ax = plt.subplots()
        ax.plot(dfobz.index, dfobz.rmse_z, '-o')
        ax.set_xlabel(r'$z_{true} \:$ ($\mu m$)')
        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_pdo + '/{}_binned_rmsez_by_z.png'.format(method))
        if show_plots:
            plt.show()
        plt.close()

raise ValueError('ha')
# ----------------------------------------------------------------------------------------------------------------------
# --- --- MINIMUM PARTICLE TO PARTICLE SPACING (MIN_DX)

if export_min_dx or plot_min_dx:

    # read percent overlap dataframe
    dfo = pd.read_excel(path_results + '/percent-overlap/{}_percent_overlap.xlsx'.format(method))

    # binning
    columns_to_bin = ['z_true', 'min_dx']

    # apply filters
    dfo = dfo[dfo['error'].abs() < error_limit]
    dfo = dfo[(dfo['z_true'] < -depth_of_focus) | (dfo['z_true'] > depth_of_focus)]

    # create directories for files
    path_min_dx = path_results + '/min_dx/error-limit-{}_exclude-dof-{}_min-num-{}'.format(error_limit,
                                                                                           depth_of_focus,
                                                                                           min_num_per_bin)
    if not os.path.exists(path_min_dx):
        os.makedirs(path_min_dx)

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
        fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches*1.35, size_y_inches*1.5))

        for name, df in dfbicts.items():

            df_nums = df[df['num_bind'] > min_num_per_bin]

            ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
            ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
        ax2.set_xlabel(r'$\delta x_{min} \: $ (pixels)')
        ax2.set_ylabel(r'$N_{p}$')
        plt.tight_layout()
        plt.savefig(path_min_dx + '/{}_rmsez_num-binned_min_dx.png'.format(method))
        plt.show()

        # log y-scale
        fig, [ax, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches*1.35, size_y_inches*1.5))

        for name, df in dfbicts.items():

            df_nums = df[df['num_bind'] > min_num_per_bin]

            ax.plot(df_nums.bin, df_nums.rmse_z, '-o', ms=2, label=name)
            ax2.plot(df_nums.bin, df_nums.num_bind, '-o', ms=2)

        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        ax.set_yscale('log')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$z_{bin}$')
        ax2.set_xlabel(r'$\delta x_{min} \: $ (pixels)')
        ax2.set_ylabel(r'$N_{p}$')
        plt.tight_layout()
        plt.savefig(path_min_dx + '/{}_rmsez_num-binned_min_dx_log-y.png'.format(method))
        plt.show()

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
        ax.set_xlabel(r'$\delta x_{min} \: $ (pixels)')
        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        plt.tight_layout()
        plt.savefig(path_min_dx + '/{}_binned_rmsez_by_min_dx.png'.format(method))
        plt.show()

        # bin by z
        dfobz = bin.bin_local_rmse_z(df=dfo, column_to_bin='z_true', bins=bin_z, min_cm=min_cm, z_range=None,
                                       round_to_decimal=round_z, df_ground_truth=None)

        fig, ax = plt.subplots()
        ax.plot(dfobz.index, dfobz.rmse_z, '-o')
        ax.set_xlabel(r'$z_{true} \:$ ($\mu m$)')
        ax.set_ylabel(r'z   r.m.s. error ($\mu m$)')
        plt.tight_layout()
        plt.savefig(path_min_dx + '/{}_binned_rmsez_by_z.png'.format(method))
        plt.show()


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
perform_intrinsic_aberrations_assessment = True
perform_test_coord_mapping = True

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

    else:
        dfs = pd.read_excel(path_ia + '/particle_similarity_curves_{}_{}.xlsx'.format(method, unique_id))

    # --- EVALUATE

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
    fig, ax = plotting.plot_intrinsic_aberrations(dict_ia, cubic=True, quartic=True)
    ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
    ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
    ax.grid(alpha=0.125)
    ax.legend(['Data', 'Cubic', 'Quartic'])
    plt.tight_layout()
    plt.savefig(path_ia + '/intrinsic-aberrations_{}.png'.format(unique_id))
    plt.show()

    # --- EVALUATE CM GRADIENT

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



print("Analysis completed without errors.")