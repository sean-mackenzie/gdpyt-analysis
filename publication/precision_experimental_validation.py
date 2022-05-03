from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt

import analyze
from utils import fit, functions, bin, plotting
from correction import correct

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

# setup figures
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# idpt - displacement
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation'

results_path = base_path + '/results/test_displacement_precision/'
figs_path = base_path + '/figs/test_displacement_precision/'

base_dir = base_path + '/test_coords/test/step_signed'
filetype = '.xlsx'

sort_ids = ['test_id', '_coords_']
sort_dzs = ['_coords_', 'micron_step_']

files = [f for f in listdir(base_dir) if f.endswith(filetype)]
files = sorted(files, key=lambda x: float(x.split(sort_ids[0])[-1].split(sort_ids[1])[0]))
names = [float(f.split(sort_ids[0])[-1].split(sort_ids[1])[0]) for f in files]
dzs = [float(f.split(sort_dzs[0])[-1].split(sort_dzs[1])[0]) for f in files]

# other files
fpcal = base_path + '/test_coords/spct-calib-coords/calib_in-focus_coords.xlsx'
theory_diam_path = base_path + '/test_coords/spct-calib-coords/calib_spct_pop_defocus_stats.xlsx'

# ----------------------------------------------------------------------------------------------------------------------
# 1.5 SET EXPERIMENTAL AND PROCESSING PARAMETERS

# experimental
mag_eff = 10.01
microns_per_pixel = 1.6
area = (512 * microns_per_pixel)**2

# data
split_by = 'frame'
split_value = 50.5


# ----------------------------------------------------------------------------------------------------------------------
# 2. Correct initial particle distribution
measure_calibration_distribution = True
plot_calib_surface = False
in_focus_param = 'z_f'

if measure_calibration_distribution:

    # read calibration in-focus coords
    dfc = pd.read_excel(fpcal)

    # ------------------------------------------------------------------------------------------------------------------
    # get mean in-focus z
    z_f_calibration_mean = dfc[in_focus_param].mean()

    # calculate tilt angle
    tilt_x_degrees, tilt_y_degrees = analyze.calculate_plane_tilt_angle(dfc,
                                                                        microns_per_pixel,
                                                                        z=in_focus_param,
                                                                        x='x',
                                                                        y='y')

    # ------------------------------------------------------------------------------------------------------------------
    # fit plane on calibration in-focus (x, y, z units: pixels)
    points_pixels = np.stack((dfc.x, dfc.y, dfc[in_focus_param])).T
    px_pixels, py_pixels, pz_microns, fit_plane_params_calib = fit.fit_3d(points_pixels, fit_function='plane')

    # ------------------------------------------------------------------------------------------------------------------
    # plot scatter points + fitted surface
    if plot_calib_surface:

        # plot scatter x and y
        fig, [axx, axy] = plt.subplots(nrows=2)
        axx.scatter(dfc.x, dfc[in_focus_param], c=dfc.id, s=1)
        axx.set_xlabel('x (pixels)')
        axx.set_ylabel(r'$z_{f} \: (\mu m)$')
        axy.scatter(dfc.y, dfc[in_focus_param], c=dfc.id, s=1)
        axy.set_xlabel('y (pixels)')
        axy.set_ylabel(r'$z_{f} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(figs_path + 'calibration_scatter_xy.png')
        plt.show()
        plt.close()

        # plot 3D scatter + fitted plane
        fig = plotting.plot_3d_scatter_and_plane(df=dfc,
                                                 z_param=in_focus_param,
                                                 p_xyz=[px_pixels, py_pixels, pz_microns],
                                                 fit_plane_params=fit_plane_params_calib
                                                 )
        plt.savefig(figs_path + 'calibration_3d_fitted_plane.png')
        plt.show()
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# 3. Read dataframes to disk
read_dataframes = True
perform_calibration_correction = True

if read_dataframes:

    dfs = []
    initial_data = []

    for fp, name, dz in zip(files, names, dzs):

        df = pd.read_excel(join(base_dir, fp))

        # store raw data
        dfi = df[df[split_by] < split_value]
        dff = df[df[split_by] > split_value]
        i_num_rows, f_num_rows = len(dfi), len(dff)
        i_num_pids, f_num_pids = len(dfi.id.unique()), len(dff.id.unique())
        initial_data.append([name, dz, i_num_rows, i_num_pids, f_num_rows, f_num_pids])

        # compute the radial distance
        if 'r' not in df.columns:
            df['r'] = np.sqrt((256 - df.x) ** 2 + (256 - df.y) ** 2)

        # correct points by fitted plane
        if perform_calibration_correction:
            df = correct.correct_z_by_fit_function(df=df,
                                                   fit_func=functions.calculate_z_of_3d_plane,
                                                   popt=fit_plane_params_calib,
                                                   x_param='x',
                                                   y_param='y',
                                                   z_param='z',
                                                   z_corr_name='z_corr',
                                                   mean_z_at_zero=False
                                                   )
            # correct z by in-focus mean
            df['z_corr'] = df['z_corr'] - z_f_calibration_mean

        dfs.append(df)

# ----------------------------------------------------------------------------------------------------------------------
# 4. Evaluate percent measure as a function of precision (in order to find an adequate precision limit)
analyze_percent_measure_by_precision = False
read_percent_measure_by_precision = False
export_results = False

if analyze_percent_measure_by_precision:

    data_percent_measure_by_precision = []

    for dff, name, dz in zip(dfs, names, dzs):

        # split dataframe into initial and final
        dfi = dff[dff[split_by] < split_value]
        dff = dff[dff[split_by] > split_value]

        # precision @ z

        for t_id, df in zip(['init', 'final'], [dfi, dff]):

            xparam = 'id'
            pos = ['z_corr']
            num_bins = len(df[xparam].unique())
            count_column = 'counts'

            dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                                column_to_bin=xparam,
                                                                precision_columns=pos,
                                                                bins=num_bins,
                                                                round_to_decimal=0)

            # filter out particles with precision > 2 microns
            filter_precisions = [5, 2.5, 1.25, 1, 0.75, 0.5, 0.25, 0.125, 0.0625]

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
                if t_id == 'init':
                    t_id_indicator = 0
                else:
                    t_id_indicator = 1

                num_pids = len(dffp.id.unique())
                emitter_density = num_pids / area
                mean_mean_dx = dffpo.mean_dx.mean()
                mean_min_dx = dffpo.min_dx.mean()
                mean_num_dxo = dffpo.num_dxo.mean()
                data_percent_measure_by_precision.append([name, dz, t_id_indicator, filt_p, num_pids, emitter_density,
                                                          mean_mean_dx, mean_min_dx, mean_num_dxo])

    # evaluate
    dfpmp = pd.DataFrame(np.array(data_percent_measure_by_precision), columns=['name', 'dz', 'i_f', 'fprecision',
                                                                               'num_pids', 'p_density', 'mean_dx',
                                                                               'min_dx', 'num_dxo'])

    # export results
    if export_results:
        dfpmp.to_excel(results_path + 'percent_measure_by_precision.xlsx', index=False)


elif read_percent_measure_by_precision:
    dfpmp = pd.read_excel(results_path + 'percent_measure_by_precision.xlsx')

    dfpmpg = dfpmp.groupby('fprecision').mean().reset_index()

    dfpmpg['percent_meas'] = dfpmpg['num_pids'] / 80

    for pc in ['percent_meas', 'p_density', 'mean_dx', 'min_dx', 'num_dxo']:
        fig, ax = plt.subplots()
        ax.plot(dfpmpg.fprecision, dfpmpg[pc], '-o')
        ax.set_xlabel(r'z-precision $(\mu m)$')
        ax.set_ylabel(pc)
        plt.tight_layout()
        plt.savefig(figs_path + 'meas-{}_by_precision_filter.png'.format(pc))
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 4. Evaluate displacement precision (ID)
analyze_precision_per_id = True
save_plots = False
show_plots = False

if analyze_precision_per_id:

    precision_per_id_results = []
    remove_ids = []

    for dff, name, dz in zip(dfs, names, dzs):

        # split dataframe into initial and final
        dfi = dff[dff[split_by] < split_value]
        dff = dff[dff[split_by] > split_value]

        # precision @ z

        for t_id, df in zip(['init', 'final'], [dfi, dff]):

            xparam = 'id'
            pos = ['z_corr']
            num_bins = len(df[xparam].unique())
            count_column = 'counts'

            dfp_id, dfpm = analyze.evaluate_1d_static_precision(df,
                                                                column_to_bin=xparam,
                                                                precision_columns=pos,
                                                                bins=num_bins,
                                                                round_to_decimal=0)

            # filter out particles with precision > 1.25 microns
            filter_precision = 1.25
            remove_ids.append([name, dz, t_id, dfp_id[dfp_id['z_corr'] > filter_precision].id.unique()])
            dfp_id = dfp_id[dfp_id['z_corr'] < filter_precision]

            # store results
            if t_id == 'init':
                i_num_rows = len(dfp_id)
                i_num_pids = dfp_id.counts.sum()
            elif t_id == 'final':
                f_num_rows = len(dfp_id)
                f_num_pids = dfp_id.counts.sum()
                precision_per_id_results.append([name, dz, i_num_rows, i_num_pids, f_num_rows, f_num_pids])

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
                        plt.savefig(figs_path + 'z-precision-by-id_{}_dz{}_{}.png'.format(name, dz, t_id))
                    if show_plots:
                        plt.show()
                    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# 4. Evaluate displacement precision
analyze_precision = True
export_results = False
save_plots = False
show_plots = False

if analyze_precision:

    read_or_analyze = 'analyze_displacement'

    if read_or_analyze == 'analyze_displacement':

        # --------------------------------------------------------------------------------------------------------------
        # 4. Analyze precision
        meas_cols = ['true_dz', 'dz', 'pz', 'pdz', 'percent_pids', 'percent_rows', 'rmse']

        data_spct = []
        data_idpt = []

        ii = np.arange(len(names))

        for i, df, name, dz in zip(ii, dfs, names, dzs):

            # remove pids with poor precision
            exclude_ids_i = remove_ids[i * 2][3]
            exclude_ids_f = remove_ids[i * 2 + 1][3]
            # split dataframe into initial and final and filter
            dfi = df[df[split_by] < split_value]
            dfi = dfi[~dfi.id.isin(exclude_ids_i)]
            dff = df[df[split_by] > split_value]
            dff = dff[~dff.id.isin(exclude_ids_f)]

            df = pd.concat([dfi, dff], ignore_index=True)


            mdp, mdm, mdmp, percent, per_fid, rmse, num_pids_measured, i_num_filt, f_num_filt = \
                analyze.evaluate_displacement_precision(df,
                                                        group_by='id',
                                                        split_by=split_by,
                                                        split_value=split_value,
                                                        precision_columns='z',
                                                        true_dz=dz,
                                                        std_filter=2)
            if name < 10:
                data_idpt.append([dz, mdm, mdp, mdmp, percent, per_fid, rmse, num_pids_measured, i_num_filt, f_num_filt])
            else:
                data_spct.append([dz, mdm, mdp, mdmp, percent, per_fid, rmse, num_pids_measured, i_num_filt, f_num_filt])

        if len(data_idpt) > 0:
            dfp_idpt = pd.DataFrame(np.array(data_idpt), columns=meas_cols + ['num_pids', 'rows_i', 'rows_f'])
        else:
            dfp_idpt = None

        if len(data_spct) > 0:
            dfp_spct = pd.DataFrame(np.array(data_spct), columns=meas_cols + ['num_pids', 'rows_i', 'rows_f'])
        else:
            dfp_spct = None

        # --------------------------------------------------------------------------------------------------------------
        # 4. Export results

        if export_results:

            # per test and mean results
            if dfp_idpt is not None:
                dfp_idpt.to_excel(results_path + 'idpt_precision.xlsx')
                dfp_idpt.mean().to_excel(results_path + 'idpt_mean_precision.xlsx')

            if dfp_spct is not None:
                dfp_spct.to_excel(results_path + 'spct_precision.xlsx')
                dfp_spct.mean().to_excel(results_path + 'spct_mean_precision.xlsx')

        # ------------------------------------------------------------------------------------------------------------------
        # 4.1 Plot

        if save_plots or show_plots:
            # setup
            ss = 2
            ssm = 3
            cb = '#0C5DA5'
            cg = '#00B945'

            # plot rms error
            fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches*1.35))

            ax1.scatter(dfp_spct.index + 1, dfp_spct.pz, s=ss*ssm, color=cg, label='SPCT')
            ax1.scatter(dfp_idpt.index + 1, dfp_idpt.pz, s=ss*ssm, color=cb, label='IDPT')
            ax1.set_ylabel(r'$\sigma_{i} + \sigma_{f} \: (\mu m)$')
            ax1.set_ylim([-0.25, 6.05])
            ax1.legend(loc='upper left', handletextpad=0.125, labelspacing=0.35, borderaxespad=0.25)

            ax2.errorbar(dfp_spct.index + 1, dfp_spct.dz, yerr=dfp_spct.pdz, fmt='o', ms=ss, color=cg, elinewidth=0.5, capsize=1.5)
            ax2.errorbar(dfp_idpt.index + 1, dfp_idpt.dz, yerr=dfp_idpt.pdz, fmt='o', ms=ss, color=cb, elinewidth=0.5, capsize=1.5)
            ax2.plot(dfp_idpt.index + 1, dfp_idpt.true_dz.unique(), color='black', linestyle='--', linewidth=0.5, alpha=0.5,
                     label=r'$\Delta z_{true}$')
            ax2.set_ylabel(r'$\overline{\Delta z} \pm \sigma_{\Delta z} \: (\mu m)$')
            ax2.set_ylim([-59.5, 59.5])
            ax2.legend(loc='upper left', handletextpad=0.25, markerscale=0.75)

            ax3.scatter(dfp_spct.index + 1, dfp_spct.rmse, s=ss*ssm, color=cg)
            ax3.scatter(dfp_idpt.index + 1, dfp_idpt.rmse, s=ss*ssm, color=cb)
            ax3.set_ylabel(r'$\Delta z \:$ r.m.s.e. $(\mu m)$')
            ax3.set_yscale('log')
            ax3.set_ylim(bottom=0.8, top=100)
            ax3.set_xticks(ticks=[y + 1 for y in range(len(dfp_idpt.true_dz.unique()))], labels=dfp_idpt.true_dz.unique())
            ax3.set_xlim([0.5, 4.5])
            ax3.set_xlabel(r'$\Delta z_{true} \: (\mu m)$')

            ax4 = ax3.twinx()
            ax4.scatter(dfp_spct.index + 1, dfp_spct.percent_pids*100, s=ss*ssm, marker='D', edgecolors=cg, color='white', linewidths=0.5)
            ax4.scatter(dfp_idpt.index + 1, dfp_idpt.percent_pids*100, s=ss*ssm, marker='D', edgecolors=cb, color='white', linewidths=0.5)
            ax4.set_ylabel(r'$\diamond \: \phi_{ID}$ (\%)')
            ax4.set_ylim([-4, 105])

            plt.tight_layout()
            if save_plots:
                plt.savefig(figs_path + 'idpt_spct_precision_logy.png')
            if show_plots:
                plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 5. Evaluate displacement precision dependency on spatial parameters
analyze_dependency = False

if analyze_dependency:

    per_id = True
    save_figs_dep = False
    show_figs_dep = False

    # ------------------------------------------- PRECISION (ID) ---------------------------------------------------

    for dff, name, dz in zip(dfs, names, dzs):

        # split dataframe into initial and final
        dfi = dff[dff[split_by] < split_value]
        dff = dff[dff[split_by] > split_value]

        # precision @ z

        for t_id, df in zip(['init', 'final'], [dfi, dff]):

            if per_id:
                pids = df.id.unique()
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

                        # OPTIONAL: filter precision outliers
                        dfp_id = dfp_id[dfp_id[pc] < 0.5]

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
                        if save_figs_dep:
                            plt.savefig(figs_path + 'id{}_dz{}_{}_{}-precision_dep-on_{}.png'.format(name,
                                                                                                     dz,
                                                                                                     t_id,
                                                                                                     pc,
                                                                                                     xparam))
                        if show_figs_dep:
                            plt.show()
                        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# 6. Evaluate displacement precision dependency on particle image overlap
analyze_overlap = True

if analyze_overlap:

    export_results = False
    save_figs_pdo = False
    show_figs_pdo = False

    # ---------------------------------------- PLOT GAUSSIAN DIAMETER --------------------------------------------------
    plot_theory = False
    if plot_theory:
        z_range = np.linspace(-50, 50, 250)
        fig, ax = plotting.plot_theoretical_gaussian_diameter(z_range=z_range,
                                                              theoretical_diameter_params_path=theory_diam_path,
                                                              zf_at_zero=True,
                                                              mag_eff=mag_eff)

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel('Diameter (pixels), M{}X'.format(mag_eff))
        plt.tight_layout()
        # plt.savefig(figs_path + 'confirm_gaussian_diameter_theory.png')
        plt.show()

    # ----------------------------------- PRECISION (PERCENT DIAMETER OVERLAP, ID) ---------------------------------

    pdxos = []
    ii = np.arange(len(names))

    for i, dfff, name, dz in zip(ii, dfs, names, dzs):

        # remove pids with poor precision
        exclude_ids_i = remove_ids[i * 2][3]
        exclude_ids_f = remove_ids[i * 2 + 1][3]

        # split dataframe into initial and final and filter
        dfi = dfff[dfff[split_by] < split_value]
        dfi = dfi[~dfi.id.isin(exclude_ids_i)]

        dff = dfff[dfff[split_by] > split_value]
        dff = dff[~dff.id.isin(exclude_ids_f)]

        # precision @ z

        for t_id, df in zip(['init', 'final'], [dfi, dff]):

            # --- calculate percent overlap
            dfo = analyze.calculate_particle_to_particle_spacing(test_coords_path=df,
                                                                 theoretical_diameter_params_path=theory_diam_path,
                                                                 mag_eff=mag_eff,
                                                                 z_param='z_corr',
                                                                 zf_at_zero=True,
                                                                 max_n_neighbors=5,
                                                                 true_coords_path=None,
                                                                 maximum_allowable_diameter=None)

            # save to excel
            if export_results:
                dfo.to_excel(results_path + 'id{}_dz{}_{}_percent_overlap.xlsx'.format(name, dz, t_id), index=False)

            # ---------------------------- PRECISION (PERCENT DIAMETER OVERLAP, ID) -------------------------
            per_percent_dx_diameter_id = True
            min_pdo_range = -1.75

            if per_percent_dx_diameter_id:

                xparam = 'percent_dx_diameter'
                plot_columns = ['z_corr']
                count_column = 'counts'
                df_pdxo = dfo.copy()

                # limit the not overlapped percent to -0.5
                df_pdxo[xparam] = df_pdxo[xparam].where(df_pdxo[xparam] > min_pdo_range, min_pdo_range)

                # append to list for collection analysis
                pdxos.append(df_pdxo)

                if export_results or save_figs_pdo or show_figs_pdo:
                    df_bin_pdxo_id, df_bin_pdxo = analyze.evaluate_2d_static_precision(df_pdxo,
                                                                                       column_to_bin=xparam,
                                                                                       precision_columns=plot_columns,
                                                                                       bins=8,
                                                                                       round_to_decimal=4)

                # save to excel
                if export_results:
                    df_bin_pdxo_id.to_excel(results_path + 'id{}_dz{}_{}_binned-id_percent_overlap.xlsx'.format(name, dz, t_id), index=False)

                # plot bin(percent dx diameter, id)
                if save_figs_pdo or show_figs_pdo:
                    for pc in plot_columns:
                        fig, ax = plt.subplots()

                        ax.plot(df_bin_pdxo[xparam], df_bin_pdxo[pc], '-o')

                        ax.set_xlabel(xparam)
                        ax.set_ylabel('{} precision'.format(pc))

                        axr = ax.twinx()
                        axr.plot(df_bin_pdxo[xparam], df_bin_pdxo[count_column], '-s', markersize=2, alpha=0.25)
                        axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
                        axr.set_ylim([0, int(np.round(df_bin_pdxo[count_column].max() + 6, -1))])

                        plt.tight_layout()
                        if save_figs_pdo:
                            plt.savefig(figs_path + 'id{}_dz{}_{}_{}-precision_dep-on_{}.png'.format(name,
                                                                                                     dz,
                                                                                                     t_id,
                                                                                                     pc,
                                                                                                     xparam))
                        if show_figs_pdo:
                            plt.show()
                        plt.close()

    # --------------------------------------- PRECISION (Z, R, ID) ---------------------------------------------
    per_z_r_id = True
    export_results = False

    if per_z_r_id:

        dfo = pd.concat(pdxos, ignore_index=True)

        columns_to_bin = ['z', 'percent_dx_diameter']
        # bin_pdo = np.append(np.arange(-1, 1.01, .2))
        bins = [[5, 20, 50, 80, 95], np.arange(-1.6, 1.01, .20)]
        pos = ['z_corr']

        df_bin_z_r_id, df_bin_z_r = analyze.evaluate_3d_static_precision(dfo,
                                                                         columns_to_bin=columns_to_bin,
                                                                         precision_columns=pos,
                                                                         bins=bins,
                                                                         round_to_decimals=[3, 4])

        if export_results:
            df_bin_z_r_id.to_excel(results_path + 'binned-id-z-percent_overlap-id.xlsx', index=True)
            df_bin_z_r.to_excel(results_path + 'binned-id-z-percent_overlap.xlsx', index=True)

        # filter minimum counts per bin
        df_bin_z_r = df_bin_z_r[df_bin_z_r['counts'] > 300]

        # plot bin(z, r)
        count_column = 'counts'
        plot_columns = pos
        xparam = 'percent_dx_diameter'
        pparams = 'z'
        line_plots = np.unique(df_bin_z_r[pparams].to_numpy())
        lbls = ['-45', '-30', '0', '30', '45']

        for pc in plot_columns:
            fig, ax = plt.subplots()
            # axr = ax.twinx()

            for lpt, lbl in zip(line_plots, lbls):
                dfpt = df_bin_z_r[df_bin_z_r[pparams] == lpt]

                ax.plot(dfpt[xparam], dfpt[pc], '-o', label=lbl)

                # axr.plot(dfpt[xparam], dfpt[count_column], '-s', markersize=2, alpha=0.25)

            ax.set_xlabel(xparam)
            ax.set_ylabel('{} precision'.format(pc))
            ax.legend(title=r'$\Delta z$')

            # axr.set_ylabel(r'$N_{p} \: (\#)$', color='gray')
            # axr.set_ylim([0, int(np.round(df_bin_z_r[count_column].max() + 6, -1))])
            plt.savefig(figs_path + 'collection_z-precision_by_z-pdo-no-nums-wide-pdo-range.png')
            plt.tight_layout()
            plt.show()