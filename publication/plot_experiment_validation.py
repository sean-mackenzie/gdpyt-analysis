# test bin, analyze, and plot functions

from os.path import join

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd

import filter
import analyze
from correction import correct
from utils import io, plotting, modify, details

# description of code
"""
Key Note on Plotting in this Code:
    * A dictionary 'dictplot' is initialized and used to store the data for final plotting in a multi-subplot figure.

Purpose of this code:
    1. Calculate the particle distributions using GDPyT and GDPT for four known displacements of +/-30 and +/-45 microns
    2. Are analyses are performed at the 'per-particle' level.
        2.1 Because particles are unlikely initially flat due to tilt of the microscope slide.

Process of this code:
    1. Setup
        1.1 Setup file paths.
        1.2 Setup plotting style.
        1.3 Setup filters. 
        1.4 Read all test_coords.
        1.5 Filter particles by c_m (should always be 0.5)
    2. Split test_coords into initial (frames <50) and final (frames >50).
        2.1 Filter frames into initial and final groups.
            2.1.1 Initial frames are at the focal plane (dz = 0) and final frames are at known displacement.
    3. Group particles to get particle average and stdev metrics: z and cm.
    4. Filter particles by z-coord. standard deviation:
        4.1 Filter initial coordinates by stdev ~ 0.5 b/c assessment should be accurate near focal plane. 
        4.2 Filter final z-coords. by stdev ~ 1.5 b/c assessment should be accurate within +/- 1 micron unless flopping.
    5. Normalize per-particle z-coords. by distance from the per-particle average z-coord. 
        5.1 for each particle: z-coord. = z-coord. average - z-coord.
            5.1.1. This normalization enables comparison of the particle distribution across different z-coords. 
    6. Filter particles by distance from the particle-collection average.
        6.1 Filter particles > 4 standard deviations away from the collection average.
    7. Inner concatenate all datasets to get only particles that passed filters across all tests.
    8. Normalize particles by distance from the focal plane.
        8.1 Enables comparison of -45 and +45 micron-steps, for example. 
    9. Plot mirrored density figure. 
    10. Export data to excel. 
"""

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

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation'
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# experiment
path_name = join(base_dir, 'test_coords/test/step')
save_id = 'exp_validation_static_v_spc'
# meta-assessment
path_meta = join(base_dir, 'test_coords/meta-assessment/test_coords')
path_meta_calib = join(base_dir, 'test_coords/meta-assessment/calib_coords')
calib_sort_strings = ['calib_', '_coords_']
save_id_meta = 'meta-assessment'
# synthetic
path_synthetic = join(base_dir, 'test_coords/synthetic')
save_id_synthetic = 'synthetic'
# synthetic - spc use_stack_id sweep
path_synthetic_spcs = join(base_dir, 'test_coords/synthetic/spc-use-stack-ids')
save_id_synthetic_spcs = 'synthetic-spcs'

# setup I/O
sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'
drop_columns = ['stack_id', 'z_true', 'x', 'y', 'max_sim', 'error']
results_drop_columns = ['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true', 'y_true']

# setup figures
scale_fig_dim = [1, 1]
scale_fig_dim_legend_outside = [1.3, 1]
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# dictionary for plotting of pubfig
dictplot = {}
dictplot_gdpyt = {}
id_gdpyt = 1.0
dictplot_gdpt = {}
id_gdpt = 27.0

cm_filter_temp = None

# ----------------------------------------------------------------------------------------------------------------------
# Analyze calibration coords
fp_cc = join(path_meta_calib, 'calib_1_coords_1h_20X_0.5demag_2.15um_mean2calib.xlsx')
path_figs_cc = join(path_figs, 'depth-dependent-defocusing')

df_cc = pd.read_excel(fp_cc)

# plot mean intensity profile of all particles
plot_intensity_average_pids = False
if plot_intensity_average_pids:
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.5))
    dfpid = df_cc.groupby('z').mean().reset_index()
    ax1.plot(dfpid.z, dfpid.peak_int, label=r'$I_{peak}$')
    ax11 = ax1.twinx()
    ax11.plot(dfpid.z, dfpid.snr, color='#00B945', label='SNR')
    ax2.plot(dfpid.z, dfpid.mean_int, label=r'$I_{mean}$')
    ax2.plot(dfpid.z, dfpid.mean_bkg + 2 * dfpid.std_bkg, label='Peak Noise')

    ax1.set_ylabel(r'$I_{peak}$ (A.U.)')
    ax11.set_ylabel('SNR', color='#00B945')
    ax11.set_ylim([-1, 101])
    ax2.set_ylabel(r'$I$ (A.U.)')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(join(path_figs_cc, 'peak_intensity_profile_avg_pids.png'))
    plt.show()

# plot intensity profile of all particles
plot_intensity_all_pids = False
if plot_intensity_all_pids:
    pids = df_cc.id.unique()
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches*1.5))
    for pid in pids:
        dfpid = df_cc[df_cc['id'] == pid]
        ax1.plot(dfpid.z, dfpid.peak_int)
        ax2.plot(dfpid.z, dfpid.mean_int)

    ax1.set_ylabel(r'$I_{peak}$ (A.U.)')
    ax2.set_ylabel(r'$I_{mean}$ (A.U.)')
    ax2.set_xlabel(r'$z \: (\mu m)$')
    ax1.set_title('Peak intensity profile: {} particles'.format(len(pids)))
    plt.tight_layout()
    plt.savefig(join(path_figs_cc, 'peak_intensity_profile_all_pids.png'))
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Analyze synthetic - SPC use_stack_id sweep
analyze_synthetic = False

if analyze_synthetic:

    # setup - I/O
    save_figs_synthetic_spcs = False
    show_figs_synthetic_spcs = False
    export_synthetic_spcs = False

    # setup - binning
    column_to_bin_and_assess = 'z_true'
    bins = 25
    mean_bins = 1
    h_synthetic = 1  # (actual value == 100)
    round_z_to_decimal = 5
    z_range = [-65.001, 35.001]
    min_cm = 0.5

    # setup figures
    ylim_synthetic = [0, 7.5]
    ylim_synthetic_norm = [0, 0.075]
    legend_loc = 'upper left'

    # ---------------------------------
    # 1. read .xlsx files to dictionary
    dficts = io.read_dataframes(path_synthetic_spcs, sort_strings, filetype, drop_columns=None)

    # ----------------------------------------------------------------------------------------
    # Calculate uncertainty for SPCs

    # 3. calculate local z-uncertainty
    dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range,
                                                 round_z_to_decimal, dficts_ground_truth=None)

    # 4. plot methods comparison local results
    if save_figs_synthetic_spcs or show_figs_synthetic_spcs:
        label_dict = {key: {'label': key - 10} for key in list(dfbicts.keys())}
        parameter = 'rmse_z'
        fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h_synthetic, scale=scale_fig_dim, label_dict=label_dict)
        ax.set_ylim(ylim_synthetic)
        ax.set_ylabel(r'$\sigma_{z}(z)\: (\mu m)$')
        ax.legend(loc=legend_loc, title=r'$p_{ID,\: calib}$')
        plt.tight_layout()
        if save_figs_synthetic_spcs:
            plt.savefig(join(path_figs, save_id_synthetic_spcs+'_spcs_local_rmse_z.png'))
        if show_figs_synthetic_spcs:
            plt.show()
        plt.close(fig)

    # 5. export mean rmse_z
    if export_synthetic_spcs:
        # calculate mean rmse-z (bins == 1)
        dfmicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, mean_bins, min_cm, z_range,
                                                     round_z_to_decimal, dficts_ground_truth=None)

        # calculate mean measurement results and export to excel
        dfm = analyze.calculate_bin_measurement_results(dfmicts)
        io.export_df_to_excel(dfm, path_name=join(path_results, save_id_synthetic_spcs + '_spcs_mean_measurement_results'),
                              include_index=True, index_label='test_id', filetype='.xlsx',
                              drop_columns=results_drop_columns)

    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------
    # Combine SPC tests and calculate the average uncertainty

    # 6. stack dficts into a dictionary with one key
    dficts = modify.stack_dficts([dficts], keys=[11.0])

    # 7. calculate local z-uncertainty
    dfbicts_spc = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range,
                                                 round_z_to_decimal, dficts_ground_truth=None)

    # 8. plot methods comparison local results
    if save_figs_synthetic_spcs or show_figs_synthetic_spcs:
        parameter = 'rmse_z'
        fig, ax = plotting.plot_dfbicts_local(dfbicts_spc, parameter, h_synthetic, scale=scale_fig_dim)
        ax.set_ylim(ylim_synthetic)
        ax.set_ylabel(r'$\sigma_{z}(z)\: (\mu m)$')
        plt.tight_layout()
        if save_figs_synthetic_spcs:
            plt.savefig(join(path_figs, save_id_synthetic_spcs+'_combined_local_rmse_z.png'))
        if show_figs_synthetic_spcs:
            plt.show()
        plt.close(fig)

    # 9. export mean rmse_z
    if export_synthetic_spcs:
        # calculate mean rmse-z (bins == 1)
        dfmicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, mean_bins, min_cm, z_range,
                                                     round_z_to_decimal, dficts_ground_truth=None)

        # calculate mean measurement results and export to excel
        dfm = analyze.calculate_bin_measurement_results(dfmicts)
        io.export_df_to_excel(dfm, path_name=join(path_results, save_id_synthetic_spcs + '_spc_mean_measurement_results'),
                              include_index=True, index_label='test_id', filetype='.xlsx',
                              drop_columns=results_drop_columns)

    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # Plot binned z-uncertainty for synthetic static and combined spcs

    # setup - I/O
    save_figs_synthetic = False
    show_figs_synthetic = False
    export_synthetic = False

    # 1. read .xlsx files to dictionary
    dficts = io.read_dataframes(path_synthetic, sort_strings, filetype, drop_columns=None)

    # 2. filter (if needed)
    # dficts = filter.dficts_filter(dficts, ['cm'], [0.5], ['greaterthan'], copy=True)

    # 3. calculate local z-uncertainty
    dfbicts_static = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range,
                                                 round_z_to_decimal, dficts_ground_truth=None)

    # 4. stack dfbicts static and spc into a single dictionary
    dfbicts_synthetic = modify.stack_dficts([dfbicts_static, dfbicts_spc], keys=[1.0, 11.0])

    # 5. plot methods comparison local results
    if save_figs_synthetic or show_figs_synthetic:
        label_dict = {key: {'label': lbl} for (key, lbl) in zip(list(dfbicts_synthetic.keys()), ['GDPyT', 'GDPT'])}
        parameter = 'rmse_z'
        fig, ax = plotting.plot_dfbicts_local(dfbicts_synthetic, parameter, h_synthetic, scale=scale_fig_dim,
                                              label_dict=label_dict)
        ax.set_ylim(ylim_synthetic)
        ax.set_ylabel(r'$\sigma_{z}(z)\: (\mu m)$')
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        if save_figs_synthetic:
            plt.savefig(join(path_figs, save_id_synthetic + '_synthetic_local_rmse_z.png'))
        if show_figs_synthetic:
            plt.show()
        plt.close(fig)

    # 9. export mean rmse_z
    if export_synthetic:
        # calculate mean rmse-z (bins == 1)
        dfmicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, mean_bins, min_cm, z_range,
                                                     round_z_to_decimal, dficts_ground_truth=None)

        # calculate mean measurement results and export to excel
        dfm = analyze.calculate_bin_measurement_results(dfmicts)
        io.export_df_to_excel(dfm, path_name=join(path_results, save_id_synthetic + '_static_mean_measurement_results'),
                              include_index=True, index_label='test_id', filetype='.xlsx',
                              drop_columns=results_drop_columns)


    # Add data to dict_plot
    dictplot_gdpyt.update({'synthetic_rmse_z': dfbicts_synthetic[1.0]})
    dictplot_gdpt.update({'synthetic_rmse_z': dfbicts_synthetic[11.0]})

    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Analyze meta-assessment
analyze_meta = False

if analyze_meta:
    # setup
    per_particle_correction = True
    show_meta_plots = False
    save_meta_plots = False
    export_meta_results = False
    save_calib_curve_plots = False
    show_calib_curve_plots = False

    # calib correction coords
    cficts = io.read_files('df', path_meta_calib, calib_sort_strings, filetype, startswith=calib_sort_strings[0])

    # read details
    cficts_details = details.parse_filename_to_details(path_meta_calib, calib_sort_strings, filetype,
                                                       startswith=calib_sort_strings[0])
    cficts_details = details.read_dficts_coords_to_details(cficts, cficts_details, calib=True)

    # test coords
    dficts = io.read_files('df', path_meta, sort_strings, filetype, startswith=sort_strings[0])
    labels = list(dficts.keys())

    # ----------------------------------------------------------------------------------------------------------------------
    # Apply filters to remove outliers

    # filter particles with c_m < __
    cm_filter = 0.0
    if cm_filter > 0.01:
        dficts = filter.dficts_filter(dficts, keys=['cm'], values=[cm_filter], operations=['greaterthan'])

    # filter particles with errors > h/10 (single rows)
    apply_barnkob_filter = False
    filter_pids_from_all_frames = False
    if apply_barnkob_filter:
        barnkob_error_filter = 0.1  # filter error < h/10 used by Barnkob and Rossi in 'A fast robust algorithm...'
        meas_vols = np.array([x[1]['meas_vol'] for x in cficts_details.items()])
        meas_vols_inverse = 1 / meas_vols
        dficts = modify.dficts_new_column(dficts, new_columns=['percent_error'], columns=['error'],
                                          multipliers=meas_vols_inverse)
        dficts, dficts_filtered = filter.dficts_filter(dficts, keys=['percent_error'],
                                                       values=[[-barnkob_error_filter, barnkob_error_filter]],
                                                       operations=['between'], return_filtered=True)

        # filter particles with errors > h/10 (all rows for particles where any row error > h/10)
        if filter_pids_from_all_frames:
            for name, df in dficts_filtered.items():
                pids_filter = df.id.unique()
                dficts = filter.dficts_filter(dficts, keys=['id'], values=[pids_filter], operations=['notin'],
                                              only_keys=[name])

    # filter particles that appear in too few frames (only useful for dynamic templates)
    filter_num_frames = False
    min_num_frames = 20
    if filter_num_frames:
        for name, df in cficts.items():
            dfc = df.groupby('id').count()
            pids_filter_num_frames = dfc[dfc['frame'] < min_num_frames].index.to_numpy()
            cficts = filter.dficts_filter(cficts, keys=['id'], values=[pids_filter_num_frames], operations=['notin'],
                                          only_keys=[name])
            dficts = filter.dficts_filter(dficts, keys=['id'], values=[pids_filter_num_frames], operations=['notin'],
                                          only_keys=[name])

    # ----------------------------------------------------------------------------------------------------------------------
    # correct particle coordinates

    # setup
    path_figs_in_focus_correction = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/figs/meta-assessment/calibration-in-focus-correction'

    # find sub-image in-focus z-coordinate using interpolated peak intensity
    include_quartiles = 2
    cficts = correct.calc_calib_in_focus_z(cficts, dficts, per_particle_correction=per_particle_correction,
                                           only_quartiles=include_quartiles, show_z_focus_plot=False,
                                           show_particle_plots=False,
                                           num_particle_plots=1, save_plots=False, save_path=path_figs_in_focus_correction,
                                           round_to_decimals=4)

    # correct test coords based on calibration corrected coordinates
    dficts = correct.correct_z_by_in_focus_z(cficts, dficts, per_particle_correction=per_particle_correction)

    # ----------------------------------------------------------------------------------------------------------------------
    # calibration curve

    # data for dictplot
    dictplot_gdpyt.update({'calib_curve': dficts[id_gdpyt]})
    dictplot_gdpt.update({'calib_curve': dficts[id_gdpt]})

    # setup
    cm_filter_temp = 0.50
    xlim_meta = [-55, 60.5]
    xticks_meta = [-50, -25, 0, 25, 50]

    if save_calib_curve_plots or show_calib_curve_plots:
        save_id_calib_curve_spcs = 'calibration_curve_spcs'

        # re-sort dficts so GDPyT is plotted on top
        dficts = modify.dficts_sort(dficts)
        colors_calib_curve = ['#00B945', '#0C5DA5']
        id_counter = list(dficts.keys())[1:]

        for id in id_counter:
            fig, ax = plt.subplots()
            df_temp = dficts[1.0]
            df_temp = df_temp[df_temp['cm'] > cm_filter_temp]
            ax.scatter(df_temp.z_true, df_temp.z, s=0.5, color=colors_calib_curve[1], label='GDPyT', zorder=3.5)

            df_temp = dficts[id]
            df_temp = df_temp[df_temp['cm'] > cm_filter_temp]
            ax.scatter(df_temp.z_true, df_temp.z, s=0.5, color=colors_calib_curve[0], label='GDPT', zorder=3)

            ax.set_ylabel(r'$z_{measured}\: (\mu m)$')
            ax.set_ylim(xlim_meta)
            ax.set_yticks(xticks_meta)
            ax.set_xlabel(r'$z_{true}\: (\mu m)$')
            ax.set_xlim(xlim_meta)
            ax.set_xticks(xticks_meta)
            # ax.grid(alpha=0.125)
            ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.5, 1, 0), markerscale=2, borderpad=0.1,
                      handletextpad=0.05, borderaxespad=0.1)
            # ax.set_title('Calibration Curve: ID={}'.format(id))
            plt.tight_layout()
            if save_calib_curve_plots:
                plt.savefig(join(path_figs, save_id_meta + '_z-calibration_cm={}.png'.format(cm_filter_temp)))
            if show_calib_curve_plots:
                plt.show()

    # ----------------------------------------------------------------------------------------------------------------------
    # calculate z-uncertainties: bin by number of bins

    # binning
    column_to_bin_and_assess = 'z_true'
    if not bins:
        num_bins = 25
    else:
        num_bins = bins
    min_cm = cm_filter
    round_z_to_decimal = 6

    # calculate local rmse_z
    dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins=num_bins, min_cm=cm_filter_temp,
                                                 z_range=None, round_to_decimal=round_z_to_decimal,
                                                 dficts_ground_truth=None)

    # data for dictplot
    dictplot_gdpyt.update({'meta_rmse_z': dfbicts[id_gdpyt]})
    dictplot_gdpt.update({'meta_rmse_z': dfbicts[id_gdpt]})

    # plots binned z-uncertainty
    if save_meta_plots or show_meta_plots:
        h_meas_vol = 105
        h_meas_vols = [1, h_meas_vol]
        ylim_metas = [[0, 15], [0, 0.15]]
        xlim_meta = [-52.5, 57.5]
        ylabel_metas = [r'$\sigma_{z}\: (\mu m)$', r'$\sigma_{z}/h$']
        label_dict = {key: {'label': lbl} for (key, lbl) in zip(list(dfbicts.keys()), ['GDPyT', 'GDPT'])}
        save_id_metas = ['', '_norm']

        # z-uncertainty (microns)
        parameter = 'rmse_z'
        for hs, ylabels, ylims, saveids in zip(h_meas_vols, ylabel_metas, ylim_metas, save_id_metas):
            fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h=hs, scale=scale_fig_dim, label_dict=label_dict)

            ax.set_ylabel(ylabels)
            ax.set_ylim(ylims)
            ax.set_xlabel(r'$z_{true} \: (\mu m)$')  #
            ax.set_xlim(xlim_meta)
            ax.set_xticks(xticks_meta)
            ax.legend(loc='upper left')  # , fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0))
            plt.tight_layout()
            if save_meta_plots:
                plt.savefig(join(path_figs, save_id_meta + '_z_uncertainty{}_{}bins_cm={}.png'.format(saveids, num_bins, cm_filter_temp)))
            if show_meta_plots:
                plt.show()

        if export_meta_results:
            # export the binned results
            dfb_export = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
            io.export_df_to_excel(dfb_export, path_name=join(path_results, save_id_meta + '_{}bins_measurement_results_cm={}'.format(num_bins, cm_filter_temp)),
                                  include_index=True, index_label='bin_z', filetype='.xlsx',
                                  drop_columns=results_drop_columns[:-2])

            # calculate local rmse_z with bin == 1
            dfmicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins=1, min_cm=cm_filter_temp,
                                                         z_range=None, round_to_decimal=round_z_to_decimal,
                                                         dficts_ground_truth=None)
            # calculate mean measurement results and export to excel
            dfm = analyze.calculate_bin_measurement_results(dfmicts)
            io.export_df_to_excel(dfm, path_name=join(path_results, save_id_meta + '_mean_measurement_results_cm={}'.format(cm_filter_temp)),
                                  include_index=True, index_label='test_id', filetype='.xlsx',
                                  drop_columns=results_drop_columns[:-2])


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Plot calibration particle-to-test particle similarity

analyze_calib_to_test_sim = False

if analyze_calib_to_test_sim:
    # setup
    show_calib_to_test_sim_plots = False
    save_calib_to_test_sim_plots = False
    export_calib_to_test_sim = False
    save_id_calib_to_test_sim = 'calib_to_test_similarity'

    path_calib_to_test_sim_calib = join(base_dir, 'test_coords/similarity/spc-calib-to-test-particles/calib_coords')
    path_calib_to_test_sim_test = join(base_dir, 'test_coords/similarity/spc-calib-to-test-particles/test_coords')

    simcficts = io.read_files('df', path_calib_to_test_sim_calib, calib_sort_strings, filetype,
                              startswith=calib_sort_strings[0])
    simdficts = io.read_files('df', path_calib_to_test_sim_test, sort_strings, filetype, startswith=sort_strings[0])

    simcficts = correct.calc_calib_in_focus_z(simcficts, simdficts, per_particle_correction=per_particle_correction,
                                              only_quartiles=include_quartiles, show_z_focus_plot=False,
                                              show_particle_plots=False,
                                              num_particle_plots=3, save_plots=False,
                                              save_path=path_figs_in_focus_correction,
                                              round_to_decimals=4)

    # correct test coords based on calibration corrected coordinates
    simdficts = correct.correct_z_by_in_focus_z(simcficts, simdficts, per_particle_correction=per_particle_correction)
    simdfbicts = analyze.calculate_bin_local(simdficts, column_to_bin='z_true', bins=num_bins, min_cm=cm_filter_temp,
                                             round_to_decimal=round_z_to_decimal, true_num_particles=None, z0=0,
                                             take_abs=False)

    # data for dictplot
    dictplot_gdpyt.update({'calib_to_test_sim': dfbicts[id_gdpyt]})
    dictplot_gdpt.update({'calib_to_test_sim': simdfbicts[id_gdpt]})

    # plots
    if show_calib_to_test_sim_plots or save_calib_to_test_sim_plots:

        fig, ax = plt.subplots()

        ax.scatter(dfbicts[id_gdpyt].index, dfbicts[id_gdpyt].cm, s=1, label='GDPyT')
        ax.plot(dfbicts[id_gdpyt].index, dfbicts[id_gdpyt].cm)

        ax.scatter(simdfbicts[id_gdpt].index, simdfbicts[id_gdpt].cm, s=1, label='GDPT')
        ax.plot(simdfbicts[id_gdpt].index, simdfbicts[id_gdpt].cm)

        ax.set_ylabel(r'$\left \langle S \left(p^{cal}_{i}, p^{test}_{j} \right) \right \rangle$')
        ax.set_xlabel(r'$z \: (\mu m)$')  #
        ax.set_xlim(xlim_meta)
        ax.set_xticks(xticks_meta)
        ax.set_ylim([0.5, 1.0])
        ax.legend(loc='lower left')  # , fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0))
        plt.tight_layout()
        if save_calib_to_test_sim_plots:
            plt.savefig(join(path_figs, save_id_calib_to_test_sim + '_{}bins_cm={}.png'.format(num_bins, cm_filter_temp)))
        if show_calib_to_test_sim_plots:
            plt.show()

    if export_calib_to_test_sim:
        dfb_gdpyt = dfbicts[id_gdpyt]
        dfb_gdpyt.insert(0, 'filename', id_gdpyt)
        dfb_gdpt = simdfbicts[id_gdpt]
        dfb_gdpt.insert(0, 'filename', id_gdpt)
        dfm = pd.concat([dfb_gdpyt, dfb_gdpt], ignore_index=False)
        io.export_df_to_excel(dfm, path_name=join(path_results,
                                                  save_id_calib_to_test_sim + '_{}bins_measurement_results_cm={}'.format(num_bins, cm_filter_temp)),
                              include_index=True, index_label='bin_z', filetype='.xlsx',
                              drop_columns=drop_columns[:-2] + ['rmse_z', 'error'])


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Plot calibration stacks self-similarity

plot_calibration_stacks_self_similarity = False

if plot_calibration_stacks_self_similarity:
    # setup
    path_self_sim_static = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/similarity/calibration-stack-self-similarity/static_meta-assessment_avg2imgs_calib-stack_middle_self_similarity.xlsx'
    path_self_sim_spc = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/similarity/calibration-stack-self-similarity/spc_meta-assessment_avg2imgs_calib-stack_middle_self_similarity.xlsx'
    spc_stack_id = 27

    # read
    dfstatic = pd.read_excel(path_self_sim_static)
    dfspc = pd.read_excel(path_self_sim_spc)

    # data processing
    dfstatic = dfstatic.groupby('z').mean()
    dfspc = dfspc[dfspc['id'] == spc_stack_id]

    if plot_calibration_stacks_self_similarity:
        fig, ax = plt.subplots()
        for df in [dfstatic, dfspc]:
            pass


# ----------------------------------------------------------------------------------------------------------------------
"""Azure: #069AF3
Dark Green: #054907
Blue: '#0C5DA5'
Green: #00B945"""
# ----------------------------------------------------------------------------------------------------------------------
# Plot figures of combined datasets
show_synthetic_meta_plots = False
save_synthetic_meta_plots = False
norm_x = True
h1 = 1  # 100
h2 = 1  # 105

if show_synthetic_meta_plots:

    lbl = ['Synthetic', 'Real', 'Synthetic', 'Real']
    clr = ['#069AF3', '#0C5DA5', 'limegreen', '#00B945']
    zo = [2.5, 2]
    scatter_size = 10

    fig, ax = plt.subplots(figsize=(size_x_inches * 1, size_y_inches))

    dfb1 = dictplot_gdpyt['synthetic_rmse_z']
    dfb2 = dictplot_gdpyt['meta_rmse_z']

    if norm_x:
        dfb1x = (dfb1.index - np.mean(dfb1.index)) / h1 + 0.5
        dfb2x = (dfb2.index - np.mean(dfb2.index)) / h2 + 0.5
    else:
        dfb1x = dfb1.index / h1
        dfb2x = dfb2.index / h2

    gys, = ax.plot(dfb1x, dfb1.rmse_z, linestyle='--', color=clr[0], label=lbl[0], zorder=zo[0])
    ax.scatter(dfb1x, dfb1.rmse_z, s=scatter_size, color=clr[0], zorder=zo[0])
    gym, = ax.plot(dfb2x, dfb2.rmse_z, linestyle='-', color=clr[1], label=lbl[1], zorder=zo[0])
    ax.scatter(dfb2x, dfb2.rmse_z, s=scatter_size, color=clr[1], zorder=zo[0])

    dfb1 = dictplot_gdpt['synthetic_rmse_z']
    dfb2 = dictplot_gdpt['meta_rmse_z']

    if norm_x:
        dfb1x = (dfb1.index - np.mean(dfb1.index)) / h1 + 0.5
        dfb2x = (dfb2.index - np.mean(dfb2.index)) / h2 + 0.5
    else:
        dfb1x = dfb1.index / h1
        dfb2x = dfb2.index / h2

    gps, = ax.plot(dfb1x, dfb1.rmse_z, linestyle='--', color=clr[2], label=lbl[2], zorder=zo[1])
    ax.scatter(dfb1x, dfb1.rmse_z, s=scatter_size, color=clr[2], zorder=zo[1])
    gpm, = ax.plot(dfb2x, dfb2.rmse_z, linestyle='-', color=clr[3], label=lbl[3], zorder=zo[1])
    ax.scatter(dfb2x, dfb2.rmse_z, s=scatter_size, color=clr[3], zorder=zo[1])

    ax.set_xlabel(r'$z_{true}\: (\mu m)$')
    ax.set_ylabel(r'$\sigma_{z}(z)\: (\mu m)$')
    ax.set_ylim([0, 10])

    if h1 > 1:
        if norm_x:
            ax.set_xlim([-0.0125, 1.0125])
            ax.set_xticks([0, 0.5, 1])
        else:
            ax.set_xlim([-0.55, 0.55])
            ax.set_xticks([-0.5, 0, 0.5])
    else:
        if norm_x:
            ax.set_xlim([-55, 55])
            ax.set_xticks([-50, -25, 0, 25, 50])
        else:
            ax.set_xlim([-65, 57.5])
            ax.set_xticks([-50, -25, 0, 25, 50])

    ax.grid(alpha=0.125)

    # legend
    handles, labels = ax.get_legend_handles_labels()

    gdpyt_legend = ax.legend(handles=[gys, gym], loc='upper left', bbox_to_anchor=(0.005, 1.0, 1, 0), title='GDPyT')
    ax.add_artist(gdpyt_legend)

    gdpt_legend = ax.legend(handles=[gps, gpm], loc='upper left', bbox_to_anchor=(0.005, 0.7, 1, 0), title='GDPT')
    ax.add_artist(gdpt_legend)

    plt.tight_layout()
    if save_synthetic_meta_plots:
        plt.savefig(join(path_figs, 'combined_synthetic_meta_units.png'))
    if show_synthetic_meta_plots:
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Data analysis of experimental validation and plotting the mirrored violin figure

# experimental details
h = 105
true_dz = [-45, -30.0, 30.0, 45, -45, -30.0, 30.0, 45]

# setup plotting
save_violin_plots = False
show_violin_plots = True
plot_violin_parameter = 'dz_mean'  # options: 'error_dx_mean'
plot_violin_position_at = 'true'
plot_absolute_displacements = False
plot_quartile = False
plot_whiskers = False
plot_true_dz_vlines = True
plot_zeroline = False
# below naming for (GDPyT and SPC) and (Positive and Negative): density distribution color, quartile color, median color
cn_gdpyt, qn_gdpyt, mn_gdpyt = 'cornflowerblue', 'royalblue', 'blue'
cp_gdpyt, qp_gdpyt, mp_gdpyt = '#0C5DA5', 'cornflowerblue', '#7BC8F6'  # 'Azure:#069AF3'; Light Blue:#7BC8F6; Paler Blue:#0343DF
cn_spc, qn_spc, mn_spc = 'palegreen', 'limegreen', 'darkgreen'
cp_spc, qp_spc, mp_spc = '#00B945', 'lime', '#054907'  # '#054907'=Dark Green;
eclr = None  # 'black'
density_widths = 0.2
median_marker = [0, 0, 1, 1, 0, 0, 1, 1]  # carets: [8, 8, 9, 9, 8, 8, 9, 9]; ticks: [0, 0, 1, 1, 0, 0, 1, 1]
median_marker_size = 25  # carets=12
true_linestyle = ':'
true_linewidth = 0.75
true_alpha = 0.25
xlim_min, xlim_max = 0.0, 0.5  # (0.2, 0.5)
ylim_error = [0.0, 10.5]
yticks_meta = [0, 5, 10]
ylim_percent_meas = [0, 105]

# filters
frame_i = 51  # separate initial frames
frame_f = 50  # separate final frames
std_filter_initial = 2  # filters initial particles by stdev of z-coord (standard = 0.5)
std_filter_final = 2  # filter final particles by stdev of z-coord (standard = 1.25)
barnkob_error_filter = h / 10 * 2  # error filter used by Barnkob and Rossi in 'A fast robust algorithm...'
std_filter_dz = 2  # filters individual particles by stdev of z-coord. from the collection average (std = 3)

# read .xlsx files to dictionary
drop_columns_test = ['stack_id', 'z_true', 'max_sim', 'error']
dficts = io.read_dataframes(path_name, sort_strings, filetype, drop_columns=drop_columns_test)
dficts_ids = list(dficts.keys())

# ----------------------------------------------------------------------------------------------------------------------
# generate legends for manual copying to other figures
plot_legends = False
if plot_legends:
    x = np.arange(4)
    y = np.arange(4)
    colors = ['#0C5DA5', 'cornflowerblue', '#00B945', 'palegreen']
    labels = ['+z', '-z', '+z', '-z']
    # colors = ['#0C5DA5', '#0C5DA5', '#00B945', '#00B945']
    # labels = [r'$\sigma/h$', r'$\phi_{ID}$', r'$\sigma/h$', r'$\phi_{ID}$']
    markers = ['s', 'o', 's', 'o']
    fig, ax = plt.subplots()
    for i in range(len(x)):
        ax.scatter(x[i], y[i], marker=markers[i], color=colors[i], label=labels[i])
    ax.legend(ncol=1, title='GDPT')
    plt.savefig(join(path_figs, 'legend_title_GDPT_fig.png'))
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 2. Filter by c_m and split into initial and final frames
analyze_inital_and_final = True

if analyze_inital_and_final:

    df_nofilters = modify.stack_dficts_by_key(dficts, drop_filename=False)
    initial_num_particles = {}
    for i in df_nofilters.filename.unique():
        initial_num_particles.update({i: len(df_nofilters[df_nofilters['filename'] == i].id.unique())})

    if not cm_filter_temp:
        cm_filter_temp = 0.5
        xlim_meta = [-55, 60.5]
        xticks_meta = [-50, -25, 0, 25, 50]

    dficts = filter.dficts_filter(dficts, ['cm'], [cm_filter_temp], ['greaterthan'], copy=True)
    dficts_i = filter.dficts_filter(dficts, keys=['frame'], values=[frame_i], operations=['lessthan'], copy=True)
    dficts_f = filter.dficts_filter(dficts, keys=['frame'], values=[frame_f], operations=['greaterthan'], copy=True)

    dfi = modify.stack_dficts_by_key(dficts_i, drop_filename=False)
    dff = modify.stack_dficts_by_key(dficts_f, drop_filename=False)

    initial_num_f_measurements = {}
    for i in dff.filename.unique():
        initial_num_f_measurements.update({i: len(dff[dff['filename'] == i].z)})

    # ----------------------------------------------------------------------------------------------------------------------
    # Filter out rows (particle ID, frame) where z_error (z_true - z_meas) > 0.1 / h
    """dfd['error_dz_mean'] = 0
    
    for i, tdz in zip(dfd.filename.unique(), true_dz):
        # error dz is displacement of every particle from the group mean
        avg_dz_mean = dfd[dfd['filename'] == i].dz_mean.mean()
        dfd['error_dz_mean'][dfd['filename'] == i] = dfd['dz_mean'][dfd['filename'] == i] - avg_dz_mean"""
    # ----------------------------------------------------------------------------------------------------------------------

    drop_cols = ['frame']
    dfi = dfi.drop(columns=drop_cols)
    # dff = dff.drop(columns=drop_cols)

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3. Group particles to get particle average and stdev metrics: z and cm
analyze_calibration = True

if analyze_calibration:

    analyze_per_particle = True
    analyze_per_particle_then_per_average = False
    by = ['filename', 'id']
    keep_cols = ['z', 'cm']

    if analyze_per_particle:

        # plot the distribution of particles along x- and y-axes
        save_initial_particle_distribution = False
        show_initial_particle_distribution = True
        if save_initial_particle_distribution or show_initial_particle_distribution:
            save_id_initial_particle_distributions = 'initial_raw_distribution'

            if analyze_per_particle_then_per_average:

                keep_cols_distribution = ['x', 'y', 'z', 'cm']
                dfi_temp = dfi.copy()
                dfi_temp = analyze.df_calc_mean_std_count(df=dfi_temp, by=by, keep_cols=keep_cols_distribution, std=True, count=True, return_original_df=False)
                for i in dfi_temp.index.get_level_values('filename').unique():
                    df_temp = dfi_temp[dfi_temp.index.get_level_values('filename') == i]
                    x1 = df_temp.x_mean
                    x2 = df_temp.y_mean
                    y = df_temp.z_mean
                    yerr = df_temp.z_std

                    fig, [ax1, ax2] = plt.subplots(nrows=2)
                    for ax, xs, xlbs in zip([ax1, ax2], [x1, x2], ['x (pixels)', 'y (pixels)']):
                        ax.errorbar(xs, y, yerr=yerr, fmt="o", ms=0.5, ecolor='lightgray', elinewidth=1, capsize=1, alpha=0.5)
                        ax.set_xlabel(xlbs)
                        ax.set_ylabel(r'z $(\mu m)$')
                        ax.grid(alpha=0.125)

                    ax1.set_title('ID{}: initial raw distribution'.format(i))
                    plt.tight_layout()
                    if save_initial_particle_distribution:
                        plt.savefig(join(path_figs, 'test_displacement', 'initial-particle-distributions',
                                         save_id_initial_particle_distributions + '_id{}.png'.format(i)))
                    if show_initial_particle_distribution:
                        plt.show()
            else:
                for i in dfi['filename'].unique():
                    df_temp = dfi[dfi['filename'] == i]
                    x1 = df_temp.x
                    x2 = df_temp.y
                    y = df_temp.z
                    clr = df_temp.id

                    fig, [ax1, ax2] = plt.subplots(nrows=2)
                    for ax, xs, xlbs in zip([ax1, ax2], [x1, x2], ['x (pixels)', 'y (pixels)']):
                        ax.scatter(xs, y, c=clr, s=0.5, alpha=0.25)
                        ax.set_xlabel(xlbs)
                        ax.set_ylabel(r'z $(\mu m)$')
                        ax.grid(alpha=0.125)

                    ax1.set_title('ID{}: initial raw all distribution'.format(i))
                    plt.tight_layout()
                    if save_initial_particle_distribution:
                        plt.savefig(join(path_figs, 'test_displacement', 'initial-all-particles-distribution',
                                         save_id_initial_particle_distributions + '_id{}.png'.format(i)))
                    if show_initial_particle_distribution:
                        plt.show()

        # plot the distribution of particles along x- and y-axes
        save_final_particle_distribution = False
        show_final_particle_distribution = True
        if save_final_particle_distribution or show_final_particle_distribution:
            save_id_final_particle_distributions = 'final_raw_distribution'

            if analyze_per_particle_then_per_average:
                keep_cols_distribution = ['x', 'y', 'z', 'cm']
                dff_temp = dff.copy()
                dff_temp = analyze.df_calc_mean_std_count(df=dff_temp, by=by, keep_cols=keep_cols_distribution,
                                                          std=True, count=True, return_original_df=False)
                for i in dff_temp.index.get_level_values('filename').unique():
                    df_temp = dff_temp[dff_temp.index.get_level_values('filename') == i]
                    x1 = df_temp.x_mean
                    x2 = df_temp.y_mean
                    y = df_temp.z_mean
                    yerr = df_temp.z_std

                    fig, [ax1, ax2] = plt.subplots(nrows=2)
                    for ax, xs, xlbs in zip([ax1, ax2], [x1, x2], ['x (pixels)', 'y (pixels)']):
                        ax.errorbar(xs, y, yerr=yerr, fmt="o", ms=0.5, ecolor='lightgray', elinewidth=1, capsize=1,
                                    alpha=0.5)
                        ax.set_xlabel(xlbs)
                        ax.set_ylabel(r'z $(\mu m)$')
                        ax.grid(alpha=0.125)
                        if np.max(y) - np.min(y) < 5:
                            ax.set_ylim([np.mean(y) - 2.5, np.mean(y) + 2.5])

                    ax1.set_title('ID{}: final raw distribution'.format(i))
                    plt.tight_layout()
                    if save_final_particle_distribution:
                        plt.savefig(join(path_figs, 'test_displacement', 'final-particle-distributions',
                                         save_id_final_particle_distributions + '_id{}.png'.format(i)))
                    if show_final_particle_distribution:
                        plt.show()
            else:
                for i in dff['filename'].unique():
                    df_temp = dff[dff['filename'] == i]
                    x1 = df_temp.x
                    x2 = df_temp.y
                    y = df_temp.z
                    clr = df_temp.id

                    fig, [ax1, ax2] = plt.subplots(nrows=2)
                    for ax, xs, xlbs in zip([ax1, ax2], [x1, x2], ['x (pixels)', 'y (pixels)']):
                        ax.scatter(xs, y, c=clr, s=0.5, alpha=0.25)
                        ax.set_xlabel(xlbs)
                        ax.set_ylabel(r'z $(\mu m)$')
                        ax.grid(alpha=0.125)

                    ax1.set_title('ID{}: final raw all distribution'.format(i))
                    plt.tight_layout()
                    if save_initial_particle_distribution:
                        plt.savefig(join(path_figs, 'test_displacement', 'final-all-particles-distribution',
                                         save_id_final_particle_distributions + '_id{}.png'.format(i)))
                    if show_initial_particle_distribution:
                        plt.show()


        # add columns (mean, std, counts) to each row in df
        dfi = dfi.drop(columns=['x', 'y'])
        dff = dff.drop(columns=['x', 'y'])
        dfi = analyze.df_calc_mean_std_count(df=dfi, by=by, keep_cols=keep_cols, std=True, count=True)
        dff = analyze.df_calc_mean_std_count(df=dff, by=by, keep_cols=keep_cols, std=True, count=True, return_original_df=True)

        # map dfi values to dff
        # z_mean
        dfi_map_mean = dfi.reset_index()
        dfi_map_mean['map_mean'] = dfi_map_mean['filename'].astype(str) + '_' + dfi_map_mean['id'].astype(str)
        dfi_map_mean = dfi_map_mean.set_index('map_mean')
        map_mean_dict = dfi_map_mean.z_mean.round(decimals=4).to_dict()
        # z_std
        dfi_map_std = dfi.reset_index()
        dfi_map_std['map_std'] = dfi_map_std['filename'].astype(str) + '_' + dfi_map_std['id'].astype(str)
        dfi_map_std = dfi_map_std.set_index('map_std')
        map_std_dict = dfi_map_std.z_std.round(decimals=4).to_dict()

        # create columns for mapping
        dff['i_z_mean'] = dff['filename'].astype(str) + '_' + dff['id'].astype(str)
        dff['i_z_std'] = dff['filename'].astype(str) + '_' + dff['id'].astype(str)

        # create mapping dictionary to remove unmapped rows
        dff_remove = dff.copy()
        dff_remove['unmapped'] = np.nan
        dff_remove = dff_remove.set_index('i_z_mean')
        unmapping_dict = dff_remove.unmapped.to_dict()

        # map z_i_mean and z_i_std to dff
        dff = dff.replace({'i_z_mean': map_mean_dict})
        dff = dff.replace({'i_z_std': map_std_dict})

        # map nan to an unmapped rows and remove
        dff = dff.replace({'i_z_mean': unmapping_dict})
        dff = dff.replace({'i_z_std': unmapping_dict})
        dff_initial_num_particles = len(dff)
        dff = dff.dropna()
        dff_mapping_errors = dff_initial_num_particles - len(dff)
        print('{}/{} ({}%) rows removed due to mapping errors'.format(dff_mapping_errors, dff_initial_num_particles,
                                                                      np.round(dff_mapping_errors / dff_initial_num_particles * 100, 2)))

        # create a 'true_z_displacement' column
        mapping_dict_true_dz = {dficts_ids[i]: true_dz[i] for i in range(len(dficts_ids))}
        dff['true_dz'] = dff['filename']
        dff = dff.replace({'true_dz': mapping_dict_true_dz})
        dff['true_z'] = dff['i_z_mean'] + dff['true_dz']
        dff['meas_dz'] = dff['z'] - dff['i_z_mean']
        dff['error_dz'] = dff['z'] - dff['true_z']

        # filter out final rows with errors > h / 10
        dff_initial_num_particles = len(dff)
        dff = dff[dff['error_dz'].abs() < barnkob_error_filter]

        # print # of particles filtered out by Barnkob filter
        dff_post_barnkob_filter_num_particles = len(dff)
        dff_percenter_barnkob_filtered = np.round(dff_post_barnkob_filter_num_particles / dff_initial_num_particles * 100, 1)
        print('{}/{} ({}%) particles remaining after Barnkob filtering (err < {} um) per-particle calibration '
              'correction'.format(dff_post_barnkob_filter_num_particles, dff_initial_num_particles,
                                  dff_percenter_barnkob_filtered, barnkob_error_filter))

        if analyze_per_particle_then_per_average:
            # re-organize dff to fit following data analysis
            dff = dff.drop(columns=['z_mean', 'z_std', 'z_counts', 'i_z_mean', 'i_z_std', 'true_dz', 'true_z', 'meas_dz', 'error_dz'])
            dff = analyze.df_calc_mean_std_count(df=dff, by=by, keep_cols=keep_cols, std=True, count=True, return_original_df=False)

    else:
        dfi = analyze.df_calc_mean_std_count(df=dfi, by=by, keep_cols=keep_cols, std=True, count=True)
        dff = analyze.df_calc_mean_std_count(df=dff, by=by, keep_cols=keep_cols, std=True, count=True)

    if analyze_per_particle is False or analyze_per_particle_then_per_average is True:
        names_i = {x: 'i_' + x for x in list(dfi.columns)}
        names_f = {x: 'f_' + x for x in list(dff.columns)}
        dffi = dfi.rename(columns=names_i)
        dfff = dff.rename(columns=names_f)

        dfd = dffi.join([dfff])
        dfd = dfd.dropna()
        dfd['dz_mean'] = dfd['f_z_mean'] - dfd['i_z_mean']
        dfd['dz_std'] = dfd['f_z_std'] + dfd['i_z_std']
    else:
        dfd = None

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Perform measurement assessment using the average particle z-coordinate
analyze_displacement = False

if analyze_displacement:

    if dfd is not None:
        # ----------------------------------------------------------------------------------------------------------------------
        # 4. Filter particles by z-coord. standard deviation

        # filter
        apply_initial_std_filter = False
        if apply_initial_std_filter:
            dfd = dfd[dfd['i_z_std'] < std_filter_initial]
            dfd = dfd[dfd['f_z_std'] < std_filter_final]

        # ----------------------------------------------------------------------------------------------------------------------
        # 5. Normalize per-particle z-coords. by distance from the per-particle average z-coord

        # reset index to retain groupby columns
        dfd = dfd.reset_index()

        # calculate distance from group average
        dfd['avg_dz_mean'] = 0
        dfd['error_dz_mean'] = 0
        dfd['true_dz'] = 0
        for i, tdz in zip(dfd.filename.unique(), true_dz):
            # error dz is displacement of every particle from the group mean
            avg_dz_mean = dfd[dfd['filename'] == i].dz_mean.mean()
            dfd['avg_dz_mean'][dfd['filename'] == i] = avg_dz_mean
            dfd['error_dz_mean'][dfd['filename'] == i] = dfd['dz_mean'][dfd['filename'] == i] - avg_dz_mean
            dfd['true_dz'][dfd['filename'] == i] = tdz

        # ----------------------------------------------------------------------------------------------------------------------
        # ----- PLOT INITIAL PARTICLE DISTRIBUTION -----
        plot_initial_distribution_error_bars = False
        if plot_initial_distribution_error_bars:
            dfd_initial_mean = dfd.groupby(by='filename').mean()
            dfd_dz_mean = dfd_initial_mean.dz_mean.to_numpy()
            dfd_ids = dfd_initial_mean.index.to_numpy()
            dfd_initial_std = dfd.groupby(by='filename').std().dz_mean.to_numpy()
            fig, ax = plt.subplots()
            i = 0
            for id, id_dz_mean in zip(dfd_ids, dfd_dz_mean):
                ax.errorbar(id, id_dz_mean, yerr=dfd_initial_std[i], fmt="o")
                i += 1
            ax.set_xticks(ticks=dfd_ids, labels=[int(x) for x in dfd_ids])
            plt.show()

        # ----------------------------------------------------------------------------------------------------------------------
        # 6. Filter particles by distance from the particle-collection average.

        # filter particles with true_dz_errors > 0.1 / measurement depth
        apply_bias_filter = False
        if apply_bias_filter:
            pass

        # filter particles with errors > 0.1 / measurement depth
        apply_barnkob_filter = False
        if apply_barnkob_filter:
            dfd = dfd[dfd['error_dz_mean'].abs() < barnkob_error_filter]

        # dfd = dfd[dfd['error_dz_mean'].abs() < dfd['dz_mean'].std() * std_filter_dz]

        # RE-CALCULATE distance from group average after filtering
        if apply_bias_filter or apply_barnkob_filter:
            dfd['avg_dz_mean'] = 0
            dfd['true_dz_mean'] = 0
            for i, tdz in zip(dfd.filename.unique(), true_dz):
                # error dz is displacement of every particle from the group mean
                avg_dz_mean = dfd[dfd['filename'] == i].dz_mean.mean()
                dfd['error_dz_mean'][dfd['filename'] == i] = dfd['dz_mean'][dfd['filename'] == i] - avg_dz_mean
                dfd['avg_dz_mean'][dfd['filename'] == i] = avg_dz_mean
                dfd['true_dz_mean'][dfd['filename'] == i] = tdz

        # ----------------------------------------------------------------------------------------------------------------------
        # 7. Inner concatenate all datasets to get only particles that passed filters across all tests.
        final_num_particles = []
        percent_particles = []
        percent_particles_ids = []
        for i in dfd.filename.unique():
            percent_particles_ids.append(i)
            final_num_particles.append(len(dfd[dfd['filename'] == i].dz_mean))
            percent_particles.append(len(dfd[dfd['filename'] == i].dz_mean) / initial_num_particles[i] * 100)
            print('{}/{} ({}%) particles remaining in {}'.format(len(dfd[dfd['filename'] == i].dz_mean),
                                                                 initial_num_particles[i],
                                                                 np.round(len(dfd[dfd['filename'] == i].dz_mean) / initial_num_particles[i] * 100, 1),
                                                                 i))

        # OUTPUT:
        # export per-particle, per-test, per-method particle displacements to Excel
        export_test_results = False
        if export_test_results:
            drop_results_columns = ['f_cm_counts', 'i_cm_counts']
            io.export_df_to_excel(dfd, join(path_results, save_id + '_results'), include_index=True, index_label='index',
                                  filetype='.xlsx', drop_columns=drop_results_columns)

            dfdm = dfd.groupby('filename').mean()
            io.export_df_to_excel(dfdm, join(path_results, save_id + '_mean_results'), include_index=True, index_label='index',
                                  filetype='.xlsx', drop_columns=drop_results_columns)

        # ----------------------------------------------------------------------------------------------------------------------
        # 8. Calculate the mean binned rmse-z for each test.

        save_test_rmse_z = False
        show_test_rmse_z = False
        export_test_rmse_z_results = False
        save_id_test_rmse_z = 'test_rmse_z'

        # create new columns to match bin_local_rmse_z method:
        dfd['z'] = dfd['dz_mean']
        dfd['z_true'] = dfd['true_dz']
        dfd['cm'] = dfd['f_cm_mean']

        # binned rmse_z
        column_to_bin_and_assess = 'true_dz'
        bins = 1
        dficts_final = modify.split_df_and_merge_dficts(dfd, keys=dficts_ids, column_to_split='filename', splits=dficts_ids, round_to_decimal=1)
        dfbicts_final = analyze.calculate_bin_local_rmse_z(dficts_final, column_to_bin_and_assess, bins, cm_filter_temp, z_range=None, round_to_decimal=5, dficts_ground_truth=None)

        # plot rmse_z
        if save_test_rmse_z or show_test_rmse_z:
            labels_test_rmse_z = ['GDPyT', None, None, None, 'GDPT']
            colors = ['#0C5DA5', '#0C5DA5', '#0C5DA5', '#0C5DA5', '#00B945', '#00B945', '#00B945', '#00B945']

            parameter = 'rmse_z'

            fig, ax = plotting.plot_dfbicts_local(dfbicts_final, parameter, h=1, colors=colors, scale=scale_fig_dim)

            # ax.legend(labels_test_rmse_z, loc='upper left', fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0))
            ax.set_ylim(ylim_error)
            ax.set_yticks(yticks_meta)
            ax.set_xticks(xticks_meta)
            plt.tight_layout()
            if save_test_rmse_z:
                plt.savefig(join(path_figs, save_id_test_rmse_z+'_global.png'))
            if show_test_rmse_z:
                plt.show()

        # stack dfbicts into dataframe for plotting later
        dfb_final = modify.stack_dficts_by_key(dfbicts_final, drop_filename=True)

        # export test rmse_z results
        if export_test_rmse_z_results:
            io.export_df_to_excel(dfb_final, join(path_results, save_id_test_rmse_z + '_mean_results'), include_index=False,
                                  index_label=None, filetype='.xlsx', drop_columns=None)

        # ----------------------------------------------------------------------------------------------------------------------
        # 8. Normalize particles by distance from the focal plane.

        group_means = []
        for i in dfd.filename.unique():
            group_means.append(dfd[dfd['filename'] == i].dz_mean.mean())

        # the mean z-displacement for each test (from the data)
        found_positions = np.array(group_means)
        print('Determined displacements: {}'.format(found_positions))

        # the true z-displacement for each test (experimental control)
        true_positions = np.array(true_dz)

        # normalize by the measurement depth
        relative_found_pos = found_positions / h
        relative_true_pos = true_positions / h

        # take the absolute value
        abs_rel_found_pos = np.abs(relative_found_pos)
        abs_rel_true_pos = np.abs(relative_true_pos)

        # ----------------------------------------------------------------------------------------------------------------------
        # 9. Plot mirrored density figure.

        # ensure high contrast colors
        if plot_absolute_displacements:
            positions = np.flip(abs_rel_found_pos)
            true_vlines = abs_rel_true_pos
            plot_dens_dirs = true_dz
            fclr = [cn_gdpyt, cn_gdpyt, cp_gdpyt, cp_gdpyt, cn_spc, cn_spc, cp_spc, cp_spc]
            qlrs = [qn_gdpyt, qn_gdpyt, qp_gdpyt, qp_gdpyt, qn_spc, qn_spc, qp_spc, qp_spc]
            clrs = [mn_gdpyt, mn_gdpyt, mp_gdpyt, mp_gdpyt, mn_spc, mn_spc, mp_spc, mp_spc]
            xlbl = r'$|z/h|$'
            xlim = [xlim_min, xlim_max]
            median_marker.reverse()
        else:
            if plot_violin_position_at == 'true':
                positions = np.flip(true_positions)
                true_vlines = true_positions
                density_widths = 25
                median_marker = [1, 1, 1, 1, 0, 0, 0, 0]  # carets: [9, 9, 8, 8, 8, 8, 9, 9]
                plot_dens_dirs = [-1, -1, -1, -1, 1, 1, 1, 1]
            elif plot_violin_position_at == 'relative_true':
                positions = np.flip(relative_true_pos)
                true_vlines = relative_true_pos
                plot_dens_dirs = [1, 1, -1, -1, -1, -1, 1, 1]
            elif plot_violin_position_at == 'relative_found':
                positions = np.flip(relative_found_pos)
                true_vlines = relative_true_pos
                plot_dens_dirs = [1, 1, -1, -1, -1, -1, 1, 1]
            fclr = [cp_gdpyt, cp_gdpyt, cp_gdpyt, cp_gdpyt, cp_spc, cp_spc, cp_spc, cp_spc]
            qlrs = [qp_gdpyt, qp_gdpyt, qp_gdpyt, qp_gdpyt, qp_spc, qp_spc, qp_spc, qp_spc]
            clrs = [mp_gdpyt, mp_gdpyt, mp_gdpyt, mp_gdpyt, mp_spc, mp_spc, mp_spc, mp_spc]
            xlbl = r'$z/h$'
            xlim = [-xlim_max, xlim_max]

        # reverse the order of the array so SPC are plotted behind GDPyT
        dfs = [dfd[dfd['filename'] == i] for i in np.flip(dfd.filename.unique())]
        plot_dens_dirs.reverse()
        fclr.reverse()
        qlrs.reverse()
        clrs.reverse()

        # ----------------------------------------------------------------------------------------------------------------------
        # ----- scatter + histogram -----
        save_kde_plots = False
        show_kde_plots = False
        save_id_kde = 'kernel_density_estimation'

        for i in range(len(dfs)):
            del fig, ax
            df_temp_id = dfs[i]['filename'].unique()[0]
            df_temp_tdz = dfs[i]['true_dz'].unique()[0]

            fig = plt.figure()
            xscatter = dfs[i]['i_z_mean'].to_numpy()
            yscatter = dfs[i]['f_z_mean'].to_numpy()
            cscatter = dfs[i]['error_dz_mean'].to_numpy()

            fig, ax, ax_histx, ax_histy = plotting.scatter_hist(xscatter, yscatter, fig, color=cscatter, scatter_size=1,
                                                                kde=True)

            ax.set_xlim([np.mean(xscatter) - 10, np.mean(xscatter) + 10])
            ax.set_ylim([np.mean(yscatter) - 10, np.mean(yscatter) + 10])
            ax.set_xlabel(r'$z_{initial} \: (\mu m)$')
            ax.set_ylabel(r'$z_{final} \: (\mu m)$')
            ax_histx.set_title('ID{}: dz={}'.format(df_temp_id, df_temp_tdz))
            plt.tight_layout()
            if save_kde_plots:
                plt.savefig(join(path_figs, save_id_kde + '_id{}_dz{}.png'.format(df_temp_id, df_temp_tdz)))
            if show_kde_plots:
                plt.show()
        # ----------------------------------------------------------------------------------------------------------------------

        scalex, scaley = 1, 1.35
        fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * scalex, size_y_inches * scaley),
                                       gridspec_kw={'height_ratios': [1, 3]})

        for i in range(len(dfs)):
            fig, ax2 = plotting.plot_violin([dfs[i][plot_violin_parameter].to_numpy()], positions=[positions[i]],
                                            density_directions=[plot_dens_dirs[i]],
                                            axis_quartiles=1, widths=density_widths, bw_method=None,
                                            facecolors=[fclr[i]], edgecolor=eclr, clrs=[clrs[i]], qlrs=[qlrs[i]],
                                            plot_quartile=plot_quartile, plot_whiskers=plot_whiskers,
                                            median_marker=median_marker[i], median_marker_size=median_marker_size,
                                            fig=fig, ax2=ax2)

        if plot_true_dz_vlines:
            for true_disp in np.unique(true_vlines):
                ax2.axvline(x=true_disp, ymin=0, ymax=1, color='black', linestyle=true_linestyle, linewidth=true_linewidth,
                            alpha=true_alpha, zorder=1.5)

        if plot_zeroline:
            ax2.axhline(y=0, xmin=0, xmax=1, color='black', linestyle=true_linestyle, linewidth=true_linewidth,
                        alpha=true_alpha, zorder=1.5)

        if plot_violin_position_at == 'true':
            ax2.set_xlabel(r'$z_{true}\: (\mu m)$')
        else:
            ax2.set_xlabel(r'$|z/h|$')
        if plot_violin_parameter == 'error_dz_mean':
            ax2.set_ylabel(r'$z-z_{avg}\: (\mu m)$')
        elif plot_violin_parameter == 'dz_mean':
            ax2.set_ylabel(r'$z_{measured}\: (\mu m)$')

        ax2.set_xticks(xticks_meta)
        ax2.set_yticks(xticks_meta)

        # UPPER PLOT

        # calculate error / sqrt(# of samples)
        if plot_violin_position_at == 'true':
            positions = np.flip(positions)
            error_z = dfb_final['rmse_z'].to_numpy()
            fclr_ereror = np.flip(fclr)
            fclr_percent = np.flip(fclr)
        else:
            abs_rel_true_pos = np.flip(abs_rel_true_pos)
            error_z = (positions - abs_rel_true_pos)  # / np.sqrt(2)
            fclr_ereror = fclr
            fclr_percent = fclr

        if plot_true_dz_vlines:
            for true_disp in np.unique(true_vlines):
                ax1.axvline(x=true_disp, ymin=0, ymax=1, color='black', linestyle=true_linestyle, linewidth=true_linewidth,
                            alpha=true_alpha, zorder=2)

        ax12 = ax1.twinx()
        for i in range(len(dfs)):
            ax1.scatter(positions[i], np.abs(error_z[i]), c=fclr_ereror[i], s=11, marker=r'$\bullet$', linewidth=0.25,
                        zorder=2.5)
            ax12.scatter(positions[i], percent_particles[i], c=fclr_percent[i], s=13, marker=r'$\diamond$', linewidth=0.25,
                         label=positions[i], zorder=2.5)
            ax12.scatter(positions[i], percent_particles[i], c='white', s=10, marker=r'$\blacklozenge$', linewidth=0.25,
                         zorder=2.0)
        ax1.set_ylabel(r'$\bullet \: \sigma_{z} \: (\mu m)$')
        ax1.set_ylim(ylim_error)
        ax1.set_yticks(yticks_meta)
        ax12.set_ylabel(r'$\diamond \: \phi_{ID}$')
        ax12.set_ylim(ylim_percent_meas)

        # ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1)
        # plt.legend([r'$-z$', '', r'$+z$'], loc='lower left')
        plt.tight_layout()

        if eclr:
            ec = 'Y'
        else:
            ec = 'N'

        if save_violin_plots:
            plt.savefig(join(path_figs, '{}_cm{}_no-filters_{}ec.png'.format(save_id, cm_filter_temp, ec)))
        if show_violin_plots:
            fig.show()

    else:
        # Assess measurement uncertainty using individual particle measurements (test, frame, id)
        # ----------------------------------------------------------------------------------------------------------------------
        # 8. Calculate the mean binned rmse-z for each test.

        save_test_rmse_z = False
        show_test_rmse_z = False
        export_test_rmse_z_results = False
        save_id_test_rmse_z = 'test_rmse_z'

        # binned rmse_z
        column_to_bin_and_assess = 'true_dz'
        bins = 1
        dficts_final = modify.split_df_and_merge_dficts(dff, keys=dficts_ids, column_to_split='filename', splits=dficts_ids,
                                                        round_to_decimal=1)
        dfbicts_final = analyze.calculate_bin_local_rmse_z(dficts_final, column_to_bin_and_assess, bins, cm_filter_temp,
                                                           z_range=None, round_to_decimal=6, dficts_ground_truth=None)

        # plot rmse_z
        if save_test_rmse_z or show_test_rmse_z:
            labels_test_rmse_z = ['GDPyT', None, None, None, 'GDPT']
            colors = ['#0C5DA5', '#0C5DA5', '#0C5DA5', '#0C5DA5', '#00B945', '#00B945', '#00B945', '#00B945']

            parameter = 'rmse_z'

            fig, ax = plotting.plot_dfbicts_local(dfbicts_final, parameter, h=1, colors=colors, scale=scale_fig_dim)

            # ax.legend(labels_test_rmse_z, loc='upper left', fancybox=True, shadow=False, bbox_to_anchor=(1.01, 1.0, 1, 0))
            ax.set_ylim(ylim_error)
            ax.set_yticks(yticks_meta)
            ax.set_xticks(xticks_meta)
            plt.tight_layout()
            if save_test_rmse_z:
                plt.savefig(join(path_figs, save_id_test_rmse_z + '_per-frame_global.png'))
            if show_test_rmse_z:
                plt.show()

        # stack dfbicts into dataframe for plotting later
        dfb_final = modify.stack_dficts_by_key(dfbicts_final, drop_filename=True)

        # export test rmse_z results
        if export_test_rmse_z_results:
            io.export_df_to_excel(dfb_final, join(path_results, save_id_test_rmse_z + '_per-frame_mean_results'), include_index=False,
                                  index_label=None, filetype='.xlsx', drop_columns=None)

        # ----------------------------------------------------------------------------------------------------------------------
        # 8. Normalize particles by distance from the focal plane.

        group_means = []
        group_ids = []
        percent_particles = []
        for i in dff.filename.unique():
            group_means.append(dff[dff['filename'] == i].meas_dz.mean())
            percent_particles.append(np.round(len(dff[dff['filename'] == i].z) / initial_num_f_measurements[i] * 100, 1))
            group_ids.append(i)

        # the mean z-displacement for each test (from the data)
        found_positions = np.array(group_means)
        print('Determined displacements: {}'.format(found_positions))

        # the true z-displacement for each test (experimental control)
        true_positions = np.array(true_dz)

        # normalize by the measurement depth
        relative_found_pos = found_positions / h
        relative_true_pos = true_positions / h

        # take the absolute value
        abs_rel_found_pos = np.abs(relative_found_pos)
        abs_rel_true_pos = np.abs(relative_true_pos)

        # ----------------------------------------------------------------------------------------------------------------------
        # 9. Plot mirrored density figure.
        plot_violin_parameter = 'meas_dz'

        # ensure high contrast colors
        if plot_absolute_displacements:
            positions = np.flip(abs_rel_found_pos)
            true_vlines = abs_rel_true_pos
            plot_dens_dirs = true_dz
            fclr = [cn_gdpyt, cn_gdpyt, cp_gdpyt, cp_gdpyt, cn_spc, cn_spc, cp_spc, cp_spc]
            qlrs = [qn_gdpyt, qn_gdpyt, qp_gdpyt, qp_gdpyt, qn_spc, qn_spc, qp_spc, qp_spc]
            clrs = [mn_gdpyt, mn_gdpyt, mp_gdpyt, mp_gdpyt, mn_spc, mn_spc, mp_spc, mp_spc]
            xlbl = r'$|z/h|$'
            xlim = [xlim_min, xlim_max]
            median_marker.reverse()
        else:
            if plot_violin_position_at == 'true':
                positions = np.flip(true_positions)
                true_vlines = true_positions
                density_widths = 25
                median_marker = [1, 1, 1, 1, 0, 0, 0, 0]  # carets: [9, 9, 8, 8, 8, 8, 9, 9]
                plot_dens_dirs = [-1, -1, -1, -1, 1, 1, 1, 1]
            elif plot_violin_position_at == 'relative_true':
                positions = np.flip(relative_true_pos)
                true_vlines = relative_true_pos
                plot_dens_dirs = [1, 1, -1, -1, -1, -1, 1, 1]
            elif plot_violin_position_at == 'relative_found':
                positions = np.flip(relative_found_pos)
                true_vlines = relative_true_pos
                plot_dens_dirs = [1, 1, -1, -1, -1, -1, 1, 1]
            fclr = [cp_gdpyt, cp_gdpyt, cp_gdpyt, cp_gdpyt, cp_spc, cp_spc, cp_spc, cp_spc]
            qlrs = [qp_gdpyt, qp_gdpyt, qp_gdpyt, qp_gdpyt, qp_spc, qp_spc, qp_spc, qp_spc]
            clrs = [mp_gdpyt, mp_gdpyt, mp_gdpyt, mp_gdpyt, mp_spc, mp_spc, mp_spc, mp_spc]
            xlbl = r'$z/h$'
            xlim = [-xlim_max, xlim_max]

        # reverse the order of the array so SPC are plotted behind GDPyT
        dfs = [dff[dff['filename'] == i] for i in np.flip(dff.filename.unique())]
        plot_dens_dirs.reverse()
        fclr.reverse()
        qlrs.reverse()
        clrs.reverse()

        # ----------------------------------------------------------------------------------------------------------------------
        # ----- scatter + histogram -----
        save_kde_plots = False
        show_kde_plots = False
        if save_kde_plots or show_kde_plots:

            plotx = 'z'
            plotc = 'error_dz'
            save_id_kde = 'per-frame_kernel_density_estimation'
            distance_from_mean = 12.5

            for i in range(len(dfs)):
                df_temp_id = dfs[i]['filename'].unique()[0]
                df_temp_tdz = dfs[i]['true_dz'].unique()[0]

                fig = plt.figure()
                xscatter = dfs[i][plotx].to_numpy()
                yscatter = dfs[i][plot_violin_parameter].to_numpy()
                cscatter = dfs[i][plotc].to_numpy()

                fig, ax, ax_histx, ax_histy = plotting.scatter_hist(xscatter, yscatter, fig, color=cscatter, scatter_size=1,
                                                                    kde=True, distance_from_mean=distance_from_mean)

                ax.set_xlim([np.mean(xscatter) - distance_from_mean, np.mean(xscatter) + distance_from_mean])
                ax.set_ylim([np.mean(yscatter) - distance_from_mean, np.mean(yscatter) + distance_from_mean])
                ax.set_xlabel(r'$z_{measured} \: (\mu m)$')
                ax.set_ylabel(r'$\Delta z_{measured} \: (\mu m)$')
                ax_histx.set_title('ID{}: dz={}'.format(df_temp_id, df_temp_tdz))
                plt.tight_layout()
                if save_kde_plots:
                    plt.savefig(join(path_figs, save_id_kde + '_id{}_dz{}.png'.format(df_temp_id, df_temp_tdz)))
                if show_kde_plots:
                    plt.show()

        # ----------------------------------------------------------------------------------------------------------------------
        # Mirrored violin plot
        save_violin_plots = True
        show_violin_plots = True
        bw_method = 0.125

        if save_violin_plots or show_violin_plots:

            scalex, scaley = 1, 1.35
            fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * scalex, size_y_inches * scaley),
                                           gridspec_kw={'height_ratios': [1, 3]})

            for i in range(len(dfs)):
                fig, ax2 = plotting.plot_violin([dfs[i][plot_violin_parameter].to_numpy()], positions=[positions[i]],
                                                density_directions=[plot_dens_dirs[i]],
                                                axis_quartiles=1, widths=density_widths, bw_method=bw_method,
                                                facecolors=[fclr[i]], edgecolor=eclr, clrs=[clrs[i]], qlrs=[qlrs[i]],
                                                plot_quartile=plot_quartile, plot_whiskers=plot_whiskers,
                                                median_marker=median_marker[i], median_marker_size=median_marker_size,
                                                fig=fig, ax2=ax2)

            if plot_true_dz_vlines:
                for true_disp in np.unique(true_vlines):
                    ax2.axvline(x=true_disp, ymin=0, ymax=1, color='black', linestyle=true_linestyle,
                                linewidth=true_linewidth,
                                alpha=true_alpha, zorder=1.5)

            if plot_zeroline:
                ax2.axhline(y=0, xmin=0, xmax=1, color='black', linestyle=true_linestyle, linewidth=true_linewidth,
                            alpha=true_alpha, zorder=1.5)

            if plot_violin_position_at == 'true':
                ax2.set_xlabel(r'$z_{true}\: (\mu m)$')
            else:
                ax2.set_xlabel(r'$|z/h|$')
            if plot_violin_parameter == 'error_dz_mean':
                ax2.set_ylabel(r'$z-z_{avg}\: (\mu m)$')
            elif plot_violin_parameter == 'meas_dz':
                ax2.set_ylabel(r'$z_{measured}\: (\mu m)$')

            ax2.set_xticks(xticks_meta)
            ax2.set_yticks(xticks_meta)

            # UPPER PLOT

            # calculate error / sqrt(# of samples)
            if plot_violin_position_at == 'true':
                positions = np.flip(positions)
                error_z = dfb_final['rmse_z'].to_numpy()
                fclr_ereror = np.flip(fclr)
                fclr_percent = np.flip(fclr)
            else:
                abs_rel_true_pos = np.flip(abs_rel_true_pos)
                error_z = (positions - abs_rel_true_pos)  # / np.sqrt(2)
                fclr_ereror = fclr
                fclr_percent = fclr

            if plot_true_dz_vlines:
                for true_disp in np.unique(true_vlines):
                    ax1.axvline(x=true_disp, ymin=0, ymax=1, color='black', linestyle=true_linestyle,
                                linewidth=true_linewidth,
                                alpha=true_alpha, zorder=2)

            ax12 = ax1.twinx()
            for i in range(len(dfs)):
                ax1.scatter(positions[i], np.abs(error_z[i]), c=fclr_ereror[i], s=11, marker=r'$\bullet$', linewidth=0.25,
                            zorder=2.5)
                ax12.scatter(positions[i], percent_particles[i], c=fclr_percent[i], s=13, marker=r'$\diamond$',
                             linewidth=0.25,
                             label=positions[i], zorder=2.5)
                ax12.scatter(positions[i], percent_particles[i], c='white', s=10, marker=r'$\blacklozenge$', linewidth=0.25,
                             zorder=2.0)
            ax1.set_ylabel(r'$\bullet \: \sigma_{z} \: (\mu m)$')
            ax1.set_ylim(ylim_error)
            ax1.set_yticks(yticks_meta)
            ax12.set_ylabel(r'$\diamond \: \phi_{ID} \: (\%)$')
            ax12.set_ylim(ylim_percent_meas)

            # ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1)
            # plt.legend([r'$-z$', '', r'$+z$'], loc='lower left')
            plt.tight_layout()

            if eclr:
                ec = 'Y'
            else:
                ec = 'N'

            if save_violin_plots:
                plt.savefig(join(path_figs, 'per-frame_{}_cm{}_bw{}.png'.format(save_id, cm_filter_temp, bw_method)))
            if show_violin_plots:
                fig.show()



    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# calculate mean values (not sure what this is used for)
"""
initial_vals = analyze.calculate_mean_value(dficts_i, output_var='z', input_var='frame', span=(0, 50))
finals_vals = analyze.calculate_mean_value(dficts_f, output_var='z', input_var='frame', span=(50, 100))

z0 = np.mean(initial_vals[:, 1])
zf_std = np.mean(finals_vals[:, 2]) * 2

dfd = pd.DataFrame(finals_vals, columns=['id', 'dz', 'z_std'])
dfd.loc[:, 'dz'] = dfd.loc[:, 'dz'] - z0
dfd['z_true'] = np.array([-45, -30, 30, 45])
"""

# ----------------------------------------------------------------------------------------------------------------------
# 3D scatter plot in-focus calibration coordinates
"""
# import calibration coordinates to correct for in-focus position

# read calibration in-focus coords

# file path
fp_in_focus = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/calibration/calib_in-focus_coords_z-micrometer-v2.xlsx'
# read excel to disk
dfcf = io.read_excel(path_name=fp_in_focus, filetype='.xlsx')

ids = dfcf.id.unique()
z_fs = dfcf.z_f.to_numpy()

# create the mapping dictionary
mapping_dict = {ids[i]: z_fs[i] for i in range(len(ids))}

# read test coords

# file path
fp_test = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/test/test_coords_id1_45micron_step_towards.xlsx'
# read excel to disk
df = io.read_excel(path_name=fp_test, filetype='.xlsx')

# insert the mapped values
df['z_f'] = df['id'].copy()
df['z_f'] = df['z_f'].map(mapping_dict)

# correct for initial displacement
df['zc'] = df['z_f'] - df['z']

# plot 3D scatter of all particle coordinates
dfi = df.loc[df['frame'] < 49].copy()
dfg = dfi.groupby('id').mean()

plt.style.use(['science', 'ieee', 'scatter'])
fig, ax = plotting.plot_scatter_3d(dfg, fig=None, ax=None, elev=20, azim=-40, color='tab:blue', alpha=1)
fig, ax = plotting.plot_scatter_3d(df=[dfg.x, dfg.y, dfg.z_f], fig=fig, ax=ax, color='tab:red', elev=20, azim=-40, alpha=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.show()

# get final displacement
dff = df.loc[df['frame'] > 52].copy()
"""
# ----------------------------------------------------------------------------------------------------------------------

j = 1