# plot 2/7/22 membrane characterizations

# imports
from os.path import join, exists
from os import makedirs
import numpy as np
from skimage import exposure
from skimage.io import imread, imsave
import pandas as pd
from scipy.optimize import curve_fit, minimize

import matplotlib.pyplot as plt
from matplotlib import cm

import filter
import analyze
from correction import correct
from utils import io, modify, details, boundary, functions, fit, plotting, verify, bin

# description of code
"""
Purpose of this code:
    1. 

Process of this code:
    1. Setup
        1.1 Setup file paths.
        1.2 Setup plotting style.
        1.3 Setup filters. 
        1.4 Read all test_coords.
        1.5 Filter particles by c_m (should always be 0.5)
    2. 
"""

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

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.07.22_membrane_characterization'
path_results = join(base_dir, 'analysis')
path_figs = join(base_dir, 'figs')

# experimental details
microns_per_pixel = 1.6
padding_during_gdpyt_test_calib = 10
padding_during_gdpyt_in_focus_calib = 30

# ----------------------------------------------------------------------------------------------------------------------
# boundary
path_image_to_mask = join(base_dir, 'images/calibration/calib_55.tif')
xc, yc, r_edge = 498, 253, 500
circle_coords = [xc, yc, r_edge]
acceptance_boundary_pixels = 5

path_mask_boundary = join(path_results, 'boundary')
path_figs_boundary = path_mask_boundary
fp_mask_boundary = join(path_mask_boundary, 'mask_boundary')
fp_mask_edge = join(path_mask_boundary, 'mask_edge')

save_mask_boundary = False
save_boundary_images = False
show_boundary_images = False

if not exists(path_mask_boundary):
    makedirs(path_mask_boundary)

mask_dict = {
    'path_mask_boundary': path_mask_boundary,
    'path_image_to_mask': path_image_to_mask,
    'padding_during_gdpyt': padding_during_gdpyt_in_focus_calib,
    'circle_coords': circle_coords,
    'acceptance_boundary_pixels': acceptance_boundary_pixels,
    'save_mask_boundary': save_mask_boundary,
    'save_boundary_images': save_boundary_images,
    'show_boundary_images': show_boundary_images,
}

# ----------------------------------------------------------------------------------------------------------------------
# in-focus calibration correction

path_calib_coords = join(base_dir, 'test_coords/meta/calib_coords')
calib_sort_strings = ['calib_', '_coords_']
filetype = '.xlsx'
save_path = join(base_dir, 'analysis/correction')
save_plots = False

io_dict = {
    'path_calib_coords': path_calib_coords,
    'calib_sort_string': calib_sort_strings,
    'filetype': filetype,
    'save_plots': save_plots,
    'save_path': save_path,
}

exp_dict = {
    'microns_per_pixel': microns_per_pixel,
    'calibration_direction': 'towards',
}

# experiment
path_name = join(base_dir, 'test_coords/tests/pos')
save_id = 'exp_memb'
# meta-assessment
path_meta = join(base_dir, 'test_coords/meta/test_coords_pos_odds')
path_meta_calib = join(base_dir, 'test_coords/meta/calib_coords')
meta_sort_strings = ['meta_id', '_coords_']
calib_sort_strings = ['calib_', '_coords_']
save_id_meta = 'meta-assessment'

# setup I/O
sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'
drop_columns = ['stack_id', 'z_true', 'max_sim', 'error']
results_drop_columns = ['frame', 'id', 'stack_id', 'z_true', 'z', 'x', 'y', 'x_true', 'y_true']

# setup figures
scale_fig_dim = [1, 1]
scale_fig_dim_legend_outside = [1.3, 1]
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
requires_correction = False

if requires_correction:

    new_auto_correction = False
    if new_auto_correction:

        # mask boundary
        mask_dict = boundary.compute_boundary_mask(mask_dict)

        # calculate per particle corrections
        correction_results = correct.perform_correction(io_dict, mask_dict, exp_dict)
        df_ppc = correction_results['per_particle_corrections']
        z_correction_name = correction_results['z_correction_name']

    else:
        # read per-particle correction
        df_ppc = pd.read_excel(join(io_dict['save_path'], 'corrections', 'per_particle_corrections.xlsx'),
                               usecols=[1, 2, 3, 4])
        # read correction results
        correction_results = correct.read_correction_params(
            join(io_dict['save_path'], 'results', 'correction_results.xlsx'))

    # parse boundary particle IDs (CAREFUL: NEEDS MANUAL TUNING)
    boundary_pids_str = correction_results['boundary_pids'].split(' ')
    boundary_pids = [int(i) for i in boundary_pids_str if i.isdigit()]
    boundary_pids.append(34)
    # parse interior particle IDs (CAREFUL: NEEDS MANUAL TUNING)
    interior_pids_str = correction_results['interior_pids'].split(' ')
    interior_pids = [int(i) for i in interior_pids_str if i.isdigit()]
    messed_strings = [42, 64, 82, 100, 118, 136, 154, 172, 190, 208, 226, 244, 262]
    for m in messed_strings:
        interior_pids.append(m)

else:
    df_ppc = None
    boundary_pids = []
    interior_pids = []

# ----------------------------------------------------------------------------------------------------------------------
# Analyze meta-assessment
analyze_meta = False

if analyze_meta:
    # setup
    per_particle_correction = False
    save_meta_plots, show_meta_plots = False, True
    save_calib_curve_plots, show_calib_curve_plots = False, True
    export_meta_results = False

    # test coords
    dficts = io.read_files('df', path_meta, meta_sort_strings, filetype, startswith=meta_sort_strings[0])
    labels = list(dficts.keys())

    # test details
    h_meas_vol = 30  # 141
    h_meas_vols = [30, 59]  # [1, h_meas_vol]

    # ----------------------------------------------------------------------------------------------------------------------
    # correct particle coordinates
    if per_particle_correction:
        dficts_corr = {}
        for name, df in dficts.items():
            dfc = correct.correct_from_mapping(df_to_correct=df, df_particle_corrections=df_ppc, z_params=['z', 'z_true'])
            dficts_corr.update({name: df})

        dficts_orig = dficts.copy()
        del dficts
        dficts = dficts_corr

    # ----------------------------------------------------------------------------------------------------------------------
    # Apply filters to remove outliers
    cm_filter = 0.95
    apply_barnkob_filter = False
    filter_pids_from_all_frames = False
    filter_num_frames = True
    min_num_frames = 15

    # need to shift z_true
    dficts = filter.dficts_filter(dficts, keys=['z_true'], values=[[28.99, 59.01]], operations=['between'],
                                  copy=True, only_keys=None, return_filtered=False)

    if any([bool(cm_filter), apply_barnkob_filter, filter_pids_from_all_frames, filter_num_frames]):

        cficts = None
        cficts_details = None

        # filter particles with c_m < __
        if cm_filter > 0.01:
            dficts = filter.dficts_filter(dficts, keys=['cm'], values=[cm_filter], operations=['greaterthan'])

        # filter particles with errors > h/10 (single rows)
        if apply_barnkob_filter:
            barnkob_error_filter = 0.1  # filter error < h/10 used by Barnkob and Rossi in 'A fast robust algorithm...'
            meas_vols_inverse = 1 / h_meas_vol
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
        if filter_num_frames:

            if cficts is not None:
                for name, df in cficts.items():
                    dfc = df.groupby('id').count()
                    pids_filter_num_frames = dfc[dfc['frame'] < min_num_frames].index.to_numpy()
                    cficts = filter.dficts_filter(cficts, keys=['id'], values=[pids_filter_num_frames],
                                                  operations=['notin'],
                                                  only_keys=[name])
                    dficts = filter.dficts_filter(dficts, keys=['id'], values=[pids_filter_num_frames],
                                                  operations=['notin'],
                                                  only_keys=[name])
            else:
                for name, df in dficts.items():
                    dfc = df.groupby('id').count()
                    pids_filter_num_frames = dfc[dfc['frame'] < min_num_frames].index.to_numpy()
                    dficts = filter.dficts_filter(dficts, keys=['id'], values=[pids_filter_num_frames],
                                                  operations=['notin'],
                                                  only_keys=[name])

    # ----------------------------------------------------------------------------------------------------------------------
    # calibration curve

    # setup
    xparam = 'z_true'
    yparam = 'z'

    if save_calib_curve_plots or show_calib_curve_plots:

        # re-sort dficts so GDPyT is plotted on top
        dficts = modify.dficts_sort(dficts)
        colors_calib_curve = ['#0C5DA5']
        id_counter = list(dficts.keys())

        for id in id_counter:
            fig, ax = plt.subplots()
            df_temp = dficts[id]
            ax.scatter(df_temp[xparam], df_temp[yparam], s=0.5, color=colors_calib_curve[0], label='GDPyT', zorder=3.5)

            ax.set_ylabel(r'$z_{measured}\: (\mu m)$')
            ax.set_xlabel(r'$z_{true}\: (\mu m)$')
            ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.90, 1, 0), markerscale=2, borderpad=0.1,
                      handletextpad=0.05, borderaxespad=0.1)
            plt.tight_layout()
            if save_calib_curve_plots:
                plt.savefig(join(path_figs, 'meta', save_id_meta + '_z-calibration_cm={}.png'.format(cm_filter)))
            if show_calib_curve_plots:
                plt.show()

    # ----------------------------------------------------------------------------------------------------------------------
    # calculate z-uncertainties: bin by number of bins
    # binning
    num_bins = 31
    column_to_bin_and_assess = 'z_true'
    round_z_to_decimal = 6

    # calculate local rmse_z
    dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins=num_bins, min_cm=cm_filter,
                                                 z_range=None, round_to_decimal=round_z_to_decimal,
                                                 dficts_ground_truth=None)

    # plots binned z-uncertainty
    if save_meta_plots or show_meta_plots:
        ylabel_metas = [r'$\sigma_{z}\: (\mu m)$', r'$\sigma_{z}/h$']
        label_dict = {key: {'label': lbl} for (key, lbl) in zip(list(dfbicts.keys()), ['GDPyT', 'GDPyT', 'GDPyT', 'GDPyT', 'GDPyT'])}
        save_id_metas = ['', '_norm']

        # z-uncertainty (microns)
        parameter = 'rmse_z'
        for hs, ylabels, saveids in zip(h_meas_vols, ylabel_metas, save_id_metas):
            fig, ax = plotting.plot_dfbicts_local(dfbicts, parameter, h=hs, scale=scale_fig_dim, label_dict=label_dict)

            ax.set_ylabel(ylabels)
            ax.set_xlabel(r'$z_{true} \: (\mu m)$')  #
            ax.legend(loc='upper left')
            plt.tight_layout()
            if save_meta_plots:
                plt.savefig(join(path_figs, 'meta',
                                 save_id_meta + '_z_uncertainty{}_{}bins_cm={}.png'.format(saveids, num_bins,
                                                                                           cm_filter)))
            if show_meta_plots:
                plt.show()

        if export_meta_results:
            # export the binned results
            dfb_export = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
            io.export_df_to_excel(dfb_export, path_name=join(path_results,
                                                             save_id_meta + '_{}bins_measurement_results_cm={}'.format(
                                                                 num_bins, cm_filter)),
                                  include_index=True, index_label='bin_z', filetype='.xlsx',
                                  drop_columns=results_drop_columns[:-2])

            # calculate local rmse_z with bin == 1
            dfmicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins=1,
                                                         min_cm=cm_filter,
                                                         z_range=None, round_to_decimal=round_z_to_decimal,
                                                         dficts_ground_truth=None)
            # calculate mean measurement results and export to excel
            dfm = analyze.calculate_bin_measurement_results(dfmicts)
            io.export_df_to_excel(dfm, path_name=join(path_results,
                                                      save_id_meta + '_mean_measurement_results_cm={}'.format(
                                                          cm_filter)),
                                  include_index=True, index_label='test_id', filetype='.xlsx',
                                  drop_columns=results_drop_columns[:-2])

# ----------------------------------------------------------------------------------------------------------------------
# Analyze membrane deflections

# parameters
frames_per_second = 24.444

# file paths
test_dir = join(base_dir, 'test_coords/tests/pos')
save_dir = join(base_dir, 'analysis')
save_verify = join(save_dir, 'tests/verify')
save_static = join(save_dir, 'eval_static')
save_bin_deflections = join(save_dir, 'tests/bin-deflections')
save_plots_tests, show_plots_tests = False, True

# test details
fps = ['test_id0_coords_0.0mm_pos.xlsx', 'test_id1_coords_0.25mm_pos.xlsx',  # 'test_id12_coords_second_0.25mm_pos.xlsx',
       'test_id2_coords_2.25mm_pos.xlsx', 'test_id4_coords_4.25mm_pos.xlsx', 'test_id6_coords_6.25mm_pos.xlsx',
       'test_id8_coords_8.25mm_pos.xlsx', 'test_id10_coords_10.25mm_pos.xlsx', 'test_id11_coords_11.25mm_pos.xlsx']  # ,
pressure = [0, 0.25, 2.25, 4.25, 6.25, 8.25, 10.25, 11.25]
dfids = [0, 0.25, 2.25, 4.25, 6.25, 8.25, 10.25, 11.25]

# filters
calib_lower_limit, calib_upper_limit = 1, 150
filter_cm = 0.01
filter_num_frames = 10
filter_z_std = 1

# read, filter, and process test results
dfcs = {}  # per-frame dataframes
dfgs = {}  # grouped dataframes
dfgbs = {}  # grouped boundary dataframes
dfgis = {}  # grouped interior dataframes

# toggles
analyze_test_data = True
perform_correction = False

if analyze_test_data:

    # ----------------------------------- PER-FRAME ANALYSIS
    for fp, _id in zip(fps, dfids):
        # read dataframe
        fp = join(test_dir, fp)
        df = pd.read_excel(fp)

        # filter
        df = df[df['z'] < calib_upper_limit - 0.1]   # NOTE: this calibration limit filter is important but why?
        df = df[df['z'] > calib_lower_limit + 0.1]
        df = df[df['cm'] > filter_cm]

        # perform per-particle correction
        if perform_correction:
            # uniformize particle ID's
            df, mapping_dict, pids_not_mapped = correct.correct_nonuniform_particle_ids_with_padding(
                baseline=df_ppc,
                coords=df,
                baseline_padding=padding_during_gdpyt_in_focus_calib,
                coords_padding=padding_during_gdpyt_test_calib,
                threshold=5,
                save_path=save_verify,
                save_id=_id
            )

            # per-particle, in-focus correction
            dfc = correct.correct_from_mapping(df_to_correct=df, df_particle_corrections=df_ppc, z_params=['z', 'z_true'])

            # flip z-dir. so membrane deflection is in +z dir.
            dfc['z_corr'] = dfc['z_corr'] * -1

            # setup post-processing
            drop_columns_stats = ['frame', 'stack_id', 'z_true', 'z_true_corr', 'max_sim', 'error']

        else:
            dfc = df
            drop_columns_stats = ['frame', 'stack_id', 'z_true', 'max_sim', 'error']

        # scale units
        dfc['x'] = dfc['x'] * microns_per_pixel
        dfc['y'] = dfc['y'] * microns_per_pixel
        dfcs.update({_id: dfc})

        # ----------------------------------- PER-PARTICLE-ID ANALYSIS

        dfg = modify.groupby_stats(df=dfc, group_by='id', drop_columns=drop_columns_stats)

        # reset index to maintain 'id' column
        dfg = dfg.reset_index()

        # filter - number of frames
        dfg = dfg[dfg['z_counts'] > filter_num_frames]

        # filter - z standard deviation (note: std is TWO (2) standard deviations)
        dfg = dfg[dfg['z_std'] < filter_z_std]

        # save
        dfgs.update({_id: dfg})

        # filter - boundary & interior particles
        if len(boundary_pids) > 1:
            dfgb = dfg[dfg.id.isin(boundary_pids)]
            dfgbs.update({_id: dfgb})
        if len(interior_pids) > 1:
            dfgi = dfg[dfg.id.isin(interior_pids)]
            dfgis.update({_id: dfgi})

    # ----------------------------------- PER-PARTICLE-ID ANALYSIS ACROSS DIFFERENT TESTS
    dfgts = modify.stack_dficts_by_key(dfgs, drop_filename=False)

    pids = dfgts.id.unique()

    tids = len(dfgts.filename.unique())
    clrs = np.arange(0, tids)

    for pid in pids:

        dfpid = dfgts[dfgts['id'] == pid]
        dfpid = dfpid.sort_values('filename')
        lbls = list(dfpid.filename.unique())

        dfpid['x0'] = dfpid[dfpid['filename'] == 0.0].x
        dfpid['y0'] = dfpid[dfpid['filename'] == 0.0].y
        dfpid['cm0'] = dfpid[dfpid['filename'] == 0.0].cm

        dfpid['dx'] = dfpid.x - dfpid.x0
        dfpid['dy'] = dfpid.y - dfpid.y0
        dfpid['dcm'] = dfpid.cm - dfpid.cm0

        if np.max([dfpid.dx.max(), dfpid.dy.max()]) > 1.5 or np.abs(dfpid.dcm.max()) > 0.55:

            fig, [ax1, ax2] = plt.subplots(nrows=2, figsize=(size_x_inches, size_y_inches * 2))

            ax1.errorbar(dfpid.filename, dfpid.cm, yerr=dfpid.cm_std, fmt='o', color='gray', ms=3, elinewidth=2, capsize=4, alpha=0.5)
            ax1.scatter(dfpid.filename, dfpid.cm, c=dfpid.filename)
            ax1.set_xlabel(r'$ID_{test}$')
            ax1.set_ylabel(r'$c_{m}$')
            ax1.set_title(r'$p_{ID} = $' + '{}'.format(pid))

            ax2.scatter(dfpid.dx, dfpid.dy, c=dfpid.filename, s=10)

            ax2.set_xlabel(r'$x \: (\mu m)$')
            ax2.set_xlim([-7, 7])
            ax2.set_ylabel(r'$y \: (\mu m)$')
            ax2.set_ylim([-7, 7])
            ax2.grid(alpha=0.25)

            plt.tight_layout()
            plt.show()
            #plt.savefig(join(save_static, 'eval_static_cm-x-y_pid{}.png'.format(pid)))
            plt.close()

    j = 1
    # -----------------------------------

    raise ValueError('ha')



    # ----------------------------------- PER-TEST ANALYSIS

    # all particles
    dfgts = modify.stack_dficts_by_key(dfgs, drop_filename=False)
    dfgts = dfgts.groupby('filename').mean()
    dfgts = dfgts.reset_index()

    if df_ppc is not None:
        dfv = verify.verify_particle_ids_across_sets(dfgts, baseline=df_ppc, save_plots=False, save_results=False, save_path=save_verify)

    if len(boundary_pids) > 1:
        dfcts = modify.stack_dficts_by_key(dfcs, drop_filename=False)
        dfcts = dfcts[dfcts.id.isin(boundary_pids)]
        drop_columns_stats = ['frame', 'id', 'stack_id', 'z_true', 'z_true_corr', 'max_sim', 'error']
        dfcgts = modify.groupby_stats(df=dfcts, group_by='filename', drop_columns=drop_columns_stats)
        dfcgts = dfcgts.reset_index()

        dfgbts_full = modify.stack_dficts_by_key(dfgbs, drop_filename=False)
        dfgbts = dfgbts_full.groupby('filename').mean()
        dfgbts = dfgbts.reset_index()

    if len(interior_pids) > 1:
        dfgits_full = modify.stack_dficts_by_key(dfgis, drop_filename=False)
        dfgits = dfgits_full.groupby('filename').mean()
        dfgits = dfgits.reset_index()


    # ----------------------------------- EXPORT RESULTS
    # setup
    save_results_tests = False

    if save_results_tests:

        save_path_compareb_tests = join(save_dir, 'tests', 'compare-boundary-tests')
        save_path_compareb_tests_particles = join(save_dir, 'tests', 'compare-boundary-tests-particles')
        save_path_compareb_particles_per_test = join(save_dir, 'tests', 'compare-boundary-particles-per-test')

        save_path_comparei_particles_per_test = join(save_dir, 'tests', 'compare-interior-particles-per-test')
        save_path_fit_deflections = join(save_dir, 'tests', 'fit-deflections')

        for name, df in dfgis.items():
            df.to_excel(join(save_path_comparei_particles_per_test, 'df_id{}.xlsx'.format(name)))

    # ----------------------------------- PLOTTING FUNCTIONS


    def plot_dist_errorbars(df, xparam, yparam, errparam, fig=None, ax=None, style=None):
        if style == 'scatter':
            plt.style.use(['science', 'scatter'])
        elif style == 'muted':
            plt.style.use(['science', 'muted'])
        else:
            plt.style.use(['science', 'ieee', 'std-colors'])

        if fig is None:
            fig, ax = plt.subplots()

        ax.errorbar(df[xparam], df[yparam], yerr=df[errparam], fmt='o', ms=1, elinewidth=1, capsize=2, alpha=0.75)

        return fig, ax


    def plot_dist_errorbars_sum_uncertainty(df, xparam, yparam, errparam, fig=None, ax=None, style=None):
        """
        Colors:
            Blue: #0C5DA5
            Green: #00B945
            Red: #FF9500
            Orange: #FF2C00
            Light Blue: #7BC8F6

        :param df:
        :param xparam:
        :param yparam:
        :param errparam:
        :param fig:
        :param ax:
        :param style:
        :return:
        """

        if fig is None:
            fig, ax = plt.subplots()

        dfmean = df.groupby(xparam).mean()
        dfstd = df.groupby(xparam).std() * 2
        sum_uncertainty = dfmean[errparam] + dfstd[yparam]
        yparam_mean = dfmean[yparam].mean()
        yparam_std = dfmean[yparam].std() * 2
        yparam_mean_std = dfstd[yparam].mean()

        ax.errorbar(dfmean.index, dfmean[yparam], yerr=sum_uncertainty, fmt='o', color='#7BC8F6', ms=1, elinewidth=2,
                    capsize=3, alpha=0.75)
        ax.errorbar(dfmean.index, dfmean[yparam], yerr=dfstd[yparam], fmt='o', color='#0C5DA5', ms=1, elinewidth=1,
                    capsize=2, alpha=1)

        ax.set_title('{} = {} +/- {} '.format(yparam, np.round(yparam_mean, 1), np.round(yparam_std + yparam_mean_std, 2))
                     + r'$\mu m$')

        return fig, ax


    def multi_scatter(dfg, save_id_tests):
        fig = plt.figure(figsize=(6.5, 5))
        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(dfg.x, dfg.y, dfg.z_corr, c=dfg.z_corr, s=4, vmin=-5, vmax=25)
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (\mu m)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (\mu m)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (\mu m)$')
                ax.set_ylabel(r'$y \: (\mu m)$')
                ax.get_zaxis().set_ticklabels([])
        plt.suptitle("Test ID {}".format(_id), y=0.875)
        plt.subplots_adjust(hspace=-0.1, wspace=0.15)
        if save_plots_tests:
            plt.savefig(join(save_dir, 'tests/test_{}_id{}_3D_scatter_multi-view.png'.format(save_id_tests, _id)))
            plt.close()
        if show_plots_tests:
            plt.show()


    # ----------------------------------- PLOTTING: BOUNDARY ORIENTATION
    plot_orientation = False
    if plot_orientation:

        # ----------------------------------- PLOTTING: PER-TEST ANALYSIS OF BOUNDARY ORIENTATION

        # read tilt correction parameters
        path_correction_params = join(save_path, 'corrections', 'correction_params.xlsx')
        correction_params = correct.read_correction_params(path_correction_params, parse_popt=True)
        popt = correction_params['popt']
        z_param, z_corr_tilt_name = 'z_corr', 'z_corr_tilt'

        # plot 2D scatter along x-y axes: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
        for name, df in dfgbs.items():
            # Calculate the per-particle correction for all particles
            dfc = correct.correct_z_by_fit_function(df,
                                                    fit_func=functions.calculate_z_of_3d_plane,
                                                    popt=popt,
                                                    x_param='x',
                                                    y_param='y',
                                                    z_param=z_param,
                                                    z_corr_name=z_corr_tilt_name)

            # tilt axis measured from in-focus particles
            x_line = np.linspace(df.x.min(), df.x.max())
            y_line = np.linspace(df.y.min(), df.y.max())
            zx_line = functions.calculate_z_of_3d_plane(x=x_line, y=np.zeros_like(x_line), popt=popt)
            zy_line = functions.calculate_z_of_3d_plane(x=np.zeros_like(y_line), y=y_line, popt=popt)
            zz_line = functions.calculate_z_of_3d_plane(x=x_line, y=y_line, popt=popt)

            # correct tilt plane for z-offset
            zx_line = zx_line + correction_params['d']
            zy_line = zy_line + correction_params['d']
            zz_line = zz_line + correction_params['d']

            # fit line to boundary particles
            popt_z_corr, pcov1, ff1 = fit.fit(df.y, df[z_param], fit_function=functions.line)
            popt_z_corr_tilt, pcov2, ff2 = fit.fit(dfc.y, dfc[z_corr_tilt_name], fit_function=functions.line)

            fig, [ax1, ax2] = plt.subplots(nrows=2, figsize=(3.5, 3.3))
            # ax1.scatter(df.x, df.z, color='black', s=2, marker='.', label='z')
            ax1.scatter(df.x, df[z_param], color='cornflowerblue', s=2, label=r'$z_f$')
            ax1.scatter(dfc.x, dfc[z_corr_tilt_name], color='red', s=4, marker='d', label=r'$z_{corrected}$')
            # ax1.plot(x_line, zx_line, color='gray', alpha=0.25, linestyle='-.', label=r'$z_{plane, x}$')
            # ax1.plot(x_line, zz_line, color='gray', alpha=0.5, linestyle='--', label=r'$z_{plane, xy}$')
            ax1.set_xlabel('x')
            ax1.set_ylabel('z')
            ax1.set_title(
                'Tilt correction demonstration on test ID{} boundary particles {} \n (not used)'.format(name, z_param),
                fontsize=6)
            ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1))
            # data points
            # ax2.scatter(df.y, df.z, color='black', s=2, marker='.')
            ax2.scatter(df.y, df[z_param], color='cornflowerblue', s=2)
            ax2.scatter(dfc.y, dfc[z_corr_tilt_name], color='red', s=4, marker='d')
            # tilt planes
            # ax2.plot(y_line, zy_line, color='gray', alpha=0.25, linestyle=':', label=r'$z_{plane, y}$')
            # ax2.plot(y_line, zz_line, color='gray', alpha=0.5, linestyle='--')
            # fitted lines
            ax2.plot(df.y, functions.line(df.y, popt_z_corr[0], popt_z_corr[1]),
                     color='tab:blue', alpha=0.5, linestyle='--', label=r'$fit_{z_{corr}}$')
            ax2.plot(dfc.y, functions.line(dfc.y, popt_z_corr_tilt[0], popt_z_corr_tilt[1]),
                     color='tab:red', alpha=0.5, linestyle='--', label=r'$fit_{z_{corr \: tilt}}$')
            ax2.set_xlabel('y')
            ax2.set_ylabel('z')
            ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1))
            plt.tight_layout(h_pad=0.125)
            plt.savefig(join(save_dir, 'tests', 'demo_tilt_correction_on_boundary_{}_2d_xy_id{}.png'.format(z_param, name)))
            plt.show()

    # ----------------------------------- PLOTTING: BOUNDARY POINTS
    plot_boundaries = False
    if plot_boundaries:

        # plot z per-particle
        pids = dfgbts_full.id.unique()
        for pid in pids:
            dfpid = dfgbts_full[dfgbts_full['id'] == pid]

            fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
            ax1.plot(dfpid.filename, dfpid.z, color='#0C5DA5', alpha=0.25)
            ax1.errorbar(dfpid.filename, dfpid.z, yerr=dfpid.z_std, fmt='o', ms=1, elinewidth=1, capsize=2, alpha=1, color='#0C5DA5')
            ax1.set_ylabel(r'$z_{raw} \: (\mu m)$')
            ax2.plot(dfpid.filename, dfpid.z_corr, color='#0C5DA5', alpha=0.25)
            ax2.errorbar(dfpid.filename, dfpid.z_corr, yerr=dfpid.z_corr_std, fmt='o', ms=1, elinewidth=1, capsize=2, alpha=1, color='#0C5DA5')
            ax2.set_ylabel(r'$z_{corrected} \: (\mu m)$')
            ax2.set_xlabel(r'$H \: (mm)$')
            plt.tight_layout()
            if save_plots_tests:
                plt.savefig(join(save_path_compareb_tests_particles, 'errorbars_per-particle_id{}_z_corr.png'.format(pid)))
            if show_plots_tests:
                plt.show()
            plt.close()


        fig, ax = plot_dist_errorbars_sum_uncertainty(dfgbts_full, xparam='id', yparam='z_corr', errparam='z_corr_std', fig=None, ax=None, style=None)
        ax.set_xlabel(r'$p_{ID}$')
        ax.set_ylabel(r'$z_{corrected} \: (\mu m)$')
        plt.tight_layout()
        if save_plots_tests:
            plt.savefig(join(save_path_compareb_tests, 'errorbars_per-particle_z_corr.png'))
            plt.close()
        if show_plots_tests:
            plt.show()

        fig, ax = plot_dist_errorbars_sum_uncertainty(dfgbts_full, xparam='id', yparam='z', errparam='z_std',
                                                      fig=None, ax=None, style=None)
        ax.set_xlabel(r'$p_{ID}$')
        ax.set_ylabel(r'$z_{raw} \: (\mu m)$')
        plt.tight_layout()
        if save_plots_tests:
            plt.savefig(join(save_path_compareb_tests, 'errorbars_per-particle_z_raw.png'))
            plt.close()
        if show_plots_tests:
            plt.show()

        # ----------------------------------- PLOTTING: PER-TEST ANALYSIS

        # plot error bars: boundary particles: z_corr(filename) + z_corr_std(filename)
        fig, ax = plot_dist_errorbars(dfcgts, 'filename', 'z_corr', 'z_corr_std')
        fig, ax = plot_dist_errorbars(dfgbts, 'filename', 'z_corr', 'z_corr_std', fig, ax)
        ax.set_xlabel(r'$p_{ID}$')
        ax.set_ylabel(r'$z_{corrected} \: (\mu m)$')
        ax.legend([r'$p_{N}$', r'$p_{ID}$'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        if save_plots_tests:
            plt.savefig(join(save_path_compareb_tests, 'errorbars_z_corr_collection.png'))
            plt.close()
        if show_plots_tests:
            plt.show()

        # plot error bars: boundary particles: z_corr(filename) + z_corr_std(filename)
        fig, ax = plot_dist_errorbars(dfgbts, 'filename', 'z_corr', 'z_corr_std')
        ax.set_xlabel(r'$p_{ID}$')
        ax.set_ylabel(r'$z_{corrected} \: (\mu m)$')
        plt.tight_layout()
        if save_plots_tests:
            plt.savefig(join(save_path_compareb_tests, 'errorbars_z_corr.png'))
            plt.close()
        if show_plots_tests:
            plt.show()

        # plot error bars: boundary particles: z(filename) + z_std(filename)
        fig, ax = plot_dist_errorbars(dfgbts, 'filename', 'z', 'z_std')
        #ax.set_xlabel(r'$Test \: ID$')
        ax.set_xlabel(r'$p_{ID}$')
        ax.set_ylabel(r'$z_{raw} \: (\mu m)$')
        plt.tight_layout()
        if save_plots_tests:
            plt.savefig(join(save_path_compareb_tests, 'errorbars_z.png'))
            plt.close()
        if show_plots_tests:
            plt.show()

        # ----------------------------------- PLOTTING: PER-PARTICLE-ID ANALYSIS COMPARING TESTS

        # plot error bars: boundary particles: z_corr(pid) + z_corr_std(pid)
        plt.style.use(['science', 'muted'])
        fig, ax = plt.subplots(figsize=(size_x_inches * 2, size_y_inches))
        for name, df in dfgbs.items():
            fig, ax = plot_dist_errorbars(df, 'id', 'z_corr', 'z_corr_std', fig, ax, style='muted')
        ax.set_xlabel(r'$Test \: ID$')
        ax.set_ylabel(r'$z_{corrected} \: (\mu m)$')
        ax.legend(dfgbs.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(alpha=0.125)
        plt.tight_layout()
        if save_plots_tests:
            plt.savefig(join(save_path_compareb_tests_particles, 'errorbars_z_corr.png'))
            plt.close()
        if show_plots_tests:
            plt.show()

        # plot error bars: boundary particles: z_corr(pid) + z_corr_std(pid)
        plt.style.use(['science', 'muted'])
        fig, ax = plt.subplots(figsize=(size_x_inches * 2, size_y_inches))
        for name, df in dfgbs.items():
            fig, ax = plot_dist_errorbars(df, 'id', 'z', 'z_std', fig, ax, style='muted')
        ax.set_xlabel(r'$Test \: ID$')
        ax.set_ylabel(r'$z_{raw} \: (\mu m)$')
        ax.legend(dfgbs.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(alpha=0.125)
        plt.tight_layout()
        if save_plots_tests:
            plt.savefig(join(save_path_compareb_tests_particles, 'errorbars_z.png'))
            plt.close()
        if show_plots_tests:
            plt.show()

        # ----------------------------------- PLOTTING: PER-PARTICLE-ID ANALYSIS

        for name, df in dfgbs.items():
            # plot error bars: boundary particles: z_corr(pid) + z_corr_std(pid)
            fig, ax = plot_dist_errorbars(df, 'id', 'z_corr', 'z_corr_std')
            ax.set_xlabel(r'$Test \: ID$')
            ax.set_ylabel(r'$z_{corrected} \: (\mu m)$')
            if save_plots_tests:
                plt.savefig(join(save_path_compareb_particles_per_test, 'errorbars_z_corr_id{}.png'.format(name)))
                plt.close()
            if show_plots_tests:
                plt.show()

            # plot error bars: boundary particles: z_corr(pid) + z_corr_std(pid)
            fig, ax = plot_dist_errorbars(df, 'id', 'z', 'z_std')
            ax.set_xlabel(r'$Test \: ID$')
            ax.set_ylabel(r'$z_{raw} \: (\mu m)$')
            if save_plots_tests:
                plt.savefig(join(save_path_compareb_particles_per_test, 'errorbars_z_id{}.png'.format(name)))
                plt.close()
            if show_plots_tests:
                plt.show()

    # ----------------------------------- PLOTTING: INTERIOR POINTS
    plot_interiors = False
    if plot_interiors:

        # define fitting function
        poisson = 0.5
        h = 20
        a = 800
        xc_microns, yc_microns = xc * microns_per_pixel, yc * microns_per_pixel


        def spherical_uniformly_loaded_clamped_plate(r, PE):
            return PE * (12 * (1 - poisson ** 2)) / (64 * h ** 3) * (a ** 2 - r ** 2) ** 2


        def spherical_uniformly_loaded_simply_supported_plate(r, PE):
            return PE * (12 * (1 - poisson ** 2)) / (64 * h ** 3) * (a ** 2 - r ** 2) * \
                   ((5 + poisson) / (1 + poisson) * a ** 2 - r ** 2)


        names = []
        PE_cs = []
        PE_sss = []
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches))
        for name, df in dfgis.items():
            # fix z-offset at boundaries
            z_boundary_mean_manual = 2
            z_boundary_mean = dfgbts[dfgbts['filename'] == name].z_corr.to_numpy()
            df['z_corr_offset'] = df.z_corr - z_boundary_mean_manual

            # generate a "radius" column based on particle's x, y coord. and known sphere center: xc, yc, rc.
            df['r'] = np.sqrt((xc_microns - df.x) ** 2 + (yc_microns - df.y) ** 2)

            # fit
            popt_c, pcov_c = curve_fit(spherical_uniformly_loaded_clamped_plate, df.r, df.z_corr_offset)
            popt_ss, pcov_ss = curve_fit(spherical_uniformly_loaded_simply_supported_plate, df.r, df.z_corr_offset)

            # store
            PE_c, PE_ss = popt_c[0], popt_ss[0]
            PE_cs.append(PE_c)
            PE_sss.append(PE_ss)
            names.append(name)

            # plot
            r_space = np.linspace(0, a)
            # z_fit = spherical_uniformly_loaded_clamped_plate(r_space, popt[0])
            z_fit = spherical_uniformly_loaded_simply_supported_plate(r_space, PE_ss)
            ax.plot(r_space, z_fit, label=name)
            ax.scatter(df.r, df.z_corr_offset, s=1)

        ax.set_xlabel(r'$r \: (\mu m)$')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$ID_{test}$')
        plt.tight_layout()
        plt.savefig(join(save_path_fit_deflections, 'test_deflections.png'))
        plt.show()

        PE_cs = np.array(PE_cs)
        PE_sss = np.array(PE_sss)
        names = np.array(names)
        rho = 1000
        g = 9.81
        names = names * rho * g * 1e-3

        fig, ax = plt.subplots()
        ax.plot(names, PE_cs, '-o', label='clamped')
        ax.plot(names, PE_sss, '-o', label='supported')
        ax.set_xlabel(r'$\Delta P \: (Pa)$')
        ax.set_ylabel(r'$P / E$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(join(save_path_fit_deflections, 'dP_linearity.png'))
        plt.show()

        fig, ax = plt.subplots()
        E = 6e6  # 6 MPa
        ax.plot(names, PE_cs * E, '-o', label='clamped')
        ax.plot(names, PE_sss * E, '-o', label='supported')
        ax.set_xlabel(r'$\Delta P \: (Pa)$')
        ax.set_ylabel(r'$Pressure \: (Pa)$')
        ax.set_title(r'$E \: = \: $' + ' {} '.format(E*1e-6) + r'$ \: MPa$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(join(save_path_fit_deflections, 'P_linearity.png'))
        plt.show()

    # ----------------------------------- PLOTTING: ALL POINTS
    plot_all_particles = False
    filter_z, z_f = True, 60
    filter_radial = False

    if plot_all_particles:

        plt.style.use(['science', 'muted'])
        fig, ax = plt.subplots(figsize=(size_x_inches*1.2, size_y_inches))

        dfbicts = {}
        for name, df in dfgs.items():

            # filter z to remove focal plane bias errors
            df = df[df['z'] > z_f]

            # scatter plot particle locations
            """fig2, ax2 = plt.subplots()
            sc = ax2.scatter(df.x, df.y, c=df.z)
            ax2.set_xlabel(r'$x \: (\mu m)$')
            ax2.set_xlim([0, 850])
            ax2.set_ylabel(r'$y \: (\mu m)$')
            ax2.set_title(r'$ID_{test} = $' + ' {}'.format(name))
            cbar = plt.colorbar(sc)
            cbar.set_label(r'$z \: (\mu m)$')
            plt.tight_layout()
            if save_plots_tests:
                plt.savefig(join(save_bin_deflections, 'figs/pre-bin_scatter_z_id{}.png'.format(name)))"""

            # plot only particles at locations where x-axis is parallel to radius
            if filter_radial:
                filter_y_min, filter_y_max = 180 * microns_per_pixel, 330 * microns_per_pixel
                df = df[(df['y'] > filter_y_min) & (df['y'] < filter_y_max)]

            # bin by x
            df = bin.bin_by_column(df, column_to_bin='x', number_of_bins=10, round_to_decimal=1)
            dfg = df.groupby('bin').mean()
            dfg_std = df.groupby('bin').std()

            # store data
            dfg_std = dfg_std.rename(columns={'z': 'z_bin_std'})
            dfg_std = dfg_std.fillna(value=0, axis=0)
            dfg = dfg.join([dfg_std[['z_bin_std']]])
            dfbicts.update({name: dfg})

            # plot deflection with x
            ax.errorbar(dfg.x, dfg.z, yerr=dfg.z_bin_std, fmt='.', ms=1, elinewidth=1, capsize=3, alpha=0.75)
            ax.scatter(dfg.x, dfg.z, s=4, label=name)

        ax.set_xlabel(r'$x \: (\mu m)$')
        ax.set_xlim([0, 850])
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.legend(title=r'$H \: (mm)$', loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(join(save_bin_deflections, 'figs/collection_z_binned_x_errorbars.png'))
        plt.show()

        dfbs = modify.stack_dficts_by_key(dfbicts, drop_filename=False)
        dfbs = dfbs.reset_index()
        dfbs.to_excel(join(save_bin_deflections, 'data/df_z_binned_x.xlsx'))


# ----------------------------------- Z BY BINNED X ANALYSIS

# ---------------- Z BINNED X
# setup
path_zbx = join(save_bin_deflections, 'data/df_z_binned_x.xlsx')

# read z binned x dataframe
df = pd.read_excel(path_zbx, index_col=0)

# z binned x: data extents
spacing = 10
filter_x_min, filter_x_max = df.x.min()-spacing, df.x.max()+spacing * 2
filter_y_min, filter_y_max = df.y.min()-spacing * 2, df.y.max()+spacing
padd_diff = 0  # padding_during_gdpyt_in_focus_calib - padding_during_gdpyt_test_calib
crop_y_diff = 200

# z binned x: bin extents
bin_min, bin_max = df.bin.min(), df.bin.max()
fit_bins = np.round(np.linspace(bin_min, bin_max, 10), 1)

# inspect z bin x for a single test ID
"""dft = df[df['filename'] == 10.25]
fig, ax = plt.subplots()
sc = ax.scatter(dft.x, dft.y, c=dft.z, s=5)
plt.colorbar(sc)
ax.set_xlim([filter_x_min, filter_x_max])
ax.set_ylim([filter_y_min, filter_y_max])
plt.tight_layout()
plt.show()
plt.close()"""

# ---------------- PER PARTICLE CORRECTION

# read per-particle correction
df_ppc = pd.read_excel(join(io_dict['save_path'], 'corrections', 'per_particle_corrections.xlsx'), usecols=[1, 2, 3, 4])

# plot ppc distribution
"""fig, ax = plt.subplots()
sc = ax.scatter(df_ppc.x, df_ppc.y, c=df_ppc.z_f_calc, s=5)
plt.colorbar(sc)
ax.axvline(padd_diff, color='blue')
ax.axhline(padd_diff, color='blue')
ax.axvline(filter_x_min / microns_per_pixel + padd_diff, color='red')
ax.axvline(filter_x_max / microns_per_pixel + padd_diff, color='red')
ax.axhline(filter_y_min / microns_per_pixel + padd_diff + crop_y_diff, color='red')
ax.axhline(filter_y_max / microns_per_pixel + padd_diff + crop_y_diff, color='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.savefig(join(save_bin_deflections, 'corrections/ppc_sampling_space.png'))
plt.show()
plt.close()"""

# resolve padding difference
df_ppc['x'] = df_ppc['x'] - padd_diff
df_ppc['y'] = df_ppc['y'] - crop_y_diff - padd_diff

# scale units
df_ppc['x'] = df_ppc['x'] * microns_per_pixel
df_ppc['y'] = df_ppc['y'] * microns_per_pixel

# filter to fit z binned x
df_ppc = df_ppc[(df_ppc['x'] > filter_x_min) & (df_ppc['x'] < filter_x_max)]
df_ppc = df_ppc[(df_ppc['y'] > filter_y_min) & (df_ppc['y'] < filter_y_max)]

# plot ppc bins
"""fig, ax = plt.subplots()
sc = ax.scatter(df_ppc.x, df_ppc.y, c=df_ppc.z_f_calc, s=5)
plt.colorbar(sc)
for b in fit_bins:
    ax.axvline(b, color='black', linewidth=0.5, alpha=0.125)
ax.set_xlim([filter_x_min, filter_x_max])
ax.set_ylim([filter_y_min, filter_y_max])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.savefig(join(save_bin_deflections, 'corrections/ppc_sampling_bins.png'))
plt.show()
plt.close()"""

# bin by x
dfb_ppc = bin.bin_by_list(df_ppc, column_to_bin='x', bins=fit_bins, round_to_decimal=1)
dfbm_ppc = dfb_ppc.groupby('bin').mean()
dfb_ppc_std = dfb_ppc.groupby('bin').std()
dfbc_ppc = dfb_ppc.groupby('bin').count()

# plot ppc z bin x
"""fig, ax = plt.subplots()
sc = ax.scatter(dfbm_ppc.x, dfbm_ppc.y, c=dfbm_ppc.z_f_calc, s=5)
plt.colorbar(sc)
ax.set_xlim([filter_x_min, filter_x_max])
ax.set_ylim([filter_y_min, filter_y_max])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.savefig(join(save_bin_deflections, 'corrections/ppc_sampling_bin_z.png'))
plt.show()
plt.close()"""

# ---------------- CALCULATE CORRECTION

# offset
dfbm_ppc['z_f_calc'] = dfbm_ppc['z_f_calc'] - dfbm_ppc['z_f_calc'].min()

# fit line
popt_line, pcov, fit_func = fit.fit(x=dfbm_ppc.x, y=dfbm_ppc.z_f_calc, fit_function=functions.line)
a_line, b_line = popt_line[0], popt_line[1]

# fit parabola
popt_parabola, pcov, fit_func = fit.fit(x=dfbm_ppc.x, y=dfbm_ppc.z_f_calc, fit_function=functions.parabola)
a_parabola, b_parabola, c_parabola = popt_parabola[0], popt_parabola[1], popt_parabola[2]
fit_bins_sampling = np.linspace(fit_bins.min(), fit_bins.max(), 100)

# setup
plt.style.use(['science', 'ieee', 'muted'])
clrs = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
fids = df.filename.unique()

# plot ppc fit
"""fig, ax = plt.subplots()
#ax.scatter(dft.bin, dft.z, s=5, label=r'$ID_{test}=0$')
ax.errorbar(dfbm_ppc.x, dfbm_ppc.z_f_calc, yerr=dfb_ppc_std.z_f_calc, color='#FF2C00', fmt='o',
            ms=2, elinewidth=1, capsize=3, alpha=1, label='ppc')
# ax.plot(fit_bins, functions.line(fit_bins, a_line, b_line), color='gray', linestyle='--', linewidth=1, label=r'$Fit_{line}$')
ax.plot(fit_bins_sampling, functions.parabola(fit_bins_sampling, a_parabola, b_parabola, c_parabola), color='black',
        linestyle='--', linewidth=1, label=r'$Fit_{parabola}$')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.legend()
plt.tight_layout()
plt.savefig(join(save_bin_deflections, 'corrections/ppc_fit_bin_z.png'))
plt.show()"""

# z offset is taken from average z-coord. of boundary particles from in-focus correction calibration
z_offset = 58.5  # z_offset = 58.5 from in-focus

# plot corrected z binned x
"""
fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))

for fid in fids:

    dfid = df[df['filename'] == fid]

    # adjust z for offset
    dfid['z'] = dfid['z'] - z_offset

    # apply ppc correction
    dfid['z'] = dfid['z'] + functions.parabola(dfid.x, a_parabola, b_parabola, c_parabola)

    # plot
    clr = next(clrs)
    ax.errorbar(dfid.x, dfid.z, yerr=dfid.z_bin_std, fmt='.', ms=4, elinewidth=1, capsize=3, alpha=0.75, color=clr, label=fid)

    # plot fitted parabola
    popt_def, pcov, fit_func = fit.fit(x=dfid.x, y=dfid.z, fit_function=functions.parabola)
    a_def, b_def, c_def = popt_def[0], popt_def[1], popt_def[2]
    ax.plot(fit_bins_sampling, functions.parabola(fit_bins_sampling, a_def, b_def, c_def), color=clr, alpha=0.25, linewidth=0.5)

# radial centerline
ax.axvline(850, 0.025, 0.975, color='gray', linewidth=0.5, label=r'$r=0$', linestyle='--', zorder=1.5)
ax.axhline(0, 0.025, 0.95, color='lightgray', linewidth=0.25, linestyle='--')
ax.set_xlabel(r'$x \: (\mu m)$')
ax.set_xlim([-25, 900])
ax.set_ylim([-1, 35])
ax.set_ylabel(r'$z_{corrected} \: (\mu m)$')
ax.legend(title=r'$H \: (mm)$', loc='upper left', bbox_to_anchor=(1, 1))
ax.set_title(r'$z_{offset} = $' + ' {} '.format(z_offset) + r'$\mu m$')
plt.tight_layout()
plt.savefig(join(save_bin_deflections, 'corrections/z_bin_x_ppc_correction_fit_parabola.png'))
plt.show()
plt.close()
"""


# max deflection
skip_fids = [0.25, 6.25]
max_zs = []
names = []
for fid in fids:
    dfid = df[df['filename'] == fid]
    max_zs.append(np.max(dfid.z + functions.parabola(dfid.x, a_parabola, b_parabola, c_parabola)) - z_offset)
    names.append(fid)

# fit line
fit_names = [fn for fn, fz in zip(names, max_zs) if fn not in skip_fids]
fit_zs = [fz for fn, fz in zip(names, max_zs) if fn not in skip_fids]
popt_line, pcov, fit_func = fit.fit(x=fit_names, y=fit_zs, fit_function=functions.line)
a_line, b_line = popt_line[0], popt_line[1]
line_extrapolat = np.linspace(0, 11.25, 10)
z_intercept = b_line
x_intercept = -b_line / a_line

# plot max deflection
"""
fig, ax = plt.subplots()
ax.plot(names, max_zs, '-o', label=r'$Data$')
ax.plot(line_extrapolat, functions.line(line_extrapolat, a_line, b_line), color='black', linewidth=1, linestyle='--', label=r'$Fit$')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

ax.set_title(r'$x_{intercept}, z_{intercept} = $' + ' {}, {} '.format(np.round(x_intercept, 1), np.round(z_intercept, 1)) + r'$\mu m$')
ax.set_xlabel(r'$H \: (\mu m)$')
ax.set_ylim([-1, 32])
ax.set_ylabel(r'$z_{corrected, max} \: (\mu m)$')
ax.legend()
plt.tight_layout()
plt.savefig(join(save_bin_deflections, 'corrections/z_max_bin_x_ppc_correction.png'))
plt.show()
"""

# fit spherical uniformly loaded w/ clamped boundary conditions
inst = functions.fSphericalUniformLoad()
inst.r = 400e-6
inst.h = 20e-6
E = 6e6
inst.poisson = 0.5

# constants
rho = 1000
g = 9.81

# adjust fit space
# line_z_extrapolat = np.linspace(x_intercept, 11.25, 10)
# line_x_extrapolat = np.linspace(0, 11.25 - x_intercept, 10)

fig, ax = plt.subplots()
# fids = np.array(fids) - x_intercept
pressures = fids * 1e-3 * rho * g
print(pressures)
ax.plot(fids, inst.spherical_uniformly_loaded_clamped_plate(P=pressures, E=E) * 1e6, '-o', label=r'$Theory_{clamped}$')
ax.plot(fids, inst.spherical_uniformly_loaded_simply_supported_plate(P=pressures, E=E) * 1e6, '-o', label=r'$Theory_{supported}$')

ax.plot(np.array(names), max_zs - z_intercept, '-o', label=r'$Data$')
ax.plot(line_extrapolat, functions.line(line_extrapolat, a_line, b_line) - z_intercept, color='black', linewidth=1, linestyle='--', label=r'$Fit$')

ax.set_xlabel(r'$\Delta H \: (\mu m)$')
# ax.set_ylim([-1, 35])
ax.set_ylabel(r'$z_{max} \: (mm)$')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(join(save_bin_deflections, 'figs/fit_spherical_uniform_load_z_max_r=400um.png'))
plt.show()

j = 1