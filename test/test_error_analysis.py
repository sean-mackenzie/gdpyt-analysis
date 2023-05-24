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
           'FINAL-04.21-22_IDPT_1um-calib_5um-test'

path_test_coords = join(base_dir, 'coords/test-coords')
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_calib_spct_pop = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/coords/calib-coords/calib_spct_pop_defocus_stats.xlsx'
path_calib_spct_stats = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/coords/calib-coords/calib_spct_stats_.xlsx'
path_test_spct_pop = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-04.12.22-SPCT-5umStep-meta/coords/calib-coords/calib_spct_pop_defocus_stats_11.06.21_z-micrometer-v2_5umMS__sim-sym.xlsx'
path_test_calib_coords = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-04.12.22-SPCT-5umStep-meta/coords/calib-coords/calib_correction_coords_11.06.21_z-micrometer-v2_5umMS__sim-sym.xlsx'
path_similarity = join(base_dir, 'similarity')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

method = 'spct'
microns_per_pixel = 1.6
mag_eff_c = 10

# ----------------------------------------------------------------------------------------------------------------------
# 1. ANALYZE TEST COORDS

# read test coords (where z is based on 1-micron steps and z_true is 5-micron steps every 3 frames)
analyze_cov = False

if analyze_cov:
    fp = path_results + '/test_coords_spct-corrected-and-filtered.xlsx'
    dft = pd.read_excel(fp, index_col=0)

    # add squared error column
    dft['sq_error'] = dft['error'] ** 2

    # --- coefficient of variation z (full z-range)
    """
    'cov_z' describes all particle measurements as a function of axial position (z).
    """
    cov_z = functions.calculate_coefficient_of_variation(dft.sq_error.to_numpy())

    # ---

    # --- coefficient of variation z (for several z_true ranges)
    column_to_bin = 'z_true'
    bin_z_trues = [-25, 0, 25]
    round_to_decimal = 0

    dfb_z_true = bin.bin_by_list(dft, column_to_bin, bin_z_trues, round_to_decimal)

    cov_z_binzs = []
    for bn in bin_z_trues:
        dfb = dfb_z_true[dfb_z_true['bin'] == bn]
        cov_z_binzs.append(functions.calculate_coefficient_of_variation(dfb.sq_error.to_numpy()))

    # --- coefficient of variation z (per id) #1
    # 'cov_z_by_pid' describes the "mean C.o.V." of all particles as a function of axial position (z).
    bin_pids = dft.id.unique()
    cov_z_by_pids = []
    for pid in bin_pids:
        dfpid = dft[dft['id'] == pid]
        cov_z_by_pids.append(functions.calculate_coefficient_of_variation(dfpid.sq_error.to_numpy()))
    cov_z_by_pid = np.mean(cov_z_by_pids)

    # group by particle ID
    """
    # --- coefficient of variation z (per id) #2
    # 'cov_z_by_id' describes the "mean C.o.V." of all particles as a function of axial position (z).
    column_to_bin = 'id'
    column_to_count = 'z_corr'
    bin_id = dft.id.unique()
    round_to_decimal = 0
    column_to_error = 'error'

    cov_z_by_id = analyze.cov_1d_localization_error(dft,
                                                    column_to_bin,
                                                    column_to_count,
                                                    bin_id,
                                                    round_to_decimal,
                                                    column_to_error,
                                                    )
    """

    # --- coefficient of variation id (per z_true)
    """
    'cov_id_by_z_true' describes the "mean C.o.V." of all particles in a single image.
    """
    column_to_bin = 'z_true'
    bin_all_z_true = dft.z_true.unique()
    round_to_decimal = 0

    dfb_z_true = bin.bin_by_list(dft, column_to_bin, bin_all_z_true, round_to_decimal)

    cov_id_binzs = []
    for bn in bin_all_z_true:
        dfb = dfb_z_true[dfb_z_true['bin'] == bn]
        cov_id_binzs.append(functions.calculate_coefficient_of_variation(dfb.sq_error.to_numpy()))
    cov_id_binz = np.mean(cov_id_binzs)

    # --- coefficient of variation id (per z_true)
    column_to_bin = 'z_true'
    column_to_count = 'id'
    bin_z_true = dft.z_true.unique()
    round_to_decimal = 1
    column_to_error = 'error'

    cov_id_by_z_true = analyze.cov_1d_localization_error(dft,
                                                         column_to_bin,
                                                         column_to_count,
                                                         bin_z_true,
                                                         round_to_decimal,
                                                         column_to_error,
                                                         )

    # --- coefficient of variation id (per z_corr)
    """
    'cov_id_by_z_true' describes the "mean C.o.V." of all particles in a binned z_corr axial depth.
    """
    column_to_bin = 'z_corr'
    column_to_count = 'id'
    bin_z_corr = 22
    round_to_decimal = 2
    column_to_error = 'error'

    cov_id_by_z_corr = analyze.cov_1d_localization_error(dft,
                                                         column_to_bin,
                                                         column_to_count,
                                                         bin_z_true,
                                                         round_to_decimal,
                                                         column_to_error,
                                                         )



    """
    Plot bar chart
    """

    # setup
    labels = [r'$z$', r'$p_{i}$']
    x_pos = np.arange(len(labels))
    data = [cov_z, cov_id_by_z_true]

    fig, ax = plt.subplots()

    vbars = ax.bar(x_pos, data, align='center')

    ax.set_xticks(x_pos, labels=labels)
    ax.set_ylabel(r'$C.o.V. \: (\epsilon_{z})$')

    # Label with specially formatted floats
    ax.bar_label(vbars, fmt='%.2f')
    ax.set_ylim(top=5.0)
    plt.tight_layout()
    plt.savefig(path_figs + '/cov_pid_vs_z_bar.png')
    plt.show()

    """
    Relationship plots
    """
    xylim = 62.5
    xyticks = [-50, -25, 0, 25, 50]

    # C.o.V. per z_bin
    fig, ax = plt.subplots()
    ax.plot(bin_z_trues, cov_z_binzs, '-o')
    ax.set_xlabel(r'$z_{bin}$')
    ax.set_xlim([-xylim, xylim])
    ax.set_xticks(ticks=xyticks, labels=xyticks)
    ax.set_ylabel(r'$C.o.V. \: (\epsilon_{z})$')
    ax.set_ylim([-0.125, 2.65])
    ax.set_yticks([0, 0.5, 1, 1.5, 2.0, 2.5])
    plt.tight_layout()
    plt.savefig(path_figs + '/cov_z_for_z-bins.png')
    plt.show()

    # C.o.V. per image
    fig, ax = plt.subplots()
    ax.scatter(bin_all_z_true, cov_id_binzs)
    ax.set_xlabel(r'$z$')
    ax.set_xlim([-xylim, xylim])
    ax.set_xticks(ticks=xyticks, labels=xyticks)
    ax.set_ylabel(r'$C.o.V. \: (\epsilon_{z})$')
    ax.set_ylim([-0.125, 4.25])
    ax.set_yticks([0, 1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(path_figs + '/cov_pid_for_z-true.png')
    plt.show()

    # C.o.V. per particle as a function of z
    fig, ax = plt.subplots()
    ax.scatter(bin_pids, cov_z_by_pids)
    ax.set_xlabel(r'$p_{ID}$')
    ax.set_ylabel(r'$C.o.V. \: (\epsilon_{z})$')
    ax.set_ylim([-0.125, 4.5])
    ax.set_yticks([0, 1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(path_figs + '/cov_z_for_all-pids.png')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 2. ANALYZE SIGNAL-LIMITED ERROR

analyze_signal_limited_error = False

if analyze_signal_limited_error:
    # step 1. merge SPCT pid stats (particle intensity, r (x-y) precision)
    # step 2. plot the axial or lateral precision for each particle as a function of their intensity.
    # step 3. fit a power-law decay curve to the data.
    # step 4. the power-law's exponent indicates the relative contribution of intensity to localization error/precision.

    fp_spct_pid_stats = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/coords/calib-coords/calib_spct_stats_.xlsx'

    dfc = pd.read_excel(fp_spct_pid_stats)

    dfc['gauss_rc'] = np.sqrt((dfc.gauss_xc - 256) ** 2 + (dfc.gauss_yc - 256) ** 2)

    # ----- STEP 1. MERGE SPCT PID STATS AND TEST COORDS

    # VERIFICATION #1 - average frames per particle ID
    dfc_counts = dfc.groupby('id').count().reset_index()

    # --- V.1 filter: min_num_counts

    # V.1.a find bad particles
    min_num_counts = 50
    remove_ids = dfc_counts[dfc_counts['z_corr'] < min_num_counts].id.unique()

    # V.1.b remove bad particles from dfc
    dfc = dfc[~dfc.id.isin(remove_ids)]

    # V.1.c recalculate group by's
    dfc_counts = dfc.groupby('id').count().reset_index()
    dfc_mean = dfc.groupby('id').mean().reset_index()

    # plot
    fig, ax = plt.subplots()

    ax.scatter(dfc.id, dfc.z_corr, c=dfc.id, s=1, marker='.', alpha=0.5)
    ax.scatter(dfc_mean.id, dfc_mean.z_corr, s=3, color='black')
    ax.set_xlabel(r'$p_{ID}$')
    ax.set_ylabel(r'$z$')

    axr = ax.twinx()
    axr.scatter(dfc_counts.id, dfc_counts.z_corr, s=3, marker='d', color='blue', alpha=0.75)
    axr.set_ylabel('Counts', color='blue')

    plt.tight_layout()
    plt.savefig(path_figs + '/calib-spct-stats_zcorr-counts_by_pid.png')
    plt.show()

    # ---

    # VERIFICATION #2 - plot distribution of mean and peak signal counts around the mean
    column_to_bin = 'mean_int'
    column_to_count = 'id'

    ctb_mean = dfc_mean[column_to_bin].mean()
    ctb_std = dfc_mean[column_to_bin].std()
    bins = [ctb_mean + i * ctb_std for i in [-1.5, -0.75, 0, 0.75, 1.5]]

    round_to_decimal = 1
    return_groupby = True

    dfm, dfstd = bin.bin_generic(dfc_mean, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

    # plot bins
    fig, ax = plt.subplots()
    ax.scatter(dfm.bin, dfm['count_' + column_to_count])
    ax.set_xlabel(r'$\overline{I} \: (A.U.)$')
    ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig(path_figs + '/mean_int_binned_by_stdevs_from_mean.png')
    plt.show()

    # plot distance from mean
    column_to_bin_dist = column_to_bin + '_dist'
    bins_dist = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 550]
    dfc_mean[column_to_bin_dist] = np.abs(dfc_mean[column_to_bin] - ctb_mean)

    dfm, dfstd = bin.bin_generic(dfc_mean, column_to_bin_dist, column_to_count, bins_dist, round_to_decimal, return_groupby)

    fig, ax = plt.subplots()
    ax.scatter(dfm.bin, dfm['count_' + column_to_count])
    ax.set_xlabel(r'$I_{i} - \overline{I} \: (A.U.)$')
    ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig(path_figs + '/pid_intensity_distribution_from_mean.png')
    plt.show()

    # ----- STEP 2. PLOT MEAN SIGNAL COUNTS AS A FUNCTION OF LATERAL PRECISION
    dfc_std = dfc.groupby('id').std().reset_index()

    def power_law(x, a, k):
        return a * x ** k


    fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.25, size_y_inches))

    # fit mean_int
    popt, pcov = curve_fit(power_law, dfc_mean.mean_int, dfc_std.gauss_rc)
    fit_int = np.linspace(0, dfc_mean.mean_int.max())

    ax1.scatter(dfc_mean.mean_int, dfc_std.gauss_rc, s=1)
    ax1.plot(fit_int, power_law(fit_int, *popt), color='black', label='k={}'.format(np.round(popt[1], 2)))
    ax1.set_xlabel(r'$\overline{I}_{i} \: (A.U.)$')
    ax1.set_ylabel(r'$\sigma^p_{r} \: (\mu m)$')
    ax1.legend()

    # fit peak_int
    popt, pcov = curve_fit(power_law, dfc_mean.peak_int, dfc_std.gauss_rc)
    fit_int = np.linspace(0, dfc_mean.peak_int.max())

    ax2.scatter(dfc_mean.peak_int, dfc_std.gauss_rc, s=1)
    ax2.plot(fit_int, power_law(fit_int, *popt), color='black', label='k={}'.format(np.round(popt[1], 2)))
    ax2.set_xlabel(r'$I^{i}_{peak} \: (A.U.)$')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path_figs + '/fit-power-law_r-precision_by_signal-counts.png')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 3. ANALYZE TEST COORDS

analyze_focal_plane_bias_errors = True

if analyze_focal_plane_bias_errors:
    # step 1.
    # step 2.
    # step 3.
    # step 4.

    fp_test_coords = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.21-22_IDPT_1um-calib_5um-test/coords/test-coords/test_coords_idpt-corrected-on-test.xlsx'
    dft = pd.read_excel(fp_test_coords)

    # ---

    # merge SPCT particle ID's with IDPT test particle ID's
    correct_nonuniform_spct_particle_ids = False
    if correct_nonuniform_spct_particle_ids:
        """
        Note: the results of this function are not used until 'B. PLOT SPCT STATS PAIRED WITH TEST COORD DATA'
        """

        fp_spct_pid_stats = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/FINAL-04.25.22_SPCT_1um-calib_5um-test/coords/calib-coords/calib_spct_stats_.xlsx'
        dfc = pd.read_excel(fp_spct_pid_stats)

        # merge spct stats with test coords data using NearestNeighbors
        baseline = dft
        coords = dfc
        coords_orig, mapping_dict, cids_not_mapped = correct.correct_nonuniform_particle_ids(baseline,
                                                                                             coords,
                                                                                             threshold=5,
                                                                                             dropna=True,
                                                                                             save_path=path_figs,
                                                                                             save_id='_',
                                                                                             shift_baseline_coords=[3, 5]
                                                                                             )
        raise ValueError('Stop here to confirm results.')

    # ---

    # filter range so test matches calibration range
    zlim = [-50, 60]
    dft = dft[(dft['z_true'] > zlim[0]) & (dft['z_true'] < zlim[1])]

    # add columns
    dft['r'] = np.sqrt((dft.x - 256) ** 2 + (dft.y - 256) ** 2)
    dft['abs_error'] = dft['error'].abs()
    dft['sq_error'] = dft['error'] ** 2

    # ---

    # dataframe slicing on errors
    axial_exclusion_range = 5
    error_lower_bound = 5
    error_fpb_limit = axial_exclusion_range * 2

    # OPTIONAL: get dataframe that excludes region near the focal plane
    dft = dft[(dft['z_true'] < -axial_exclusion_range) | (dft['z_true'] > axial_exclusion_range)]

    # store original test_coords
    dft_original = dft.copy()

    # get IDs of particles where error is always < 5 microns
    pids_errors = dft[dft['error'].abs() > error_lower_bound].id.unique()
    df = dft[~dft.id.isin(pids_errors)]

    # get IDs of particles where focal plane bias errors occur
    pids_fpb_errors = dft[dft['error'].abs() > error_fpb_limit].id.unique()
    dft = dft[dft.id.isin(pids_fpb_errors)]

    # get dataframe of only focal plane bias errors
    df_fpb_errors = dft[dft['error'].abs() > error_fpb_limit]

    # ---

    # --- A. PLOT TEST COORD DATA

    show_figs = False
    save_figs = False

    if any([save_figs, show_figs]):

        # --- Part A. Plot focal plane bias errors wrt spatial coordinates (x, y, r)

        # Figure 1a-b. plot x-, y-, and r-coordinates of focal plane bias errors
        plot_fpb_error_spatial_distribution = False

        if plot_fpb_error_spatial_distribution:

            # groupby original dataframe to get location of all particles
            dfto = dft_original.groupby('id').mean()

            # Figure 1a. plot x, y location of focal plane bias errors
            fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))

            ax.scatter(dfto.x, dfto.y, s=5, marker='s', color='gray', alpha=0.125, label='all')
            ax.scatter(df_fpb_errors.x, df_fpb_errors.y, c=df_fpb_errors.id, s=3, label='fpb errors')

            ax.set_xlabel('x (pixels)')
            ax.set_xlim([0, 512])
            ax.set_ylabel('y (pixles)')
            ax.set_ylim([0, 512])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_figs + '/fpb-errors_scatter-xy.png')
            if show_figs:
                plt.show()

            # ---

            # Figure 1b. histogram
            params = ['x', 'y', 'r']
            num_bins = 5

            # plot

            for density in [True, False]:

                fig, ax = plt.subplots(ncols=len(params), sharey=True, figsize=(size_x_inches*2, size_y_inches))

                for i, param in enumerate(params):

                    # get fpb errors per param
                    x = df_fpb_errors[param].to_numpy()
                    mu = np.mean(x)
                    sigma = np.std(x)

                    # the histogram of the data
                    n, bins, patches = ax[i].hist(x, num_bins, density=density)

                    # add a 'best fit' line
                    if density:
                        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
                        ax[i].plot(bins, y, '--', color='black')

                    ax[i].set_xlabel(param)

                if density:
                    ax[0].set_ylabel('Probability Density')
                    plt.tight_layout()
                    save_id = '/fpb-errors_histogram-pdf.png'
                else:
                    ax[0].set_ylabel('Counts')
                    plt.tight_layout()
                    save_id = '/fpb-errors_histogram-counts.png'

                if save_figs:
                    plt.savefig(path_figs + save_id)
                if show_figs:
                    plt.show()

        # ---

        # --- Part B. Plot focal plane bias errors wrt ... ?

        # Figure 1. plot z-coordinates of focal plane bias errors per particle
        plot_fpb_error_occurrence = False

        if plot_fpb_error_occurrence:

            fig_lims = [zlim[0] - 5, zlim[1] + 5]
            fig, ax = plt.subplots(figsize=(size_x_inches * 1.1, size_y_inches))

            for pid in dft.id.unique():
                dfpid = dft[dft['id'] == pid]
                dfpid = dfpid[dfpid['error'].abs() > error_fpb_limit]

                p1, = ax.plot(dfpid.z_true, dfpid.z, 'o', ms=3, label=pid)
                ax.plot(dfpid.z_true, dfpid.z_true, 's', ms=2, color=p1.get_color(), alpha=0.125)

            ax.plot(zlim, zlim, linestyle='--', color='gray', alpha=0.125)

            ax.set_xlabel(r'$z_{true} \: (\mu m)$')
            ax.set_xlim(fig_lims)
            ax.set_ylabel(r'$z \: (\mu m)$')
            ax.set_ylim(fig_lims)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            if save_figs:
                plt.savefig(path_figs + '/error-analysis_fpb-error-z-occurrences-per-pid.png')
            if show_figs:
                plt.show()

            # export dataframe of measurements where focal plane bias error occurs
            export_fpb_errors = False
            if export_fpb_errors:
                df_fpb_errors.to_excel(
                    path_figs + '/df_focal-plane-bias-errors_err-greater-than-{}.xlsx'.format(error_fpb_limit))

        # ---

        # Figure 2. groupby z_true; plot params
        plot_test_params = False

        if plot_test_params:
            params = ['error', 'abs_error', 'sq_error', 'cm']

            dfz = df.groupby('z_true').mean().reset_index()
            dfzstd = df.groupby('z_true').std().reset_index()

            dftz = dft.groupby('z_true').mean().reset_index()
            dftzstd = dft.groupby('z_true').std().reset_index()

            for param in params:

                fig, ax = plt.subplots()

                ax.errorbar(dfz.z_true, dfz[param], yerr=np.sqrt(dfzstd[param]),
                            elinewidth=1, capsize=2, label=r'$$\epsilon_{z}|_{h}<5$$')

                ax.errorbar(dftz.z_true, dftz[param], yerr=np.sqrt(dftzstd[param]),
                            elinewidth=1, capsize=2, label=r'$\epsilon_{f.p.b.}$')

                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(param)
                ax.legend()
                plt.tight_layout()
                plt.savefig(path_figs + '/error-analysis_fpb-errors_{}_by_z-true_sqrt-errorbars.png'.format(param))
                if show_figs:
                    plt.show()

        # ---

        # Figure 3. mean error + counts (r bin)
        plot_sq_error_and_counts_by_r = False

        if plot_sq_error_and_counts_by_r:

            column_to_bin = 'r'
            column_to_count = 'id'
            bins = np.arange(50, 351, 50)
            round_to_decimal = 1
            return_groupby = True

            # setup
            xparam = 'bin'
            yparam1 = 'sq_error'
            yparam2 = 'count_' + column_to_count

            dfs = [df, dft]
            lbls = [r'$\epsilon_{z}|_{h}<5$', r'$\epsilon_{f.p.b.}$']

            fig, [axr, ax] = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]},
                                          figsize=(size_x_inches, size_y_inches * 1.125))

            for dff, lbl in zip(dfs, lbls):

                dfm, dfstd = bin.bin_generic(dff, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

                # counts
                axr.plot(dfm.bin, dfm[yparam2], '-s', ms=1, alpha=0.85)

                # mean squared error
                ax.errorbar(dfm.bin, dfm[yparam1], yerr=dfstd[yparam1] / 10, lolims=True,
                            fmt='-o', ms=2, elinewidth=0.5, capsize=2, alpha=0.85, label=lbl)

            axr.set_ylabel(r'$Counts \: (\#)$')
            axr.set_ylim(bottom=-50)

            ax.set_xlabel(r'$r \: (\mu m)$')
            ax.set_ylabel(r'$\epsilon_{z}^2 \: [0, \frac {\sigma_{z}} {10}] \: (\mu m)$')
            ax.set_yscale('log')
            ax.legend()

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_figs + '/error-analysis_fpb-errors_mean-sq-error-counts_by_r.png')
            if show_figs:
                plt.show()

        # ---

        # ---

        # --- B. PLOT SPCT STATS PAIRED WITH TEST COORD DATA
        plot_fpb_errors_spct = False

        if plot_fpb_errors_spct:

            fp_spct_pid_stats = path_figs + '/uniformize-spct-pids/calib_spct_stats_uniformize-ids-on-test-coords.xlsx'
            dfc = pd.read_excel(fp_spct_pid_stats)

            # get IDs of particles where error is always < 5 microns
            df = dfc[~dfc.id.isin(pids_errors)]

            # get IDs of particles where focal plane bias errors occur
            dft = dfc[dfc.id.isin(pids_fpb_errors)]

            # ---

            # params to plot

            params = ['peak_int', 'mean_int', 'snr', 'nsv',
                      'solidity', 'thinness_ratio',
                      'gauss_A', 'gauss_dia_x_y', 'gauss_sigma_x_y',
                      'min_dx', 'num_dxo', 'percent_dx_diameter',
                      ]

            # ---

            # Figure 1. groupby z_true and plot quantities
            plot_spct_groupby_params = True

            if plot_spct_groupby_params:

                # 1. plot the number of particles per z_true
                dfz_counts = df.groupby('z_true').count().reset_index()
                dftz_counts = dft.groupby('z_true').count().reset_index()

                fig, ax = plt.subplots()

                ax.plot(dfz_counts.z_true, dfz_counts['peak_int'], '-o', ms=2, color=sciblue, label=r'$$\epsilon_{z}|_{h}<5$$')
                ax.plot(dftz_counts.z_true, dftz_counts['peak_int'], '-o', ms=2, color=scigreen, label=r'$\epsilon_{f.p.b.}$')

                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$N_{p} \: (\#)$')
                ax.legend()
                plt.tight_layout()
                if save_figs:
                    plt.savefig(path_figs + '/error-analysis_fpb-errors_spct_num-particles_by_z-true.png')
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # 2. plot params
                dfz = df.groupby('z_true').mean().reset_index()
                dfzstd = df.groupby('z_true').std().reset_index()

                dftz = dft.groupby('z_true').mean().reset_index()
                dftzstd = dft.groupby('z_true').std().reset_index()

                for param in params:
                    fig, ax = plt.subplots()

                    # plot transparent error bars
                    ax.errorbar(dfz.z_true, dfz[param], yerr=dfzstd[param], fmt='.', ms=0.5, elinewidth=0.5, capsize=1,
                                color=lighten_color(sciblue, amount=0.75), alpha=0.65)
                    ax.errorbar(dftz.z_true, dftz[param], yerr=dftzstd[param], fmt='.', ms=0.5, elinewidth=0.5, capsize=1,
                                color=lighten_color(scigreen, amount=0.75), alpha=0.65)

                    # plot solid lines
                    ax.plot(dfz.z_true, dfz[param], color=sciblue, linewidth=3, label=r'$$\epsilon_{z}|_{h}<5$$')
                    ax.plot(dftz.z_true, dftz[param], color=scigreen, linewidth=3, label=r'$\epsilon_{f.p.b.}$')

                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.set_ylabel(param)
                    ax.legend()

                    # modifiers
                    if param in ['gauss_dia_x_y', 'gauss_sigma_x_y']:
                        ax.set_ylim([0.875, 1.125])

                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_figs + '/error-analysis_fpb-errors_spct-{}_by_z-true.png'.format(param))
                    if show_figs:
                        plt.show()
                    plt.close()

            # ---

            # Figure 2. plot per-fpb-error-particle ID params vs. groupby low error particles
            plot_spct_params_per_fpb_pid = True

            if plot_spct_params_per_fpb_pid:

                # calculate groupby stats for low error particles
                dfz = df.groupby('z_true').mean().reset_index()
                dfzstd = df.groupby('z_true').std().reset_index()

                # determine the number of figures (5 fpb error particles per figure)
                fpb_pids = dft.id.unique()
                num_fpb_pids = len(fpb_pids)
                num_plots_per_fig = 5
                num_figs = int(np.ceil(num_fpb_pids / num_plots_per_fig))

                for param in params:

                    for i in range(num_figs):

                        # get fpb particle ID's for this figure only
                        fpb_pids_this_fig = fpb_pids[i * num_plots_per_fig: (i + 1) * num_plots_per_fig]

                        # plot
                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches * 1.25))

                        # plot groupby error bars for low error particles
                        ax.errorbar(dfz.z_corr, dfz[param], yerr=dfzstd[param], fmt='.', ms=0.5, elinewidth=0.5, capsize=1,
                                    color=lighten_color('gray', amount=0.85), alpha=0.35)
                        ax.plot(dfz.z_corr, dfz[param], color='gray', linewidth=1, alpha=0.5, label=r'$$\epsilon_{z}|_{h}<5$$')

                        for pid in fpb_pids_this_fig:

                            # get spct dataframe of this particle only
                            dfpid = dft[dft['id'] == pid]

                            # get dataframe of only focal plane bias errors for this particle only
                            dfpid_fpb_errors = df_fpb_errors[df_fpb_errors['id'] == pid]

                            # plot this particle
                            p1, = ax.plot(dfpid.z_corr, dfpid[param], '-o', ms=2, linewidth=1,
                                          label=r'$p_{ID}$' + ' {}'.format(pid))

                            # plot where the focal plane bias error occurs
                            for j in range(len(dfpid_fpb_errors)):
                                ax.axvline(dfpid_fpb_errors.iloc[j].z_true + (np.random.rand(1)[0] - 0.5) * 3,
                                           linestyle='--', linewidth=0.75, color=lighten_color(p1.get_color(), 1.15))

                        ax.set_xlabel(r'$z \: (\mu m)$')
                        ax.set_ylabel(param)
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                        # modifiers
                        if param in ['gauss_dia_x_y', 'gauss_sigma_x_y']:
                            ax.set_ylim([0.5, 1.225])

                        plt.tight_layout()
                        plt.savefig(path_figs +
                                    '/error-analysis_fpb-errors_spct-{}_by_z-true_fpb-pid-group{}.png'.format(param, i))
                        if show_figs:
                            plt.show()
                        plt.close()

    # ---

    # ---

    # --- C. PLOT IDPT STATS PAIRED WITH TEST COORD DATA
    plot_idpt_stats = False

    if plot_idpt_stats:

        method = 'idpt'

        fp_idpt_pid_stats = path_calib_coords + '/calib_idpt_stats_11.06.21_z-micrometer-v2_match-location.xlsx'
        dfc = pd.read_excel(fp_idpt_pid_stats)

        # get IDs of particles where error is always < 5 microns
        df = dfc[~dfc.id.isin(pids_errors)]

        # get IDs of particles where focal plane bias errors occur
        dft = dfc[dfc.id.isin(pids_fpb_errors)]

        # ---

        # params to plot

        xparam = 'z_corr'
        groupby_param = 'z_true'

        params = ['peak_int', 'mean_int', 'snr', 'nsv', 'nsv_signal', 'percent_dx_diameter']
        params_groupby_id = ['counts', 'contour_area', 'contour_diameter', 'mean_dx', 'min_dx', 'mean_dxo', 'num_dxo']

        lbls = [r'$\epsilon_{z}|_{h}<5$', r'$\epsilon_{f.p.b.}$']

        # ---

        # ---

        show_figs = False
        save_figs = False

        if any([save_figs, show_figs]):

            # plot averages across particles of high and low error
            plot_idpt_groupby_params = False

            if plot_idpt_groupby_params:

                # Figure 1. bar chart of "static" parameters (due to IDPT static contours across all images)

                # collect data
                low_error_data = []
                low_error_std = []
                fpb_error_data = []
                fpb_error_std = []

                for param in params_groupby_id:

                    if param == 'counts':
                        low_error_data.append(len(df.id.unique()))
                        low_error_std.append(0)
                        fpb_error_data.append(len(dft.id.unique()))
                        fpb_error_std.append(0)
                    else:
                        low_error_data.append(df[param].mean())
                        low_error_std.append(df[param].std())
                        fpb_error_data.append(dft[param].mean())
                        fpb_error_std.append(dft[param].std())

                # numpy array for math operations
                low_error_data = np.array(low_error_data)
                fpb_error_data = np.array(fpb_error_data)

                # setup
                labels = params_groupby_id
                x_pos = np.arange(len(labels))
                width = 0.4

                fig, ax = plt.subplots(figsize=(size_x_inches * 1.65, size_y_inches))

                rects1 = ax.bar(x_pos - width / 4, low_error_data / low_error_data, width / 2, label=lbls[0])
                rects2 = ax.bar(x_pos + width / 2, np.round(fpb_error_data / low_error_data, 2), width, label=lbls[1])

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.bar_label(rects2, padding=2)
                ax.set_ylabel(lbls[1] + r'$/$' + lbls[0])
                ax.set_ylim([0.5, 1.375])
                ax.set_xticks(x_pos, labels)
                ax.legend()

                fig.tight_layout()
                if save_figs:
                    plt.savefig(path_figs + '/idpt-spct_groupby-params_bar.png')
                if show_figs:
                    plt.show()
                plt.close()

                # ---

                # Figure 2. groupby z_true and plot quantities

                # setup

                # dataframes
                dfz = df.groupby('z_true').mean().reset_index()
                dfzstd = df.groupby('z_true').std().reset_index()

                dftz = dft.groupby('z_true').mean().reset_index()
                dftzstd = dft.groupby('z_true').std().reset_index()

                for param in params:
                    fig, ax = plt.subplots()

                    # plot transparent error bars
                    ax.errorbar(dfz[xparam], dfz[param], yerr=dfzstd[param], fmt='.', ms=0.5, elinewidth=0.5, capsize=1,
                                color=lighten_color(sciblue, amount=0.75), alpha=0.5)
                    ax.errorbar(dftz[xparam], dftz[param], yerr=dftzstd[param], fmt='.', ms=0.5, elinewidth=0.5, capsize=1,
                                color=lighten_color(scigreen, amount=0.75), alpha=0.4)

                    # plot solid lines
                    ax.plot(dfz[xparam], dfz[param], color=sciblue, linewidth=3, label=r'$$\epsilon_{z}|_{h}<5$$')
                    ax.plot(dftz[xparam], dftz[param], color=scigreen, linewidth=3, label=r'$\epsilon_{f.p.b.}$')

                    ax.set_xlabel(r'$z \: (\mu m)$')
                    ax.set_ylabel(param)
                    ax.legend()

                    plt.tight_layout()
                    if save_figs:
                        plt.savefig(path_figs + '/error-analysis_fpb-errors_spct-{}_by_z-true.png'.format(param))
                    if show_figs:
                        plt.show()
                    plt.close()

        # ---

        # --- --- PERCENT DIAMETER OVERLAP

        plot_percent_diameter_overlap = True

        if plot_percent_diameter_overlap:

            # --- read each percent diameter overlap dataframe (if available)
            calculate_percent_overlap = False
            export_percent_diameter_overlap = False

            param_diameter = 'contour_diameter'
            max_n_neighbors = 3

            if calculate_percent_overlap:

                if param_diameter == 'contour_diameter':
                    popt_contour = analyze.fit_contour_diameter(path_calib_spct_stats, fit_z_dist=15, show_plot=True)
                elif param_diameter == 'gauss_diameter':
                    popt_contour = None
                else:
                    raise ValueError('Parameter for percent diameter overlap is not understood.')

                dfo = analyze.calculate_particle_to_particle_spacing(
                                test_coords_path=dft_original,
                                theoretical_diameter_params_path=None,
                                mag_eff=mag_eff_c,
                                z_param='z_true',
                                zf_at_zero=True,
                                zf_param=None,
                                max_n_neighbors=max_n_neighbors,
                                true_coords_path=dfc,
                                maximum_allowable_diameter=None,
                                popt_contour=popt_contour,
                                param_percent_diameter_overlap=param_diameter
                    )

                # save to excel
                if export_percent_diameter_overlap:

                    # create directories for files
                    if not os.path.exists(path_results + '/percent-overlap'):
                        os.makedirs(path_results + '/percent-overlap')

                    dfo.to_excel(
                        path_results + '/percent-overlap/{}_percent_overlap_{}.xlsx'.format(method, param_diameter),
                        index=False)

            else:
                dfo = pd.read_excel(
                    path_results + '/percent-overlap/{}_percent_overlap_{}.xlsx'.format(method, param_diameter))

            # ---

            # plot low error vs. fpb error on particle spacing and overlap data

            # get IDs of particles where error is always < 5 microns
            df = dfo[~dfo.id.isin(pids_errors)]

            # get IDs of particles where focal plane bias errors occur
            dft = dfo[dfo.id.isin(pids_fpb_errors)]

            # get dataframe of only focal plane bias errors
            dft = dft[dft['error'].abs() > error_fpb_limit]

            # ---

            # params to plot

            xparam = 'z_true'
            groupby_param = 'z_true'
            params = ['contour_diameter', 'mean_dx', 'min_dx', 'mean_dxo', 'num_dxo', 'percent_dx_diameter']
            lbls = [r'$\epsilon_{z}|_{h}<5$', r'$\epsilon_{f.p.b.}$']

            # ---

            # ---

            show_figs = True
            save_figs = True

            if any([save_figs, show_figs]):

                # Figure 1. groupby z_true and plot quantities
                plot_idpt_groupby_params = False

                if plot_idpt_groupby_params:
                    # dataframes
                    dfz = df.groupby(groupby_param).mean().reset_index()
                    dfzstd = df.groupby(groupby_param).std().reset_index()

                    dftz = dft.groupby(groupby_param).mean().reset_index()
                    dftzstd = dft.groupby(groupby_param).std().reset_index()

                    for param in params:
                        fig, ax = plt.subplots()

                        # plot transparent error bars
                        ax.errorbar(dfz[xparam], dfz[param], yerr=dfzstd[param], fmt='o', ms=3, elinewidth=0.5, capsize=2,
                                    color=lighten_color(sciblue, amount=1), alpha=1, label=r'$$\epsilon_{z}|_{h}<5$$')
                        ax.errorbar(dftz[xparam], dftz[param], yerr=dftzstd[param], fmt='s', ms=3, elinewidth=0.5, capsize=2,
                                    color=lighten_color(scigreen, amount=1), alpha=1, label=r'$\epsilon_{f.p.b.}$')

                        ax.set_xlabel(r'$z \: (\mu m)$')
                        ax.set_ylabel(param)
                        ax.legend()

                        plt.tight_layout()
                        if save_figs:
                            plt.savefig(path_figs + '/error-analysis_fpb-errors_idpt-spacing_{}_by_z.png'.format(param))
                        if show_figs:
                            plt.show()
                        plt.close()

                # Figure 2. plot error as a function of params
                plot_error_by_spacing = True

                if plot_error_by_spacing:

                    for param in params:

                        fig, ax = plt.subplots(figsize=(size_x_inches * 1.15, size_y_inches))

                        ax.scatter(df[param], df.error.abs() / df.error.abs().max(), s=2, alpha=0.125, label=lbls[0])
                        ax.scatter(dft[param], dft.error.abs() / dft.error.abs().max(), s=7, marker='x',
                                   color=lighten_color(scired, 1.1),
                                   label=lbls[1])

                        ax.set_xlabel(param)
                        ax.set_ylabel(r'$|\epsilon_{z} / \epsilon_{z, max}| \: (\mu m)$')
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        plt.savefig(
                            path_figs + '/error-analysis_fpb-errors_abs-error_by_{}.png'.format(param))
                        if show_figs:
                            plt.show()

j = 1
print("Analysis completed with errors.")