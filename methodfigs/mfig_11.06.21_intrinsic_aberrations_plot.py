# imports
from os.path import join
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import scipy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from utils.plotting import lighten_color

import analyze
from utils import io, plotting, bin, functions

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

mag_eff = 10.01
NA_eff = 0.45
microns_per_pixel = 1.6
size_pixels = 16  # microns
image_length = 512
padding = 5
xc, yc = image_length / 2 + padding, image_length / 2 + padding

z_range = [-50, 55]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
min_cm = 0.5

# ---

# dataset alignment
z_zero_from_calibration = 49.9  # 50.0
z_zero_of_calib_id_from_calibration = 49.6  # the in-focus position of calib particle in test set.

z_zero_from_test_img_center = 68.6  # 68.51
z_zero_of_calib_id_from_test = 68.1  # the in-focus position of calib particle in calib set.

# ---

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_intrinsic_aberrations_plot'
path_read = join(base_dir, 'data')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')
filetype = '.xlsx'
fig_type = '.png'

# ---


# METHOD: IDPT or SPCT
method = 'idpt'

# --- INTRINSIC ABERRATIONS ASSESSMENT
calc_ia = False
if calc_ia:

    min_num_frames = 14.5
    print('MIN NUMBER OF SPCT FRAMES = {}'.format(min_num_frames))

    # file names
    fni = '{}_particle_similarity_curves_c1umSteps_t5umSteps_i.a.'.format(method)
    df = pd.read_excel(join(path_read, fni + filetype))

    if method == 'spct':
        fpp = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_intrinsic_aberrations_plot/tracking/spct/test-spct_micrometer_5um-i.a./test_coords_ter-v2_1umSteps__i.a._cter-v2_5umSteps_i.a._2022-09-06 19:17:01.724134.xlsx'
        dff = pd.read_excel(fpp)

        dffc = dff.groupby('id').count().reset_index()
        spct_likely_single_ids = dffc[dffc['z'] > min_num_frames].id.unique()
        df = df[df.id.isin(spct_likely_single_ids)]

    # filtering / modifications
    df['r'] = np.sqrt((df['x'] - xc) ** 2 + (df['y'] - yc) ** 2)

    # adjust z values
    zf = 0
    df = df[(df['z_true'] > 0) & (df['z_true'] < 102)]

    # CORRECTION #3: shift 'z_true', 'z_est', and 'z_cm' according to z_f
    df['z_cm'] = df['z_cm'] - 50
    df['z_true'] = df['z_true'] - 50
    df['z_est'] = df['z_est'] - 50

    # evaluate
    dict_ia = analyze.evaluate_intrinsic_aberrations(df,
                                                     z_f=zf,
                                                     min_cm=min_cm,
                                                     param_z_true='z_true',
                                                     param_z_cm='z_cm')

    dict_ia = analyze.fit_intrinsic_aberrations(dict_ia)
    io.export_dict_intrinsic_aberrations(dict_ia, path_results, unique_id=method)

    # plot
    fig, ax = plotting.plot_intrinsic_aberrations(dict_ia, cubic=True, quartic=True, alpha=0.0625)
    ax.set_xlabel(r'$z_{raw} \: (\mu m)$')
    ax.set_ylabel(r'$S_{max}(z_{l}) / S_{max}(z_{r})$')
    ax.grid(alpha=0.125)
    # ax.set_ylim([-0.15, 0.15])
    ax.legend(['Data', 'Cubic', 'Quartic'])
    plt.tight_layout()
    plt.savefig(path_figs + '/{}_intrinsic-aberrations_raw.png'.format(method))
    plt.show()
    plt.close()

# ---

# --- POST-PROCESS PLOT INTRINSIC ABERRATIONS
plot_ia = False
if plot_ia:

    # file names
    fni = 'ia_values_{}'.format(method)
    df = pd.read_excel(join(path_results, fni + filetype))

    # ---

    # filter
    df = df[(df['zs'] > -50) & (df['zs'] < 50)]

    # ---

    # sensitivity
    analyze_sensitivity = False
    if analyze_sensitivity:
        df_sens = analyze.evaluate_ia_sensitivity(df)
        df_sens.to_excel(join(path_results, 'ia_sensitivity_{}'.format(method) + filetype))

    # plotting

    # setup
    x_label = r'$z \: (\mu m)$'
    x_lim = [-53.5, 53.5]
    x_ticks = [-50, 0, 50]
    y_label = r'$C_{m}(z < z_{f}) / C_{m}(z > z_{f})$'

    if method == 'idpt':
        y_lim = [-0.045, 0.045]
        y_ticks = [-0.04, -0.02, 0, 0.02, 0.04]
    elif method == 'spct':
        y_lim = [-0.065, 0.065]
        y_ticks = [-0.05, 0, 0.05]
    else:
        raise ValueError("Need to define method as 'idpt' or 'spct'.")

    # sensitivity plot
    plot_sensitivity = False
    if plot_sensitivity:
        fig, ax = plt.subplots()
        ax.plot(df_sens.index, df_sens.true_positive_rate, '-o', ms=4,
                label=r'$TPR=$' + ' {}'.format(np.round(df_sens.sensitivity.mean(), 3)))
        ax.set_xlabel(x_label)
        ax.set_xlim(x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(y_label)
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_figs + '/{}_post_ia_sensitivity.png'.format(method))
        plt.show()
        plt.close()

    # ---

    # scatter plot
    plot_scatter = False
    if plot_scatter:
        fig, ax = plt.subplots()
        ax.scatter(df.zs, df.cms, s=1, alpha=0.15)
        ax.set_xlabel(x_label)
        ax.set_xlim(x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(y_label)
        plt.tight_layout()
        # plt.savefig(path_figs + '/{}_post_ia_scatter.png'.format(method))
        plt.show()
        plt.close()

    # ---

    # error bars plot
    plot_error_bars = False
    if plot_error_bars:
        dfm = df.groupby('zs').mean().reset_index()
        dfs = df.groupby('zs').std().reset_index()

        fig, ax = plt.subplots()
        ax.errorbar(dfm.zs, dfm.cms, yerr=dfs.cms, fmt='-o', capsize=2, elinewidth=1)
        ax.set_xlabel(x_label)
        ax.set_xlim(x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_lim)
        ax.set_yticks(y_ticks)

        plt.tight_layout()
        plt.savefig(path_figs + '/{}_post_ia_errorbars.png'.format(method))
        plt.show()
        plt.close()

    # ---

    # plot i.a. as a function of r
    plot_bin_by_r = False
    if plot_bin_by_r:

        # bin by number of bins
        bin_r_by_num = True
        if bin_r_by_num:

            # bin by 'r'
            column_to_bin = 'r'
            bins = 5
            df = bin.bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=1)
            bins_r = df.bin.unique()
            bins_r.sort()

            # error bars plot
            plot_error_bars_r = True
            if plot_error_bars_r:
                fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.75))

                for br in bins_r:
                    dfb = df[df['bin'] == br]

                    dfm = dfb.groupby('zs').mean().reset_index()
                    dfs = dfb.groupby('zs').std().reset_index()

                    ax.errorbar(dfm.zs, dfm.cms, yerr=dfs.cms, fmt='-o', ms=2, capsize=2, elinewidth=1,
                                label=int(np.round(br * microns_per_pixel, 0)))

                ax.axhline(y=0, linewidth=0.5, linestyle='--', color='black', alpha=0.25)

                ax.legend(title=r'$r^{\delta} \: (\mu m)$')
                ax.set_xlabel(x_label)
                ax.set_xlim(x_lim)
                ax.set_xticks(x_ticks)
                ax.set_ylabel(y_label)
                ax.set_ylim(y_lim)
                ax.set_yticks(y_ticks)

                plt.tight_layout()
                plt.savefig(path_figs + '/{}_post_ia_bins-{}-r_errorbars-.png'.format(method, bins))
                plt.show()
                plt.close()

        # ---

        # bin by list
        bin_r_by_list = True
        if bin_r_by_list:

            # reconfigure plot
            clrs = [sciblue, scigreen, scigreen, scired]
            ms = 4
            y_lim = [-0.065, 0.065]
            y_ticks = [-0.05, 0, 0.05]

            # for IDPT
            # bins: [125, 145, 425, 450]

            # bin by 'r'
            column_to_bin = 'r'
            bins = np.array([125, 130, 440, 450]) / microns_per_pixel  # [125, 275, 450]
            df = bin.bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=1)
            bins_r = df.bin.unique()
            bins_r.sort()

            # error bars plot
            plot_error_bars_r = True
            if plot_error_bars_r:
                fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.75))

                for br, clr in zip(bins_r, clrs):

                    if 85 < br < 270:
                        continue

                    dfb = df[df['bin'] == br]

                    print("Number of particles in bin {} = {}".format(br, len(dfb.groupby('id').count())))
                    dfm = dfb.groupby('zs').mean().reset_index()
                    dfs = dfb.groupby('zs').std().reset_index()

                    ax.errorbar(dfm.zs, dfm.cms, yerr=dfs.cms,
                                fmt='-o', ms=ms, capsize=2.5, elinewidth=1,
                                # color=clr,
                                label=int(np.round(br * microns_per_pixel, 0)))

                ax.legend(title=r'$r^{\delta} \: (\mu m)$')
                ax.set_xlabel(x_label)
                ax.set_xlim(x_lim)
                ax.set_xticks(x_ticks)
                ax.set_ylabel(y_label)
                ax.set_ylim(y_lim)
                ax.set_yticks(y_ticks)

                plt.tight_layout()
                plt.savefig(path_figs + '/{}_post_ia_list-bins-{}-r_errorbars-v3.png'.format(method, len(bins)))
                plt.show()
                plt.close()

    # ---

    # ---

    # define radial dependence for multi-plots
    rl, rr, dr = 120, 450, 1
    r_bins = [rl, rl + dr, rr - dr, rr]

    # 3 subplots: (1) scatter, raw; (2) fill between, fitted curve; (3) error bars, radial dependence
    plot_raw_fit_radial = True
    if plot_raw_fit_radial:

        # processing
        df['clr'] = 1
        df['clr'] = df.clr.where(((df.cms > 0) & (df.zs > 0)) | ((df.cms < 0) & (df.zs < 0)), 0)
        dftp = df[df['clr'] > 0.5]
        dffp = df[df['clr'] < 0.5]

        # setup
        y1_label = r'$S_{fp}$'
        y2_label = r'$S_{fp}$'
        y3_label = r'$S_{fp}$'

        y1l, d1l = 0.08, 0.0
        y2l, d2l = 0.05, 0.0
        y3l, d3l = 0.08, 0.0

        y1_lim = [-y1l - d1l, y1l + d1l]
        y1_ticks = [-y1l, 0, y1l]
        y2_lim = [-y2l - d2l, y2l + d2l]
        y2_ticks = [-y2l, 0, y2l]
        y3_lim = [-y3l - d3l, y3l + d3l]
        y3_ticks = [-y3l, 0, y3l]

        clr1 = 'black'
        clr2 = sciblue
        clrs_radial = [sciblack, scired]

        scatter1_alpha = 0.15
        scatter1_zorder = 3.3
        scatter1_label_true_positive = 'TP'
        scatter1_label_false_positive = 'FP'
        scatter1_cmap_true_positive = 'RdYlGn_r'
        scatter1_cmap_false_positive = 'RdYlGn'
        plot2_linestyle = '-'
        plot2_linewidth = 0.75
        plot2_zorder = 3.4
        plot2_alpha = 1.0
        plot2_label = 'Fit'
        plot2_std_alpha = 0.25
        plot2_std_linestyle = '-'
        plot2_std_linewidth = 0.25
        fill2_zorder = 3.0
        fill2_alpha = 0.25
        grid2_alpha = 0.25

        # radial dependence
        grid3_alpha = 0.25
        ms = 3
        capsize = 1.5
        elinewidth = 1

        # ---

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                            figsize=(size_x_inches * 1, size_y_inches * 1.5),
                                            gridspec_kw={'height_ratios': [1, 0.5, 0.5]})

        # (1) scatter, raw
        ax1.scatter(dftp.zs, dftp.cms, c=dftp.clr, s=1, marker='.',  # label=scatter1_label_true_positive,
                    cmap=scatter1_cmap_false_positive,
                    alpha=scatter1_alpha, zorder=scatter1_zorder)
        ax1.scatter(dffp.zs, dffp.cms, c=dffp.clr, s=1, marker='.',  # label=scatter1_label_false_positive,
                    cmap=scatter1_cmap_true_positive,
                    alpha=scatter1_alpha, zorder=scatter1_zorder)
        ax1.scatter(dftp.iloc[0].zs, dftp.iloc[0].cms + 1, c=dftp.iloc[0].clr, s=1, marker='.',
                    label=scatter1_label_true_positive,
                    cmap=scatter1_cmap_false_positive, alpha=1)
        ax1.scatter(dffp.iloc[0].zs, dffp.iloc[0].cms + 1, c=dffp.iloc[0].clr, s=1, marker='.',
                    label=scatter1_label_false_positive,
                    cmap=scatter1_cmap_true_positive, alpha=1)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=3)
        ax1.set_ylabel(y1_label)
        ax1.set_ylim(y1_lim)
        ax1.set_yticks(y1_ticks)
        ax1.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # ---

        # (2) fill between, fitted curve
        dfim = df.groupby('zs').mean().reset_index()
        dfis = df.groupby('zs').std().reset_index()

        # fit curve to mean value
        imx = dfim.zs.to_numpy()
        imy = dfim.cms.to_numpy()
        imy_upper = dfim.cms.to_numpy() + dfis.cms.to_numpy()
        imy_lower = dfim.cms.to_numpy() - dfis.cms.to_numpy()

        # fit
        popti, pcov = curve_fit(functions.cubic_slide, imx, imy)
        popti_upper, pcov = curve_fit(functions.cubic_slide, imx, imy_upper)
        popti_lower, pcov = curve_fit(functions.cubic_slide, imx, imy_lower)

        # resample
        fmx = np.linspace(imx.min(), imx.max(), 100)
        fmy = functions.cubic_slide(fmx, *popti)
        fmy_upper = functions.cubic_slide(fmx, *popti_upper)
        fmy_lower = functions.cubic_slide(fmx, *popti_lower)

        # idpt
        ax2.plot(fmx, fmy, linestyle=plot2_linestyle, color=clr2, zorder=plot2_zorder, label=plot2_label,
                 linewidth=plot2_linewidth, alpha=plot2_alpha)
        ax2.plot(fmx, fmy_upper, linestyle=plot2_std_linestyle, color=clr2, alpha=plot2_std_alpha,
                 linewidth=plot2_std_linewidth, zorder=fill2_zorder)
        ax2.plot(fmx, fmy_lower, linestyle=plot2_std_linestyle, color=clr2, alpha=plot2_std_alpha,
                 linewidth=plot2_std_linewidth, zorder=fill2_zorder)
        ax2.fill_between(fmx, y1=fmy_upper, y2=fmy_lower, color=clr2, ec='none',
                         alpha=fill2_alpha, zorder=fill2_zorder)

        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.grid(alpha=grid2_alpha)
        ax2.set_ylabel(y2_label)
        ax2.set_ylim(y2_lim)
        ax2.set_yticks(y2_ticks)
        ax2.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # ---

        # (3) error bars, radial dependence

        # bin by 'r'
        column_to_bin = 'r'
        bins = np.array(r_bins) / microns_per_pixel
        df = bin.bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=1)
        bins_r = df.bin.unique()
        bins_r.sort()

        for br, clr in zip([bins_r[0], bins_r[-1]], clrs_radial):
            dfb = df[df['bin'] == br]

            print("Number of particles in bin {} = {}".format(br, len(dfb.groupby('id').count())))
            dfm = dfb.groupby('zs').mean().reset_index()
            dfs = dfb.groupby('zs').std().reset_index()

            ax3.errorbar(dfm.zs, dfm.cms, yerr=dfs.cms,
                         fmt='-o', ms=ms, capsize=capsize, elinewidth=elinewidth,
                         color=clr,
                         label=int(np.round(br * microns_per_pixel, 0)))

        ax3.grid(alpha=grid3_alpha)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r^{\delta} \: (\mu m)$')
        ax3.set_ylabel(y3_label)
        ax3.set_ylim(y3_lim)
        ax3.set_yticks(y3_ticks)
        ax3.set_xlabel(x_label)
        ax3.set_xlim(x_lim)
        ax3.set_xticks(x_ticks)
        ax3.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.tight_layout()
        plt.savefig(path_figs + '/{}_post_ia_raw_fit_radial'.format(method) + fig_type)
        plt.show()
        plt.close()

    # ---

    # 2 subplots: (2) scatter, raw; fill between, fitted curve; ; (3) error bars, radial dependence
    plot_fit_on_raw_and_radial = True
    if plot_fit_on_raw_and_radial:

        # processing
        df['clr'] = 1
        df['clr'] = df.clr.where(((df.cms > 0) & (df.zs > 0)) | ((df.cms < 0) & (df.zs < 0)), 0)
        dftp = df[df['clr'] > 0.5]
        dffp = df[df['clr'] < 0.5]

        # setup
        y2_label = r'$S_{fp}$'
        y3_label = r'$S_{fp}$'

        y2l, d2l = 0.08, 0.0
        y3l, d3l = 0.08, 0.0

        y2_lim = [-y2l - d2l, y2l + d2l]
        y2_ticks = [-y2l, 0, y2l]
        y3_lim = [-y3l - d3l, y3l + d3l]
        y3_ticks = [-y3l, 0, y3l]

        plot_tp_fp = False
        plot_tp_fp_quadrants = True
        scatter1_color = sciblue
        scatter1_label = 'Exp.'
        scatter1_size = 0.75
        scatter1_markerscale = 3.5
        scatter1_alpha = 0.15
        scatter1_zorder = 3.3
        scatter1_label_true_positive = 'TP'
        scatter1_label_false_positive = 'FP'
        scatter1_cmap_true_positive = 'Blues_r'  # 'RdYlGn_r'
        scatter1_cmap_false_positive = 'Blues_r'  # 'RdYlGn'

        plot2_color = 'black'
        plot2_linestyle = '-'
        plot2_linewidth = 0.5
        plot2_zorder = 3.4
        plot2_alpha = 1.0
        plot2_label = 'Fit'
        plot2_std_alpha = 0.0
        plot2_std_linestyle = '-'
        plot2_std_linewidth = 0.25
        fill2_alpha = 0.0
        fill2_zorder = 1.01
        fill2_alpha_quadrants = 0.09
        fill2_zorder_quadrants = 0.1
        grid2_alpha = 0.25

        # radial dependence
        clrs_radial = ['black', 'red']
        grid3_alpha = 0.25
        ms = 2
        capsize = 1.5
        elinewidth = 0.75

        # ---

        # plot
        fig, (ax2, ax3) = plt.subplots(nrows=2, sharex=True,
                                       figsize=(size_x_inches * 1, size_y_inches * 1.25),
                                       gridspec_kw={'height_ratios': [1, 0.5]})

        # (1) scatter, raw
        if plot_tp_fp:
            ax2.scatter(dftp.zs, dftp.cms, c=dftp.clr, s=1, marker='.',  # label=scatter1_label_true_positive,
                        cmap=scatter1_cmap_false_positive,
                        alpha=scatter1_alpha, zorder=scatter1_zorder)
            ax2.scatter(dffp.zs, dffp.cms, c=dffp.clr, s=1, marker='.',  # label=scatter1_label_false_positive,
                        cmap=scatter1_cmap_true_positive,
                        alpha=scatter1_alpha, zorder=scatter1_zorder)
            ax2.scatter(dftp.iloc[0].zs, dftp.iloc[0].cms + 1, c=dftp.iloc[0].clr, s=1, marker='.',
                        label=scatter1_label_true_positive,
                        cmap=scatter1_cmap_false_positive, alpha=1)
            ax2.scatter(dffp.iloc[0].zs, dffp.iloc[0].cms + 1, c=dffp.iloc[0].clr, s=1, marker='.',
                        label=scatter1_label_false_positive,
                        cmap=scatter1_cmap_true_positive, alpha=1)
        else:
            ax2.scatter(df.zs, df.cms, s=scatter1_size, marker='.',
                        color=scatter1_color,
                        alpha=scatter1_alpha, zorder=scatter1_zorder)
            ax2.scatter(df.iloc[0].zs, df.iloc[0].cms + 1, s=scatter1_size, marker='.', label=scatter1_label,
                        color=scatter1_color, alpha=1)

        # ---

        # (2) fill between, fitted curve
        dfim = df.groupby('zs').mean().reset_index()
        dfis = df.groupby('zs').std().reset_index()

        # fit curve to mean value
        imx = dfim.zs.to_numpy()
        imy = dfim.cms.to_numpy()
        imy_upper = dfim.cms.to_numpy() + dfis.cms.to_numpy()
        imy_lower = dfim.cms.to_numpy() - dfis.cms.to_numpy()

        # fit
        popti, pcov = curve_fit(functions.cubic_slide, imx, imy)
        popti_upper, pcov = curve_fit(functions.cubic_slide, imx, imy_upper)
        popti_lower, pcov = curve_fit(functions.cubic_slide, imx, imy_lower)

        # resample
        fmx = np.linspace(imx.min(), imx.max(), 100)
        fmy = functions.cubic_slide(fmx, *popti)
        fmy_upper = functions.cubic_slide(fmx, *popti_upper)
        fmy_lower = functions.cubic_slide(fmx, *popti_lower)

        # idpt
        ax2.plot(fmx, fmy, linestyle=plot2_linestyle, color=plot2_color, zorder=plot2_zorder, label=plot2_label,
                 linewidth=plot2_linewidth, alpha=plot2_alpha)
        ax2.plot(fmx, fmy_upper, linestyle=plot2_std_linestyle, color=plot2_color, alpha=plot2_std_alpha,
                 linewidth=plot2_std_linewidth, zorder=fill2_zorder)
        ax2.plot(fmx, fmy_lower, linestyle=plot2_std_linestyle, color=plot2_color, alpha=plot2_std_alpha,
                 linewidth=plot2_std_linewidth, zorder=fill2_zorder)
        ax2.fill_between(fmx, y1=fmy_upper, y2=fmy_lower, color=plot2_color, ec='none',
                         alpha=fill2_alpha, zorder=fill2_zorder)

        if plot_tp_fp_quadrants:
            # fill green: True Positive
            ax2.fill_between([x_lim[0], 0], y1=y2_lim[1], y2=0, color=scigreen, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants,
                             label=scatter1_label_true_positive,
                             )
            ax2.fill_between([0, x_lim[1]], y1=0, y2=y2_lim[0], color=scigreen, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants)
            # fill red: False Positive
            ax2.fill_between([x_lim[0], 0], y1=0, y2=y2_lim[0], color=scired, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants)
            ax2.fill_between([0, x_lim[1]], y1=y2_lim[1], y2=0, color=scired, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants,
                             label=scatter1_label_false_positive,
                             )

        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=scatter1_markerscale)
        ax2.grid(alpha=grid2_alpha)
        ax2.set_ylabel(y2_label)
        ax2.set_ylim(y2_lim)
        ax2.set_yticks(y2_ticks)
        ax2.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # ---

        # (3) error bars, radial dependence

        # bin by 'r'
        column_to_bin = 'r'
        bins = np.array(r_bins) / microns_per_pixel
        df = bin.bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=1)
        bins_r = df.bin.unique()
        bins_r.sort()

        for br, clr, lbl_ind, zord in zip([bins_r[0], bins_r[-1]], clrs_radial, [r'$<$', r'$>$'], [3.2, 3.1]):
            dfb = df[df['bin'] == br]

            print("Number of particles in bin {} = {}".format(br, len(dfb.groupby('id').count())))
            dfm = dfb.groupby('zs').mean().reset_index()
            dfs = dfb.groupby('zs').std().reset_index()

            ax3.errorbar(dfm.zs, dfm.cms, yerr=dfs.cms,
                         fmt='-o', ms=ms, capsize=capsize, elinewidth=elinewidth,
                         color=clr,
                         zorder=zord,
                         label=lbl_ind + '{}'.format(int(np.round(br * microns_per_pixel, 0))))

        ax3.grid(alpha=grid3_alpha)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')
        ax3.set_ylabel(y3_label)
        ax3.set_ylim(y3_lim)
        ax3.set_yticks(y3_ticks)
        ax3.set_xlabel(x_label)
        ax3.set_xlim(x_lim)
        ax3.set_xticks(x_ticks)
        ax3.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.tight_layout()
        plt.savefig(path_figs + '/{}_post_ia_fit_on_raw_and_radial'.format(method) + fig_type)
        plt.show()
        plt.close()

    # ---

# ---

# --- COMPARE IDPT AND SPCT

compare_ia = False
if compare_ia:

    # file paths
    path_compare_results = join(base_dir, 'results', 'compare', 'values')
    path_compare_figs = join(base_dir, 'results', 'compare', 'figs')

    # file names
    fni = 'ia_values_idpt'
    fns = 'ia_values_spct'

    dfi = pd.read_excel(join(path_compare_results, fni + filetype))
    dfs = pd.read_excel(join(path_compare_results, fns + filetype))

    # ---

    # filter
    dfi = dfi[(dfi['zs'] > -50) & (dfi['zs'] < 50)]
    dfs = dfs[(dfs['zs'] > -50) & (dfs['zs'] < 50)]

    # ---

    # plotting

    # setup
    x_label = r'$z \: (\mu m)$'
    x_lim = [-52.5, 52.5]
    x_ticks = [-50, 0, 50]
    y_label = r'$C_{m}(z < z_{f}) / C_{m}(z > z_{f})$'
    y_lim = [-0.0575, 0.0575]
    y_ticks = [-0.05, 0, 0.05]

    # error bars plot
    compare_error_bars = False
    if compare_error_bars:
        dfim = dfi.groupby('zs').mean().reset_index()
        dfis = dfi.groupby('zs').std().reset_index()
        dfsm = dfs.groupby('zs').mean().reset_index()
        dfss = dfs.groupby('zs').std().reset_index()

        fig, ax = plt.subplots()
        ax.errorbar(dfim.zs, dfim.cms, yerr=dfis.cms,
                    fmt='-o', capsize=2, elinewidth=1, zorder=3.3, color=scired,
                    label=r'$I_{i}^{c}$',
                    )
        ax.errorbar(dfsm.zs, dfsm.cms, yerr=dfss.cms,
                    fmt='-o', capsize=2, elinewidth=1, zorder=3.1, color=sciblack,
                    label=r'$I_{0}^{c}$',
                    )

        ax.grid(alpha=0.125)
        ax.legend()

        ax.set_xlabel(x_label)
        ax.set_xlim(x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_lim)
        ax.set_yticks(y_ticks)

        plt.tight_layout()
        plt.savefig(path_compare_figs + '/compare_post_ia_errorbars.png')
        plt.show()
        plt.close()

    # ---

    # plot fitted curves with confidence intervals
    compare_fitted_confidence_plots = False
    if compare_fitted_confidence_plots:
        # idpt fitting

        dfim = dfi.groupby('zs').mean().reset_index()
        dfis = dfi.groupby('zs').std().reset_index()

        # fit curve to mean value
        imx = dfim.zs.to_numpy()
        imy = dfim.cms.to_numpy()
        imy_upper = dfim.cms.to_numpy() + dfis.cms.to_numpy()
        imy_lower = dfim.cms.to_numpy() - dfis.cms.to_numpy()

        # fit
        popti, pcov = curve_fit(functions.cubic_slide, imx, imy)
        popti_upper, pcov = curve_fit(functions.cubic_slide, imx, imy_upper)
        popti_lower, pcov = curve_fit(functions.cubic_slide, imx, imy_lower)

        # resample
        fmx = np.linspace(imx.min(), imx.max(), 100)
        fmy = functions.cubic_slide(fmx, *popti)
        fmy_upper = functions.cubic_slide(fmx, *popti_upper)
        fmy_lower = functions.cubic_slide(fmx, *popti_lower)

        # ---

        # spct fitting

        dfsm = dfs.groupby('zs').mean().reset_index()
        dfss = dfs.groupby('zs').std().reset_index()
        smx = dfsm.zs.to_numpy()
        smy = dfsm.cms.to_numpy()
        smy_upper = dfsm.cms.to_numpy() + dfss.cms.to_numpy()
        smy_lower = dfsm.cms.to_numpy() - dfss.cms.to_numpy()
        popts, pcov = curve_fit(functions.cubic_slide, smx, smy)
        popts_upper, pcov = curve_fit(functions.cubic_slide, smx, smy_upper)
        popts_lower, pcov = curve_fit(functions.cubic_slide, smx, smy_lower)
        fsmx = np.linspace(smx.min(), smx.max(), 100)
        fsmy = functions.cubic_slide(fsmx, *popts)
        fsmy_upper = functions.cubic_slide(fsmx, *popts_upper)
        fsmy_lower = functions.cubic_slide(fsmx, *popts_lower)

        # ---

        print(dfim.cms.abs().mean())
        print(dfsm.cms.abs().mean())
        print(dfis.cms.mean())
        print(dfss.cms.mean())
        raise ValueError()

        # plotting

        # setup
        y_lim = [-0.055, 0.055]
        y_ticks = [-0.05, 0, 0.05]
        spct_clr = sciblack
        y_label = r'$S_{fp}$'

        # plot
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        # spct
        ax1.plot(fsmx, fsmy, color=spct_clr, label=r'$I_{o}^{c}$', linestyle='-', zorder=3.5)
        ax1.plot(fsmx, fsmy_upper, color=spct_clr, alpha=0.25, linewidth=0.25, linestyle='-', zorder=2.1)
        ax1.plot(fsmx, fsmy_lower, color=spct_clr, alpha=0.25, linewidth=0.25, linestyle='-', zorder=2.1)
        ax1.fill_between(fsmx, y1=fsmy_upper, y2=fsmy_lower, color=spct_clr, ec='none', alpha=0.125, zorder=2.1)
        ax1.plot(fsmx, fsmy, color=spct_clr, linestyle='-', zorder=3.5)

        # idpt
        ax2.plot(fmx, fmy, color=scired, label=r'$I_{i}^{c}$', zorder=3.3)
        ax2.plot(fmx, fmy_upper, color=scired, alpha=0.25, linewidth=0.25, zorder=3.3)
        ax2.plot(fmx, fmy_lower, color=scired, alpha=0.25, linewidth=0.25, zorder=3.3)
        ax2.fill_between(fmx, y1=fmy_upper, y2=fmy_lower, color=scired, ec='none', alpha=0.125, zorder=3.3)

        ax1.grid(alpha=0.25)
        ax2.grid(alpha=0.25)

        ax1.legend()
        ax2.legend()

        ax1.set_ylabel(y_label)
        ax1.set_ylim(y_lim)
        ax1.set_yticks(y_ticks)
        ax1.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax2.set_xlabel(x_label)
        ax2.set_xlim(x_lim)
        ax2.set_xticks(x_ticks)
        ax2.set_ylabel(y_label)
        ax2.set_ylim(y_lim)
        ax2.set_yticks(y_ticks)
        ax2.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.tight_layout()
        plt.savefig(path_compare_figs + '/compare_post_ia_confidence_intervals.png')
        plt.show()
        plt.close()

    # ---

    # plot fitted curves with confidence intervals
    only_idpt_fitted_confidence_plots = False
    if only_idpt_fitted_confidence_plots:
        dfim = dfi.groupby('zs').mean().reset_index()
        dfis = dfi.groupby('zs').std().reset_index()
        dfsm = dfs.groupby('zs').mean().reset_index()
        dfss = dfs.groupby('zs').std().reset_index()

        # fit curve to mean value
        imx = dfim.zs.to_numpy()
        imy = dfim.cms.to_numpy()
        imy_upper = dfim.cms.to_numpy() + dfis.cms.to_numpy()
        imy_lower = dfim.cms.to_numpy() - dfis.cms.to_numpy()

        # fit
        popti, pcov = curve_fit(functions.cubic_slide, imx, imy)
        popti_upper, pcov = curve_fit(functions.cubic_slide, imx, imy_upper)
        popti_lower, pcov = curve_fit(functions.cubic_slide, imx, imy_lower)

        # resample
        fmx = np.linspace(imx.min(), imx.max(), 100)
        fmy = functions.cubic_slide(fmx, *popti)
        fmy_upper = functions.cubic_slide(fmx, *popti_upper)
        fmy_lower = functions.cubic_slide(fmx, *popti_lower)

        # ---

        # plotting

        # setup
        y_lim = [-0.04, 0.04]
        y_ticks = [-0.025, 0, 0.025]

        # plot
        fig, ax = plt.subplots()
        ax.plot(fmx, fmy, color=scired, label='fit', zorder=3.3)
        ax.plot(fmx, fmy_upper, color=scired, zorder=3.3, alpha=0.25, linewidth=0.5)
        ax.plot(fmx, fmy_lower, color=scired, zorder=3.3, alpha=0.25, linewidth=0.5)
        ax.fill_between(fmx, y1=fmy_upper, y2=fmy_lower, color=scired, alpha=0.125)

        ax.grid(alpha=0.125)
        ax.legend()

        ax.set_xlabel(x_label)
        ax.set_xlim(x_lim)
        ax.set_xticks(x_ticks)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_lim)
        ax.set_yticks(y_ticks)

        plt.tight_layout()
        plt.savefig(path_compare_figs + '/idpt_post_ia_confidence_intervals.png')
        plt.show()
        plt.close()

# ---

# --- 03/25/23 - Calculate astigmatic similarity (S_fp) distribution for each axial position, z
compute_ia_dist = False
if compute_ia_dist:
    """
    Note on chi-squared assessment of distribution:
        * if chi-squared ~ n, then "good" fit. (~ mean "on the order of")
        * if chi-squared >> n, then poor fit.
        * (n = number of bins)
    
    For reduced chi-squared assessment:
        * if reduced chi-squared ~ 1, then "good" fit. 
        * the quantitative measure of this is: probability(theoretical reduced chi-squared > observed red. chi-squared)
            * a 5% significance level is generally used (i.e., the probability should be greater than 5%). 
    """

    # inputs
    save_fig = False
    num_bins = 11  # need to be small enough such that multiple samples per bin but not too small.
    num_constraints = 3  # do you know the mean and standard deviation should be, a-priori? If not, num_constraints = 3
    degrees_of_freedom = num_bins - num_constraints

    # file names
    fni = 'ia_values_{}'.format(method)
    df = pd.read_excel(join(path_results, fni + filetype))

    # ---

    # filter
    df = df[(df['zs'] > -50) & (df['zs'] < 50)]

    # ---

    # evaluate distribution of 'cms' at each axial position
    dist_by_z = True
    if dist_by_z:

        dist_names = ['norm']
        unique_zs = df['zs'].sort_values().unique()

        data = []
        for unique_z in unique_zs:
            # get all 'cms' values for this 'zs'
            sfp = df[df['zs'] == unique_z]['cms'].to_numpy()

            # fit the normal distribution to the distribution of 'cms' values
            # popt, pcov = curve_fit(normal_distribution, dfz)
            mu, std = norm.fit(sfp)

            # count number of values outside 3 standard deviations from the mean
            sfp_outliers = sfp[np.abs(sfp - mu) > std * 3]
            num_outliers = len(sfp_outliers)
            percent_outliers = num_outliers / len(sfp) * 100

            # calculate chi-squared

            # method from internet (uses frequency)
            chi_square_statistics = []
            percentile_bins = np.linspace(0, 100, num_bins)  # equi-distant bins of observed data (originally, 11)
            percentile_cutoffs = np.percentile(sfp, percentile_bins)
            observed_frequency, bins = (np.histogram(sfp, bins=percentile_cutoffs))
            cum_observed_frequency = np.cumsum(observed_frequency)

            # plot
            fig, ax = plt.subplots()
            axr = ax.twinx()
            ax.hist(sfp, bins=num_bins, alpha=0.75, color='gray')

            # Loop through candidate distributions
            for dist_id, distribution in enumerate(dist_names):
                # Set up distribution and get fitted distribution parameters
                dist = getattr(scipy.stats, distribution)
                param = dist.fit(sfp)
                loc_norm, scale_norm = dist.fit(sfp)

                # Get expected counts in percentile bins
                # cdf of fitted distribution across bins
                cdf_fitted = dist.cdf(percentile_cutoffs, *param)
                expected_frequency = []
                for bin in range(len(percentile_bins) - 1):
                    expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
                    expected_frequency.append(expected_cdf_area)

                # Chi-square Statistics
                expected_frequency = np.array(expected_frequency) * np.size(sfp)
                cum_expected_frequency = np.cumsum(expected_frequency)
                ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
                chi_square_statistics.append(ss)

                # reduced chi-square
                chi_square_reduced = ss / degrees_of_freedom  # NOTE: the expected value is = 1.

                if degrees_of_freedom == 5:
                    if chi_square_reduced < 1.25:
                        prob_chi = 28
                    elif chi_square_reduced < 1.5:
                        chi_adj = (chi_square_reduced - 1.25) / 0.25 * 9
                        prob_chi = 28 - chi_adj
                    elif chi_square_reduced < 1.75:
                        chi_adj = (chi_square_reduced - 1.5) / 0.25 * 7
                        prob_chi = 19 - chi_adj
                    elif chi_square_reduced < 2:
                        chi_adj = (chi_square_reduced - 1.75) / 0.25 * 4
                        prob_chi = 12 - chi_adj
                        prob_chi = 8
                    elif chi_square_reduced < 3:
                        chi_adj = (chi_square_reduced - 2) / 1 * 7
                        prob_chi = 8 - chi_adj
                    else:
                        prob_chi = 0
                else:
                    prob_chi = 0

                # ---

                # Plot probability density function

                # PDF(inside: 3 * sigma): ~99.7% of measurements
                x = np.linspace(mu - std * 3, mu + std * 3, 100)
                p = dist.pdf(x, *param)
                axr.plot(x, p, linewidth=1, color='g',
                         label=r'$(\tilde{\chi}^2_{o}, \mu, \sigma)=$' +
                               '({}, {}, {})'.format(np.round(chi_square_reduced, 1),
                                                     np.round(loc_norm, 3),
                                                     np.round(scale_norm, 3),
                                                     )
                         )

                # PDF(outside: 3 * sigma): ~0.3% of measurements num_outliers / len(sfp) * 100
                x = np.linspace(mu - std * 5, mu - std * 3, 10)
                axr.plot(x, dist.pdf(x, *param), linewidth=1, color='r',
                         label='{}/{} outliers \n ({}\%)'.format(num_outliers, len(sfp), np.round(percent_outliers, 1)))
                x = np.linspace(mu + std * 3, mu + std * 5, 10)
                axr.plot(x, dist.pdf(x, *param), linewidth=1, color='r')

                # store results
                data.append([unique_z, dist_id,
                             loc_norm, scale_norm,
                             ss, chi_square_reduced, prob_chi,
                             num_outliers, num_bins,
                             ])

            # plt.title(title)
            ax.set_ylabel('Counts')
            ax.set_xlabel(r'$S_{fp}$')
            ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
            axr.set_ylabel('PDF')
            axr.set_ylim(bottom=0, top=axr.get_ylim()[1] * 1.15)
            axr.legend(loc='upper left',
                       handlelength=1, borderpad=0.25, labelspacing=0.35, handletextpad=0.4, borderaxespad=0.25)
            ax.set_title('z={} '.format(unique_z) + r'$\mu m$' + ': ' +
                         r'$Prob(\tilde{\chi}^2 > \tilde{\chi}^2_{o})=$' + ' {}\%'.format(np.round(prob_chi, 1)))
            plt.tight_layout()
            if save_fig:
                plt.savefig(join(path_figs, 'fit-{}-Sfp-dist_to_z={}.png'.format(method, unique_z)))
            # plt.show()
            plt.close()

        # structure results
        df_res = pd.DataFrame(np.vstack(data),
                              columns=['z', 'dids', 'mu', 'sigma', 'chi', 'chi_red', 'prob_chi', 'outliers', 'bins'])
        dfg_res = df_res.groupby('dids').mean()

        # export results
        df_res.to_excel(join(path_results,
                             '{}-chisquares-mean-sigma_by_dist-and-z_bins={}.xlsx'.format(method, num_bins)),
                        index=False)
        dfg_res.to_excel(join(path_results,
                              '{}-mean-chisquares-mean-sigma_by_dist_bins={}.xlsx'.format(method, num_bins)),
                         index=False)

        print(dfg_res)

    # ---

    # evaluate normalized distribution of 'cms' across all axial positions

    dist_across_all_z = False
    if dist_across_all_z:

        dist_names = ['norm']
        unique_zs = df['zs'].sort_values().unique()

        # normalize distribution of 'cms' at each axial position by subtracting the mean 'cms'
        sfp_norms = []
        for unique_z in unique_zs:
            # get all 'cms' values for this 'zs'
            sfp = df[df['zs'] == unique_z]['cms'].to_numpy()
            mu, std = norm.fit(sfp)
            sfp_norms.append(sfp - mu)
        sfp_norms = np.hstack(sfp_norms)

        # evaluate distributions
        mu, std = norm.fit(sfp_norms)

        # count number of values outside 3 standard deviations from the mean
        sfp_outliers = sfp_norms[np.abs(sfp_norms - mu) > std * 3]
        num_outliers = len(sfp_outliers)
        percent_outliers = num_outliers / len(sfp_norms) * 100

        chi_square_statistics = []
        # 11 equi-distant bins of observed Data
        percentile_bins = np.linspace(0, 100, num_bins)
        percentile_cutoffs = np.percentile(sfp_norms, percentile_bins)
        observed_frequency, bins = (np.histogram(sfp_norms, bins=percentile_cutoffs))
        cum_observed_frequency = np.cumsum(observed_frequency)

        # plot
        fig, ax = plt.subplots()
        axr = ax.twinx()
        ax.hist(sfp_norms, bins=num_bins, color='gray', alpha=0.5)

        # Loop through candidate distributions
        for dist_id, distribution in enumerate(dist_names):
            dist = getattr(scipy.stats, distribution)
            param = dist.fit(sfp_norms)
            loc_norm, scale_norm = dist.fit(sfp_norms)

            # Get expected counts in percentile bins
            # cdf of fitted distribution across bins
            cdf_fitted = dist.cdf(percentile_cutoffs, *param)
            expected_frequency = []
            for bin in range(len(percentile_bins) - 1):
                expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
                expected_frequency.append(expected_cdf_area)

            # Chi-square Statistics
            expected_frequency = np.array(expected_frequency) * np.size(sfp_norms)
            cum_expected_frequency = np.cumsum(expected_frequency)
            ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
            chi_square_statistics.append(ss)

            # PDF
            x = np.linspace(mu - std * 3, mu + std * 3, 100)
            p = dist.pdf(x, *param)
            axr.plot(x, p, linewidth=0.75, color='green', label='({}, \n {}, \n {})'.format(np.round(ss, 1),
                                                                                            np.round(loc_norm, 4),
                                                                                            np.round(scale_norm, 4))
                     )

            # PDF(outside: 3 * sigma): ~0.3% of measurements num_outliers / len(sfp) * 100
            x = np.linspace(mu - std * 5, mu - std * 3, 10)
            axr.plot(x, dist.pdf(x, *param), linewidth=0.75, color='r',
                     label='{}/{} outliers \n ({}\%)'.format(num_outliers, len(sfp_norms),
                                                             np.round(percent_outliers, 1)))
            x = np.linspace(mu + std * 3, mu + std * 5, 10)
            axr.plot(x, dist.pdf(x, *param), linewidth=0.75, color='r')

        # plt.title(title)
        ax.set_ylabel('Counts')
        ax.set_xlabel(r'$S_{fp}(z) - \overline{S_{fp}}$')
        axr.set_ylabel('PDF')
        axr.set_ylim(bottom=0)
        axr.legend(loc='upper left', bbox_to_anchor=(1.1, 1), title=r'$(\chi^2_{s}, \mu, \sigma)$', handlelength=1.25)
        plt.tight_layout()
        plt.savefig(join(path_figs, 'fit-{}-norm-Sfp-dist_bins={}.png'.format(method, num_bins)))
        plt.close()

# ---

# compare standard deviation of normal distribution of S_fp
compare_ia_dist = False
if compare_ia_dist:
    # file names
    fni = 'idpt-chisquares-mean-sigma_by_dist-and-z'
    fns = 'spct-chisquares-mean-sigma_by_dist-and-z'

    dfi = pd.read_excel(join(path_results, fni + filetype))
    dfs = pd.read_excel(join(path_results, fns + filetype))

    i_bins = dfi['bins'].iloc[0]
    s_bins = dfs['bins'].iloc[0]

    if i_bins != s_bins:
        raise ValueError('The number of bins must be equal to compare IDPT and SPCT')


    # ---

    def fline(x, a, b):
        return a * x + b


    popt_i, pcov = curve_fit(fline, dfi.z, dfi.mu)
    popt_s, pcov = curve_fit(fline, dfs.z, dfs.mu)

    # plot
    ms = 1
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

    ax1.plot(dfi.z, dfi.mu, '-o', ms=ms, label='IDPT')
    ax1.plot(dfs.z, dfs.mu, '-o', ms=ms, label='SPCT')
    ax1.plot(dfi.z, fline(dfi.z, *popt_i), linewidth=0.75, linestyle='--', color='navy',
             label='{}'.format(np.round(popt_i[0], 5)))
    ax1.plot(dfs.z, fline(dfs.z, *popt_s), linewidth=0.75, linestyle='--', color='lime',
             label='{}'.format(np.round(popt_s[0], 5)))
    ax1.grid(alpha=0.25)
    ax1.set_ylabel(r'$\overline{S_{fp}}(z)$')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax2.plot(dfi.z, dfi.sigma, '-o', ms=ms, label='IDPT')
    ax2.plot(dfs.z, dfs.sigma, '-o', ms=ms, label='SPCT')
    ax2.axhline(dfi.sigma.mean(), linewidth=0.5, color='navy')
    ax2.axhline(dfs.sigma.mean(), linewidth=0.5, color='darkgreen')
    ax2.set_ylabel(r'$\sigma_{S_{fp}}(z)$')
    ax2.set_ylim(bottom=0)

    ax3.plot(dfi.z, dfi['chi'], '-o', ms=ms, label='IDPT')
    ax3.plot(dfs.z, dfs['chi'], '-o', ms=ms, label='SPCT')
    ax3.set_ylabel(r'$\chi^2_{s}$')
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(join(path_figs, 'compare_mean-sigma-chi_by_z_bins={}.png'.format(i_bins)))
    plt.show()
    plt.close()

# ---

# plot fitted cubic function for both IDPT and SPCT on the same figure
compare_ia_cubic = False
if compare_ia_cubic:

    # create figure
    fig, (ax2, ax3) = plt.subplots(nrows=2, sharex=True,
                                   figsize=(size_x_inches * 1.25, size_y_inches * 1.5),
                                   gridspec_kw={'height_ratios': [1, 0.5]})

    scatter1_colors = [sciblue, scigreen]
    plot2_colors = ['black', 'red']
    clrs_radials = [['black', 'red'], ['purple', 'blue']]

    for method, scatter1_color, plot2_color, clrs_radial in zip(['idpt', 'spct'], scatter1_colors, plot2_colors, clrs_radials):

        # file names
        fni = 'ia_values_{}'.format(method)
        df = pd.read_excel(join(path_results, fni + filetype))

        # ---

        # filter
        df = df[(df['zs'] > -50) & (df['zs'] < 50)]

        # processing
        df['clr'] = 1
        df['clr'] = df.clr.where(((df.cms > 0) & (df.zs > 0)) | ((df.cms < 0) & (df.zs < 0)), 0)
        dftp = df[df['clr'] > 0.5]
        dffp = df[df['clr'] < 0.5]

        # setup
        y2_label = r'$S_{fp}$'
        y3_label = r'$S_{fp}$'

        y2l, d2l = 0.08, 0.0
        y3l, d3l = 0.08, 0.0

        y2_lim = [-y2l - d2l, y2l + d2l]
        y2_ticks = [-y2l, 0, y2l]
        y3_lim = [-y3l - d3l, y3l + d3l]
        y3_ticks = [-y3l, 0, y3l]

        plot_tp_fp = False
        plot_tp_fp_quadrants = True
        scatter1_label = 'Exp.'
        scatter1_size = 1.5
        scatter1_markerscale = 3.5
        scatter1_alpha = 0.5
        scatter1_zorder = 3.3
        scatter1_label_true_positive = 'TP'
        scatter1_label_false_positive = 'FP'
        scatter1_cmap_true_positive = 'Blues_r'  # 'RdYlGn_r'
        scatter1_cmap_false_positive = 'Blues_r'  # 'RdYlGn'

        plot2_linestyle = '-'
        plot2_linewidth = 0.5
        plot2_zorder = 3.4
        plot2_alpha = 1.0
        plot2_label = 'Fit'
        plot2_std_alpha = 0.0
        plot2_std_linestyle = '-'
        plot2_std_linewidth = 0.25
        fill2_alpha = 0.0
        fill2_zorder = 1.01
        fill2_alpha_quadrants = 0.09
        fill2_zorder_quadrants = 0.1
        grid2_alpha = 0.75

        # radial dependence
        grid3_alpha = 0.25
        ms = 2
        capsize = 1.5
        elinewidth = 0.75

        # setup
        x_label = r'$z \: (\mu m)$'
        x_lim = [-53.5, 53.5]
        x_ticks = np.arange(-50, 51, 10)  # [-50, 0, 50]
        y_label = r'$C_{m}(z < z_{f}) / C_{m}(z > z_{f})$'

        if method == 'idpt':
            y_lim = [-0.045, 0.045]
            y_ticks = [-0.04, -0.02, 0, 0.02, 0.04]
        elif method == 'spct':
            y_lim = [-0.065, 0.065]
            y_ticks = [-0.05, 0, 0.05]
        else:
            raise ValueError("Need to define method as 'idpt' or 'spct'.")

        # define radial dependence for multi-plots
        rl, rr, dr = 120, 450, 1
        r_bins = [rl, rl + dr, rr - dr, rr]

        # ---

        # plot

        # (1) scatter, raw
        if plot_tp_fp:
            ax2.scatter(dftp.zs, dftp.cms, c=dftp.clr, s=1, marker='.',  # label=scatter1_label_true_positive,
                        cmap=scatter1_cmap_false_positive,
                        alpha=scatter1_alpha, zorder=scatter1_zorder)
            ax2.scatter(dffp.zs, dffp.cms, c=dffp.clr, s=1, marker='.',  # label=scatter1_label_false_positive,
                        cmap=scatter1_cmap_true_positive,
                        alpha=scatter1_alpha, zorder=scatter1_zorder)
            ax2.scatter(dftp.iloc[0].zs, dftp.iloc[0].cms + 1, c=dftp.iloc[0].clr, s=1, marker='.',
                        label=scatter1_label_true_positive,
                        cmap=scatter1_cmap_false_positive, alpha=1)
            ax2.scatter(dffp.iloc[0].zs, dffp.iloc[0].cms + 1, c=dffp.iloc[0].clr, s=1, marker='.',
                        label=scatter1_label_false_positive,
                        cmap=scatter1_cmap_true_positive, alpha=1)
        else:
            ax2.scatter(df.zs, df.cms, s=scatter1_size, marker='.',
                        color=scatter1_color,
                        alpha=scatter1_alpha, zorder=scatter1_zorder)
            ax2.scatter(df.iloc[0].zs, df.iloc[0].cms + 1, s=scatter1_size, marker='.', label=scatter1_label,
                        color=scatter1_color, alpha=1)

        # ---

        # (2) fill between, fitted curve
        dfim = df.groupby('zs').mean().reset_index()
        dfis = df.groupby('zs').std().reset_index()

        # fit curve to mean value
        imx = dfim.zs.to_numpy()
        imy = dfim.cms.to_numpy()
        imy_upper = dfim.cms.to_numpy() + dfis.cms.to_numpy()
        imy_lower = dfim.cms.to_numpy() - dfis.cms.to_numpy()

        # fit
        popti, pcov = curve_fit(functions.cubic_slide, imx, imy)
        popti_upper, pcov = curve_fit(functions.cubic_slide, imx, imy_upper)
        popti_lower, pcov = curve_fit(functions.cubic_slide, imx, imy_lower)

        # resample
        fmx = np.linspace(imx.min(), imx.max(), 100)
        fmy = functions.cubic_slide(fmx, *popti)
        fmy_upper = functions.cubic_slide(fmx, *popti_upper)
        fmy_lower = functions.cubic_slide(fmx, *popti_lower)

        # idpt
        ax2.plot(fmx, fmy, linestyle=plot2_linestyle, color=plot2_color, zorder=plot2_zorder, label=plot2_label,
                 linewidth=plot2_linewidth, alpha=plot2_alpha)
        ax2.plot(fmx, fmy_upper, linestyle=plot2_std_linestyle, color=plot2_color, alpha=plot2_std_alpha,
                 linewidth=plot2_std_linewidth, zorder=fill2_zorder)
        ax2.plot(fmx, fmy_lower, linestyle=plot2_std_linestyle, color=plot2_color, alpha=plot2_std_alpha,
                 linewidth=plot2_std_linewidth, zorder=fill2_zorder)
        ax2.fill_between(fmx, y1=fmy_upper, y2=fmy_lower, color=plot2_color, ec='none',
                         alpha=fill2_alpha, zorder=fill2_zorder)

        if plot_tp_fp_quadrants:
            # fill green: True Positive
            ax2.fill_between([x_lim[0], 0], y1=y2_lim[1], y2=0, color=scigreen, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants,
                             label=scatter1_label_true_positive,
                             )
            ax2.fill_between([0, x_lim[1]], y1=0, y2=y2_lim[0], color=scigreen, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants)
            # fill red: False Positive
            ax2.fill_between([x_lim[0], 0], y1=0, y2=y2_lim[0], color=scired, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants)
            ax2.fill_between([0, x_lim[1]], y1=y2_lim[1], y2=0, color=scired, ec='none',
                             alpha=fill2_alpha_quadrants, zorder=fill2_zorder_quadrants,
                             label=scatter1_label_false_positive,
                             )

        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=scatter1_markerscale)
        ax2.grid(color='k', alpha=grid2_alpha)
        ax2.set_ylabel(y2_label)
        ax2.set_ylim(y2_lim)
        ax2.set_yticks(y2_ticks)
        ax2.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # ---

        # (3) error bars, radial dependence

        # bin by 'r'
        column_to_bin = 'r'
        bins = np.array(r_bins) / microns_per_pixel
        df = bin.bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=1)
        bins_r = df.bin.unique()
        bins_r.sort()

        for br, clr, lbl_ind, zord in zip([bins_r[0], bins_r[-1]], clrs_radial, [r'$<$', r'$>$'], [3.2, 3.1]):
            dfb = df[df['bin'] == br]

            print("Number of particles in bin {} = {}".format(br, len(dfb.groupby('id').count())))
            dfm = dfb.groupby('zs').mean().reset_index()
            dfs = dfb.groupby('zs').std().reset_index()

            ax3.errorbar(dfm.zs, dfm.cms, yerr=dfs.cms,
                         fmt='-o', ms=ms, capsize=capsize, elinewidth=elinewidth,
                         color=clr,
                         zorder=zord,
                         label=lbl_ind + '{}'.format(int(np.round(br * microns_per_pixel, 0))))

        ax3.grid(alpha=grid3_alpha)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r \: (\mu m)$')
        ax3.set_ylabel(y3_label)
        ax3.set_ylim(y3_lim)
        ax3.set_yticks(y3_ticks)
        ax3.set_xlabel(x_label)
        ax3.set_xlim(x_lim)
        ax3.set_xticks(x_ticks)
        ax3.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)


    plt.tight_layout()
    plt.savefig(path_figs + '/reference-only_compare-idpt-spct_post_ia_fit_on_raw_and_radial' + fig_type)
    plt.show()
    plt.close()

    # ---


# ---

print("Analysis completed without errors.")