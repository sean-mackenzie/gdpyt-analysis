# 02.06.22 - local axial and radial displacements per membrane

# imports
import os
from os.path import join

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, Akima1DInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import analyze
from utils import boundary, functions, io, bin, plotting
from utils.plotting import lighten_color

# ---

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

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 0. Experimental Parameters

mag_eff = 5.0
numerical_aperture = 0.3
pixel_size = 16
depth_of_focus = functions.depth_of_field(mag_eff, numerical_aperture, 600e-9, 1.0, pixel_size=pixel_size * 1e-6) * 1e6
microns_per_pixel = 3.2
exposure_time = 40e-3
frame_rate = 24.444
time_per_frame = 1 / frame_rate
E_silpuran = 500e3
poisson = 0.5
t_membrane = 20e-6
t_membrane_norm = 20

# pressure application
start_frame = 39
start_time = start_frame / frame_rate

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. Processing Parameters
padding_during_idpt_test_calib = 15  # 10
image_length = 512
img_xc = 256
img_yc = 256

""" --- MEMBRANE SPECIFIC PARAMETERS --- """

# mask lower right membrane
xc_lr, yc_lr, r_edge_lr = 423, 502, 252
circle_coords_lr = [xc_lr, yc_lr, r_edge_lr]

# mask upper left membrane
xc_ul, yc_ul, r_edge_ul = 167, 35, 157
circle_coords_ul = [xc_ul, yc_ul, r_edge_ul]

# mask left membrane
xc_ll, yc_ll, r_edge_ll = 12, 289, 78
circle_coords_ll = [xc_ll, yc_ll, r_edge_ll]

# mask middle
xc_mm, yc_mm, r_edge_mm = 177, 261, 31
circle_coords_mm = [xc_mm, yc_mm, r_edge_mm]

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2.5 IMPORTANT PHYSICAL POSITIONS

# axial positions
z_f_from_calib = 140
z_offset_lr = 5
z_offset_ul = 2
z_inital_focal_plane_bias_errors = np.max([z_offset_lr, z_offset_ul]) + 5

# exclude outliers
pids_saturated = [12, 13, 18, 34, 39, 49, 66, 78]
exclude_pids = [39, 61]
bad_pids = [12, 13, 18, 34, 39, 49, 61, 66, 78]

# ---


# ----------------------------------------------------------------------------------------------------------------------
# 2. Setup Files and Variables

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_rmse'
top_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/'
analysis_dir = 'results-09.15.22_idpt-' + 'subpix'  # 'sweep-ttemp' 'subpix'
data_dir = top_dir + analysis_dir

if analysis_dir.endswith('ttemp'):
    path_data_fn = '/results/dz1_ttemp{}'
    dzs = [9, 11, 13]
    save_id = 'ttemp'
    gbl_xlbl = r'$l^{t} \: (pix.)$'
    mod = 1
elif analysis_dir.endswith('subpix'):
    path_data_fn = '/results/dz{}'
    dzs = [1, 2, 3, 4, 5, 6, 7, 8, 11, 15]
    save_id = 'dzc'
    gbl_xlbl = r'$\Delta_{c} z \: (\mu m)$'
    mod = 2
else:
    raise ValueError('Analysis dir not understood.')

# setup
tid = 1
dz_id = tid

# data
path_data = data_dir + path_data_fn.format(dz_id)

# results
path_results = join(base_dir, 'results')
if not os.path.exists(path_results):
    os.makedirs(path_results)

# ---

# ---

# ---

analyze_uncertainty = True
if analyze_uncertainty:

    plot_rmse_by_z_per_particle = False
    plot_rmse_by_r_per_particle = False
    plot_rmse_by_zr_per_particle = False
    plot_rmse_by_zr_per_particle_pos_vs_neg_dz = False
    plot_rmse_by_max_dz = True

    save_figs = True
    path_results_uncertainty = path_results

    # ---

    data_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_displacement'
    path_results_tid_coords = join(data_dir, 'results', 'dz{}'.format(tid))

    read_raw_or_reconstruction = 'raw'  # 'reconstruction'
    df = pd.read_excel(
        path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

    # ---

    # binning
    z_bins = 30
    r_bins = 11

    z_bins = np.arange(-120, 121, z_bins)
    r_bins = np.linspace(0.0, 1.0, r_bins)
    bins = 25  # 16  # phi/rmse-z (s.r.)

    # plot setup
    ms = 5

    # ---

    # create necessary columns
    df['z_true'] = df['d_dz'] - df['z_offset']
    df['error'] = df['z_corr'] - df['z_true']

    # store a copy
    dfo = df.copy()

    # -

    # rmse-z by z

    if plot_rmse_by_z_per_particle:
        dfb = bin.bin_local_rmse_z(df, column_to_bin='z_true', bins=z_bins, min_cm=0.5, z_range=None,
                                   round_to_decimal=2,
                                   df_ground_truth=None, dropna=True, error_column=None, include_xy=False)

        # plot
        fig, ax = plt.subplots()
        ax.plot(dfb.index, dfb.rmse_z, '-o', ms=ms)
        ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        # ax.set_ylim(ylim_rmse)
        ax.set_xlabel(r'$z \: (\mu m)$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(path_results_uncertainty + '/rmse_z_by_z.png')
        plt.show()
        plt.close()

        # ---

    # ---

    # ---

    # rmse-z by r
    if plot_rmse_by_r_per_particle:

        # create necessary columns
        df = dfo.copy()

        # -

        # lower right membrane
        dfmb = df[df['memb_id'] == 1]
        dfmb['nd_r'] = dfmb.r / (r_edge_lr * microns_per_pixel)
        dfb1 = bin.bin_local_rmse_z(dfmb, column_to_bin='nd_r', bins=r_bins, round_to_decimal=2)

        # upper left membrane
        dfmb = df[df['memb_id'] == 2]
        dfmb['nd_r'] = dfmb.r / (r_edge_ul * microns_per_pixel)
        dfb2 = bin.bin_local_rmse_z(dfmb, column_to_bin='nd_r', bins=r_bins, round_to_decimal=2)

        # plot

        fig, ax = plt.subplots()

        ax.plot(dfb1.index, dfb1.rmse_z, '-o', ms=ms, label=800)
        ax.plot(dfb2.index, dfb2.rmse_z, '-o', ms=ms, label=500)

        ax.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax.set_xlabel(r'$r/a$')
        # ax.set_xlim([-0.03, 1.03])
        ax.legend(title=r'$r \: (\mu m)$')
        plt.tight_layout()
        if save_figs:
            plt.savefig(path_results_uncertainty + '/norm-rmse_z_by_r.svg')
        plt.show()
        plt.close()

        # ---

    # ---

    # rmse-z by z and r
    if plot_rmse_by_zr_per_particle:

        # get original
        df = dfo.copy()

        # rmse-z by z: per-particle
        dfb_z = bin.bin_local_rmse_z(df, column_to_bin='z_true', bins=z_bins, min_cm=0.5, z_range=None,
                                     round_to_decimal=2,
                                     df_ground_truth=None, dropna=True, error_column=None, include_xy=False)

        # rmse-z by r: per-particle
        dfb_r = bin.bin_local_rmse_z(df, column_to_bin='nd_r', bins=r_bins, round_to_decimal=2)

        # plot
        fig, (ax1, ax2) = plt.subplots(nrows=2)

        ax1.plot(dfb_z.index, dfb_z.rmse_z, '-o', ms=ms)
        ax2.plot(dfb_r.index, dfb_r.rmse_z, '-o', ms=ms)

        ax1.set_ylabel(r'$\sigma_{z}^{\delta}(z) \: (\mu m)$')
        ax1.set_xlabel(r'$z \: (\mu m)$')

        ax2.set_ylabel(r'$\sigma_{z}^{\delta}(r) \: (\mu m)$')
        ax2.set_xlabel(r'$r/a$')

        plt.tight_layout()
        if save_figs:
            plt.savefig(path_results_uncertainty + '/rmse_z_by_zr.png')
        plt.show()
        plt.close()

    # ---

    # rmse-z by z and r for both positive deflections and negative deflections
    if plot_rmse_by_zr_per_particle_pos_vs_neg_dz:

        # get frames where DZ-LR is positive and negative
        dfz = pd.read_excel(
            path_data + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(dz_id))

        frames_dz_pos = dfz[dfz['rz_lr'] > 15].frame.values
        frames_dz_neg = dfz[dfz['rz_lr'] < -15].frame.values

        # get original
        df = dfo.copy()
        df = df[df['memb_id'].isin([1, 2])]
        df = df.dropna(subset=['z_corr'])
        df['z'] = df['z_corr']
        dfp = df[df['frame'].isin(frames_dz_pos)]
        dfn = df[df['frame'].isin(frames_dz_neg)]

        # export mean rmse
        dfb_rp = bin.bin_local_rmse_z(dfp, column_to_bin='nd_r', error_column='dz_error', dropna_cols='z_corr',
                                      bins=1, round_to_decimal=2, dropna=True)
        dfb_rp['dz_dir'] = 1
        dfb_rn = bin.bin_local_rmse_z(dfn, column_to_bin='nd_r', error_column='dz_error', dropna_cols='z_corr',
                                      bins=1, round_to_decimal=2, dropna=True)
        dfb_rn['dz_dir'] = -1
        dfb_compare_pos_neg = pd.concat([dfb_rp, dfb_rn])
        dfb_compare_pos_neg = dfb_compare_pos_neg[['dz_dir', 'cm', 'drg', 'dz_error', 'rmse_z', 'num_bind', 'num_meas',
                                                   'd_dz_corr', 'd_p', 'nd_p', 'd_k', 'nd_k']]
        dfb_compare_pos_neg.to_excel(path_results_uncertainty + '/mean-rmse_z_compare_pos_vs_neg_dz.xlsx')

        # ---

        # rmse-z by z: per-particle
        dfb_zp = bin.bin_local_rmse_z(dfp, column_to_bin='z_true', error_column='dz_error', dropna_cols='z_corr',
                                      bins=z_bins, round_to_decimal=2, dropna=True)
        dfb_zn = bin.bin_local_rmse_z(dfn, column_to_bin='z_true', error_column='dz_error', dropna_cols='z_corr',
                                      bins=z_bins, round_to_decimal=2, dropna=True)

        # rmse-z by r: per-particle
        dfb_rp = bin.bin_local_rmse_z(dfp, column_to_bin='nd_r', error_column='dz_error', dropna_cols='z_corr',
                                      bins=r_bins, round_to_decimal=2, dropna=True)
        dfb_rn = bin.bin_local_rmse_z(dfn, column_to_bin='nd_r', error_column='dz_error', dropna_cols='z_corr',
                                      bins=r_bins, round_to_decimal=2, dropna=True)

        # plot
        fig, (ax1, ax2) = plt.subplots(nrows=2)

        ax1.plot(dfb_zp.index, dfb_zp.rmse_z, '-o', ms=ms, label=r'$\Delta z > 15$')
        ax1.plot(dfb_zn.index, dfb_zn.rmse_z, '-o', ms=ms, label=r'$\Delta z < -15$')

        ax2.plot(dfb_rp.index, dfb_rp.rmse_z, '-o', ms=ms)
        ax2.plot(dfb_rn.index, dfb_rn.rmse_z, '-o', ms=ms)

        ax1.legend()
        ax1.set_ylabel(r'$\sigma_{z}^{\delta}(z) \: (\mu m)$')
        ax1.set_xlabel(r'$z \: (\mu m)$')

        ax2.set_ylabel(r'$\sigma_{z}^{\delta}(r) \: (\mu m)$')
        ax2.set_xlabel(r'$r/a$')

        plt.tight_layout()
        if save_figs:
            plt.savefig(path_results_uncertainty + '/rmse_z_by_zr_pos_vs_neg_dz.png')
        plt.show()
        plt.close()


    # ---

    if plot_rmse_by_max_dz:
        df = pd.read_excel(
            path_data + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest_vertical.xlsx'.format(dz_id))

        dflr = df[df.memb_id.isin([1])]
        dful = df[df.memb_id.isin([2])]

        df = df[df.memb_id.isin([1, 2])]

        # -

        # create necessary columns
        column_to_bin = 'rz'
        column_to_count = 'frame'
        round_to_decimal = 2
        return_groupby = True

        # -

        # local binning
        dfm, dfstd = bin.bin_generic(df, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

        # plot: rmse_z(w)
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(dfm.bin, dfm.fit_percent_meas, '-o', ms=ms)
        ax2.plot(dfm.bin, dfm.fit_rmse, '-o', ms=ms)

        ax1.set_ylabel(r'$\phi_{S.R.}$')
        ax2.set_xlabel(r'$w_{o} \: (\mu m)$')
        ax2.set_ylabel(r'$\sigma_{S.R.} \: (\mu m)$')
        plt.tight_layout()
        plt.savefig(path_results_uncertainty + '/bin_rmsez-and-nef-percent_by_z-max_both-lr-ul_fix-ylabel.png')
        plt.show()
        plt.close()

        # ---

        # plot: rmse_z(w) / w
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(dfm.bin, dfm.fit_percent_meas, '-o', ms=ms)
        ax2.plot(dfm.bin, dfm.fit_rmse / dfm.bin.abs(), '-o', ms=ms)

        ax1.set_ylabel(r'$\phi_{S.R.}$')
        ax2.set_xlabel(r'$w_{o} \: (\mu m)$')
        ax2.set_ylabel(r'$\sigma_{S.R.} / w_{o}$')
        ax2.set_ylim([-0.01, 0.1])
        plt.tight_layout()
        # plt.savefig(path_results_uncertainty + '/bin_rmsez-and-nef-percent_by_z-max_both-lr-ul.png')
        plt.show()
        plt.close()

        # ---

        # local binning
        dfmlr, dfstdlr = bin.bin_generic(dflr, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)
        dfmul, dfstdul = bin.bin_generic(dful, column_to_bin, column_to_count, bins, round_to_decimal, return_groupby)

        # plot: rmse_z(w)
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(dfmlr.bin, dfmlr.fit_percent_meas, '-o', ms=ms)
        ax2.plot(dfmlr.bin, dfmlr.fit_rmse, '-o', ms=ms, label=r'$800$')

        ax1.plot(dfmul.bin, dfmul.fit_percent_meas, '-o', ms=ms)
        ax2.plot(dfmul.bin, dfmul.fit_rmse, '-o', ms=ms, label=r'$500$')

        ax1.set_ylabel(r'$\phi_{S.R.}$')
        ax2.set_xlabel(r'$w_{o} \: (\mu m)$')
        ax2.set_ylabel(r'$\sigma_{S.R.}$')
        ax2.legend(title=r'$r \: (\mu m)$')
        plt.tight_layout()
        # plt.savefig(path_results_uncertainty + '/bin_rmsez-and-nef-percent_by_z-max_both-lr-ul.png')
        plt.show()
        plt.close()

    # ---

# ---

compare_mean_uncertainty = False
if compare_mean_uncertainty:
    # write data
    path_results_global = join(path_results, 'global-rmse')
    if not os.path.exists(path_results_global):
        os.makedirs(path_results_global)

    # read data
    fn_mean_rmse_sr = 'mean-surface-reconstruction-rmse_both-lr-ul.xlsx'
    fn_mean_rmse_pp = 'mean-per-particle_rmse.xlsx'

    # compute aggregated global compute
    compute_aggregated_global_rmse = False
    if compute_aggregated_global_rmse:
        # processing
        dfsr = []
        dfpp = []
        for dz_id in dzs:
            df_sr = pd.read_excel(data_dir + path_data_fn.format(dz_id) + '/uncertainty/' + fn_mean_rmse_sr)
            df_pp = pd.read_excel(data_dir + path_data_fn.format(dz_id) + '/uncertainty/' + fn_mean_rmse_pp)

            dfsr.append(df_sr)
            dfpp.append(df_pp)

        dfsr = pd.concat(dfsr)
        dfpp = pd.concat(dfpp)

        dfsr['bin'] = dzs
        dfpp['bin'] = dzs

        # export
        dfsr.to_excel(path_results_global + '/compare-rmse-percent-meas-sr_by_{}.xlsx'.format(save_id))
        dfpp.to_excel(path_results_global + '/compare-rmse-percent-meas-pp_by_{}.xlsx'.format(save_id))

        # ---

        # plotting

        # setup
        x = 'bin'
        y11 = 'rmse_z'
        y12 = 'fit_rmse'
        y21 = 'percent_meas'
        y22 = 'fit_percent_meas'

        # plot: rmse per-particle and rmse surface-reconstruction
        compare_pp_sr = True
        if compare_pp_sr:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]})

            p1, = ax2.plot(dfpp[x] * mod, dfpp[y11], '-o', label=r'$\sigma_{i}$')
            p2, = ax2.plot(dfsr[x] * mod, dfsr[y12], '-o', label=r'$\sigma_{S.R.}$')

            # ax2.plot(dfpp[x], dfpp[y21], '-o', color=p1.get_color(), label='P.P.')
            ax1.plot(dfsr[x] * mod, dfsr[y22], '-o', color=p2.get_color(), label='S.R.')

            ax1.set_ylabel(r'$\overline{\phi}$')
            ax2.legend()
            ax2.set_ylabel(r'$\overline{\sigma_{z}} \: (\mu m)$')
            ax2.set_xlabel(gbl_xlbl)

            plt.tight_layout()
            # plt.savefig(path_results_global + '/compare-rmse-percent-meas-pp-sr_by_{}.png'.format(save_id))
            plt.show()

        # ---

        # plot: rmse per-particle and rmse surface-reconstruction
        compare_dz_sr = True
        if compare_dz_sr:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

            p2, = ax2.plot(dfsr[x] * mod, dfsr[y12], '-o', label=r'$\sigma_{S.R.}$')
            ax1.plot(dfsr[x] * mod, dfsr[y22], '-o', color=p2.get_color(), label='S.R.')

            ax1.set_ylabel(r'$\overline{\phi}$')
            ax2.set_ylabel(r'$\overline{\sigma_{S.R.}} \: (\mu m)$')
            ax2.set_xlabel(gbl_xlbl)

            plt.tight_layout()
            # plt.savefig(path_results_global + '/compare-rmse-percent-meas-sr_by_{}.png'.format(save_id))
            plt.show()

        # ---

    # ---

    else:

        # surface reconstruction
        dfsr_dzc = pd.read_excel(path_results_global + '/compare-rmse-percent-meas-sr_by_dzc.xlsx')
        dfsr_ttemp = pd.read_excel(path_results_global + '/compare-rmse-percent-meas-sr_by_ttemp.xlsx')

        # per-particle
        # dfpp_dzc = pd.read_excel(path_results_global + '/compare-rmse-percent-meas-pp_by_dzc.xlsx')
        # dfpp_ttemp = pd.read_excel(path_results_global + '/compare-rmse-percent-meas-pp_by_ttemp.xlsx')

        # ---

        # plotting

        # setup
        x = 'bin'
        y1 = 'fit_rmse'
        y2 = 'fit_percent_meas'

        # plot
        fig, axs = plt.subplots(2, 2, layout="constrained")

        axs[0, 0].plot(dfsr_dzc[x] * 2, dfsr_dzc[y2], '-o')
        axs[1, 0].plot(dfsr_dzc[x] * 2, dfsr_dzc[y1], '-o', label=r'$l^{t} = 11 \: pix.$')
        axs[1, 0].legend(loc='upper left', handlelength=1.2, handletextpad=0.4)

        axs[0, 1].plot(dfsr_ttemp[x], dfsr_ttemp[y2], '-s')
        axs[1, 1].plot(dfsr_ttemp[x], dfsr_ttemp[y1], '-s', label=r'$\Delta_{c} z = 2 \: \mu m$')
        axs[1, 1].legend(loc='upper left', handlelength=1.2, handletextpad=0.4)

        axs[0, 0].set_ylim([0.62, 0.88])
        axs[0, 1].set_ylim([0.62, 0.88])

        axs[1, 0].set_ylim([2.9, 4.3])
        axs[1, 1].set_ylim([2.9, 4.3])

        axs[0, 0].set_ylabel(r'$\overline{\phi}_{S.R.}$')
        axs[1, 0].set_ylabel(r'$\overline{\sigma}_{S.R.} \: (\mu m)$')

        axs[1, 0].set_xlabel(r'$\Delta_{c} z \: (\mu m)$')
        axs[1, 1].set_xlabel(r'$l^{t} \: (pix.)$')

        axs[0, 0].set_xticklabels([])
        axs[0, 0].tick_params(axis='both', which='minor',
                              bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        axs[1, 0].tick_params(axis='both', which='minor',
                              bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # right sub plots

        axs[0, 1].set_yticklabels([])
        axs[0, 1].set_xticks([9, 11, 13], labels=[])
        axs[1, 1].set_xticks([9, 11, 13])
        axs[1, 1].set_yticklabels([])

        axs[0, 1].tick_params(axis='both', which='minor',
                              bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        axs[1, 1].tick_params(axis='both', which='minor',
                              bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=2 / 72, wspace=0.15,  hspace=0.1)
        plt.savefig(path_results_global + '/compare-rmse-percent-meas-sr_by_dzc-and-ttemp.png')
        plt.show()


# ---

# ---

analyze_pressure_pretension_others = False
if analyze_pressure_pretension_others:

    path_results_pk = join(path_results, 'pressure-pretension-others')
    if not os.path.exists(path_results_pk):
        os.makedirs(path_results_pk)

    # ---

    # read
    fp = '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(dz_id)
    df = pd.read_excel(path_data + fp)

    # ---

    # plot: pressure / pre-tension by deflection

    memb_strs = ['lr', 'ul']
    split_pretension = ['all']  # ['low', 'high', 'all']
    split_value = 50
    scatter_size = 1

    for splits in split_pretension:

        for mstr in memb_strs:

            path_results_memb_pk = join(path_results_pk, 'memb-{}'.format(mstr))
            if not os.path.exists(path_results_memb_pk):
                os.makedirs(path_results_memb_pk)

            path_results_split_pk = join(path_results_memb_pk, 'pretension-{}'.format(splits))
            if not os.path.exists(path_results_split_pk):
                os.makedirs(path_results_split_pk)

            # define plot variables
            x = 'rz_{}'.format(mstr)
            xb = 'rz_{}_abs'.format(mstr)
            y1 = 'fit_{}_pressure'.format(mstr)
            y2 = 'fit_{}_pretension'.format(mstr)
            y3 = 'theta_{}_deg'.format(mstr)
            y3b = 'theta_{}_peak_deg'.format(mstr)

            # split
            if splits == 'low':
                dfs = df[df[y2] < split_value]
            elif splits == 'high':
                dfs = df[df[y2] > split_value]
            else:
                dfs = df.copy()

            # get positive and negative deflections
            dfp = dfs[dfs['rz_{}'.format(mstr)] > 0]
            dfn = dfs[dfs['rz_{}'.format(mstr)] < 0]

            # ---

            # plot each variable
            for logx in [False, True]:
                for y in [y1, y2, y3, y3b]:
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(size_x_inches * 2, size_y_inches * 1))
                    ax1.scatter(dfp[xb], dfp[y].abs(), s=scatter_size)
                    ax1.scatter(dfn[xb], dfn[y].abs(), s=scatter_size)
                    ax1.set_ylabel(y)
                    ax1.set_yscale('log')
                    ax1.set_xlabel(xb)

                    ax2.scatter(dfp[x], dfp[y].abs(), s=scatter_size, label=r'$+\Delta w_{o}$')
                    ax2.scatter(dfn[x], dfn[y].abs(), s=scatter_size, label=r'$-\Delta w_{o}$')
                    ax2.set_ylabel(y)
                    ax2.set_yscale('log')
                    ax2.set_xlabel(x)
                    ax2.legend()

                    if logx:
                        ax1.set_xscale('log')
                        ax2.set_xscale('log')

                    plt.tight_layout()
                    plt.savefig(path_results_split_pk + '/logy_{}_by_{}_logx{}.png'.format(y, x, logx))
                    plt.close()

            # ---

            # specialized plots
            if splits == 'all':
                # plot center deflection normalized by loading as a function of in-plane tension
                fig, ax2 = plt.subplots()
                ax2.scatter(dfp[y2], dfp[x].abs() / dfp[y1].abs(), s=scatter_size, label=r'$+\Delta w_{o}$')
                ax2.scatter(dfn[y2], dfn[x].abs() / dfn[y1].abs(), s=scatter_size, label=r'$-\Delta w_{o}$')
                ax2.set_ylabel(r'$\Delta w_{o} / P$')
                ax2.set_yscale('log')
                ax2.set_xlabel(r'$k$')
                ax2.legend()
                plt.tight_layout()
                plt.savefig(path_results_split_pk + '/peak-def-norm-P_by_k.png')
                plt.close()

                # plot peak deflection (x) by pressure (y1)
                fig, ax2 = plt.subplots()
                ax2.scatter(dfp[y1].abs(), dfp[x].abs(), s=scatter_size, label=r'$+\Delta w_{o}$')
                ax2.scatter(dfn[y1].abs(), dfn[x].abs(), s=scatter_size, label=r'$-\Delta w_{o}$')
                ax2.set_ylabel(r'$\Delta w_{o}$')
                ax2.set_yscale('log')
                ax2.set_xlabel(r'$P$')
                ax2.set_xscale('log')
                ax2.legend()
                plt.tight_layout()
                plt.savefig(path_results_split_pk + '/peak-def_by_P.png')
                plt.close()

                # plot pressure (y1) by pre-tension (y2)
                fig, ax2 = plt.subplots()
                ax2.scatter(dfp[y2].abs(), dfp[y1].abs(), s=scatter_size, label=r'$+\Delta w_{o}$')
                ax2.scatter(dfn[y2].abs(), dfn[y1].abs(), s=scatter_size, label=r'$-\Delta w_{o}$')
                ax2.set_ylabel(r'$P$')
                ax2.set_yscale('log')
                ax2.set_xlabel(r'$k$')
                ax2.set_xscale('log')
                ax2.legend()
                plt.tight_layout()
                plt.savefig(path_results_split_pk + '/P_by_k.png')
                plt.close()

                # plot pre-tension (y2) by pressure (y1)
                fig, ax2 = plt.subplots()
                ax2.scatter(dfp[y1].abs(), dfp[y2].abs(), s=scatter_size, label=r'$+\Delta w_{o}$')
                ax2.scatter(dfn[y1].abs(), dfn[y2].abs(), s=scatter_size, label=r'$-\Delta w_{o}$')
                ax2.set_ylabel(r'$k_{eff}$')
                ax2.set_yscale('log')
                ax2.set_xlabel(r'$P$')
                ax2.set_xscale('log')
                ax2.legend()
                plt.tight_layout()
                plt.savefig(path_results_split_pk + '/k-eff_by_P.png')
                plt.close()

# ---

# ---

analyze_overlapping_pids_spatially = False
if analyze_overlapping_pids_spatially:
    # ---

    data_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_displacement'
    path_results_tid_coords = join(data_dir, 'results', 'dz{}'.format(tid), 'includes_saturated_pids')
    read_raw_or_reconstruction = 'raw'  # 'reconstruction'
    df = pd.read_excel(
        path_results_tid_coords + '/df_{}_local_displacement_lr-ul_nd.xlsx'.format(read_raw_or_reconstruction))

    # ---

    pids_of_interest = [61, 62, 65, 66]

    df = df[df['frame'] > start_frame]

    dfo = df[df.id.isin(pids_of_interest)][['frame', 'id', 'xg', 'yg', 'rg', 'z_corr', 'cm', 'nd_theta']]
    dfo = dfo.rename(columns={'xg': 'x', 'yg': 'y'})
    dfo = analyze.calculate_particle_to_particle_spacing(test_coords_path=dfo,
                                                         theoretical_diameter_params_path=None,
                                                         mag_eff=None,
                                                         max_n_neighbors=4,
                                                         param_percent_diameter_overlap=None,
                                                         )

    # plot relative z position
    dfpid_lower = dfo[dfo['id'] == 61].reset_index()
    dfg = dfo.groupby('frame').mean().reset_index()

    fig, (ax2, ax3) = plt.subplots(nrows=2, sharex=True,
                                        figsize=(size_x_inches, size_y_inches),
                                        gridspec_kw={'height_ratios': [1, 1]})
    # ax1.plot(dfg.frame, dfg.nd_theta, '-', color=sciblack)

    for pid in [62, 65, 66]:
        dfpid = dfo[dfo['id'] == pid].reset_index()

        arr_z = dfpid.z_corr - dfpid_lower.z_corr

        ax2.plot(dfpid.frame, arr_z, '-o', ms=1, label=pid)
        ax3.plot(dfpid.frame,
                 (dfpid_lower.rg - dfpid.rg) * microns_per_pixel * np.tan(np.deg2rad(dfpid.nd_theta)),
                 '-o', ms=1, label=pid)
    # ax1.set_ylabel(r'$\theta$')
    ax2.set_ylabel(r'$\delta z_{i} = z_{i} - z_{61}$')
    ax2.set_ylim([-5.5, 5.5])
    ax3.legend()
    ax3.set_ylabel(r'$\delta z_{\theta} = (r_{i} - r_{61}) \cdot \tan(\theta_{i})$')
    plt.tight_layout()
    plt.show()

    # ---

    # evaluate in-plane and out-of-plane distance
    dfg = dfo.groupby('frame').mean()
    dfstd = dfo.groupby('frame').std()

    # plot
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows=7, sharex=True,
                                                            figsize=(size_x_inches * 1.2, size_y_inches * 2),
                                                            gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1]})
    ax1.plot(dfg.index, dfg.z_corr, '-o', ms=1)
    ax2.plot(dfg.index, dfg.cm, '-o', ms=1)
    ax3.plot(dfg.index, dfg.mean_dx, '-o', ms=1, label=r'$\overline{\delta x}$')
    ax4.plot(dfstd.index, dfstd.mean_dx, '-o', ms=1)
    ax5.plot(dfg.index, dfg.min_dx, '-o', ms=1, color=scigreen, label=r'$\delta x_{min}$')
    ax6.plot(dfstd.index, dfstd.min_dx, '-o', ms=1, color=scigreen)
    ax7.plot(dfg.index, dfg.nd_theta, '-o', ms=1, color=sciblack)

    ax1.set_ylabel('z')
    ax2.set_ylabel('Cm')
    ax3.set_ylabel(r'$\overline{\delta x}_{mean}$')
    ax4.set_ylabel(r'$\sigma_{xy} (\delta x)$')
    ax5.set_ylabel(r'$\overline{\delta x}_{min}$')
    ax6.set_ylabel(r'$\sigma_{xy} (\delta x)$')
    ax7.set_ylabel(r'$\theta (deg.)$')
    ax7.set_xlabel('Frame')

    plt.suptitle(r'$\overline{\delta x} = $' +
                 ' {} '.format(np.round(dfg.mean_dx.mean(), 1)) +
                 r'$\pm$' +
                 ' {}'.format(np.round(dfg.mean_dx.std(), 2) * 2) + ', ' +
                 r'$\overline{\delta x}_{min} = $' +
                 ' {} '.format(np.round(dfg.min_dx.mean(), 1)) +
                 r'$\pm$' +
                 ' {}'.format(np.round(dfg.min_dx.std(), 2) * 2) + '\n' +
                 '(2 stdev)'
                 )
    plt.tight_layout()
    plt.show()

    j = 1

# ---

print("Analysis completed without errors.")