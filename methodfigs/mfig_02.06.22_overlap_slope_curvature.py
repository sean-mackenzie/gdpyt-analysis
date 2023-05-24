# 02.06.22 - local axial and radial displacements per membrane

# imports
import os
from os.path import join
import itertools

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, Akima1DInterpolator

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import analyze
from utils import boundary, functions, io, bin
from utils.plotting import lighten_color
from utils.functions import fSphericalUniformLoad, fNonDimensionalNonlinearSphericalUniformLoad

# ---

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee'])  # , 'std-colors'
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
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
frame_rate = 24.444
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

# axial positions
z_f_from_calib = 140
z_inital_focal_plane_bias_errors = 0
z_i_mean_allowance = 2.5

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

"""fpp = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_local_displacement/results/dz1_includes_saturated_pids/df_raw_local_displacement_lr-ul_EXCLUDE-NONE.xlsx'
df = pd.read_excel(fpp)
pids_g = [61, 62, 65, 66]
for pid in pids_g:
    dfpid = df[df['id'] == pid]
    print("pid {} has length {}".format(pid, len(dfpid)))
raise ValueError()"""

# ---

# setup file paths

calculate_non_dimensional = True
if calculate_non_dimensional:

    # which test to analyze
    dz_id = 1
    apply_error_filter = False

    # ---

    # 1. DATA DIRECTORY - WHERE TO READ "RAW" FILES

    # upper most dir
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization'
    dir_local_disp = '/methodfigs/02.06.22_local_displacement/results'

    # read from: methofigs/local_displacement

    # read local displacement
    data_dir1 = base_dir + dir_local_disp + '/dz{}'.format(dz_id)
    dft = pd.read_excel(data_dir1 + '/df_raw_local_displacement_lr-ul_EXCLUDE-NONE.xlsx')
    # dft = dft[['frame', 't', 'id', 'z_corr', 'cm', 'xg', 'yg', 'r', 'rg', 'drg']]
    # dft = dft[dft['frame'] > 0]

    # read pids per membrane
    fp_pids = base_dir + dir_local_disp + '/df_pids_per_membrane.xlsx'
    df_pids = pd.read_excel(fp_pids)
    pids_lr = df_pids.iloc[0, 1:].dropna().values
    pids_ul = df_pids.iloc[1, 1:].dropna().values

    # ---

    # read from: analyses/results... --> data processing
    data_dir2 = base_dir + '/experiments/02.06.22_membrane_characterization/analyses/results-09.15.22_idpt-subpix'
    path_read2 = data_dir2 + '/results/dz{}'.format(dz_id)

    # read dataframe
    dfr = pd.read_excel(path_read2 + '/id{}_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'.format(dz_id))
    dfr = dfr[['frame', 'time',
               'rz_lr', 'fit_lr_pressure', 'fit_lr_pretension',
               'rz_ul', 'fit_ul_pressure', 'fit_ul_pretension',
               ]]

    # ---

    # 2. BASE DIRECTORY - WHERE TO SAVE RESULTS

    path_results = base_dir + '/methodfigs/02.06.22_overlap_slope_curvature/results/dz{}'.format(dz_id)
    path_results_deflection = path_results + '/def-single-frame'
    path_results_deflections = path_results + '/def-multi-frames'

    if not os.path.exists(path_results):
        os.makedirs(path_results)
    if not os.path.exists(path_results_deflection):
        os.makedirs(path_results_deflection)
    if not os.path.exists(path_results_deflections):
        os.makedirs(path_results_deflections)

    # ---

    # ---

    # 3. FUNCTION DESCRIPTION
    """
    Function process:
        1. Initialize plate model 
        2. Calculate non-dimensional pressure and pretension from fitted (1) pressure load, (2) pretension: ['dfr']
        3. Get particle test coords of only particles of interest: ['dft'] of pID == 50, 52, 55. 
        4. Append nd_P, nd_k columns to particle test coords: ['dft']
        5. Calculate nd_z, nd_slope, nd_curvature using nd_P, nd_k. 
        6. Calculate dimensional quantities: z, slope, curvature. 
        7. Calculate error: z_corr - d_z
    """

    # df_test = dft
    # membrane_radius = r_edge_lr
    # df_results = dfr
    # p_col = 'fit_lr_pressure'
    # k_col = 'fit_lr_pretension'

    # define: initialize plate model
    def initialize_plate_model(df_test, membrane_radius, df_results, p_col, k_col,
                               nonlinear_only=True, exclude_outside_membrane_radius=True):
        """ dft = initialize_plate_model(df_test, membrane_radius, df_results, p_col, k_col) """
        a_membrane = membrane_radius * microns_per_pixel
        fND = fNonDimensionalNonlinearSphericalUniformLoad(r=a_membrane * 1e-6,
                                                           h=t_membrane,
                                                           youngs_modulus=E_silpuran,
                                                           poisson=poisson)

        # 2. calculate non-dimensional pressure and pre-tension
        nd_P, nd_k = fND.non_dimensionalize_p_k(d_p0=df_results[p_col].to_numpy(),
                                                d_n0=df_results[k_col].to_numpy()
                                                )
        df_results['nd_p'] = nd_P
        df_results['nd_k'] = nd_k

        # 4. Append nd_P, nd_k columns to 'dft'

        # 4.1 - columns to be mapped
        df_test['nd_p'] = df_test['frame']
        df_test['nd_k'] = df_test['frame']

        # 4.2 - create mapping dict
        mapper_dict = df_results[['frame', 'nd_p', 'nd_k']].set_index('frame').to_dict()

        # 4.3 - map nd_P, nd_k to 'dft' by 'frame'
        df_test = df_test.replace({'nd_p': mapper_dict['nd_p']})
        df_test = df_test.replace({'nd_k': mapper_dict['nd_k']})

        # 4.4 - replace nd_k == 0 with 0.001
        if nonlinear_only:
            df_test = df_test[df_test['nd_k'] > 0.0001]
        # dft['nd_k'] = dft['nd_k'].where(dft['nd_k'] > 0.001, 0.001)

        if exclude_outside_membrane_radius:
            df_test = df_test[df_test['r'] < membrane_radius]

        # -

        # 5. Calculate nd_z, nd_slope, nd_curvature using nd_P, nd_k.

        nd_P = df_test['nd_p'].to_numpy()
        nd_k = df_test['nd_k'].to_numpy()

        # 5.1 - calculate non-dimensional r
        # z_ev = (dfrlr.z_corr.to_numpy() - z_lr_offset) * 1e-6
        nd_r = df_test['r'].to_numpy() * microns_per_pixel / a_membrane
        nd_z = fND.nd_nonlinear_clamped_plate_p_k(nd_r, nd_P, nd_k)
        nd_theta = fND.nd_nonlinear_theta(nd_r, nd_P, nd_k)
        nd_curve = fND.nd_nonlinear_curvature_lva(nd_r, nd_P, nd_k)

        df_test['nd_r'] = nd_r
        df_test['nd_rg'] = df_test['rg'].to_numpy() * microns_per_pixel / a_membrane
        df_test['nd_dr'] = df_test['drg'] * microns_per_pixel / t_membrane_norm
        df_test['nd_dz'] = nd_z
        df_test['nd_dz_corr'] = (df_test['z_corr'] + df_test['z_offset']) / t_membrane_norm
        df_test['d_dz_corr'] = df_test['z_corr'] + df_test['z_offset']
        df_test['d_dz'] = nd_z * t_membrane_norm
        df_test['nd_theta'] = nd_theta
        df_test['nd_curve'] = nd_curve

        # 7. Calculate error: (z_corr - d_z); and squared error for rmse
        df_test['dz_error'] = df_test['d_dz_corr'] - df_test['d_dz']
        df_test['z_rmse'] = df_test['dz_error'] ** 2

        # -

        return df_test, fND


    # -

    # define: get plate deflection for a single frame
    def get_plate_deflection_for_frame(frame, df_test, fND, membrane_radius):
        """ df_test_fr, d_r, d_z = get_plate_deflection_for_frame(frame, df_test, fND, df_results, p_col, k_col) """

        df_test_fr = df_test[df_test['frame'] == frame].reset_index()

        nd_P = df_test_fr.loc[0, 'nd_p']
        nd_k = df_test_fr.loc[0, 'nd_k']

        nd_r = np.linspace(0, 1, 250)
        nd_z = fND.nd_nonlinear_clamped_plate_p_k(nd_r, nd_P, nd_k)

        d_r = nd_r * membrane_radius
        d_z = nd_z * t_membrane_norm

        return df_test_fr, d_r, d_z


    # ---

    # ---   OVERLAP, SLOPE, CURVATURE
    plot_overlap_slope_curvature = True
    if plot_overlap_slope_curvature:

        z_error_limit = 10
        p_ids = [pids_lr]
        g_ids = list(itertools.chain(*p_ids))

        # processing

        # get particle test coords of p_ids
        dft = dft[dft.id.isin(g_ids)]

        # get plate model results
        dft, fND = initialize_plate_model(df_test=dft,
                                          membrane_radius=r_edge_lr,
                                          df_results=dfr,
                                          p_col='fit_lr_pressure',
                                          k_col='fit_lr_pretension',
                                          nonlinear_only=False,
                                          )

        # filter 'dz_error'
        if apply_error_filter:
            dft = dft[dft['dz_error'].abs() < z_error_limit]

        # correct some quantities: slope, curvature.
        dft['nd_theta'] = dft['nd_theta'].where(dft['nd_theta'].abs() < 15, np.nan)
        dft['nd_curve'] = dft['nd_curve'].where(dft['nd_curve'].abs() < 15, np.nan)

        # ---

        # plot local error
        plot_local_rmse_and_error = False
        if plot_local_rmse_and_error:
            ycol = 'dz_error'
            xcols = ['nd_r', 'nd_dr', 'nd_dz', 'nd_theta', 'nd_curve']
            xlbls = [r'$r/a$', r'$\Delta r / t_{m}$', r'$\Delta z / t{m}$', r'$Slope Angle \: (\theta)$',
                     'Surface Curvature']

            for column_to_bin, xlbl in zip(xcols, xlbls):
                column_to_count = 'nd_dz'
                bins = 13
                round_to_decimal = 1
                return_groupby = True

                dfm, dfstd = bin.bin_generic(dft, column_to_bin, column_to_count, bins, round_to_decimal,
                                             return_groupby)
                dfm['z_rmse'] = np.sqrt(dfm['z_rmse'])

                # plot - rmse and error
                fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25),
                                                    gridspec_kw={'height_ratios': [0.5, 1, 1]})

                ax0.plot(dfm[column_to_bin], dfm['count_' + column_to_count], '-o')
                ax0.set_ylabel(r"$N_{p}^{'} \: (\#)$")

                ax1.plot(dfm[column_to_bin], dfm[ycol], '-o')
                ax1.set_ylabel(r'$\epsilon_{\Delta z} \: (\mu m)$')

                ax2.plot(dfm[column_to_bin], dfm['z_rmse'], '-o')
                ax2.set_xlabel(xlbl)
                ax2.set_ylabel(r'$\sigma_{\Delta z} \: (\mu m)$')

                plt.tight_layout()
                plt.savefig(path_results + '/dz-rmse-error_bin_{}.png'.format(column_to_bin))
                plt.show()

        # ---

        # plot scatter
        plot_all_simple_vars = False
        if plot_all_simple_vars:
            xcols = ['t', 'nd_r', 'nd_dr', 'nd_dz', 'nd_theta', 'nd_curve']
            ycolss = [['dz_error', 'nd_dr', 'nd_dz', 'nd_theta', 'nd_curve'],
                      ['dz_error'],
                      ['dz_error'],
                      ['dz_error'],
                      ['dz_error'],
                      ['dz_error'],
                      ]

            p_ids = [# [50, 52, 55],
                     [61, 62, 65, 66],
                     # [87, 88, 89],
                     # [11, 12, 13, 17, 19, 22, 23, 25, 28, 30, 32, 33, 37, 38, 39, 40, 44],
                     ]

            for pids in p_ids:
                for xcol, ycols in zip(xcols, ycolss):
                    dfg = dft.groupby(xcol).mean().reset_index()

                    for ycol in ycols:
                        fig, ax = plt.subplots()
                        for pid in pids:
                            dfpid = dft[dft['id'] == pid]

                            ax.scatter(dfpid[xcol], dfpid[ycol], s=5, label=pid)

                        ax.plot(dfg[xcol], dfg[ycol], '-', color=sciblack, label='Avg.')

                        ax.set_xlabel(xcol)
                        ax.set_ylabel(ycol)
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        plt.savefig(
                            path_results + '/scatter-{}_by_{}_for-pids-{}_compare-avg.png'.format(ycol, xcol, pids))
                        plt.show()

        # ---

        # pubfig: custom
        plot_pubfig_gids = False
        if plot_pubfig_gids:
            # plot average of all pids on feature for comparison
            plot_feature_average = False

            # particle ID's of interest
            p_ids = [[61, 62, 65, 66]]
            clrs = [scigreen, sciblue, scired, sciorange]  # [scired, sciorange, scigreen, sciblue]
            zorders = [3.1, 3.4, 3.2, 3.3]

            # ---

            # filtering

            # filter: start time
            dft = dft[dft['t'] >= start_time]

            # filter: very last frame (which is maybe an error? it looks like its actually frame = 0)
            dft = dft[dft['frame'] < 200]

            # ---

            # plotting

            # pubfig

            x = 't'
            y1 = 'cm'
            y2 = 'z_corr'
            ms = 0.5

            for pids in p_ids:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                               figsize=(size_x_inches * 0.85, size_y_inches * 0.55))

                # pids.reverse()
                for pid, clr, zo in zip(pids, clrs, zorders):

                    dfpid = dft[dft['id'] == pid]
                    ax1.plot(dfpid[x], dfpid[y1], 'o', ms=ms, color=clr, zorder=zo, label=pid)
                    ax2.plot(dfpid[x], dfpid[y2], 'o', ms=ms, color=clr, zorder=zo)

                # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=1.5, title=r'$p_{i}$')
                ax1.set_ylabel(r'$C_{m}$')
                ax1.set_ylim([0.95, 1.0])
                ax1.set_yticks([0.95, 1])
                ax2.set_ylabel(r'$z \: (\mu m)$')
                ax2.set_ylim([-100, 100])
                ax2.set_yticks([-75, 0, 75])
                ax2.set_xlabel(r'$t \: (s)$', labelpad=0.5)

                ax1.tick_params(axis='both', which='minor',
                                bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                ax2.tick_params(axis='both', which='minor',
                                bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labeltop=False, labelleft=False, labelright=False)

                plt.tight_layout()
                plt.savefig(
                    path_results + '/pubfig_{}-{}_by_{}_pids{}_0.55Y.png'.format(y1, y2, x, pids))
                plt.show()

            # ---

            # plot supporting figures
            plot_supporting = False
            if plot_supporting:
                xcols = ['t', 'nd_r', 'nd_dr', 'nd_dz']  # , 'nd_theta', 'nd_curve']
                ycolss = [['cm', 'd_dz_corr', 'nd_dz', 'drg', 'nd_dr',
                           'dz_error'],  # , 'nd_theta', 'nd_curve'],
                          ['dz_error'],
                          ['dz_error'],
                          ['dz_error'],
                          ['dz_error'],
                          ['dz_error'],
                          ]

                for pids in p_ids:
                    for xcol, ycols in zip(xcols, ycolss):
                        dfg = dft.groupby(xcol).mean().reset_index()

                        for ycol in ycols:
                            fig, ax = plt.subplots()
                            for pid in pids:
                                dfpid = dft[dft['id'] == pid]

                                ax.scatter(dfpid[xcol], dfpid[ycol], s=5, label=pid)

                            if plot_feature_average:
                                ax.plot(dfg[xcol], dfg[ycol], '-', color=sciblack, label='Avg.')

                            ax.set_xlabel(xcol)
                            ax.set_ylabel(ycol)
                            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            plt.tight_layout()
                            plt.savefig(
                                path_results + '/scatter-{}_by_{}_for-pids-{}_compare-avg.png'.format(ycol, xcol, pids))
                            plt.show()

    # ---

    # ---

    # --- PLOT DEFLECTION PROFILES
    plot_deflection_profiles = True
    if plot_deflection_profiles:
        # get plate models for LR and UL

        # LR membrane
        dft_lr = dft[dft.id.isin(pids_lr)]
        dft_lr, fND_lr = initialize_plate_model(df_test=dft_lr,
                                                membrane_radius=r_edge_lr,
                                                df_results=dfr,
                                                p_col='fit_lr_pressure',
                                                k_col='fit_lr_pretension',
                                                )

        # UL membrane
        dft_ul = dft[dft.id.isin(pids_ul)]
        dft_ul, fND_ul = initialize_plate_model(df_test=dft_ul,
                                                membrane_radius=r_edge_ul,
                                                df_results=dfr,
                                                p_col='fit_ul_pressure',
                                                k_col='fit_ul_pretension',
                                                )

        # ---

        """
        peak positive deflection frames: 145, 197, 86
        peak negative deflection frames: 170, 115, 51
        """

        # plot variables
        px = 'r'
        py = 'd_dz_corr'

        # plot single frame
        plot_single_frame = False
        if plot_single_frame:
            frois = [50, 51, 52,
                     85, 86, 87,
                     114, 115, 116,
                     144, 145, 146,
                     169, 170, 171,
                     196, 197, 198,
                     ]

            for froi in frois:
                # get results for this frame
                df_test_fr_lr, dr_lr, dz_lr = get_plate_deflection_for_frame(frame=froi,
                                                                             df_test=dft_lr,
                                                                             fND=fND_lr,
                                                                             membrane_radius=r_edge_lr,
                                                                             )
                df_test_fr_ul, dr_ul, dz_ul = get_plate_deflection_for_frame(frame=froi,
                                                                             df_test=dft_ul,
                                                                             fND=fND_ul,
                                                                             membrane_radius=r_edge_ul,
                                                                             )
                # filtering

                # filter r
                df_test_fr_lr = df_test_fr_lr[df_test_fr_lr[px] < r_edge_lr]
                df_test_fr_ul = df_test_fr_ul[df_test_fr_ul[px] < r_edge_ul]

                # filter z
                if df_test_fr_lr[py].mean() > 0:
                    df_test_fr_lr = df_test_fr_lr[df_test_fr_lr[py] > 0 - z_i_mean_allowance]
                    df_test_fr_ul = df_test_fr_ul[df_test_fr_ul[py] > 0 - z_i_mean_allowance]
                else:
                    df_test_fr_lr = df_test_fr_lr[df_test_fr_lr[py] < 0 + z_i_mean_allowance]
                    df_test_fr_ul = df_test_fr_ul[df_test_fr_ul[py] < 0 + z_i_mean_allowance]

                # ---

                # plotting

                # setup
                ms = 3

                # plot
                fig, ax = plt.subplots()

                ax.plot(df_test_fr_lr[px] * microns_per_pixel, df_test_fr_lr[py], 'o', ms=ms,
                        color=lighten_color(sciblue, 1), label='800')
                ax.plot(dr_lr * microns_per_pixel, dz_lr, color=lighten_color(sciblue, 1.25))

                ax.plot(df_test_fr_ul[px] * microns_per_pixel, df_test_fr_ul[py], 'o', ms=ms,
                        color=lighten_color(scigreen, 1), label='500')
                ax.plot(dr_ul * microns_per_pixel, dz_ul, color=lighten_color(scigreen, 1.25))

                ax.set_xlabel(r'$r \: (\mu m)$')
                ax.set_ylabel(r'$\Delta z \: (\mu m)$')
                ax.legend(title=r'$r \: (\mu m)$')

                plt.tight_layout()
                plt.savefig(path_results_deflection + '/lr-ul_dz_by_r_fr{}.png'.format(froi))
                plt.show()

        # ---

        # plot multiple frames on same figure
        plot_multi_frames = False
        if plot_multi_frames:
            frois = [51,
                     86,
                     115,
                     145,
                     170,
                     # 197,
                     ]

            for froi in frois:

                # plot setup
                ms = 3
                fig, ax = plt.subplots()

                # time shift setup
                n_frames = 3
                d_frames = -4

                for dframe in np.flip(np.arange(n_frames)):

                    # shift frame
                    froi_i = froi + dframe * d_frames

                    # get results for this frame
                    df_test_fr_lr, dr_lr, dz_lr = get_plate_deflection_for_frame(frame=froi_i,
                                                                                 df_test=dft_lr,
                                                                                 fND=fND_lr,
                                                                                 membrane_radius=r_edge_lr,
                                                                                 )
                    df_test_fr_ul, dr_ul, dz_ul = get_plate_deflection_for_frame(frame=froi_i,
                                                                                 df_test=dft_ul,
                                                                                 fND=fND_ul,
                                                                                 membrane_radius=r_edge_ul,
                                                                                 )
                    # filtering

                    # filter r
                    df_test_fr_lr = df_test_fr_lr[df_test_fr_lr[px] < r_edge_lr]
                    df_test_fr_ul = df_test_fr_ul[df_test_fr_ul[px] < r_edge_ul]

                    # filter z
                    if df_test_fr_lr[py].mean() > 0:
                        df_test_fr_lr = df_test_fr_lr[df_test_fr_lr[py] > 0 - z_i_mean_allowance]
                        df_test_fr_ul = df_test_fr_ul[df_test_fr_ul[py] > 0 - z_i_mean_allowance]
                    else:
                        df_test_fr_lr = df_test_fr_lr[df_test_fr_lr[py] < 0 + z_i_mean_allowance]
                        df_test_fr_ul = df_test_fr_ul[df_test_fr_ul[py] < 0 + z_i_mean_allowance]

                    # ---

                    # plotting

                    ax.plot(df_test_fr_lr[px] * microns_per_pixel, df_test_fr_lr[py], 'o', ms=ms,
                            color=lighten_color(sciblue, 1), label='800')
                    ax.plot(dr_lr * microns_per_pixel, dz_lr, color=lighten_color(sciblue, 1.25))

                    ax.plot(df_test_fr_ul[px] * microns_per_pixel, df_test_fr_ul[py], 'o', ms=ms,
                            color=lighten_color(scigreen, 1), label='500')
                    ax.plot(dr_ul * microns_per_pixel, dz_ul, color=lighten_color(scigreen, 1.25))

                # ---

                # figure labels
                ax.set_xlabel(r'$r \: (\mu m)$')
                ax.set_ylabel(r'$\Delta z \: (\mu m)$')
                ax.legend(title=r'$r \: (\mu m)$')

                plt.tight_layout()
                plt.savefig(path_results_deflections +
                            '/lr-ul_dz_by_r_fr{}-dfr{}-n.png'.format(froi, d_frames, n_frames))
                plt.show()

        # ---

        # stack all frames into a dataframe and export
        plot_all_frames = True
        if plot_multi_frames:

            for froi in dft_lr.frame.unique():

                # get results for this frame
                df_test_fr_lr, dr_lr, dz_lr = get_plate_deflection_for_frame(frame=froi_i,
                                                                             df_test=dft_lr,
                                                                             fND=fND_lr,
                                                                             membrane_radius=r_edge_lr,
                                                                             )
                df_test_fr_ul, dr_ul, dz_ul = get_plate_deflection_for_frame(frame=froi_i,
                                                                             df_test=dft_ul,
                                                                             fND=fND_ul,
                                                                             membrane_radius=r_edge_ul,
                                                                             )

    # ---

    # ---

    # --- PLOT: PRESSURE, PRE-TENSION, FITTED PEAK DEFLECTION by TIME
    plot_p_k_by_t = False
    if plot_p_k_by_t:

        # filter > start time
        dfr = dfr[dfr['time'] >= start_time]

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

        ax1.plot(dfr.time, dfr.fit_lr_pressure, 'o', ms=1, label='LR')
        ax1.plot(dfr.time, dfr.fit_ul_pressure, 'd', ms=1, alpha=0.5, label='UL')

        ax2.plot(dfr.time, dfr.fit_lr_pretension, 'o', ms=1, label='LR')
        ax2.plot(dfr.time, dfr.fit_ul_pretension, 'd', ms=1, alpha=0.5, label='UL')

        ax3.plot(dfr.time, dfr.rz_lr, 'o', ms=1, label='LR')
        ax3.plot(dfr.time, dfr.rz_ul, 'd', ms=1, alpha=0.5, label='UL')

        ax3.legend(markerscale=2, borderpad=0.1, labelspacing=0.1, handletextpad=0.1)

        ax1.set_ylabel('P')
        ax1.set_ylim([-1500, 32500])
        ax2.set_ylabel('k')
        ax2.set_ylim([-2.5, 60])
        ax3.set_ylabel('z')
        ax3.set_xlabel('t')

        plt.tight_layout()
        plt.savefig(path_results + '/pressure_pre-tension_peak-fitted-z-def_by_time_zoom.png')
        plt.show()

    # ---

# ---

print("Analysis completed without errors. ")