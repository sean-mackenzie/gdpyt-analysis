# imports
from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import warnings

from utils import io, functions, bin, fit, plotting
from utils.plotting import lighten_color
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

sciblue = '#0C5DA5'
scigreen = '#00B945'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
           'results-04.24.22_meta_gauss-xyc-is-absolute'

path_test_coords = join(base_dir, 'coords/test-coords')
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_correction = base_dir + 'correction/'
path_similarity = join(base_dir, 'similarity')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

method = 'spct'
microns_per_pixel = 1.6
image_dimensions = (512, 512)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#       A. CALIB COORDS

read_calib_coords = True

# if read_calib_coords:
dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)

# ----------------------------------------------------------------------------------------------------------------------
# 2. CORRECT TEST COORDS

fit_calibration_surface = True

if fit_calibration_surface:
    # setup
    param_zf = 'zf_from_peak_int'
    kx, ky = 2, 2

    # 1. FIT PLANE AND PLOT RAW POINTS + FITTED PLANE (NO CORRECTION)
    dict_fit_plane, fig_xy, fig_xyz, fig_plane = correct.inspect_calibration_surface(dfcpid,
                                                                                     param_zf,
                                                                                     microns_per_pixel)

    fig_xy.savefig(path_figs + '/zf_scatter_xy.png')
    plt.close(fig_xy)
    fig_xyz.savefig(path_figs + '/zf_scatter_xyz.png')
    plt.close(fig_xyz)
    fig_plane.savefig(path_figs + '/zf_fit-3d-plane.png')
    plt.close(fig_plane)

    # 2. FIT SMOOTH 2D SPLINE AND PLOT RAW POINTS + FITTED SURFACE (NO CORRECTION)
    bispl_c, rmse_c = fit.fit_3d_spline(x=dfcpid.x,
                                        y=dfcpid.y,
                                        z=dfcpid[param_zf],
                                        kx=kx,
                                        ky=ky)

    fig, ax = plotting.scatter_3d_and_spline(dfcpid.x, dfcpid.y, dfcpid[param_zf],
                                             bispl_c,
                                             cmap='RdBu',
                                             grid_resolution=30,
                                             view='multi')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_zlabel(r'$z_{f} \: (\mu m)$')
    plt.suptitle('fit RMSE = {}'.format(np.round(rmse_c, 3)))
    plt.savefig(path_figs + '/calibration_fit-spline_kx{}_ky{}.png'.format(kx, ky))
    # plt.show()
    plt.close()

    # 3. CORRECT PARTICLES Z-POSITION BY FITTED PLANE
    dfcpid_corrected = correct.correct_z_by_plane_tilt(dfcal=dfcpid,
                                                       dftest=None,
                                                       param_zf=param_zf,
                                                       param_z=param_zf,
                                                       param_z_true=None,
                                                       popt_calib=dict_fit_plane['popt_pixels'],
                                                       params_correct=None)

    # 4. RE-FIT SMOOTH 2D SPLINE AND PLOT CORRECTED Z-POINTS + FITTED SURFACE: COMPARE RMSE OF FITS
    """
    Purpose: Compare the RMSE of the 2D spline before and after planar tilt correction. 
    """
    bispl_c2, rmse_c2 = fit.fit_3d_spline(x=dfcpid_corrected.x,
                                          y=dfcpid_corrected.y,
                                          z=dfcpid_corrected['z_corr'],
                                          kx=kx,
                                          ky=ky)

    fig, ax = plotting.scatter_3d_and_spline(dfcpid_corrected.x, dfcpid_corrected.y, dfcpid_corrected['z_corr'],
                                             bispl_c2,
                                             cmap='RdBu',
                                             grid_resolution=30,
                                             view='multi')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_zlabel(r'$z_{f} \: (\mu m)$')
    plt.suptitle('fit RMSE = {}'.format(np.round(rmse_c, 3)))
    plt.savefig(path_figs + '/calibration_fit-spline_kx{}_ky{}_after-tilt-correction.png'.format(kx, ky))
    # plt.show()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#       B.  TEST COORDS

read_test_coords = True

if read_test_coords:
    dft = io.read_test_coords(path_test_coords)

# ----------------------------------------------------------------------------------------------------------------------
# 2. CORRECT TEST COORDS

"""
Two Parts: (A, B)

Part A: Uniformize p ID's between calib coords and test coords.
    1. Inputs: calibration dataframe (id, x, y) and test coords (id, x, y)
    2. Method: Perform NearestNeighbor to identify identical particles based on (x, y) coordinates.
        NOTE: if there is a difference in image padding then a different function must be applied.
    3. Outputs: plots figure, exports dataframes, returns corrected dataframe.
    
Part B: (4 steps) C.1, C.2, C.3, and C.4
    1. correct z_true value to account for 3 images per z-step.
    2. map z in-focus from pid_defocus to test_coords.
    3. subtract z in-focus from z and z_true to get each particle's corrected axial position.
        * Reference: Copeland et al. (2021)
    4. normalize the error by the focal plane direction (+/- z_true) such that negative errors indicate the error is 
    closer to the focal plane and positive errors indicate the error is further from the focal plane than z_true.
    
NOTE: it is best to run this once and then replace the original test coords with the uniformized test coords.
"""
perform_correction = False
perform_uniformization = False

if perform_correction:

    column_zf = 'zf_from_peak_int'

    if perform_uniformization:
        save_id = 'map_calib-IDs_to_test-coords'
        threshold = 3

        # store original test coords
        dfot = dft.copy()

        # modify calib coords
        dfcpid_locs = dfcpid[['id', 'x', 'y']]

        dft, mapping_dict, pids_not_mapped = correct.correct_nonuniform_particle_ids(baseline=dfcpid_locs,
                                                                                     coords=dft,
                                                                                     threshold=threshold,
                                                                                     save_path=path_correction,
                                                                                     save_id=save_id,
                                                                                     )

        # --- 2.1 (C.1) MAP IN-FOCUS Z TO TEST COORDS FOR EACH PARTICLE ID IN CALIBRATION COORDS.
        # This method is performed by Copeland et al. (2021) and is the basis for this decision.

        # create arrays of unique_identifier and value to map
        calibration_pids = dfcpid.id.values
        calibration_pids_zf = dfcpid[column_zf].values

        # create a new column in dft to map values to
        dft['zf'] = dft['id']
        dft = dft[dft['id'].isin(calibration_pids)]

        # create the mapping dictionary
        zf_mapping_dict = {calibration_pids[i]: calibration_pids_zf[i] for i in range(len(calibration_pids))}

        # insert the mapped values
        dft.loc[:, 'zf'] = dft.loc[:, 'zf'].map(zf_mapping_dict)

        # --- 2.2 (C.2) CORRECT Z_TRUE: TRANSFORM FROM TEST IMAGE COORDINATES TO CALIBRATION COORDINATES
        # because the calibration stack flatten so there are 3 images per z-step.
        dft['z_true'] = np.floor(dft.z_true / 3) + 1
        dft['error'] = dft.z - dft.z_true

        # --- 2.3 (C.3) CORRECT Z_TRUE AND Z BY IN-FOCUS Z
        dft['z_true'] = dft.z_true - dft.zf
        dft['z'] = dft.z - dft.zf

        # --- 2.4 (C.4) NORMALIZE Z-ERROR BY FOCAL PLANE DIRECTION (+/-)
        """
        Normalize the error by the focal plane direction (+/- z_true) such that negative errors indicate the error is
        closer to the focal plane and positive errors indicate the error is further from the focal plane than z_true.
        """
        dft['error'] = dft.error * dft.z_true / dft.z_true.abs()

        # --- 2.5 EXPORT
        dft.to_excel(path_correction + 'test_coords_uniformized_z-corrected_normalized-errors.xlsx', index=False)

        # --- 2.6 PLOT TEST COORDS FOR VALIDATION

        # for SPCT, get test_coord dataframe of the calibration particle tested on itself
        pid_calib = dft.stack_id.unique()[0]
        dft_calib = dft[dft['id'] == pid_calib]

        fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(12, 3))

        ax1.scatter(dfot.z_true, dfot.z, s=0.5, color='gray', label='raw')
        ax1.set_xlabel(r'$z_{true, raw} \: (\mu m)$')
        ax1.set_ylabel(r'$z_{raw} \: (\mu m)$')
        ax1.set_title('{} particles, {} rows'.format(len(dfot.id.unique()), len(dfot)))
        ax1.legend(loc='center right')

        ax2.scatter(dft.z_true, dft.z, s=0.5, label='corrected')
        ax2.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax2.set_ylabel(r'$z_{corr} \: (\mu m)$')
        ax2.set_title('{} particles, {} rows'.format(len(dft.id.unique()), len(dft)))
        ax2.legend(loc='center right')

        ax3.scatter(dft_calib.z_true, dft_calib.z, s=0.5, color='tab:green', label='calib.')
        ax3.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax3.set_ylabel(r'$z_{corr} \: (\mu m)$')
        ax3.legend(loc='center right')

        plt.tight_layout()
        plt.savefig(path_figs + '/test-coords_original-corrected-calibration.png')
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 2. LATERAL AND AXIAL DEPENDENCE ON AXIAL LOCALIZATION ERROR

analyze_error = False

if analyze_error:

    # setup
    error_threshold = 7.5
    xlim = [-57.5, 62.5]
    xyticks = [-50, -25, 0, 25, 50]
    yerr_lims = [-7.5, 7.5]
    yerr_ticks = [-5, 0, 5]

    # test-coords without (mostly) focal plane bias errors
    df_error = dft[dft['error'].abs() < error_threshold]
    df_error = df_error.sort_values('z_true')

    # --- PLOTTING

    # FIGURE 1: all errors by z_true & fit quadratic
    fig_1 = False
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
    fig_2 = True

    if fig_2:
        if 'r' not in df_error.columns:
            df_error['r'] = np.sqrt(
                (df_error.x - image_dimensions[0] / 2) ** 2 +
                (df_error.y - image_dimensions[1] / 2) ** 2
            )

        # bin(r)
        column_to_bin = 'r'
        column_to_count = 'id'
        bins_r = 7
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
    fig_3 = True

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
    fig_4 = True

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
    fig_5 = True
    plot_fit = False

    if fig_5:
        columns_to_bin = ['r', 'z_true']
        column_to_count = 'id'
        bins = [bins_r, 27]
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

print("Analysis completed without errors.")