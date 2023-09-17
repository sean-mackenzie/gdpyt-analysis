# plot self-similarity as a function of z-step (SPCT)
import os
from os.path import join
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# OPTICS
zf = -4
mag_eff = 10.0
numerical_aperture = 0.3
pixel_size = 16
depth_of_focus = functions.depth_of_field(mag_eff, numerical_aperture, 600e-9, 1.0, pixel_size=pixel_size * 1e-6) * 1e6

# ----------------------------------------------------------------------------------------------------------------------
# LOAD FILES

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/effect_of_subpixel_position'

path_pubfig = base_dir + '/pubfigs'

# ---

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE LOCAL
plot_local_rmse = False
correct_MicroSIG_xy_error = True
MicroSIG_xy_error = 1

if plot_local_rmse:

    path_test_coords = base_dir + '/test_coords/ct+3_rescale-0'
    path_results = base_dir + '/results/results-ct+3_MicroSIG_-1pixel_correction_rescale-0'

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    fnxs = [37.5, 37.5, 37.5, 37.5, 37.6, 37.75, 38]
    fnys = [37.5, 37.6, 37.75, 38, 37.6, 37.75, 38]

    # processing
    column_to_bin = 'z_true'
    bins = 20
    min_cm = 0.0
    z_range = None
    round_to_decimal = 4
    df_ground_truth = None
    dropna = True
    error_column = None
    include_xy = True
    xy_colss = [['x', 'y'], ['xm', 'ym'], ['xg', 'yg'], ['gauss_xc', 'gauss_yc']]

    # plot setup
    ms = 3

    # iterate
    for xy_cols in xy_colss:

        # define which columns to drop rows
        dropna_cols = xy_cols

        # figure
        fig, (axx, axy, axz) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.65),
                                            gridspec_kw={'height_ratios': [1, 1, 2]})

        # results
        dfgs = []

        for fnx, fny in zip(fnxs, fnys):
            df = pd.read_excel(path_test_coords + '/test_coords_stats_x{}_y{}.xlsx'.format(fnx, fny))

            if correct_MicroSIG_xy_error:
                df['xg'] = df['xg'] + 0
                df['yg'] = df['yg'] + 0
                df['x_true'] = df['x_true'] - MicroSIG_xy_error
                df['y_true'] = df['y_true'] - MicroSIG_xy_error

            dfrmse = bin.bin_local_rmse_z(df,
                                          column_to_bin,
                                          bins,
                                          min_cm,
                                          z_range,
                                          round_to_decimal,
                                          df_ground_truth,
                                          dropna,
                                          dropna_cols,
                                          error_column,
                                          include_xy,
                                          xy_cols,
                                          )

            # store results
            dfrmse['group'] = 1
            dfg = dfrmse.groupby('group').mean()
            dfgs.append(dfg)

            # plot
            axx.plot(dfrmse.index, dfrmse.rmse_x, '-o', ms=ms)
            axy.plot(dfrmse.index, dfrmse.rmse_y, '-o', ms=ms)
            axz.plot(dfrmse.index, dfrmse.rmse_z, '-o', ms=ms, label='({}, {})'.format(fnx, fny))

        axx.set_ylabel(r'$\sigma_{x}^{\delta} \: (pix.)$')
        axy.set_ylabel(r'$\sigma_{y}^{\delta} \: (pix.)$')
        axz.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')

        axx.set_ylim([-0.1, 1.1])
        axy.set_ylim([-0.1, 1.1])

        axz.set_xlabel(r'$z \: (\mu m)$')
        axz.legend(loc='upper left', title=r'$(x, y)$')  # , bbox_to_anchor=(1, 1)
        plt.tight_layout()
        # plt.savefig(path_results + '/rmse-xyz_x{}_y{}_by_z.svg'.format(xy_cols[0], xy_cols[1]))
        plt.show()

        # ---

        dfrmse_mean = pd.concat(dfgs)
        dfrmse_mean = dfrmse_mean[
            ['x_true', xy_cols[0], 'rmse_x', 'y_true', xy_cols[1], 'rmse_y', 'z_true', 'z', 'rmse_z']]
        # dfrmse_mean.to_excel(path_results + '/dfrmse_mean_x{}_y{}.xlsx'.format(xy_cols[0], xy_cols[1]), index=False)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE MEAN
plot_mean_rmse = False

if plot_mean_rmse:
    path_results = base_dir + '/compare'
    df = pd.read_excel(path_results + '/compare_ct+3_rescale-0.xlsx')

    # calib position
    cx, cy = 37.5 - MicroSIG_xy_error, 37.5 - MicroSIG_xy_error

    # add column
    df['cx_true'] = cx
    df['cy_true'] = cy

    # calculate sub-pixel displacement
    df['dx'] = df['x_true'] - df['cx_true']
    df['dy'] = df['y_true'] - df['cy_true']
    df['dxy'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

    # plot setup
    x = 'dxy'
    df = df.sort_values(x)
    ms = 4

    # export
    # df.to_excel(path_results + '/compare-dxy_ct+3_rescale-0.xlsx')

    # plot
    fig, (axx, axy, axz) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches),
                                        gridspec_kw={'height_ratios': [1, 1, 1]})

    axx.plot(df[x], df.rmse_xg, '-o', ms=ms, label='cc')
    axx.plot(df[x], df.rmse_gauss_xc, '-o', ms=ms, label='int.')
    axy.plot(df[x], df.rmse_yg, '-o', ms=ms, label='cc')
    axy.plot(df[x], df.rmse_gauss_yc, '-o', ms=ms, label='int.')
    axz.plot(df[x], df.rmse_z, '-o', ms=ms)

    axx.set_ylabel(r'$\sigma_{x}^{\delta} \: (pix.)$')
    axy.set_ylabel(r'$\sigma_{y}^{\delta} \: (pix.)$')
    axz.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')

    axx.set_ylim([0.2, 0.535])
    axy.set_ylim([0.2, 0.535])

    # axx.legend(loc='upper left')  # , bbox_to_anchor=(1, 1)
    axy.legend(loc='center right')  # , bbox_to_anchor=(1, 1)

    axz.set_xlabel(r'$\Delta(xy_{t} - xy_{c}) \: (pix.)$')
    plt.tight_layout()
    # plt.savefig(path_results + '/rmse-xyz_by_dxy-subpix.svg')
    plt.show()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# PLOT Z_MEAS BY Z_TRUE - ON THE SAME FIGURE
plot_z_by_z_true = False

if plot_z_by_z_true:

    path_test_coords = base_dir + '/test_coords/ct+3_rescale-0'
    path_results = base_dir + '/compare'

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    fnxs = [37.5, 37.5, 38]
    fnys = [37.5, 37.75, 38]

    cx, cy = 37.5, 37.5

    # plot setup
    ms = 0.2
    clrs = [sciblue, scired, sciblack]
    fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 1.1))

    # iterate
    for fnx, fny, clr in zip(fnxs, fnys, clrs):
        df = pd.read_excel(path_test_coords + '/test_coords_stats_x{}_y{}.xlsx'.format(fnx, fny))

        # calc sub-pixel dist.
        dxy = np.sqrt((fnx - cx) ** 2 + (fny - cy) ** 2)
        lbl_dxy = np.round(dxy, 3)
        lbl_dx_dy = '({}, {}, {})'.format(np.round(fnx - cx, 2), np.round(fny - cy, 2), lbl_dxy)

        # plot
        ax.scatter(df.z_true, df.z, s=ms, marker='o', color=clr, label=lbl_dx_dy)

    ax.fill_between([-depth_of_focus + zf, depth_of_focus + zf],
                    -depth_of_focus + zf, depth_of_focus + zf,  # df.z_true.min(), df.z_true.max(),
                    color='gray', ec='none', ls='--', lw=0.75,
                    alpha=0.0875, zorder=0.9)  # , label='D.o.F.'

    ax.fill_between([-depth_of_focus / 2 + zf, depth_of_focus / 2 + zf],
                    -depth_of_focus / 2 + zf, depth_of_focus / 2 + zf,  # df.z_true.min(), df.z_true.max(),
                    color='gray', ec='none', ls='--', lw=0.75,
                    alpha=0.25, zorder=0.9)  # , label='D.o.F.'

    ax.set_ylabel(r'$z \: (\mu m)$')
    ax.set_xlabel(r'$z_{true} \: (\mu m)$')
    ax.legend(loc='upper left', markerscale=5,
              title=r'$(\Delta x, \Delta y, \Delta r)$',
              borderpad=0.4, labelspacing=0.5, handletextpad=0.3)  # , bbox_to_anchor=(1, 1)
    plt.tight_layout()
    plt.savefig(path_pubfig + '/plot_z_by_z-true_2DoFs.png')
    plt.show()

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# PLOT Z_MEAS BY Z_TRUE - ON INDIVIDUAL FIGURES
plot_z_by_z_true_per_dxy = False

if plot_z_by_z_true_per_dxy:

    path_test_coords = base_dir + '/test_coords/ct+3_rescale-0'

    fnxs = [37.5, 37.5, 37.6, 37.5, 37.75, 37.5, 38]
    fnys = [37.5, 37.6, 37.6, 37.75, 37.75, 38, 38]
    cx, cy = 37.5, 37.5

    # plot setup
    clrs = [sciblue, scigreen, sciorange, scired, scigray, scipurple, sciblack]
    ms = 1

    # store dead zones
    dz_dead_zones = []
    dxys = []

    # iterate
    for fnx, fny, clr in zip(fnxs, fnys, clrs):
        df = pd.read_excel(path_test_coords + '/test_coords_stats_x{}_y{}.xlsx'.format(fnx, fny))

        # calculate sub-pixel distance
        dxy = np.sqrt((fnx - cx) ** 2 + (fny - cy) ** 2)
        dxys.append(dxy)
        lbl_dxy = np.round(dxy, 3)
        lbl_dx_dy = '({}, {}, {})'.format(np.round(fnx - cx, 2), np.round(fny - cy, 2), lbl_dxy)

        # calculate axial dead zone
        dz_dead_zone = df[df['z'] > zf].z.min() - df[df['z'] < zf].z.max()
        dz_dead_zones.append(dz_dead_zone)
        lbl_dead_zone = np.round(dz_dead_zone, 3)

        # plot
        fig, ax = plt.subplots(figsize=(size_y_inches * 0.6, size_y_inches * 0.6))

        ax.scatter(df.z_true, df.z, s=ms, marker='o', color=clr, label=lbl_dx_dy)  # label=lbl_dx_dy)

        """ax.fill_between([-depth_of_focus / 2 + zf, depth_of_focus / 2 + zf],
                        -depth_of_focus / 2 + zf, depth_of_focus / 2 + zf,
                        color='gray', ec='none', ls='--', lw=0.75,
                        alpha=0.25, zorder=0.9)"""

        ax.set_ylabel(r'$z \: (\mu m)$')
        dz_view = 2
        ax.set_ylim([-depth_of_focus + zf - dz_view, depth_of_focus + zf + dz_view])
        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
        ax.set_xlim([-depth_of_focus + zf - dz_view, depth_of_focus + zf + dz_view])
        ax.tick_params(axis='both', which='minor',
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        """ax.legend(loc='lower right', markerscale=1,
                  title=r'$\Delta z_{dead zone} \: (\mu m)$',
                  borderpad=0.4, labelspacing=0.5, handletextpad=0.3)"""
        """ax.legend(loc='lower right', markerscale=1,
                  title=r'$(\Delta x, \Delta y, \Delta r)$',
                  borderpad=0.4, labelspacing=0.5, handletextpad=0.3)  # , bbox_to_anchor=(1, 1)"""
        plt.tight_layout()
        plt.savefig(path_pubfig + '/plot_z_by_z-true_dr{}_zoom.png'.format(lbl_dxy))
        plt.show()
        plt.close()

    # ---

    # plot dead zone
    fig, ax = plt.subplots(figsize=(size_y_inches * 0.6, size_y_inches * 0.6))

    for i, clr in enumerate(clrs):
        ax.plot(dxys[i], dz_dead_zones[i], 'o', color=clr)

    ax.set_ylabel(r'$\Delta z_{dead zone} \: (\mu m)$')
    ax.set_xlabel(r'$\Delta r \: (pix.)$')
    ax.tick_params(axis='both', which='minor',
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.tight_layout()
    plt.savefig(path_pubfig + '/plot_dz-dead-zone_by_dr.png')
    plt.show()
    plt.close()

    # ---

# ---


# ----------------------------------------------------------------------------------------------------------------------
# COMPARE LOCAL
plot_local_correlation_coefficient = False

if plot_local_correlation_coefficient:

    path_test_coords = base_dir + '/test_coords/ct+3_rescale-0'
    path_results = base_dir + '/results/results-ct+3_MicroSIG_-1pixel_correction_rescale-0'

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # fnxs = [37.5, 37.5, 37.5, 37.5, 37.6, 37.75, 38]
    # fnys = [37.5, 37.6, 37.75, 38, 37.6, 37.75, 38]

    fnxs = [37.5, 37.5, 37.5, 37.6, 37.75, 38]
    fnys = [37.5, 37.75, 38, 37.6, 37.75, 38]

    # fnxs = [37.5, 37.6, 38]
    # fnys = [37.5, 37.6, 38]

    cx, cy = 37.5, 37.5

    # processing
    column_to_bin = 'z_true'
    bins = 20
    min_cm = 0.0
    z_range = None
    round_to_decimal = 4
    df_ground_truth = None
    dropna = False
    dropna_cols = 'z'
    error_column = None
    include_xy = True
    xy_colss = [['x', 'y']]

    # plot setup
    ms = 3

    # iterate
    for xy_cols in xy_colss:

        # figure
        fig, axz = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.65))

        for fnx, fny in zip(fnxs, fnys):
            df = pd.read_excel(path_test_coords + '/test_coords_stats_x{}_y{}.xlsx'.format(fnx, fny))

            if correct_MicroSIG_xy_error:
                df['xg'] = df['xg'] + 0
                df['yg'] = df['yg'] + 0
                df['x_true'] = df['x_true'] - MicroSIG_xy_error
                df['y_true'] = df['y_true'] - MicroSIG_xy_error

            # calc sub-pixel dist.
            dxy = np.round(np.sqrt((fnx - cx) ** 2 + (fny - cy) ** 2), 3)

            dfrmse = bin.bin_local_rmse_z(df,
                                          column_to_bin,
                                          bins,
                                          min_cm,
                                          z_range,
                                          round_to_decimal,
                                          df_ground_truth,
                                          dropna,
                                          dropna_cols,
                                          error_column,
                                          include_xy,
                                          xy_cols,
                                          )

            # plot
            axz.plot(dfrmse.index, dfrmse.cm, '-o', ms=ms, label=dxy)

        axz.set_ylabel(r'$C_{m}^{\delta}$')
        # axz.set_ylim(top=1.01)
        axz.set_xlabel(r'$z \: (\mu m)$')
        axz.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$\Delta xy_{t}$',
                   borderpad=0.3, labelspacing=0.3, handlelength=1.5, handletextpad=0.5)  #
        plt.tight_layout()
        plt.savefig(path_results + '/Cm_by_z.svg')
        plt.show()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# PLOT PUBLICATION FIGURE
plot_pubfig = False

if plot_pubfig:

    # read mean rmse-z
    path_results = base_dir + '/compare'
    df = pd.read_excel(path_results + '/compare_ct+3_rescale-0.xlsx')

    # calib position
    cx, cy = 37.5 - MicroSIG_xy_error, 37.5 - MicroSIG_xy_error

    # add column
    df['cx_true'] = cx
    df['cy_true'] = cy

    # calculate sub-pixel displacement
    df['dx'] = df['x_true'] - df['cx_true']
    df['dy'] = df['y_true'] - df['cy_true']
    df['dxy'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)
    df['dxy_lbl'] = np.round(np.sqrt(df['dx'] ** 2 + df['dy'] ** 2), 3)

    # plot setup
    x = 'dxy'
    df = df.sort_values(x)
    ms = 4

    # export
    # df.to_excel(path_results + '/compare-dxy_ct+3_rescale-0.xlsx')

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(size_x_inches, size_y_inches * 1.1),
                                   gridspec_kw={'height_ratios': [2.9, 1]})

    # plot mean rmse-z
    ax2.plot(df[x], df.rmse_z, '-', color='black', alpha=0.0)
    ax2.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax2.set_ylim([0.4, 3.7])
    ax2.set_xlabel(r'$\Delta r \: (pix.)$')
    ax2.tick_params(axis='both', which='minor',
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # store df
    df_rmse_mean = df.copy()

    # ---

    # plot local Cm

    # read correlation coefficients
    path_test_coords = base_dir + '/test_coords/ct+3_rescale-0'

    fnxs_cm = [37.5, 37.5, 37.6, 37.5, 37.75, 37.5, 38]
    fnys_cm = [37.5, 37.6, 37.6, 37.75, 37.75, 38, 38]
    # fnxs_cm = [37.5, 37.5, 37.5, 37.6, 37.75, 38]
    # fnys_cm = [37.5, 37.75, 38, 37.6, 37.75, 38]
    cx_lbl, cy_lbl = 37.5, 37.5

    # plot setup
    clrs = [sciblue, scigreen, sciorange, scired, scigray, scipurple, sciblack]
    ms = 1

    # processing
    column_to_bin = 'z_true'
    bins = 25
    min_cm = 0.0
    z_range = None
    round_to_decimal = 4
    df_ground_truth = None
    dropna = False
    dropna_cols = 'z'
    error_column = None
    include_xy = True
    xy_cols = ['x', 'y']

    for fnx, fny, clr in zip(fnxs_cm, fnys_cm, clrs):
        df = pd.read_excel(path_test_coords + '/test_coords_stats_x{}_y{}.xlsx'.format(fnx, fny))

        if correct_MicroSIG_xy_error:
            df['xg'] = df['xg'] + 0
            df['yg'] = df['yg'] + 0
            df['x_true'] = df['x_true'] - MicroSIG_xy_error
            df['y_true'] = df['y_true'] - MicroSIG_xy_error

        # calc sub-pixel dist.
        dxy = np.round(np.sqrt((fnx - cx_lbl) ** 2 + (fny - cy_lbl) ** 2), 3)
        lbl_dx_dy = '({}, {}, {})'.format(np.round(fnx - cx_lbl, 2), np.round(fny - cy_lbl, 2), dxy)

        dfrmse = bin.bin_local_rmse_z(df,
                                      column_to_bin,
                                      bins,
                                      min_cm,
                                      z_range,
                                      round_to_decimal,
                                      df_ground_truth,
                                      dropna,
                                      dropna_cols,
                                      error_column,
                                      include_xy,
                                      xy_cols,
                                      )

        # plot
        p1, = ax1.plot(dfrmse.index, dfrmse.cm, '-o', ms=ms, color=clr, label=lbl_dx_dy)
        ax2.plot(df_rmse_mean[df_rmse_mean['dxy_lbl'] == dxy].dxy,
                 df_rmse_mean[df_rmse_mean['dxy_lbl'] == dxy].rmse_z,
                 'o', ms=4, color=p1.get_color())

    ax1.set_ylabel(r'$C_{m}^{\delta}$')
    ax1.set_xlabel(r'$z_{true} \: (\mu m)$')
    """ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.15), title=r'$(\Delta x, \Delta y, \Delta r)$',
               borderpad=0.3, labelspacing=0.3, handlelength=1., handletextpad=0.5)"""
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(\Delta x, \Delta y, \Delta r)$',
               borderpad=0.3, labelspacing=0.8, handlelength=1., handletextpad=0.5)
    plt.tight_layout()
    plt.savefig(path_pubfig + '/plot_local_Cm_by_z_and_mean_rmse-z_by_dr_legend.png')
    plt.show()

    # ---

# ---

# ---

print("Analysis completed without errors.")