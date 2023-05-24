# imports
from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from utils.plotting import lighten_color


sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


mag_eff = 10.01
NA_eff = 0.45
microns_per_pixel = 1.6
size_pixels = 16  # microns

z_range = [-50, 55]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
min_cm = 0.9

# ---

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_localization_performance'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')
filetype = '.xlsx'

# file names
fni = 'bin-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-error'
fns = 'bin-dz_rmse-xy-z-percent-meas_by_z_true_units=microns_relFIJI_ec-error_corr_tilt'
fng = None

# error columns
error_columns = ['error']

for ec in error_columns:

    fpi = join(path_read, 'idpt', fni + '.xlsx')
    fps = join(path_read, 'spct', fns + '.xlsx')


    dfi = pd.read_excel(fpi)
    dfs = pd.read_excel(fps)

    # setup
    save_id = 'IDPT_SPCT'

    include_gdpt = False
    if include_gdpt:
        save_id = save_id + '_GDPT'
        fp_gdpt = join(path_read, 'gdpt', fng + '.xlsx')
        dfgdpt = pd.read_excel(fp_gdpt)

    include_spct_min_cm_other = True
    if include_spct_min_cm_other:
        save_id = save_id + '_min-cm-0.9'
        fpss = join(path_read, 'spct_min-cm-0.9', fns + '.xlsx')
        dfss = pd.read_excel(fpss)

    # ---

    # plot setup
    save_figs = True
    show_figs = True
    rmse_xy_cols = [['rmse_xyg', 'rmse_xy_gauss'], ['rmse_drg', 'rmse_gauss_dr'], ['rmse_drgf', 'rmse_gauss_drf']]
    ms = 4

    if min_cm < 0.9:
        ylim_cm = [0.7, 1.01]
        yticks_cm = [0.7, 1]
    else:
        ylim_cm = [0.87, 1.01]
        yticks_cm = [0.9, 1]

    # plot

    if save_figs or show_figs:

        for h, ps in zip([1, measurement_depth], [1, microns_per_pixel]):

            for r_col in rmse_xy_cols:

                """fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True,
                                                         figsize=(size_x_inches, size_y_inches * 1.5),
                                                         )"""

                # figure.constrained_layout.w_pad:  0.04167  # inches. Default is 3/72 inches (3 points)
                plt.rcParams['figure.constrained_layout.w_pad'] = 0.25

                fig = plt.figure(constrained_layout=True,
                                 figsize=(size_x_inches * 2, size_y_inches * 1.25))
                gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])  # create a 1x2 grid of axes (1 row, 2 columns)
                gsl = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                                       hspace=0.2, wspace=0.25)  # split the left axis into 3 rows
                gsr = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1],
                                                       hspace=0.2,
                                                       wspace=0.25)  # split the right axis into a top and bottom

                ax0 = fig.add_subplot(gsl[0, 0])  # left axis - top
                ax1 = fig.add_subplot(gsl[1, 0])  # left axis - middle
                ax2 = fig.add_subplot(gsl[2, 0])  # left axis - bottom
                ax3 = fig.add_subplot(gsr[0, 0])  # right axis - top
                ax4 = fig.add_subplot(gsr[1, 0])  # right axis - bottom

                # left column
                ax0.plot(dfi.bin / h, dfi.cm, '-o', ms=ms, label='IDPT', zorder=4)
                ax0.plot(dfs.bin / h, dfs.cm, '-o', ms=ms, label='SPCT')
                ax0.set_ylabel(r'$C^{\delta}_{m}$')
                ax0.set_ylim(ylim_cm)
                ax0.set_yticks(yticks_cm)

                ax1.plot(dfi.bin / h, dfi.percent_meas_idd, '-o', ms=ms, label='IDPT', zorder=4)
                ax1.plot(dfs.bin / h, dfs.percent_meas_idd, '-o', ms=ms, label='SPCT', zorder=3.5)
                ax1.set_ylabel(r'$\phi^{\delta}_{ID}$')
                ax1.set_ylim([0, 1.05])
                ax1.set_yticks([0, 1])

                ax2.plot(dfi.bin / h, dfi.true_percent_meas, '-o', ms=ms, label='IDPT', zorder=4)
                ax2.plot(dfs.bin / h, dfs.true_percent_meas, '-o', ms=ms, label='SPCT', zorder=3.5)
                ax2.set_ylabel(r'$\phi^{\delta}$')
                ax2.set_ylim([0, 1.05])
                ax2.set_yticks([0, 1])

                # right column
                ax3.plot(dfi.bin / h, dfi[r_col[0]] / ps, '-o', ms=ms, label='IDPT' + r'$(C_{m,min}=0.5)$', zorder=4)
                ax3.plot(dfs.bin / h, dfs[r_col[1]] / ps, '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.5)$', zorder=3.5)

                ax4.plot(dfi.bin / h, dfi.rmse_z / h, '-o', ms=ms, label='IDPT', zorder=4)
                ax4.plot(dfs.bin / h, dfs.rmse_z / h, '-o', ms=ms, label='SPCT', zorder=3.5)

                if include_gdpt:
                    ax0.plot(dfgdpt.bin / h, dfgdpt.cm, '-o', ms=ms, label='GDPT')
                    ax1.plot(dfgdpt.bin / h, dfgdpt.percent_meas_idd, '-o', ms=ms, label='GDPT', zorder=3.3)
                    ax2.plot(dfgdpt.bin / h, dfgdpt.true_percent_meas, '-o', ms=ms, label='GDPT', zorder=3.3)
                    ax3.plot(dfgdpt.bin / h, dfgdpt[r_col[1]] / ps, '-o', ms=ms, label='GDPT', zorder=3.3)
                    ax4.plot(dfgdpt.bin / h, dfgdpt.rmse_z / h, '-o', ms=ms, label='GDPT', zorder=3.5)

                if include_spct_min_cm_other:
                    ax0.plot(dfss.bin / h, dfss.cm, '-o', ms=ms, label='SPCT')
                    ax1.plot(dfss.bin / h, dfss.percent_meas_idd, '-o', ms=ms, label='SPCT', zorder=3.3)
                    ax2.plot(dfss.bin / h, dfss.true_percent_meas, '-o', ms=ms, label='SPCT', zorder=3.3)
                    ax3.plot(dfss.bin / h, dfss[r_col[1]] / ps, '-o', ms=ms, label='SPCT' + r'$(C_{m,min}=0.9)$', zorder=3.3)
                    ax4.plot(dfss.bin / h, dfss.rmse_z / h, '-o', ms=ms, label=r'$SPCT(C_{m,min}=0.9)$', zorder=3.5)

                # ax4.legend(loc='upper left')
                ax3.legend(loc='upper right')

                if h == 1:
                    ax2.set_xlabel(r'$z \: (\mu m)$')
                    ax3.set_ylabel(r'$\sigma^{\delta}_{xy} \: (\mu m)$')
                    ax4.set_ylabel(r'$\sigma^{\delta}_{z} \: (\mu m)$')
                    ax4.set_xlabel(r'$z \: (\mu m)$')
                    xlim_norm = [-50, 0, 50]
                    ax0.set_xticks(xlim_norm, labels=[])
                    ax1.set_xticks(xlim_norm, labels=[])
                    ax2.set_xticks(xlim_norm)
                    ax3.set_xticks(xlim_norm, labels=[])
                    ax4.set_xticks(xlim_norm)
                else:
                    ax2.set_xlabel(r'$z / h$')
                    ax3.set_ylabel(r'$\sigma^{\delta}_{xy} / w$')
                    ax4.set_ylabel(r'$\sigma^{\delta}_{z} / h$')
                    ax4.set_xlabel(r'$z / h$')
                    xlim_norm = [-0.5, 0, 0.5]
                    ax0.set_xticks(xlim_norm, labels=[])
                    ax1.set_xticks(xlim_norm, labels=[])
                    ax2.set_xticks(xlim_norm)
                    ax3.set_xticks(xlim_norm, labels=[])
                    ax4.set_xticks(xlim_norm)

                # fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.0, wspace=0.0)

                # plt.tight_layout()
                if save_figs:
                    plt.savefig(path_save +
                                '/{}-xyz-percent-meas-by-z_true_norm-{}_{}.png'.format(r_col[0], h, save_id))
                if show_figs:
                    plt.show()
                plt.close()

        # ---

# ---