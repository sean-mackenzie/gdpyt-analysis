# imports
from os.path import join
import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from utils.plotting import lighten_color

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'

plt.style.use(['science', 'ieee'])  # 'ieee', 'std-colors', 'nature', 'high-vis', 'bright', , 'std-colors'
fig, ax = plt.subplots()
sci_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
sci_color_cycler = ax._get_lines.prop_cycler
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

mag_eff = 10.01
NA_eff = 0.45
microns_per_pixel = 1.6
size_pixels = 16  # microns

z_range = [-50, 55]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
min_cm = 0.5

# ---

# key modifier
z_error_limit = 4

# ---

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_field-dependence-on-localization'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')
filetype = '.xlsx'

# file names
fni = 'per-particle/per-particle-rmse_z_errlim{}'.format(z_error_limit)
fnir = 'radial-error/dataframe-mean_bin-r-z_norm-z-errors_by_z_errlim{}'.format(z_error_limit)
fnirp = 'radial-error-relative-plane/dataframe-mean_bin-r-z_norm-z-errors_by_z_errlim{}'.format(z_error_limit)

fns = 'per-particle/per-particle-rmse_z_errlim{}'.format(z_error_limit)
fnsr = 'radial-error/dataframe-mean_bin-r-z_norm-z-errors_by_z_errlim{}'.format(z_error_limit)
fnsrp = 'radial-error-relative-plane/dataframe-mean_bin-r-z_norm-z-errors_by_z_errlim{}'.format(z_error_limit)

# file paths
fpi = join(path_read, 'idpt', fni + '.xlsx')
fpir = join(path_read, 'idpt', fnir + '.xlsx')
fpirp = join(path_read, 'idpt', fnirp + '.xlsx')

fps = join(path_read, 'spct', fns + '.xlsx')
fpsr = join(path_read, 'spct', fnsr + '.xlsx')
fpsrp = join(path_read, 'spct', fnsrp + '.xlsx')

dfi = pd.read_excel(fpi)
dfir = pd.read_excel(fpir)
dfirp = pd.read_excel(fpirp)

dfs = pd.read_excel(fps)
dfsr = pd.read_excel(fpsr)
dfsrp = pd.read_excel(fpsrp)

# setup


# plot
plot_z_error_rmse_by_r = True
if plot_z_error_rmse_by_r:


    dfis = [dfir, dfirp]
    dfss = [dfsr, dfsrp]
    yc1s = ['error', 'error_z_plane']

    for dfi_, dfs_, yc1 in zip(dfis, dfss, yc1s):

        r_bins = dfi_.bin_tl.unique()
        r_bins.sort()
        xc = 'z_true'
        yc2 = 'rmse_z_spec'

        # color
        rclrs = ['b' 'k', 'r']
        rclr_mod = [1, 1, 1]
        iclr_mod = [0.8, 1, 1.2]
        sclr_mod = [0.8, 1, 1.2]

        fig, axs = plt.subplots(2, 2, figsize=(size_x_inches * 1.5, size_y_inches * 1.15), constrained_layout=True)
        (ax1, ax2, ax3, ax4) = np.ravel(axs)

        for i, rb in enumerate(r_bins):
            dfibr = dfi_[dfi_['bin_tl'] == rb]
            dfsbr = dfs_[dfs_['bin_tl'] == rb]

            # IDPT (left)
            ax1.plot(dfibr[xc], dfibr[yc1], '-o', ms=2, label=rb,
                     # color=lighten_color(rclrs[i], rclr_mod[i]),
                     )
            ax2.plot(dfibr[xc], dfibr[yc2], '-o', ms=2, label=int(rb * microns_per_pixel),
                     # color=lighten_color(rclrs[i], rclr_mod[i]),
                     )

            # SPCT (right)
            ax3.plot(dfsbr[xc], dfsbr[yc1], '-o', ms=2, label=rb,
                     # color=lighten_color(rclrs[i], rclr_mod[i]),
                     )
            ax4.plot(dfsbr[xc], dfsbr[yc2], '-o', ms=2, label=rb,
                     # color=lighten_color(rclrs[i], rclr_mod[i]),
                     )

        err_lim = [-2.75, 2]
        err_ticks = [-2, 0, 2]
        z_lim = [-55, 60]
        z_ticks = [-50, 0, 50]
        rmse_lim = [0, 2.75]
        rmse_ticks = [0, 2]

        ax1.set_ylabel(r'$\epsilon_{z}^{\delta} \: (\mu m)$')
        ax1.set_ylim(err_lim)
        ax1.set_yticks(err_ticks)
        ax1.set_xlim(z_lim)
        ax1.set_xticks(z_ticks, labels=[])
        ax1.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax2.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax2.set_ylim(rmse_lim)
        ax2.set_yticks(rmse_ticks)
        # ax2.set_xlabel(r'$z \: (\mu m)$')
        ax2.set_xlim(z_lim)
        ax2.set_xticks(z_ticks, labels=[])
        ax2.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$r^{\delta} \: (\mu m)$',
                   markerscale=0.5, handlelength=1)

        ax3.set_ylabel(r'$\epsilon_{z}^{\delta} \: (\mu m)$')
        ax3.set_ylim(err_lim)
        ax3.set_yticks(err_ticks)
        ax3.set_xlim(z_lim)
        ax3.set_xticks(z_ticks)
        ax3.set_xlabel(r'$z \: (\mu m)$')
        ax3.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        ax4.set_ylabel(r'$\sigma_{z}^{\delta} \: (\mu m)$')
        ax4.set_ylim(rmse_lim)
        ax4.set_yticks(rmse_ticks)
        ax4.set_xlabel(r'$z \: (\mu m)$')
        ax4.set_xlim(z_lim)
        ax4.set_xticks(z_ticks)
        ax4.tick_params(axis='both', which='minor',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        fig.set_constrained_layout_pads(w_pad=14 / 72, h_pad=10 / 72, hspace=0, wspace=0)
        plt.savefig(path_save + '/errlim{}_z-{}-rmse_by_r-z_horiz.svg'.format(z_error_limit, yc1))
        plt.show()

    # ---

# ---

# plot x-y scatter of rmse_z per particle
plot_rmse_z_per_particle = False
if plot_rmse_z_per_particle:
    dfs = [dfi, dfs]
    dflbls = ['idpt', 'spct']
    cmaps = ['coolwarm']

    for dft_pid, dflbl in zip(dfs, dflbls):
        for cmp in cmaps:
            if cmp == 'viridis':
                cmap = mpl.cm.viridis
            elif cmp == 'plasma':
                cmap = mpl.cm.plasma
            elif cmp == 'inferno':
                cmap = mpl.cm.inferno
            elif cmp == 'gist_heat':
                cmap = mpl.cm.gist_heat
            elif cmp == 'coolwarm':
                cmap = mpl.cm.coolwarm

            # setup
            vmin, vmax = 0.5, 4
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            padding = 5

            # filters
            units = 'microns'

            # plot
            fig, ax = plt.subplots(figsize=(size_y_inches * 0.85, size_y_inches * 0.85), subplot_kw={'aspect': 'equal'})

            sc = ax.scatter(dft_pid['x'], dft_pid['y'], c=cmap(norm(dft_pid['rmse_z'])), s=10)

            if units == 'microns':
                adj_um = 0  # truthfully = 6.25
                ax.set_xlabel(r'$x (\mu m)$')
                ax.set_xticks([padding + adj_um, 256 + padding, 512 + padding - adj_um], labels=[-400, 0, 400])
                ax.set_xlim([padding, 512 + padding])
                ax.set_ylabel(r'$y (\mu m)$')
                ax.set_ylim([padding, 512 + padding])
                ax.set_yticks([padding + adj_um, 256 + padding, 512 + padding - adj_um], labels=[-400, 0, 400])
                ax.invert_yaxis()
            elif units == 'pixels':
                ax.set_xlabel(r'$X (pix.)$')
                ax.set_xlim([padding, 512 + padding])
                ax.set_xticks([padding, 512 + padding], labels=[0, 512])
                ax.set_ylabel(r'$Y (pix.)$')
                ax.set_ylim([padding, 512 + padding])
                ax.set_yticks([padding, 512 + padding], labels=[0, 512])
                ax.invert_yaxis()

            ax.tick_params(axis='both', which='minor',
                           bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            # color bar
            cbar_lbl = r'$\overline{\sigma}_{z}(i) \: (\mu m)$'
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=cax, orientation='vertical', label=cbar_lbl, extend='both')

            plt.tight_layout()
            plt.savefig(
                path_save +
                '/errlim{}_rmse-per-pid_by_x-y_cm0.5_{}_{}_{}.svg'.format(z_error_limit, units, cmp, dflbl))
            plt.show()
            plt.close()

# ---

# plot x-y scatter of rmse_z per particle
plot_histogram_of_errors = True
if plot_histogram_of_errors:
    from sklearn.neighbors import KernelDensity

    # setup
    methods = ['idpt', 'spct']
    error_columns = ['error']  # , 'error_z_plane', 'error_z_plane'
    error_limits = [4]
    yls = [4]
    dls = [0.75]
    xls = [250, 100]
    dxs = [0, 0]
    binwidth_y = 0.5
    bandwidth_y = 0.5
    scatter_size = 1.5
    lbls = [100, 300, 500]

    for mthd, xl, dx in zip(methods, xls, dxs):
        for error_column, error_limit, yl, dl in zip(error_columns, error_limits, yls, dls):

            # read file
            fnh = 'histogram/dfb-r-z-histogram-errors_by_z_ec-{}_errlim{}'.format(error_column, error_limit)
            dfb = pd.read_excel(join(path_read, mthd, fnh + '.xlsx'))

            # figure
            fig = plt.figure(figsize=(size_x_inches, size_y_inches * 0.8))
            gs = fig.add_gridspec(1, 2, width_ratios=(5, 2),
                                  left=0.1, right=0.75, bottom=0.1, top=0.9, wspace=0.075, hspace=0.075)
            gsl = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                                   hspace=0.5, wspace=0.075)
            gsr = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1],
                                                   hspace=0.5, wspace=0.075)

            ax1 = fig.add_subplot(gsl[0, 0])  # left axis - top
            ax2 = fig.add_subplot(gsl[1, 0])  # left axis - middle
            ax3 = fig.add_subplot(gsl[2, 0])  # left axis - bottom
            ax1_histy = fig.add_subplot(gsr[0, 0], sharey=ax1)  # right axis - top
            ax2_histy = fig.add_subplot(gsr[1, 0], sharey=ax2)  # right axis - middle
            ax3_histy = fig.add_subplot(gsr[2, 0], sharey=ax3)  # right axis - bottom

            ax1.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax2.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax3.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax1_histy.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax2_histy.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax3_histy.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            ax1.tick_params(axis='x', left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax2.tick_params(axis='x', left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax3.tick_params(axis='x', left=False, right=False, labeltop=False, labelleft=False, labelright=False)

            ax1_histy.tick_params(axis='x', labeltop=False)
            ax2_histy.tick_params(axis='x', labeltop=False)
            ax3_histy.tick_params(axis='x', labeltop=False)
            ax1_histy.tick_params(axis='y', labelleft=False, labelright=False)
            ax2_histy.tick_params(axis='y', labelleft=False, labelright=False)
            ax3_histy.tick_params(axis='y', labelleft=False, labelright=False)

            kde = True
            colormap = None
            centerline = True

            for bin_r, ax, ax_histy, color, lbl in zip(dfb.bin.unique(), [ax1, ax2, ax3], [ax1_histy, ax2_histy, ax3_histy],
                                                  sci_color_cycle, lbls):
                dfbr = dfb[dfb['bin'] == bin_r]
                dfbr = dfbr.sort_values('z_true')
                bin_id = int(np.round(bin_r * microns_per_pixel, 0))

                x = dfbr.z_true.to_numpy()
                y = dfbr[error_column].to_numpy()

                # ---

                # plotting

                # the scatter plot:
                ax.scatter(x, y, s=scatter_size, marker='.', c=color, cmap=colormap)
                ax.axhline(y=0, linewidth=0.25, linestyle='--', color='black', alpha=0.5)

                # y
                ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
                ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
                ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
                ny, binsy, patchesy = ax_histy.hist(y, bins=ybins, orientation='horizontal', color='gray', zorder=2.5)

                # kernel density estimation
                if kde:
                    ymin, ymax = np.min(y), np.max(y)
                    y_range = ymax - ymin
                    y_plot = np.linspace(ymin - y_range / 5, ymax + y_range / 5, 250)

                    y = y[:, np.newaxis]
                    y_plot = y_plot[:, np.newaxis]

                    kde_y = KernelDensity(kernel="gaussian", bandwidth=bandwidth_y).fit(y)
                    log_dens_y = kde_y.score_samples(y_plot)
                    scale_to_max = np.max(ny) / np.max(np.exp(log_dens_y))

                    p2 = ax_histy.fill_betweenx(y_plot[:, 0], 0, np.exp(log_dens_y) * scale_to_max,
                                                fc="None", ec=scired,
                                                zorder=2.5)
                    p2.set_linewidth(0.5)
                    # ax_histy.plot(y_plot[:, 0], np.exp(log_dens_y) * scale_to_max, linestyle='-', color=scired)

                # faux scatter plot - legend
                ax_histy.scatter(0, 20, s=scatter_size, marker='.', c=color, cmap=colormap,
                                 label=r'$r^{\delta}=$' + ' {} '.format(lbl) + r'$(\mu m)$')

            # ---

            ylim = [-yl - dl, yl + dl]
            yticks = [-yl, 0, yl]

            ax1.set_ylim(ylim)
            ax1.set_yticks(yticks)
            ax1.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
            # ax1_histy.set_xlim(0, xl + dx)
            # ax1_histy.set_xticks([0, xl])
            ax1_histy.legend(loc='upper left', bbox_to_anchor=(1, 1),
                            markerscale=2.5, borderpad=0.2, borderaxespad=0.2, labelspacing=0.2, handletextpad=0.2)

            ax2.set_ylim(ylim)
            ax2.set_yticks(yticks)
            ax2.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')
            # ax2_histy.set_xlim(0, xl + dx)
            # ax2_histy.set_xticks([0, xl])
            ax2_histy.legend(loc='upper left', bbox_to_anchor=(1, 1),
                            markerscale=2.5, borderpad=0.2, borderaxespad=0.2, labelspacing=0.2, handletextpad=0.2)

            ax3.set_ylim(ylim)
            ax3.set_yticks(yticks)
            ax3.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')

            ax3_histy.set_xlabel('counts')
            # ax3_histy.set_xlim(0, xl + dx)
            #ax3_histy.set_xticks([0, xl])
            ax3_histy.legend(loc='upper left', bbox_to_anchor=(1, 1),
                            markerscale=2.5, borderpad=0.2, borderaxespad=0.2, labelspacing=0.2, handletextpad=0.2)
            ax3.set_xlabel(r'$z \: (\mu m)$')
            ax3.set_xticks([-50, 0, 50])

            plt.savefig(
                path_save +
                '/{}_errlim{}_hist-{}_by_z_smallest.png'.format(mthd, error_limit, error_column))
            plt.show()
            plt.close()

            # ---

# ---

print("Analysis finished without errors.")