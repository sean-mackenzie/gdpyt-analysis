# imports
from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import bin, fit, functions
from utils.plotting import lighten_color

"""
Blue: #0C5DA5
Green: #00B945
"""

plt.style.use(['science', 'ieee', 'std-colors'])
sci_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

# --- structure files

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/spct-stats-grid-overlap'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')

filetype = '.xlsx'

fp_spct_stats = 'calib_spct_stats_grid-dz_spct_nl15'
fp_img_sim_spct = 'collection_similarities_grid-dz_spct_nl15'
fp_avg_img_sim_spct = 'average_similarity_grid-dz_spct_nl15'
fp_self_sim_spct = 'calib_stacks_forward_self-similarity_grid-dz_spct_nl15'
# fp_test_sim_spct = 'particle_similarity_curves_spct_grid-overlap-nl1'

# ---

# --- spct stats
plot_spct_stats = False
collection_similarity = False
self_similarity = False

save_figs = False
show_figs = False
save_fig_type = '.svg'

if any([plot_spct_stats, collection_similarity, self_similarity]):

    # --- read data
    dfstats = pd.read_excel(join(path_read, fp_spct_stats) + filetype)

    # --- process data
    dfstats = dfstats[['frame', 'id', 'z_true', 'z', 'z_corr', 'mean_int', 'bkg_mean', 'bkg_noise', 'contour_area']]
    dfstats['snr'] = (dfstats['mean_int'] - dfstats['bkg_mean']) / dfstats['bkg_noise']

    dfstats_mean = dfstats.groupby('frame').mean()
    dfstats_std = dfstats.groupby('frame').std()

    # figure 1, 2: snr, contour_area(z, z_corr)
    if plot_spct_stats:
        ms = 2
        z_params = ['z', 'z_corr']
        for z in z_params:
            fig, ax = plt.subplots()

            ax.plot(dfstats_mean[z], dfstats_mean.snr, '-o', ms=ms, color=sciblue, label='SNR')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$SNR$', color=sciblue)
            ax.set_ylim(bottom=0)
            # ax.set_yticks([0, 10, 20, 30, 40])

            axr = ax.twinx()
            axr.plot(dfstats_mean[z], dfstats_mean.contour_area, '-o', ms=ms, color=scigreen, label='Area')
            axr.set_ylabel(r'$Area \: (pix.)$', color=scigreen)
            axr.set_ylim(bottom=0)
            # axr.set_yticks([0, 100, 200, 300])

            plt.tight_layout()
            if save_figs:
                plt.savefig(path_save + '/spct_snr-area_by_{}{}'.format(z, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

    # ---

    # --- spct collection similarity
    if collection_similarity:
        # --- read data
        dfimgsim = pd.read_excel(join(path_read, fp_img_sim_spct) + filetype)
        # dfavgimgsim = pd.read_excel(join(path_read, fp_avg_img_sim_spct) + filetype)

        # --- process data
        dfimgsim_mean = dfimgsim.groupby('frame').mean()
        dfimgsim_std = dfimgsim.groupby('frame').std()

        # figure 1: cm, SNR(z)

        """z_params = ['z']
        for z in z_params:
        
            ms = 2
            fig, ax = plt.subplots()
        
            ax.plot(dfimgsim_mean[z], dfimgsim_mean.cm, '-o', ms=ms, color=sciblue, label='cm')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\overline{S}(p_{i}, p_{N})$', color=sciblue)
            ax.set_ylim(bottom=0.975)
            # ax.set_yticks([0, 10, 20, 30, 40])
        
            axr = ax.twinx()
            axr.plot(dfstats_mean[z], dfstats_mean.snr, '-o', ms=ms, color=scigreen, label='SNR')
            axr.set_ylabel(r'$SNR$', color=scigreen)
            axr.set_ylim(bottom=0)
            # axr.set_yticks([0, 100, 200, 300])
        
            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_cm-snr_by_{}{}'.format(z, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()


        # figure 2: cm, contour_area(z)

        z_params = ['z']
        for z in z_params:
        
            ms = 2
            fig, ax = plt.subplots()
        
            ax.plot(dfimgsim_mean[z], dfimgsim_mean.cm, '-o', ms=ms, color=sciblue, label='cm')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\overline{S}(p_{i}, p_{N})$', color=sciblue)
            ax.set_ylim(bottom=0.975)
            # ax.set_yticks([0, 10, 20, 30, 40])
        
            axr = ax.twinx()
            axr.plot(dfstats_mean[z], dfstats_mean.contour_area, '-o', ms=ms, color=scigreen, label='Area')
            axr.set_ylabel(r'$Area \: (pix.)$', color=scigreen)
            axr.set_ylim(bottom=0)
            # axr.set_yticks([0, 100, 200, 300])
        
            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_cm-area_by_{}{}'.format(z, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()"""

        # ---

        # figure 3: cm, (snr, contour_area)

        z = 'z'

        split_plots = ['cols', 'rows']
        for sp in split_plots:

            if sp == 'cols':
                fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches, size_y_inches))
                ms = 2.5
            else:
                fig, ax = plt.subplots(nrows=2, figsize=(size_x_inches, size_y_inches))
                ax[1].set_ylabel(r'$\overline{S}(p_{i}, p_{N})$')
                ms = 2

            ax[0].scatter(dfstats_mean.snr, dfimgsim_mean.cm, c=dfimgsim_mean[z], s=ms * 2, label='cm')
            ax[0].set_xlabel(r'$SNR$')
            ax[0].set_xlim(left=10)
            ax[0].set_ylabel(r'$\overline{S}(p_{i}, p_{N})$')
            ax[0].set_ylim([0.9735, 1.0015])

            sc = ax[1].scatter(dfstats_mean.contour_area, dfimgsim_mean.cm, c=dfimgsim_mean[z], s=ms * 2, label='cm')
            ax[1].set_xlabel(r'$Area \: (pix.)$')
            ax[1].set_xlim(left=10)
            ax[1].set_ylim([0.9735, 1.0015])

            plt.tight_layout()

            # colorbar
            plt.subplots_adjust(bottom=0.15, right=0.8, top=0.9)
            cax = plt.axes([0.85, 0.15, 0.025, 0.8])
            cbar = plt.colorbar(sc, cax=cax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()

            plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.5)

            if save_figs:
                plt.savefig(path_save + '/spct_cm_by_snr-area_{}{}'.format(sp, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # figure 4: cm, [normalized(snr, contour_area)]

        z_params = ['z']
        for z in z_params:

            ms = 2
            fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.5, size_y_inches),
                                   gridspec_kw={'width_ratios': [1, 1.2]})

            ax[0].scatter(dfstats_mean.snr / dfstats_mean.snr.max(), dfimgsim_mean.cm, c=dfimgsim_mean[z], s=ms * 2,
                          label='cm')
            ax[0].set_xlabel(r'$SNR/SNR_{max}$')
            ax[0].set_xlim(left=0)
            ax[0].set_ylabel(r'$\overline{S}(p_{i}, p_{N})$')

            sc = ax[1].scatter(dfstats_mean.contour_area / dfstats_mean.contour_area.max(), dfimgsim_mean.cm,
                               c=dfimgsim_mean[z], s=ms * 2, label='cm')
            cbar = plt.colorbar(sc, ax=ax[1], extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax[1].set_xlabel(r'$A/A_{max}$')
            ax[1].set_xlim(left=0)

            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_cm_by_normalized-snr-area{}'.format(save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # figure 5: cm, [normalized(snr, contour_area)]

        z_params = ['z']
        for z in z_params:

            ms = 2
            fig, ax = plt.subplots()

            sc = ax.scatter(
                dfstats_mean.snr / dfstats_mean.snr.max() * dfstats_mean.contour_area / dfstats_mean.contour_area.max(),
                dfimgsim_mean.cm, c=dfimgsim_mean[z], s=ms * 2, label='cm')
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_xlabel(r'$\frac{SNR}{SNR_{max}} \frac{Area}{Area_{max}}$')
            ax.set_xlim(left=0)
            ax.set_ylabel(r'$\overline{S}(p_{i}, p_{N})$')

            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_cm_by_norm-snr-area{}'.format(save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # figure 6: cm(information-to-noise ratio = [normalized(snr, contour_area)])

        z_params = ['z']
        for z in z_params:

            ms = 2
            fig, ax = plt.subplots()

            sc = ax.scatter(
                dfstats_mean.snr / dfstats_mean.snr.max() * dfstats_mean.contour_area / dfstats_mean.contour_area.max(),
                dfimgsim_mean.cm, c=dfimgsim_mean[z], s=ms * 2, label='cm')
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_xlabel(r'$INR$')
            ax.set_xlim(left=0)
            ax.set_ylabel(r'$\overline{S}(p_{i}, p_{N})$')

            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_cm_by_information-to-noise-ratio{}'.format(save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # figure 7: variance of cm(information-to-noise ratio = [normalized(snr, contour_area)])

        z_params = ['z']
        for z in z_params:

            ms = 2
            fig, ax = plt.subplots()

            sc = ax.scatter(
                dfstats_mean.snr / dfstats_mean.snr.max() * dfstats_mean.contour_area / dfstats_mean.contour_area.max(),
                np.sqrt(dfimgsim_std.cm), c=dfimgsim_mean[z], s=ms * 2, label='cm')
            cbar = plt.colorbar(sc, ax=ax, extend='both', aspect=40, label=r'$z \: (\mu m)$')
            cbar.minorticks_on()
            ax.set_xlabel(r'$INR$')
            ax.set_xlim(left=0)
            ax.set_ylabel(r'$\sigma^2 ( \overline{S} (p_{i}, p_{N}) ) $')

            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_variance-cm_by_information-to-noise-ratio{}'.format(save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # figure 8: error bars: cm(z)

        z_params = ['z']
        for z in z_params:

            ms = 2
            fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches))

            ax.errorbar(dfimgsim_mean[z], dfimgsim_mean.cm, yerr=dfimgsim_std.cm,
                        ms=ms, marker='o', capsize=2, elinewidth=1, color=sciblue, label='cm')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\overline{S}(p_{i}, p_{N})$')
            ax.set_ylim(bottom=0.975)

            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_cm_by_{}_errorbars{}'.format(z, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # figure 9: variance cm(z)

        z_params = ['z']
        for z in z_params:

            ms = 2
            fig, ax = plt.subplots()

            ax.plot(dfimgsim_mean[z], np.sqrt(dfimgsim_std.cm), '-o', ms=ms, color=sciblue, label='cm')
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_ylabel(r'$\sigma^2 ( \overline{S} (p_{i}, p_{N}) ) $')

            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_variance-cm_by_{}{}'.format(z, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

        # figure 10: normalized(SNR, area, cm, variance cm) ~ f(z)

        z_params = ['z']
        for z in z_params:

            ms = 1
            fig, ax = plt.subplots(figsize=(size_x_inches * 1.2, size_y_inches))

            ax.plot(dfimgsim_mean[z], dfstats_mean.snr / dfstats_mean.snr.max(),
                    '-o', ms=ms, label=r'$\widetilde{SNR}$')

            ax.plot(dfimgsim_mean[z], dfstats_mean.contour_area / dfstats_mean.contour_area.max(),
                    '-o', ms=ms, label=r'$\tilde{A}$')

            ax.plot(dfimgsim_mean[z], dfimgsim_mean.cm / np.max(dfimgsim_mean.cm),
                    '-o', ms=ms, label=r'$\widetilde{c_{m}}$')

            ax.plot(dfimgsim_mean[z], np.sqrt(dfimgsim_std.cm) / np.max(np.sqrt(dfimgsim_std.cm)),
                    '-o', ms=ms, label=r'$\widetilde{\sigma^2(c_{m})}$', alpha=0.5)

            ax.set_xlabel(r'$z \: (\mu m)$')
            # ax.set_ylabel(r'$\sigma^2 ( \overline{S} (p_{i}, p_{N}) ) $')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/spct_normalized-snr-area-cm-variance-cm_by_{}{}'.format(z, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

        # ---

    # ---

    # --- spct self similarity
    if self_similarity:
        # --- read data
        dfselfsim = pd.read_excel(join(path_read, fp_self_sim_spct) + filetype)

        # --- process data
        dfselfsim_mean = dfselfsim.groupby('z').mean().reset_index()
        dfselfsim_std = dfselfsim.groupby('z').std().reset_index()

        # figure 1: self similarity(z)

        ms = 2
        param_z = 'z'
        fig, ax = plt.subplots()

        ax.plot(dfselfsim_mean[param_z], dfselfsim_mean.cm, '-o', ms=ms)

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\overline {S} (z_{i}, z_{i+1})$')
        plt.tight_layout()

        if save_figs:
            plt.savefig(path_save + '/forward-self-similarity_{}{}'.format(param_z, save_fig_type))
        if show_figs:
            plt.show()
        plt.close()

        # ---

        # figure 2: self similarity(z)

        ms = 2
        param_z = 'z'
        fig, ax = plt.subplots()

        ax.errorbar(dfselfsim_mean[param_z], dfselfsim_mean.cm, yerr=dfselfsim_std.cm,
                    ms=ms, marker='o', capsize=2, elinewidth=1, color=sciblue, label='cm')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\overline {S} (z_{i}, z_{i+1})$')
        plt.tight_layout()

        if save_figs:
            plt.savefig(path_save + '/forward-self-similarity_{}_errorbars{}'.format(param_z, save_fig_type))
        if show_figs:
            plt.show()
        plt.close()

        # --

        # figure 2.5: self similarity(z) + fit line

        ms = 2
        param_z = 'z'
        x = dfselfsim_mean[param_z].to_numpy()
        y = dfselfsim_mean.cm.to_numpy()
        ystd = dfselfsim_std.cm.to_numpy()

        # fit polynomials
        x_fit = np.linspace(x.min(), x.max(), 200)
        p2 = np.poly1d(np.polyfit(x, y, 2))
        p6 = np.poly1d(np.polyfit(x, y, 6))

        # plot
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=ystd,
                    linestyle='', ms=ms, marker='o', capsize=2, elinewidth=1, color=sciblue, alpha=1, label='Data')

        ax.plot(x_fit, p2(x_fit),
                linestyle='-', linewidth=0.5, color=lighten_color(sciblue, 0.75), label=r'p$(2^{\circ})$')
        ax.plot(x_fit, p6(x_fit),
                linestyle='-', linewidth=0.5, color=scired, label=r'p$(6^{\circ})$')

        # plot vertical and horizontal lines
        ax.axvline(x=-4, linestyle='--', linewidth=0.5, color='gray', alpha=0.35, label=r'$z_{f}$')

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\overline {S} (z_{i}, z_{i+1})$')
        ax.legend(loc='upper left', markerscale=0.5, handlelength=1, borderpad=0.25,
                  labelspacing=0.25, handletextpad=0.4, borderaxespad=0.4)
        plt.tight_layout()

        if save_figs:
            plt.savefig(path_save + '/frwd_ss_and_fit_{}_errorbars{}'.format(param_z, save_fig_type))
        if show_figs:
            plt.show()
        plt.close()

        # ---

        # figure 2.75: self similarity(z) + fit line + derivative

        # poly derivative
        dp2dz = p2.deriv(m=1)
        dp6dz = p6.deriv(m=1)

        # plot
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

        # self-sim + fit
        ax1.errorbar(x, y, yerr=ystd,
                     linestyle='', ms=ms, marker='o', capsize=2, elinewidth=1, color=sciblue, alpha=1, label='Data')
        pl1, = ax1.plot(x_fit, p2(x_fit),
                        linestyle='-', linewidth=0.75, color='black', label=r'p$(2^{\circ})$')
        pl2, = ax1.plot(x_fit, p6(x_fit),
                        linestyle='-', linewidth=0.75, color=scired, label=r'p$(6^{\circ})$')

        # derivatives
        ax2.plot(x_fit, dp2dz(x_fit), color=pl1.get_color())
        ax2.plot(x_fit, dp6dz(x_fit), color=pl2.get_color())

        # zf vertical line + dS/dz horizontal line
        ax1.axvline(x=-4, linestyle='--', linewidth=0.5, color='gray', alpha=0.35, label=r'$z_{f}$')
        ax2.axvline(x=-4, linestyle='--', linewidth=0.5, color='gray', alpha=0.35)
        ax2.axhline(y=0, linewidth=0.5, color='gray', alpha=0.35)

        ax1.set_ylabel(r'$\overline {S} (z_{i}, z_{i+1})$')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # , markerscale=0.5, handlelength=1, borderpad=0.25, labelspacing=0.25, handletextpad=0.4, borderaxespad=0.4
        ax2.set_ylabel(r'$ \frac {d\overline {S} (z_{i}, z_{i+1})} {dz} $')
        ax2.set_ylim([-0.0055, 0.0055])
        ax2.set_yticks([-0.005, 0, 0.005])
        ax2.set_xlabel(r'$z \: (\mu m)$')

        plt.tight_layout()

        if save_figs:
            plt.savefig(path_save + '/frwd_ss_and_fit+deriv_{}_errorbars{}'.format(param_z, save_fig_type))
        if show_figs:
            plt.show()
        plt.close()

        # ---

        # figure 2.875: self similarity(z) + fit line + derivative

        # setup
        zf = -4
        depth_of_focus = 12
        y1lim = [0.9535, 1.0035]
        y2lim = [-0.0055, 0.0055]

        # poly derivative
        dp2dz = p2.deriv(m=1)
        dp6dz = p6.deriv(m=1)

        # plot
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

        # self-sim + fit
        ax1.errorbar(x, y, yerr=ystd,
                     linestyle='', ms=ms, marker='o', capsize=2, elinewidth=1, color=sciblue, alpha=1, label='Data')
        pl1, = ax1.plot(x_fit, p2(x_fit),
                        linestyle='-', linewidth=0.75, color='black', label=r'p$(2^{\circ})$')
        pl2, = ax1.plot(x_fit, p6(x_fit),
                        linestyle='-', linewidth=0.75, color=scired, label=r'p$(6^{\circ})$')

        # derivatives
        ax2.plot(x_fit, dp2dz(x_fit), color=pl1.get_color())
        ax2.plot(x_fit, dp6dz(x_fit), color=pl2.get_color())

        # zf vertical line + dS/dz horizontal line
        ax1.axvline(x=zf, linestyle='--', linewidth=0.5, color='gray', alpha=0.35, label=r'$z_{f}$')
        ax2.axvline(x=zf, linestyle='--', linewidth=0.5, color='gray', alpha=0.35)
        ax2.axhline(y=0, linewidth=0.5, color='gray', alpha=0.35)

        # vertical lines corresponding to depth of focus
        ax1.fill_between(x=[zf - depth_of_focus, zf + depth_of_focus], y1=y1lim[1], y2=y1lim[0],
                         color=sciblue, alpha=0.1, label='DoF')
        ax2.fill_between(x=[zf - depth_of_focus, zf + depth_of_focus], y1=y2lim[1], y2=y2lim[0],
                         color=sciblue, alpha=0.1)

        ax1.set_ylabel(r'$\overline {S} (z_{i}, z_{i+1})$')
        ax1.set_ylim(y1lim)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # , markerscale=0.5, handlelength=1, borderpad=0.25, labelspacing=0.25, handletextpad=0.4, borderaxespad=0.4
        ax2.set_ylabel(r'$ \frac {d\overline {S} (z_{i}, z_{i+1})} {dz} $')
        ax2.set_ylim(y2lim)
        ax2.set_yticks([-0.005, 0, 0.005])
        ax2.set_xlabel(r'$z \: (\mu m)$')

        plt.tight_layout()

        if save_figs:
            plt.savefig(path_save + '/frwd_ss_and_fit+deriv+dof_{}_errorbars{}'.format(param_z, save_fig_type))
        if show_figs:
            plt.show()
        plt.close()
        raise ValueError()
        # ---

        # figure 3: variance self similarity(z)

        ms = 2
        param_z = 'z'
        fig, ax = plt.subplots()

        ax.plot(dfselfsim_mean[param_z], np.sqrt(dfselfsim_std.cm), '-o', ms=ms)

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$\sigma^2 ( \overline {S} (z_{i}, z_{i+1}) ) $')
        plt.tight_layout()

        if save_figs:
            plt.savefig(path_save + '/variance-forward-self-similarity_{}{}'.format(param_z, save_fig_type))
        if show_figs:
            plt.show()
        plt.close()

        # ---

        # figure 4: gradient self similarity(z)

        ms = 2
        param_z = 'z'
        fig, ax = plt.subplots()

        ax.plot(dfselfsim_mean[param_z], np.abs(dfselfsim_mean.cm.diff()), '-o', ms=ms)

        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$ \lvert \frac {d\overline {S} (z_{i}, z_{i+1})} {dz} \rvert $')
        plt.tight_layout()

        if save_figs:
            plt.savefig(path_save + '/gradient-forward-self-similarity_{}{}'.format(param_z, save_fig_type))
        if show_figs:
            plt.show()
        plt.close()

        # ---

        # figure 5: gradient self similarity(z) + fit line

        for y_meas in ['magnitude', 'absolute']:

            if y_meas == 'magnitude':
                y = np.abs(dfselfsim_mean.cm.diff().to_numpy()[1:])
            elif y_meas == 'absolute':
                y = dfselfsim_mean.cm.diff().to_numpy()[1:]
            else:
                raise ValueError('y_meas not understood.')

            ms = 2
            param_z = 'z'
            x = dfselfsim_mean[param_z].to_numpy()[1:]

            # fit space
            x_fit = np.linspace(x.min(), x.max(), 200)

            # sliding quadratic
            popt_deg2, pcov, ff = fit.fit(x, y, fit_function=functions.quadratic_slide, bounds=None)
            y2 = functions.quadratic_slide(x_fit, *popt_deg2)

            # sliding quartic
            popt_deg4, pcov, ff = fit.fit(x, y, fit_function=functions.quartic_slide, bounds=None)
            y4 = functions.quartic_slide(x_fit, *popt_deg4)

            fig, ax = plt.subplots()

            ax.plot(x, y, 'o', ms=ms, alpha=0.5, label='Data')
            ax.plot(x_fit, y2, '--', label='Quadratic')
            ax.plot(x_fit, y4, '-.', label='Quartic')

            # plot vertical and horizontal lines
            ax.axvline(x=-4, linestyle='--', linewidth=0.5, color='gray', alpha=0.35, label=r'$z_{f}$')

            if y_meas == 'absolute':
                ax.axhline(y=0, linewidth=0.5, color='gray', alpha=0.35)
                ax.set_ylabel(r'$ \frac {d\overline {S} (z_{i}, z_{i+1})} {dz} $')
            else:
                ax.set_ylabel(r'$ \lvert \frac {d\overline {S} (z_{i}, z_{i+1})} {dz} \rvert $')

            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.legend()
            plt.tight_layout()

            if save_figs:
                plt.savefig(path_save + '/gradient-frwd-ss_and_fit_{}_{}{}'.format(y_meas, param_z, save_fig_type))
            if show_figs:
                plt.show()
            plt.close()

            # ---

        # ---

    # ---

# ---

# --- spct test particle similarity curves
similarity_curves = False

if similarity_curves:

    # setup
    save_plots = False
    show_plots = False

    # --- read data
    dftestsim = pd.read_excel(join(path_read, fp_test_sim_spct) + filetype)

    # --- process data
    pids = bin.sample_array_at_intervals(dftestsim.id.unique(), np.arange(5, 185, 10), 5)
    pids = [62]
    bins_z_true = np.arange(-30, 40, 10)
    bin_width = 5

    # ---

    # figure 0: plot particle positions + ID's
    show_positions = False
    if show_positions:
        fig, ax = plt.subplots()
        sc = ax.scatter(dftestsim.x, dftestsim.y, c=dftestsim.id, s=2, vmin=np.min(pids), vmax=np.max(pids))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(sc, label='p id')
        plt.show()
        plt.close()

    # ---

    # figure 1: cm(z) - one figure/plot per pid and z_true
    plot_sim_per_ztrue = False
    if plot_sim_per_ztrue:
        ms = 2
        param_z = 'z_cm'

        for pid in pids:

            dfpid = dftestsim[dftestsim['id'] == pid]
            pid_z_trues = dfpid.z_true.to_numpy()

            z_true_samples = bin.sample_array_at_intervals(arr_to_sample=pid_z_trues,
                                                           bins=bins_z_true,
                                                           bin_width=bin_width,
                                                           nearest_sample_to_bin=True)

            for zt in z_true_samples:

                dfpidzt = dfpid[dfpid['z_true'] == zt].sort_values(param_z).reset_index()

                fig, ax = plt.subplots()
                p1, = ax.plot(dfpidzt[param_z], dfpidzt.cm, '-o', ms=ms, color=lighten_color(sciblue, 1.2))

                ax.scatter(dfpidzt.iloc[dfpidzt.cm.idxmax()][param_z], dfpidzt.cm.max(), label=r'$z_{est.}$',
                           s=20, marker='d', color=lighten_color(sciblue, 0.9), zorder=3)

                ax.axvline(zt, linewidth=1, linestyle='--', color='black', alpha=0.25, label=r'$z_{true}$')

                ax.set_xlabel(r'$z \: (\mu m)$')
                ax.set_ylabel(r'$S(z_{i}, z)$')
                ax.set_ylim([0, 1])
                ax.legend()
                plt.tight_layout()
                if save_plots:
                    plt.savefig(path_save + '/test-particle-similarity-curves' + '/pid{}_similarity_z{}.png'.format(pid,
                                                                                                                    np.round(
                                                                                                                        zt,
                                                                                                                        1)))
                if show_plots:
                    plt.show()
                plt.close()

        # ---

    # ---

    # figure 2: cm(z) - one figure per pid (all z_true plots on same figure)

    ms = 2
    param_z = 'z_cm'
    plot_ztrue_vline = False
    plot_zmeas_vline = True
    ylim = [0.175, 1.025]

    for pid in pids:

        dfpid = dftestsim[dftestsim['id'] == pid]
        pid_z_trues = dfpid.z_true.to_numpy()

        z_true_samples = bin.sample_array_at_intervals(arr_to_sample=pid_z_trues,
                                                       bins=bins_z_true,
                                                       bin_width=bin_width,
                                                       nearest_sample_to_bin=True)

        # NOTE: these are mock (fake) labels
        lbls = ['IDPT', 'SPCT']
        multipliers = [1.3, 1.075]
        clr_mod = [1.1, 1.1]

        # set axes limits now so vline plots to scale
        fig, ax = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.65))
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$S(z_{i}, z)$')
        ax.set_ylim(ylim)

        for i, zt in enumerate(z_true_samples):

            dfpidzt = dfpid[dfpid['z_true'] == zt].sort_values(param_z).reset_index()

            z_nearest_idx = np.argmin(np.abs(dfpidzt[param_z].to_numpy() - zt))
            cm_nearest_idx = dfpidzt.iloc[z_nearest_idx].cm

            p1, = ax.plot(dfpidzt[param_z], dfpidzt.cm * multipliers[i], 'o', ms=ms, label=lbls[i])

            ax.scatter(dfpidzt.iloc[dfpidzt.cm.idxmax()][param_z], dfpidzt.cm.max() * multipliers[i],
                       s=25, marker='*', color='black', zorder=3)  # lighten_color(p1.get_color(), 1.25)

            if plot_zmeas_vline:
                ymin = 0
                ymax = (dfpidzt.cm.max() * multipliers[i] - ylim[0]) / (ylim[1] - ylim[0]) - 0.00625
                ax.axvline(dfpidzt.iloc[dfpidzt.cm.idxmax()][param_z],
                           ymin=ymin, ymax=ymax,
                           linewidth=0.5, linestyle='--', color=lighten_color(p1.get_color(), clr_mod[i]), alpha=0.5)

            if plot_ztrue_vline:
                ax.axvline(zt, ymin=0, ymax=cm_nearest_idx - .00625,
                           linewidth=1, linestyle='--', color=lighten_color(p1.get_color(), 0.8), alpha=0.5)

        ax.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(path_save + '/test-particle-similarity-curves' + '/pid{}_similarity_curves_v3.png'.format(pid))
        if show_plots:
            plt.show()
        plt.close()

# ---

# --- spct test particle similarity curves
single_particle_similarity_curve = False

if single_particle_similarity_curve:
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/test_coords/raw/' \
         'test_id11_coords_raw_spct_1501-1700_grid-no-dz-overlap-nl15.xlsx'

    df = pd.read_excel(fp)
    df = df[df['x'] < 50]

    dfm, dfstd = bin.bin_generic(df,
                                 column_to_bin='z_true',
                                 column_to_count='z',
                                 bins=65,
                                 round_to_decimal=2,
                                 return_groupby=True)

    dfm = dfm.sort_values('bin')
    dfstd = dfstd.sort_values('bin')

    fig, ax = plt.subplots()
    ax.plot(dfm.bin, dfm.cm, '-')
    # ax.errorbar(dfm.bin, dfm.cm, yerr=dfstd.cm, fmt='-o', ms=2, elinewidth=1, capsize=2)
    ax.set_xlabel(r'$z \: (\mu m)$')
    # ax.set_xlim([-10, 350 * microns_per_pixels])
    # ax.set_xticks(ticks=[0, 150, 300, 450])
    ax.set_ylabel(r'$c_{m}$')
    # ax.set_ylim([-3.25, 1.85])
    # ax.set_yticks([-3, -2, -1, 0, 1])
    plt.tight_layout()
    plt.savefig(path_save + '/spct_no-dz-overlap_similarity_coeff_by_z_plot.svg')
    plt.show()
    plt.close()

# ---

# --- spct test particle similarity curves
single_particle_spct_stats_from_no_dz_overlap_test = False

if single_particle_spct_stats_from_no_dz_overlap_test:
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/calib_coords/' \
         'calib_spct_stats_grid-dz_calib_nll2_spct_spct-cal.xlsx'
    dfstats = pd.read_excel(fp)

    # --- process data
    dfstats = dfstats[['frame', 'id', 'z_true', 'z', 'z_corr', 'mean_int', 'bkg_mean', 'bkg_noise', 'contour_area']]
    dfstats['snr'] = (dfstats['mean_int'] - dfstats['bkg_mean']) / dfstats['bkg_noise']

    dfstats_mean, dfstats_std = bin.bin_generic(dfstats,
                                                column_to_bin='z_corr',
                                                column_to_count='id',
                                                bins=25,
                                                round_to_decimal=2,
                                                return_groupby=True)

    dfstats_mean = dfstats_mean.sort_values('bin').reset_index()
    dfstats_std = dfstats_std.sort_values('bin').reset_index()

    # figure 1, 2: snr, contour_area(z, z_corr)
    ms = 2
    z_params = ['z_corr']
    for z in z_params:
        fig, ax = plt.subplots()

        ax.plot(dfstats_mean[z], dfstats_mean.snr, '-o', ms=ms, color=sciblue, label='SNR')
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$SNR$', color=sciblue)
        ax.set_ylim(bottom=0)
        # ax.set_yticks([0, 10, 20, 30, 40])

        axr = ax.twinx()
        axr.plot(dfstats_mean[z], dfstats_mean.contour_area, '-o', ms=ms, color=scigreen, label='Area')
        axr.set_ylabel(r'$Area \: (pix.)$', color=scigreen)
        axr.set_ylim(bottom=0)
        # axr.set_yticks([0, 100, 200, 300])

        plt.tight_layout()
        plt.savefig(path_save + '/spct-from-test_snr-area_binned_by_{}{}'.format(z, save_fig_type))
        plt.show()
        plt.close()

    # ---

print("Analysis completed without errors.")