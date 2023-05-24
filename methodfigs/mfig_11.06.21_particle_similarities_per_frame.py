# imports
from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# --- structure data
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/particle-similarities-in-image'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')
filetype = '.xlsx'

# experimental
microns_per_pixels = 1.6
image_pixels = 512
image_padding = 5
image_range = image_pixels + image_padding * 2

plot_new = True
if plot_new:

    # ---

    # process self-similarity

    # read self similarity
    fpi = 'calib_stacks_forward_self-similarity_11.06.21.xlsx'
    dfi = pd.read_excel(join(base_dir, 'data', fpi))

    # read collection similarity
    dfsg1 = pd.read_excel(path_read + '/col_avg_11.06.21_filtered_mean.xlsx')
    dfsg1_std = pd.read_excel(path_read + '/col_avg_11.06.21_filtered_std.xlsx')

    # read SPCT stats (note, SPCT stats are from a different analysis so pid's are likely different)
    fn = 'calib_spct_stats_11.06.21_z-micrometer-v2_1umSteps.xlsx'
    dfs = pd.read_excel(join(path_read, fn))

    # ---

    # setup

    # (A) - Process self-similarity
    dfm = dfi.groupby('z').mean().reset_index()
    dfstd = dfi.groupby('z').std().reset_index()

    # confidence bands
    Ss_lower = dfm['cm'] - dfstd['cm']
    Ss_upper = dfm['cm'] + dfstd['cm']

    # ---

    # (B) - Process collection similarity
    dfsg1 = dfsg1.iloc[:-1]
    dfsg1_std = dfsg1_std[:-1]

    # confidence bands
    Sptp_upper = dfsg1.cm.to_numpy() + dfsg1_std.cm.to_numpy()
    Sptp_lower = dfsg1.cm.to_numpy() - dfsg1_std.cm.to_numpy()
    Sptp_upper = np.where(Sptp_upper < 1, Sptp_upper, 1)

    # ---

    # (C.a) - Process number of particles per frame
    dfs = dfs[dfs['id'] < 87]
    dfgzid = dfs.groupby(['z_true', 'id']).mean().reset_index()
    dfgz = dfgzid.groupby('z_true').count().reset_index()

    # (C.b) - Process sigma_xy per grid
    calculate_sigma_xy = False
    if calculate_sigma_xy:
        px = 'gauss_sigma_x_r'
        py = 'gauss_sigma_y_r'
        pxy = 'gauss_sigma_x_y_r'
        min_num_frames = 10

        # pids with most counts
        dfg = dfs.groupby('id').count().reset_index().sort_values(pxy, ascending=False)
        pids_in_most_frames = dfg[dfg[pxy] > min_num_frames].id.values
        pids_max_sigma_x_or_y_std = [71, 85, 46, 67, 73, 7, 9, 39, 54, 55, 84, 83, 11,
                                     1, 15, 50, 0, 3, 2, 4, 22, 5, 16, 68, 69, 12, 10, 13, 14, 86,
                                     26, 44]
        pids_max_sigma_xy_std = [8, 6, 72, 0, 4, 3, 26, 10, 20, 70, 83, 24, 14, 17, 13,
                                 7, 27, 9, 2, 16, 1, 11, 12, 21, 84, 5, 15, 45,
                                 44, 85, 26, 50, 71, 73, 39, 67, 46, 54, 55, 69, 68, 83, 67]
        pids_to_plot = set(pids_in_most_frames) & set(pids_max_sigma_x_or_y_std) & set(pids_max_sigma_xy_std)
        dfp = dfs[dfs['id'].isin(pids_to_plot)]

        # split dataframe into 6 areas
        x_range = [0, image_range / 3, image_range * 2 / 3, image_range]
        y_range = [0, image_range / 3, image_range * 2 / 3, image_range]

        min_num_per_z = 2
        ii = 0
        grid_idx = []
        z_sigma_xys = []
        sigma_xys = []
        for i in range(len(x_range) - 1):
            for j in range(len(y_range) - 1):

                xmin, xmax = x_range[i], x_range[i + 1]
                ymin, ymax = y_range[j], y_range[j + 1]

                dfqx = dfp[(dfp['x'] > xmin) & (dfp['x'] < xmax)]
                dfq = dfqx[(dfqx['y'] > ymin) & (dfqx['y'] < ymax)]

                # filter by number of particles per frame
                dfgc = dfq.groupby('z_true').count().reset_index()
                valid_z_range = dfgc[dfgc['id'] >= min_num_per_z]['z_true'].values
                dfq = dfq[dfq['z_true'].isin(valid_z_range)]

                dfgxyz = dfq.groupby('z_true').mean().reset_index()

                if len(dfgxyz) > 10:
                    grid_idx.append(ii)
                    z_sigma_xys.append(dfgxyz['z_true'].to_numpy())
                    sigma_xys.append(dfgxyz['gauss_sigma_x_y_r'].to_numpy())
                ii += 1

        z_sigma_xys = np.array(z_sigma_xys)
        sigma_xys = np.array(sigma_xys)
        dict_gdx = {0: r'$1, 1$',
                    2: r'$1, 3$',
                    4: r'$2, 4$',
                    6: r'$3, 1$',
                    8: r'$3, 3$',
                    }

    # ---

    # z-offset
    z_f = 50

    # plot two subplots: (1) P-2-P similarity, (2) Self-Similarity
    plot_two_sims = False
    if plot_two_sims:
        fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.91, size_y_inches * 0.85))

        ax1.fill_between(dfsg1['z'], Sptp_lower, Sptp_upper, color=sciblue, alpha=0.125)
        ax1.fill_between(dfm['z'], Ss_lower, Ss_upper, color=scired, alpha=0.125)

        ax1.plot(dfsg1['z'], dfsg1.cm, '-o', ms=1,
                 label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')  # \Bigr \rangle
        ax1.plot(dfm['z'], dfm['cm'], '-o', ms=1, color=scired,
                 label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) $')

        ax1.set_ylabel(r'$ \langle S(\cdot, \cdot) \rangle $')
        ax1.set_ylim(bottom=0.715, top=1.01)
        ax1.set_yticks([0.8, 0.9, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, -25, 0, 25, 50])
        ax1.legend(loc='lower left', markerscale=1.5)

        plt.tight_layout()
        plt.savefig(path_save + '/compare_SPCT_p2p_with_IDPT_forward-self-similarity_sized.svg')
        plt.show()
        plt.close()

    # ---

    # TWO subplots: (1) Sigma XY, (2) P2P and Self-Similarity
    plot_sigma_xy_and_p2p_self_similarity = False
    if plot_sigma_xy_and_p2p_self_similarity:
        # THREE PLOTS

        # plot: (1) number of particles identified per frame, (2) sigma_xy, (3) p2p similarity per frame
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True,
                                             figsize=(size_x_inches * 0.85, size_y_inches * 1.0425),
                                             gridspec_kw={'height_ratios': [1, 1.4]})

        # - Sigma XY
        for gdx in [0, 2, 4, 6, 8]:
            ax0.plot(z_sigma_xys[gdx] - z_f, sigma_xys[gdx], '-',
                     linewidth=1,  # marker='.', ms=0.95,
                     label=dict_gdx[gdx])
        ax0.set_ylabel(r'$w_{x}/w_{y}$')
        ax0.legend(loc='upper left', bbox_to_anchor=(1, 1.1),
                    title=r"$(m, n)$",
                    markerscale=2,
                    handlelength=1, handletextpad=0.4, labelspacing=0.125, borderaxespad=0.3,
                    )
        ax0.tick_params(axis='y', which='minor', left=False, right=False)

        # ---

        # - P2P and Self Similarity

        ax1.fill_between(dfsg1['z'] - z_f, Sptp_lower, Sptp_upper, color=scigreen, alpha=0.125)
        ax1.fill_between(dfm['z'] - z_f, Ss_lower, Ss_upper, color=sciblue, alpha=0.125)

        ax1.plot(dfsg1['z'] - z_f, dfsg1.cm, '-o', ms=1, color=scigreen,
                 label=r'$ S_{wf} $')
        # label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $'
        ax1.plot(dfm['z'] - z_f, dfm['cm'], '-o', ms=1, color=sciblue,
                 label=r'$ S_{i}^{c}(+\Delta_c z) $')
        # label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) $'

        ax1.set_ylabel(r'$ \langle S \rangle $')  # r'$ \langle S(\cdot, \cdot) \rangle $'
        ax1.set_ylim(bottom=0.715, top=1.01)
        ax1.set_yticks([0.8, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        ax1.legend(loc='lower left', markerscale=2,
                    handlelength=0.75, handletextpad=0.6, labelspacing=0.175, borderaxespad=0.4,)

        plt.minorticks_off()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35)
        plt.savefig(path_save + '/plot_sigmaXY_and_compare_p2p-self-similarity_sized-shorthand2.svg')
        plt.show()
        plt.close()

    # ---

    plot_p2p_sim_and_num_pids = False
    if plot_p2p_sim_and_num_pids:
        # ONE PLOT - TWO Y-AXES

        # plot: (1) number of particles identified per frame, (2) p2p similarity per frame
        fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.91, size_y_inches * 0.5))

        # - Number of particles per frame
        axr = ax1.twinx()
        axr.plot(dfgz.iloc[:-1]['z_true'] - z_f, dfgz.iloc[:-1]['id'],
                 '-', color='black', alpha=0.5, zorder=2.5)  # , marker='.', ms=1
        axr.set_ylabel(r"$N_{p}^{''}$")
        axr.set_ylim(top=95)
        axr.set_yticks([40, 80])
        axr.tick_params(axis='y', which='minor', right=False)

        # - P2P Similarity
        ax1.fill_between(dfsg1['z'] - z_f, Sptp_lower, Sptp_upper,
                         color=scired, edgecolor='none', alpha=0.15, zorder=3)
        ax1.plot(dfsg1['z'] - z_f, dfsg1.cm, '-', ms=1, color=scired, zorder=3.5,
                 label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')  # \Bigr \rangle

        ax1.set_ylabel(r'$ S_{wf} $', color=lighten_color(scired, 1.1))
        ax1.set_ylim(bottom=0.7, top=1.0)
        ax1.set_yticks([0.7, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])

        plt.minorticks_off()
        plt.tight_layout()
        # plt.savefig(path_save + '/compare_SPCT_Np-NoOverlap_with_p2p-similarity_same-plot-lines.svg')
        plt.show()
        plt.close()

        # ---

        # TWO PLOTS

        # plot: (1) number of particles identified per frame, (2) p2p similarity per frame
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 0.91, size_y_inches * 0.85),
                                       gridspec_kw={'height_ratios': [1, 1.45]})

        # - Number of particles per frame
        ax0.plot(dfgz.iloc[:-1]['z_true'] - z_f, dfgz.iloc[:-1]['id'], '-o', ms=1)  # , label='N.O.'
        ax0.legend(loc='upper right', handlelength=0.5, handletextpad=0.4, markerscale=1.5)
        ax0.set_ylabel(r"$N_{p}^{''}$")
        ax0.set_ylim(top=96.5)
        ax0.set_yticks([40, 80])
        ax0.tick_params(axis='y', which='minor', left=False, right=False)

        # ---

        # - P2P Similarity
        ax1.fill_between(dfsg1['z'] - z_f, Sptp_lower, Sptp_upper, alpha=0.125)
        ax1.plot(dfsg1['z'] - z_f, dfsg1.cm, '-o', ms=1,
                 label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')  # \Bigr \rangle

        ax1.set_ylabel(r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')
        ax1.set_ylim(bottom=0.715, top=1.01)
        ax1.set_yticks([0.8, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        # ax1.legend(loc='lower left', markerscale=1.5)

        plt.minorticks_off()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35)
        # plt.savefig(path_save + '/compare_SPCT_Np-NoOverlap_with_p2p-similarity_sized.svg')
        plt.show()
        plt.close()

    # ---

    plot_single_figs = True
    if plot_single_figs:
        # plot self-similarity
        fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.85 * 0.93, size_y_inches * 0.6 * 0.93))

        ax1.fill_between(dfm['z'] - z_f, Ss_lower, Ss_upper,
                         color=lighten_color(sciblue, 1.25), edgecolor='none', alpha=0.25)
        ax1.plot(dfm['z'] - z_f, dfm['cm'], '-o', ms=1, color=sciblue,
                 label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) $')

        # ax1.set_ylabel(r'$ \bigl \langle S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) \bigr \rangle $')
        ax1.set_ylabel(r'$S_{ii}^{+}(z)$')
        # ax1.set_ylim(bottom=0.89, top=1.01)
        # ax1.set_yticks([0.9, 1.0])
        ax1.set_ylim(bottom=0.6125, top=1.04)
        ax1.set_yticks([0.7, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        # ax1.legend(loc='lower left', markerscale=1.5)

        plt.minorticks_off()
        plt.tight_layout()
        plt.savefig(path_save + '/plot_IDPT_forward-self-similarity_by_z_sized_same-yscale_no-xlabel.svg')
        plt.show()
        plt.close()

        # ---

        # plot P2P-similarity
        fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.85 * 0.93, size_y_inches * 0.6 * 0.93))

        ax1.fill_between(dfsg1['z'] - z_f, Sptp_lower, Sptp_upper,
                         color=lighten_color(scigreen, 1), ec='none', alpha=0.25)

        ax1.plot(dfsg1['z'] - z_f, dfsg1.cm, '-o', ms=1, color=lighten_color(scigreen, 1.25),
                 label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')

        # ax1.set_ylabel(r'$ \bigl \langle S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) \bigr \rangle $')
        ax1.set_ylabel(r'$S_{wf}(z)$')
        ax1.set_ylim(bottom=0.6125, top=1.04)
        ax1.set_yticks([0.7, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        # ax1.legend(loc='lower left', markerscale=1.5)

        plt.minorticks_off()
        plt.tight_layout()
        plt.savefig(path_save + '/plot_SPCT_p2p_by_z_sized_same-yscale_no-xlabel.svg')
        plt.show()
        plt.close()

        # ---

        # plot: (1) number of particles identified per frame
        plot_single_num = False
        if plot_single_num:
            fig, ax = plt.subplots(figsize=(size_x_inches * 0.85, size_y_inches * 0.8))

            # FAUX IDPT number of particles
            ax.plot(dfgz.iloc[:-1]['z_true'] - z_f, np.ones_like(dfgz.iloc[:-1]['z_true'].to_numpy()) * 87, '-',
                    marker='.', ms=2,  # color='black',
                    alpha=1, zorder=3, label='IDPT')

            # - Number of particles per frame
            ax.plot(dfgz.iloc[:-1]['z_true'] - z_f, dfgz.iloc[:-1]['id'] + 1, '-',
                    marker='.', ms=2,  # color='black',
                    alpha=1, zorder=2.5, label='N.O.')

            ax.set_ylabel(r"$N_{p}^{''}$")
            ax.set_ylim(top=95)
            ax.set_yticks([40, 80])
            # ax.tick_params(axis='y', which='minor', right=False)
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_xticks([-50, 0, 50])
            ax.legend(loc='lower center',
                      handlelength=1, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.35,)

            plt.minorticks_off()
            plt.tight_layout()
            plt.savefig(path_save + '/Np-NoOverlap_fauxIDPT_by_z.svg')
            plt.show()
            plt.close()

    # ---

    plot_num_sigma_and_p2p_similarity = False
    if plot_num_sigma_and_p2p_similarity:
        # THREE PLOTS

        # plot: (1) number of particles identified per frame, (2) sigma_xy, (3) p2p similarity per frame
        fig, (ax0, ax01, ax1) = plt.subplots(nrows=3, sharex=True,
                                             figsize=(size_x_inches * 0.91, size_y_inches * 1.25),
                                             gridspec_kw={'height_ratios': [1, 1.5, 1.5]})

        # - Number of particles per frame
        ax0.plot(dfgz.iloc[:-1]['z_true'] - z_f, dfgz.iloc[:-1]['id'],
                 '-', linewidth=1, # marker='o', ms=1,
                 )  # , label='N.O.'
        ax0.legend(loc='upper right', handlelength=0.5, handletextpad=0.4, markerscale=1.5)
        ax0.set_ylabel(r"$N_{p}^{''}$")
        ax0.set_ylim(top=96.5)
        ax0.set_yticks([40, 80])
        ax0.tick_params(axis='y', which='minor', left=False, right=False)

        # ---

        # - Sigma XY
        for gdx in [0, 2, 4, 6, 8]:
            ax01.plot(z_sigma_xys[gdx] - 50, sigma_xys[gdx],
                      '-', linewidth=1,  # marker='.', ms=0.95,
                      label=dict_gdx[gdx])
        ax01.set_ylabel(r'$w_{x}/w_{y}$')
        ax01.legend(loc='upper left', bbox_to_anchor=(1, 1.1),
                    title=r"$(m, n)$",
                    markerscale=2,
                    handlelength=1, handletextpad=0.4, labelspacing=0.125, borderaxespad=0.3,
                    )
        ax01.tick_params(axis='y', which='minor', left=False, right=False)

        # ---

        # - P2P Similarity
        ax1.fill_between(dfsg1['z'] - z_f, Sptp_lower, Sptp_upper, alpha=0.15)
        ax1.plot(dfsg1['z'] - z_f, dfsg1.cm,
                 '-', linewidth=1, # marker='o', ms=1,
                 label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')  # \Bigr \rangle

        ax1.set_ylabel(r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')
        ax1.set_ylim(bottom=0.715, top=1.01)
        ax1.set_yticks([0.8, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        # ax1.legend(loc='lower left', markerscale=1.5)

        plt.minorticks_off()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35)
        plt.savefig(path_save + '/compare_Np_sigma-xy_p2p-similarity_sized_lines.svg')
        plt.show()
        plt.close()

    # ---

    plot_Np_area_SNR = False
    if plot_Np_area_SNR:
        # TWO PLOTS: (1) Area + SNR, (2) Np for NoOverlap and IDPT

        fig, (ax, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 0.85, size_y_inches * 0.75))

        # plot: (1) number of particles identified per frame
        dfpid = dfs[dfs['id'] == 5]
        ax.plot(dfpid.z_true - z_f, dfpid.contour_area, 'o', ms=2, color='black', zorder=3)
        ax.set_ylabel(r'$A_{p}$')
        ax.set_ylim([0, 800])
        ax.set_yticks([0, 800])

        axr = ax.twinx()
        axr.plot(dfpid.z_true - z_f, (dfpid.mean_int - 110) / 4, 'o', ms=2, color=scired, zorder=2.5)
        # ax.scatter(, marker='D', edgecolors=cg, color='white', linewidths=0.5)
        axr.set_ylabel(r'$SNR$', color=scired)
        axr.set_ylim([0, 400])
        axr.set_yticks([0, 400])

        ax.tick_params(axis='y', which='minor', left=False)
        axr.tick_params(axis='y', which='minor', right=False)

        # ---

        # plot: (2) number of particles identified per frame

        # FAUX IDPT number of particles
        ax1.plot(dfgz.iloc[:-1]['z_true'] - z_f, np.ones_like(dfgz.iloc[:-1]['z_true'].to_numpy()) * 87, '-',
                marker='o', ms=2,  # color='black',
                alpha=1, zorder=3, label='IDPT')

        # - Number of particles per frame
        ax1.plot(dfgz.iloc[:-1]['z_true'] - z_f, dfgz.iloc[:-1]['id'] + 1, '-',
                marker='o', ms=2,  # color='black',
                alpha=1, zorder=2.5, label='SPCT')

        ax1.set_ylabel(r"$N_{p}^{''}$")
        ax1.set_ylim(top=95)
        ax1.set_yticks([40, 80])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1),
                  handlelength=1, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.35,)
        ax1.tick_params(axis='y', which='minor', left=False, right=False)

        plt.minorticks_off()
        plt.tight_layout()
        plt.savefig(path_save + '/Area_SNR_and_Np-NoOverlap_fauxIDPT_by_z.svg')
        plt.show()
        plt.close()

print('Analysis completed without errors.')