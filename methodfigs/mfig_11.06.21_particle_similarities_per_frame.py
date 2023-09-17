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
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_particle_similarities_per_frame'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')
filetype = '.xlsx'

# experimental
microns_per_pixels = 1.6
image_pixels = 512
image_padding = 5
image_range = image_pixels + image_padding * 2

# ---

quick_plot = False
if quick_plot:

    read_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/results/calibration-spct_dzc-1'
    fn_col_sim = 'average_similarity_11.06.21_z-micrometer-v2_1umSteps__1.xlsx'
    fn_for_sim = 'calib_stacks_forward_self-similarity_11.06.21_z-micrometer-v2_1umSteps__1_{}.xlsx'

    dfg = pd.read_excel(join(read_dir, fn_col_sim))
    dfg_cols = ('z', 'cm')
    zf = 50

    # forward similarity
    dzcs = np.arange(1, 11)

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 1.1))

    for dzc in dzcs:
        dff = pd.read_excel(join(read_dir, fn_for_sim.format(dzc)))
        dffg = dff.groupby('z').mean().reset_index()
        ax.plot(dffg['z'] - zf, dffg['cm'], '-o', ms=1, label='({}, {})'.format(dzc, np.round(dffg['cm'].mean(), 3)))

    ax.plot(dfg['z'] - zf, dfg['cm'], '-d', ms=3, color='k',
            label=r'$S_{wf}$' + '({})'.format(np.round(dfg['cm'].mean(), 3)))

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    raise ValueError()


# ---

evaluate_spct_stats = False  # False True
if evaluate_spct_stats:

    # spct stats
    save_dir = join(base_dir, 'figs/eval_spct_stats')
    fn = 'data/calib_spct_stats_11.06.21_z-micrometer-v2_1umSteps__1.xlsx'
    fp = join(base_dir, fn)

    # pertinent information about the dataset
    """
    IMPROPER TEMPLATE SIZING:
    
    2023_dataset: 'Bad' particle IDs: [1, 29, 59]
        * Note: pid 54 could be removed after z > +40 (or +90 on a 0-105 scale)
    Archive_2022 dataset: 'Bad' particle ID's: 1, 29, 54, 59, 86

    * Note: clearly the 2023 and 2022 datasets  are more related than I thought b/c their particle IDs are identical. 
    
    zf = 50
    num_pids_in_focus = 85
    exclude_ids = [1, 29, 59]
    # exclude_ids = [1, 2, 18, 29, 32, 42, 59, 63, 76]  # MORE EXCLUSIVE LOW
    # exclude_ids = [1, 2, 18, 29, 32, 42, 59, 63, 76, 66, 37, 57, 64]  # MORE EXCLUSIVE MID
    # exclude_ids = [1, 2, 18, 29, 32, 42, 59, 63, 76, 66, 37, 57, 64, 17, 75, 78, 74, 23]  # MORE EXCLUSIVE HIGH
    
    # ----
    071723:
    
    # exclude_ids = [6, 17, 58]  # IN BASELINE ID'S
    # exclude_ids = [84, 85, 86, 90, 91, 93, 95, 96, 97, 98, 99, 100, 101, 102]  # OUTSIDE OF BASELINE ID'S
    exclude_ids = [6, 17, 58, 84, 85, 86, 90, 91, 93, 95, 96, 97, 98, 99, 100, 101, 102]
    
    # ----
    071923: --> the most recent data
    """
    # ---

    zf = 50
    # exclude_ids = [6, 8, 17]  # IN BASELINE ID'S --> could include 58 which appears like a super large particle
    # exclude_ids = [84, 85, 86, 90, 91, 93, 95, 96, 97, 98, 99, 100, 101, 102]  # OUTSIDE OF BASELINE ID'S
    exclude_ids = [6, 8, 17, 84, 85, 86, 90, 91, 93, 95, 96, 97, 98, 99, 100, 101, 102]

    # ----

    clean_or_eval_self_sim = False  # False True
    if clean_or_eval_self_sim:

        cols = ['frame', 'id', 'z_corr']
        px = 'z_corr'
        params = ['contour_area', 'contour_diameter', 'solidity', 'thinness_ratio']

        # ----

        # read dataframe and get columns
        df = pd.read_excel(fp)
        cols.extend(params)
        df = df[cols].sort_values('id')

        # filter 1. pid < 87
        # df = df[df['id'] <= num_pids_in_focus].sort_values('id')

        # filter 2. pids not in exclude_ids
        df = df[~df['id'].isin(exclude_ids)]

        # get pids
        pids = df.id.unique()

        # ---

        # iterate
        dzcs = np.arange(10, 0, -1)
        max_dcmdzs = np.ones_like(dzcs) * -1  # [-0.03, -0.0375]  #

        for dzc, max_dcmdz in zip(dzcs, max_dcmdzs):

            # read self similarity
            fpi = 'calib_stacks_forward_self-similarity_11.06.21_z-micrometer-v2_1umSteps__1_{}.xlsx'.format(dzc)
            dfi = pd.read_excel(join(base_dir, 'data', fpi))

            # filter 1. pid < 87
            # dfi = dfi[dfi['id'] <= num_pids_in_focus]

            # filter 2. pids not in exclude_ids
            dfi = dfi[~dfi['id'].isin(exclude_ids)]

            # drop any rows with a Cm drop less than -0.3
            dfii = []
            for pid in pids:
                dfid = dfi[dfi['id'] == pid].reset_index()

                dfidz = dfid.diff()
                dfidz['cm'] = dfidz['cm'].fillna(0)

                #if dzc == 1:
                #    print("pid {}: min cm = {}".format(pid, np.round(dfidz[dfidz['cm'] > 0.01]['cm'].min(), 3)))

                dfid_include = dfid[dfidz['cm'] > max_dcmdz]
                dfii.append(dfid_include)

            dfi = pd.concat(dfii)
            del dfii, dfid_include

            fpi_cleaned = 'spct-cal_forward-self-sim_dzc={}_cleaned.xlsx'.format(dzc)
            dfi = dfi.drop(columns=['index']).reset_index(drop=True)
            dfi.to_excel(join(base_dir, 'data', fpi_cleaned))

        # ---

        eval_per_pid = True
        if eval_per_pid:
            num_pids = len(pids)
            num_pids_per_fig = 7
            num_figs = int(np.ceil(num_pids / num_pids_per_fig))

            for i in range(num_figs):
                pmin, pmax = int(i * num_pids_per_fig), int((i + 1) * num_pids_per_fig)

                fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.5))

                these_pids = pids[pmin:pmax]
                for pid in these_pids:
                    dfpid = df[df['id'] == pid].sort_values(px)
                    dfid = dfi[dfi['id'] == pid].sort_values('z')

                    for ax, py in zip(axs.ravel(), params):

                        if py == 'thinness_ratio':
                            ax.plot(dfid['z'] - zf, dfid['cm'], '-o', ms=1, label=pid)
                            ax.set_ylabel('cm')
                        else:
                            ax.plot(dfpid[px], dfpid[py], '-o', ms=1, label=pid)
                            ax.set_ylabel(py)

                axs[0, 0].legend(ncol=2, fontsize='x-small')
                axs[1, 1].legend(ncol=2, fontsize='x-small')
                plt.tight_layout()
                plt.savefig(join(save_dir, 'pids-{}-to-{}_with-self-sim.png'.format(np.min(these_pids), np.max(these_pids))))
                plt.close()

        # ---

    # ---

    # clean collection similarities data
    clean_or_eval_col_sim = True  # False True
    if clean_or_eval_col_sim:
        fpc_cleaned = 'spct-cal_widefield-sim_cleaned.xlsx'

        clean_col_sim = True  # False True
        if clean_col_sim:
            fnc_raw = 'collection_similarities_11.06.21_z-micrometer-v2_1umSteps__1.xlsx'

            # read dataframe and get columns
            dfc = pd.read_excel(join(base_dir, 'data', fnc_raw))
            dfc['z'] = dfc['frame'] + 1

            # filter 1. pid < 87
            # dfc = dfc[dfc['image'] <= num_pids_in_focus]
            # dfc = dfc[dfc['template'] <= num_pids_in_focus]

            # filter 2. pids not in exclude_ids
            dfc = dfc[~dfc['image'].isin(exclude_ids)]
            dfc = dfc[~dfc['template'].isin(exclude_ids)]

            # export
            dfc.to_excel(join(base_dir, 'data', fpc_cleaned), index=False)

            dfcg_m = dfc.groupby('z').mean().reset_index()
            dfcg_std = dfc.groupby('z').std().reset_index()

            dfcg_m.to_excel(join(base_dir, 'data', 'average_' + fpc_cleaned), index=False)
            dfcg_std.to_excel(join(base_dir, 'data', 'std_' + fpc_cleaned), index=False)
        else:
            dfcg_m = None
            dfcg_std = None

        eval_col_sim = True  # False True
        if eval_col_sim:

            if dfcg_m is None:
                dfm = pd.read_excel(join(base_dir, 'data', 'average_' + fpc_cleaned))
                dfstd = pd.read_excel(join(base_dir, 'data', 'std_' + fpc_cleaned))
            else:
                dfm = dfcg_m
                dfstd = dfcg_std

            px = 'z'
            py = 'cm'

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

            ax1.errorbar(dfm[px] - zf, dfm[py], yerr=dfstd[py], fmt='-o', ms=1, elinewidth=0.5, capsize=1,
                         label=np.round(dfm[py].mean(), 2))
            ax2.plot(dfm[px] - zf, dfstd[py] / dfm[py], '-o', ms=1)

            ax1.set_ylabel(r'$S_{wf}$')
            ax1.legend(title=r'$\overline{S_{wf}}$')
            ax2.set_ylabel(r'$CoV(S_{wf})$')
            ax2.set_xlabel(r'$z \: (\mu m)$')
            plt.tight_layout()
            plt.show()

    # ---

# ---

# ---

plot_new = False  # False True
if plot_new:

    z_f = 50
    depth_of_focus = 6.5
    # num_pids_in_focus = 85

    # ---

    # process self-similarity

    # read self similarity
    dzc = 1
    fpi = 'spct-cal_forward-self-sim_dzc={}_cleaned.xlsx'.format(dzc)
    dfi = pd.read_excel(join(base_dir, 'data', fpi))

    # read collection similarity
    dfsg1 = pd.read_excel(path_read + '/average_spct-cal_widefield-sim_cleaned.xlsx')
    dfsg1_std = pd.read_excel(path_read + '/std_spct-cal_widefield-sim_cleaned.xlsx')

    # ---

    # setup

    # (A) - Process self-similarity
    # dfi = dfi[dfi['id'] <= 85]
    dfm = dfi.groupby('z').mean().reset_index()
    dfstd = dfi.groupby('z').std().reset_index()
    dfcounts = dfi.groupby('z').count().reset_index()

    # confidence bands
    Ss_lower = dfm['cm'] - dfstd['cm']
    Ss_upper = dfm['cm'] + dfstd['cm']

    # coefficient of variation (CoV)
    Ss_CoV = dfstd['cm'] / dfm['cm']

    # ---

    # (B) - Process collection similarity
    dfsg1 = dfsg1.iloc[:-1]
    dfsg1_std = dfsg1_std[:-1]

    # confidence bands
    Sptp_upper = dfsg1.cm.to_numpy() + dfsg1_std.cm.to_numpy()
    Sptp_lower = dfsg1.cm.to_numpy() - dfsg1_std.cm.to_numpy()
    Sptp_upper = np.where(Sptp_upper < 1, Sptp_upper, 1)

    # coefficient of variation (CoV)
    Sptp_CoV = dfsg1_std.cm.to_numpy() / dfsg1.cm.to_numpy()

    # ---

    # plot two subplots: coefficient of variation (CoV) of (1) P-2-P similarity, (2) Self-Similarity
    plot_coefficient_of_variation = False  # True False
    if plot_coefficient_of_variation:

        plot_CV_Swf_Sii = False
        plot_CV_Np_Swf_Sii = False

        if plot_CV_Swf_Sii:
            fig, ax = plt.subplots(figsize=(size_x_inches * 1, size_y_inches * 1))

            ax.plot(dfsg1['z'] - z_f, Sptp_CoV, '-o', ms=2, color=scigreen, zorder=3.2,
                     label=r'$S_{wf}$')  # label=np.round(dfsg1['cm'].mean(), 3))

            ax.plot(dfm['z'] - z_f, Ss_CoV, '-o', ms=2, color=sciblue, zorder=3.1,
                     label=r'$S_{ii}^{+}$')  # label=np.round(dfm['cm'].mean(), 3))

            # ax.axvline(-depth_of_focus, color='gray', linestyle='--', alpha=0.5, label='2DoF')
            # ax.axvline(depth_of_focus, color='gray', linestyle='--', alpha=0.5)
            ax.axvspan(-depth_of_focus, depth_of_focus, color='gray', alpha=0.125, label='2DoF', ec=None, zorder=3)

            ax.set_ylabel(r'$CV$')  # r'$ \langle S(\cdot, \cdot) \rangle $'
            #ax.set_ylim(bottom=0.68, top=1.02)
            #ax.set_yticks([0.7, 0.8, 0.9, 1.0])
            ax.set_xlabel(r'$z \: (\mu m)$')
            ax.set_xticks([-50, -25, 0, 25, 50])
            ax.legend(loc='upper left', markerscale=1.5)  # , title=r'$\overline{S(\cdot, \cdot)}$'

            plt.tight_layout()
            plt.savefig(path_save + '/compare_CoV_SPCT_Swf_Sii_with-DOF-bar.svg')
            plt.show()
            plt.close()

        # -

        if plot_CV_Np_Swf_Sii:
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.1, size_y_inches * 1.1))

            ax0.plot(dfm['z'] - z_f, dfcounts['cm'], '-o', ms=2, color='k')

            ax1.plot(dfsg1['z'] - z_f, Sptp_CoV, '-o', ms=2, color=scigreen,
                     label=r'$S_{wf}$')  # label=np.round(dfsg1['cm'].mean(), 3))
            # r'$S_{wf}$'
            # np.round(dfsg1['cm'].mean(), 3)
            # r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')  # \Bigr \rangle

            ax1.plot(dfm['z'] - z_f, Ss_CoV, '-o', ms=2, color=sciblue,
                     label=r'$S_{ii}^{+}$')  # label=np.round(dfm['cm'].mean(), 3))
            # r'$S_{ii}^{+}$'
            # np.round(dfm['cm'].mean(), 3))
            # r'$ S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) $')

            ax0.set_ylabel(r'$N_{p}^{''}$')  # r'$ \langle S(\cdot, \cdot) \rangle $'
            ax1.set_ylabel(r'$CV$')  # r'$ \langle S(\cdot, \cdot) \rangle $'
            # ax1.set_ylim(bottom=0.68, top=1.02)
            # ax1.set_yticks([0.7, 0.8, 0.9, 1.0])
            ax1.set_xlabel(r'$z \: (\mu m)$')
            ax1.set_xticks([-50, -25, 0, 25, 50])
            ax1.legend(loc='upper left', markerscale=1.5)  # , title=r'$\overline{S(\cdot, \cdot)}$'

            plt.tight_layout()
            plt.savefig(path_save + '/compare_SPCT_p2p_with_SPCT_forward-self-similarity_CoV_labeled.svg')
            plt.show()
            plt.close()

        # -

    # ---

    # plot two subplots: (1) P-2-P similarity, (2) Self-Similarity
    plot_two_sims = False  # True False
    if plot_two_sims:
        fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.91, size_y_inches * 0.85))  # (0.91, 0.85)

        ax1.fill_between(dfsg1['z'] - z_f, Sptp_lower, Sptp_upper,
                         color=lighten_color(scigreen, 1), ec='none', alpha=0.25)
        ax1.fill_between(dfm['z'] - z_f, Ss_lower, Ss_upper,
                         color=lighten_color(sciblue, 1.25), edgecolor='none', alpha=0.25)

        ax1.plot(dfsg1['z'] - z_f, dfsg1.cm, '-o', ms=1, color=scigreen, label=np.round(dfsg1['cm'].mean(), 3))
        # r'$S_{wf}$'
        # np.round(dfsg1['cm'].mean(), 3)
        # r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')  # \Bigr \rangle

        ax1.plot(dfm['z'] - z_f, dfm['cm'], '-o', ms=1, color=sciblue, label=np.round(dfm['cm'].mean(), 3))
        # r'$S_{ii}^{+}$'
        # np.round(dfm['cm'].mean(), 3))
        # r'$ S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) $')

        ax1.set_ylabel(r'$S$')  # r'$ \langle S(\cdot, \cdot) \rangle $'
        ax1.set_ylim(bottom=0.74, top=1.02)
        ax1.set_yticks([0.8, 0.9, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, -25, 0, 25, 50])
        ax1.legend(loc='lower left', markerscale=1.5)  # , title=r'$\overline{S(\cdot, \cdot)}$'

        plt.tight_layout()
        plt.savefig(path_save + '/compare_SPCT_p2p_with_SPCT_forward-self-similarity_mean-values.svg')
        plt.show()
        plt.close()

    # ---

    # plot individual figures of: (1) P-2-P similarity, (2) Self-Similarity
    plot_single_figs = False  # True False
    if plot_single_figs:
        # plot self-similarity
        fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.85 * 0.93, size_y_inches * 0.6 * 0.93))

        ax1.fill_between(dfm['z'] - z_f, Ss_lower, Ss_upper,
                         color=lighten_color(sciblue, 1.25), edgecolor='none', alpha=0.25)
        ax1.plot(dfm['z'] - z_f, dfm['cm'], '-o', ms=1, color=sciblue,
                 label=r'$S_{ii}^{+}$')
        # r'$ S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) $')
        # ax1.set_ylabel(r'$ \bigl \langle S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) \bigr \rangle $')
        ax1.set_ylabel(r'$S_{ii}^{+}$')
        # ax1.set_ylim(bottom=0.89, top=1.01)
        # ax1.set_yticks([0.9, 1.0])
        ax1.set_ylim(bottom=0.74, top=1.02)
        ax1.set_yticks([0.8, 0.9, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        # ax1.legend(loc='lower left', markerscale=1.5)

        plt.minorticks_off()
        plt.tight_layout()
        plt.savefig(path_save + '/plot_SPCT_forward-self-similarity_by_z.svg')
        plt.show()
        plt.close()

        # ---

        # plot P2P-similarity
        fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.85 * 0.93, size_y_inches * 0.6 * 0.93))

        ax1.fill_between(dfsg1['z'] - z_f, Sptp_lower, Sptp_upper,
                         color=lighten_color(scigreen, 1), ec='none', alpha=0.25)

        ax1.plot(dfsg1['z'] - z_f, dfsg1.cm, '-o', ms=1, color=lighten_color(scigreen, 1.25),
                 label=r'$S_{wf}$')
        # r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')
        # ax1.set_ylabel(r'$ \bigl \langle S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) \bigr \rangle $')
        ax1.set_ylabel(r'$S_{wf}$')
        ax1.set_ylim(bottom=0.74, top=1.02)
        ax1.set_yticks([0.8, 0.9, 1.0])
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_xticks([-50, 0, 50])
        # ax1.legend(loc='lower left', markerscale=1.5)

        plt.minorticks_off()
        plt.tight_layout()
        plt.savefig(path_save + '/plot_SPCT_p2p_by_z.svg')
        plt.show()
        plt.close()

        # ---

    # ---

    # plot: (1) P-2-P similarity, (2) Self-Similarity as a function of dzc (calibration step size)

    plot_dzcs = False  # True False
    if plot_dzcs:

        # forward similarity
        dzcs = np.arange(1, 11)

        # plot
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 1))
        clrs = iter(plt.cm.Spectral(np.linspace(0.01, 0.99, len(dzcs))))

        for dzc in dzcs:
            # read dzc-dependent self-similarity
            fpi = 'spct-cal_forward-self-sim_dzc={}_cleaned.xlsx'.format(dzc)
            dff = pd.read_excel(join(base_dir, 'data', fpi))
            dffg = dff.groupby('z').mean().reset_index()

            ax.plot(dffg['z'] - z_f, dffg['cm'], '-o', ms=2, color=next(clrs),
                    label='{}: {}'.format(dzc, np.round(dffg['cm'].mean(), 3)))

        # ax.plot(dfsg1['z'] - z_f, dfsg1['cm'], '-o', ms=2, color='k', label=r'$S_{wf}$' + '({})'.format(np.round(dfsg1['cm'].mean(), 3)))
        ax.set_ylabel(r'$S_{ii}^{+}$')  # r'$S_{ii}^{+}$'
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$n: \: \overline{S_{ii}^{+}}$')

        plt.tight_layout()
        plt.savefig(path_save + '/plot_Sii-by-dzc_by_z_spectral.svg')
        plt.show()
        plt.close()

        # ---

    # ---

    # ---

    # THE FOLLOWING PLOTS REQUIRE SPCT_STATS

    include_spct_stats = False
    if include_spct_stats:

        # read SPCT stats (note, SPCT stats are from a different analysis so pid's are likely different)
        # fn = 'calib_spct_stats_11.06.21_z-micrometer-v2_1umSteps.xlsx' --> NOTE: archive_2022 filename
        # dfs = pd.read_excel(join(path_read, fn))

        # ---

        # (C.a) - Process number of particles per frame
        # dfs = dfs[dfs['id'] < num_pids_in_focus]
        # dfgzid = dfs.groupby(['z_true', 'id']).mean().reset_index()
        # dfgz = dfgzid.groupby('z_true').count().reset_index()

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

        # ---

    # ---

# ---

print('Analysis completed without errors.')