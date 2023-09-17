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
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/particle-similarities-in-image'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')
filetype = '.xlsx'

plot_new = False
if plot_new:

    # ---

    # process self-similarity

    # read self similarity
    fpi = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-07.29.22-idpt-tmg/tests/tm16_cm19/tracking/calibration-idpt_micrometer_5um-3/calib_stacks_forward_self-similarity_11.06.21_z-micrometer-v2_5umMS__3.xlsx'
    dfi = pd.read_excel(fpi)

    # setup
    zf = 50.0
    x = 'z_corr'
    y = 'cm'

    # center z on focal plane
    dfi['z_corr'] = dfi['z'] - zf

    # groupby stats
    dfm = dfi.groupby('z').mean()
    dfstd = dfi.groupby('z').std()

    z_shift = -1
    dfm_shift = dfi[dfi['z'] > z_shift].groupby('z').mean()
    dfstd_shift = dfi[dfi['z'] > z_shift].groupby('z').std()

    Ss_lower = dfm_shift.iloc[:][y] - dfstd_shift.iloc[:][y]
    Ss_upper = dfm_shift.iloc[:][y] + dfstd_shift.iloc[:][y]

    # ---
    """fp_exp_silp_new = 'raw/collection_similarities_11.06.21_z-micrometer-v2_1umSteps_spct'

    dfs1 = pd.read_excel(join(path_read, fp_exp_silp_new) + filetype)

    # filters
    dfs1 = dfs1[dfs1['image'] < 85]
    dfs1 = dfs1[dfs1['template'] < 85]

    print(dfs1.cm.mean())
    print(dfs1.cm.std())
    raise ValueError()"""

    # process collection similarities
    if not os.path.exists(path_read + '/col_avg_11.06.21_filtered_mean.xlsx'):
        fp_exp_silp_new = 'collection_similarities_11.06.21_z-micrometer-v2_1umSteps_spct'
        fp_exp_silp_old = 'collection_similarities_11.06.21_z-micrometer-v2_5umMS__p2p-sim-ctemp0'

        dfs1 = pd.read_excel(join(path_read, fp_exp_silp_new) + filetype)

        # filters
        dfs1 = dfs1[dfs1['image'] < 85]
        dfs1 = dfs1[dfs1['template'] < 85]

        dfsg1 = dfs1.groupby('z').mean().reset_index()
        dfsg1_std = dfs1.groupby('z').std().reset_index()

        # dfs2 = pd.read_excel(join(path_read, fp_exp_silp_old) + filetype)
        # dfsg2 = dfs2.groupby('z').mean().reset_index()
        # dfsg2_std = dfs2.groupby('z').std().reset_index()

        dfsg1.to_excel(path_read + '/col_avg_11.06.21_filtered_mean.xlsx')
        dfsg1_std.to_excel(path_read + '/col_avg_11.06.21_filtered_std.xlsx')
    else:
        dfsg1 = pd.read_excel(path_read + '/col_avg_11.06.21_filtered_mean.xlsx')
        dfsg1_std = pd.read_excel(path_read + '/col_avg_11.06.21_filtered_std.xlsx')

        dfsg1 = dfsg1.iloc[:-1]
        dfsg1_std = dfsg1_std[:-1]

        print(dfsg1.cm.mean())
        print(dfsg1_std.cm.mean())
        raise ValueError()

    # limit error bars to S=1
    Sptp_upper = dfsg1.cm.to_numpy() + dfsg1_std.cm.to_numpy()
    Sptp_lower = dfsg1.cm.to_numpy() - dfsg1_std.cm.to_numpy()

    Sptp_upper = np.where(Sptp_upper < 1, Sptp_upper, 1)
    # Sptp_lower = np.where(Sptp_lower > 0.725, Sptp_lower, 0.725)

    # ---

    # plot two subplots: (1) P-2-P similarity, (2) Self-Similarity
    fig, ax1 = plt.subplots(figsize=(size_x_inches * 0.91, size_y_inches * 0.85))

    ax1.fill_between(dfsg1['z'] - 50, Sptp_lower, Sptp_upper, color=sciblue, alpha=0.125)
    ax1.fill_between(dfm_shift.iloc[:][x], Ss_lower, Ss_upper, color=scired, alpha=0.125)

    ax1.plot(dfsg1['z'] - 50, dfsg1.cm, '-o', ms=1,
             label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{j}(z) \bigr) $')  # \Bigr \rangle
    ax1.plot(dfm.iloc[:][x], dfm.iloc[:][y], '-o', ms=1, color=scired,
             label=r'$ S \bigl( I^{c}_{i}(z), I^{c}_{i}(z+\Delta_{c}z) \bigr) $')

    ax1.set_ylabel(r'$ \langle S(\cdot, \cdot) \rangle $')
    ax1.set_ylim(bottom=0.715, top=1.01)
    ax1.set_yticks([0.8, 0.9, 1.0])

    # ax2.set_ylabel(r'$\overline{S} \left( z_{i}, z_{i+1} \right) $')
    # ax2.set_ylim(bottom=0.905, top=1.0075)
    # ax2.set_yticks([0.95, 1.00])
    ax1.set_xlabel(r'$z \: (\mu m)$')
    ax1.set_xticks([-50, -25, 0, 25, 50])
    ax1.legend(loc='lower left', markerscale=1.5)

    plt.tight_layout()
    plt.savefig(path_save + '/compare_SPCT_p2p_with_IDPT_forward-self-similarity_sized.svg')
    plt.show()
    plt.close()

    # plot two subplots: (1) P-2-P similarity, (2) Self-Similarity
    plot_two_subplots = False
    if plot_two_subplots:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(dfsg1['z'] - 50, dfsg1.cm, '-o', ms=1)
        ax1.fill_between(dfsg1['z'] - 50, Sptp_lower, Sptp_upper, alpha=0.2)
        ax2.errorbar(dfm_shift.iloc[:-1][x], dfm_shift.iloc[:-1][y], yerr=dfstd_shift.iloc[:-1][y], fmt='o', ms=1, color=sciblue,
                     capsize=2, ecolor='silver', elinewidth=1, errorevery=5,
                     label='IDPT')
        ax2.plot(dfm.iloc[:-1][x], dfm.iloc[:-1][y], 'o', ms=1, color=sciblue)

        ax1.set_ylabel(r'$ \Bigl \langle S \Bigl( I^{c}_{i}(z), I^{c}_{j}(z) \Bigr) \Bigr \rangle $')
        ax1.set_ylim(bottom=0.675, top=1.01)
        ax1.set_yticks([0.8, 1.0])

        ax2.set_ylabel(r'$\overline{S} \left( z_{i}, z_{i+1} \right) $')
        ax2.set_ylim(bottom=0.905, top=1.0075)
        ax2.set_yticks([0.95, 1.00])
        ax2.set_xlabel(r'$z \: (\mu m)$')
        ax2.set_xticks([-50, -25, 0, 25, 50])
        # ax2.legend(loc='lower center')

        plt.tight_layout()
        # plt.savefig(path_results + '/compare_SPCT_IDPT_forward-self-similarity.svg')
        plt.show()
        plt.close()


    # --- simple
    param_z = 'z'

    # plot fill between(cm) by z
    plot_fill_between_by_z = False
    if plot_fill_between_by_z:
        fig, ax = plt.subplots()
        ax.plot(dfsg1[param_z] - 50, dfsg1.cm, '-o', ms=1)
        ax.fill_between(dfsg1[param_z] - 50, dfsg1.cm - dfsg1_std.cm, dfsg1.cm + dfsg1_std.cm, alpha=0.2)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$ \Bigl \langle S \Bigl( I^{c}_{i}(z), I^{c}_{j}(z) \Bigr) \Bigr \rangle $')
        plt.tight_layout()
        # plt.savefig(path_save + '/average-particle-image-similarity_only-synthetic_{}_errorbars.png'.format(param_z))
        plt.show()
        plt.close()

    # ---

    # plot mean(cm) by z
    plot_mean_by_z = False
    if plot_mean_by_z:
        fig, ax = plt.subplots()
        ax.plot(dfsg1[param_z] - 50, dfsg1.cm, '-o', ms=1, label='Synthetic')
        # ax.plot(dfeg1[param_z] - 50, dfeg1.cm, '-o', ms=1, label='Experimental')
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
        ax.legend()
        plt.tight_layout()
        # plt.savefig(path_save + '/average-particle-image-similarity_only-synthetic_{}_plot.png'.format(param_z))
        plt.show()
        plt.close()

    # ---

    # plot error bars(cm) by z
    plot_error_bars_by_z = False
    if plot_error_bars_by_z:
        fig, ax = plt.subplots()
        ax.errorbar(dfsg1[param_z] - 50, dfsg1.cm, yerr=dfsg1_std.cm, fmt='o', ms=1, capsize=2, elinewidth=1,
                    label='Synthetic')
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
        ax.legend()
        plt.tight_layout()
        # plt.savefig(path_save + '/average-particle-image-similarity_only-synthetic_{}_errorbars.png'.format(param_z))
        plt.show()
        plt.close()

# ---

compare_real_to_synthetic = False
if compare_real_to_synthetic:
    fp_synthetic = 'collection_similarities_grid-dz_calib_nll2_spct_p2p-sim-ctemp0'
    fp_exp_silp = 'collection_similarities_11.06.21_z-micrometer-v2_5umMS__p2p-sim-ctemp0'

    dfs1 = pd.read_excel(join(path_read, fp_synthetic) + filetype)
    dfe1 = pd.read_excel(join(path_read, fp_exp_silp) + filetype)

    dfsg1 = dfs1.groupby('z').mean().reset_index()
    dfsg1_std = dfs1.groupby('z').std().reset_index()
    dfeg1 = dfe1.groupby('z').mean().reset_index()
    dfeg1_std = dfe1.groupby('z').std().reset_index()

    # --- simple
    param_z = 'z'

    fig, ax = plt.subplots()
    ax.plot(dfsg1[param_z] + 4, dfsg1.cm, '-o', ms=1, label='Synthetic')
    # ax.plot(dfeg1[param_z] - 50, dfeg1.cm, '-o', ms=1, label='Experimental')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_save + '/average-particle-image-similarity_only-synthetic_{}_plot.png'.format(param_z))
    plt.show()

    fig, ax = plt.subplots()
    ax.errorbar(dfsg1[param_z] + 4, dfsg1.cm, yerr=dfsg1_std.cm, fmt='o', ms=1, capsize=2, elinewidth=1,
                label='Synthetic')
    """ax.errorbar(dfeg1[param_z] - 50, dfeg1.cm, yerr=dfeg1_std.cm, fmt='o', ms=1, capsize=2, elinewidth=1,
                label='Experimental', alpha=0.25)"""
    ax.errorbar(dfsg1[param_z] + 4, dfsg1.cm, yerr=dfsg1_std.cm, fmt='o', ms=1, capsize=2, elinewidth=1,
                color=sciblue)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_save + '/average-particle-image-similarity_only-synthetic_{}_errorbars.png'.format(param_z))
    plt.show()


plot_old = False
if plot_old:

    path_read = path_read + '/additional'

    fp_synthetic = 'average_similarity_1id_synthetic_grid-no-overlap-nl1'
    fp_exp_glass = 'average_similarity_4id_SPCT_20X_1Xmag_0.87umNR'
    fp_exp_silp = 'average_similarity_5id_SPCT_11.06.21_z-micrometer-v2'
    fp_exp_bpe = 'average_similarity_6id_10.07.21-BPE_Pressure_Deflection_avg-sim'
    fp_exp_bpe_non_u_illum = 'average_similarity_7id_SPCT_11.02.21-BPE_Pressure_Deflection_20X_non-u-illlum'

    dfs1 = pd.read_excel(join(path_read, fp_synthetic) + filetype)
    dfe1 = pd.read_excel(join(path_read, fp_exp_glass) + filetype)
    dfe2 = pd.read_excel(join(path_read, fp_exp_silp) + filetype)
    dfe3 = pd.read_excel(join(path_read, fp_exp_bpe) + filetype)
    dfe4 = pd.read_excel(join(path_read, fp_exp_bpe_non_u_illum) + filetype)

    dfps = []
    for df in [dfs1, dfe1, dfe2, dfe3, dfe4]:
        df['z_norm'] = (df['z_corr'] - df['z_corr'].min()) / (df['z_corr'].max() - df['z_corr'].min())
        dfps.append(df)

    dfs1, dfe1, dfe2, dfe3, dfe4 = dfps[0], dfps[1], dfps[2], dfps[3], dfps[4]

    # --- plot

    # plot one by one
    ylim = [0.499, 1.01]


    # --- simple
    param_z = 'z_corr'

    fig, ax = plt.subplots()

    ax.plot(dfs1[param_z], dfs1.sim, '-o', ms=1, label='Synthetic')
    ax.plot(dfe1[param_z], dfe1.sim, '-o', ms=1, label='Glass')
    ax.plot(dfe2[param_z], dfe2.sim, '-o', ms=1, label='Elastomer')
    ax.plot(dfe3[param_z], dfe3.sim, '-o', ms=1, label='Device')
    ax.plot(dfe4[param_z], dfe4.sim, '-o', ms=1, label='Device; N.U.I.')

    ax.set_xlabel(r'$z/h$')
    ax.set_ylabel(r'$\overline {S} (p_{i}, p_{N})$')
    ax.set_ylim(ylim)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path_save + '/average-particle-image-similarity_{}.png'.format(param_z))
    plt.show()



print('Analysis completed without errors.')