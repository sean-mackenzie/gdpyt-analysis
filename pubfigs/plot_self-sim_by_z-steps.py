# plot self-similarity as a function of z-step (SPCT)
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

from utils import fit, functions, bin, io, plotting, modify, plot_collections
from utils.plotting import lighten_color

# Key words for easier searching (SHIFT + SHIFT (everywhere search), like CTRL + F)
# self-similarity, z-step, dz-step, dz, forward self similarity, z step size

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

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# LOAD FILES

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/' \
           'results-08.16.22_spct-z-steps'

path_results = base_dir + '/results'

# ---

# ----------------------------------------------------------------------------------------------------------------------
# PLOT SELF SIMILARTY BY DZ-STEP
plot_self_sim = False

if plot_self_sim:

    # setup
    self_sim = 'middle'
    min_percent_layers = 0.01
    DoF = 6.5

    # plot
    z_steps = [1, 2, 3, 5]
    calib_ids = [41, 42, 42, 42]

    fig, ax = plt.subplots()  # figsize=(size_x_inches / 2, size_y_inches / 2)

    for dz, spct_calib_id in zip(z_steps[:3], calib_ids):
        fp = base_dir + '/similarity/{}/calib_stacks_{}_self-similarity_z-step-{}.xlsx'.format(self_sim, self_sim, dz)
        dfs = pd.read_excel(fp)

        # setup
        zf = 50.0
        x = 'z_corr'
        y = 'cm'

        # processing
        """dfg = dfs.groupby('id').mean().reset_index()
        passing_ids = dfg[dfg['cm'] > 0.975].id.values
        dfs = dfs[dfs['id'].isin(passing_ids)]"""

        # get calibration particle and center z on focal plane
        if spct_calib_id is None:
            dfs = dfs.groupby('z').mean().reset_index()
        else:
            dfs = dfs[dfs['id'] == spct_calib_id]
        dfs['z_corr'] = dfs['z'] / dz - zf
        dfs = dfs[dfs['z_corr'] < 51]

        # plot
        ax.plot(dfs[x], dfs[y], '-o', ms=1, label=dz, zorder=(33 - dz) / 30)


    ax.fill_between([-DoF / 2, DoF / 2], 0.9, 1.1,
                    color='red', ec='none', alpha=0.1, label='2X D.o.F.')

    if self_sim == 'forward':
        ax.set_ylabel(r'$S \left( z_{i}, z_{i+1} \right)$')
    else:
        ax.set_ylabel(r'$\overline{S}_{(i-1, i, i+1)}$')

    ax.set_ylim(bottom=0.92, top=1.0025)
    # ax.set_ylim(bottom=0.965, top=1.00125)
    ax.set_yticks([0.95, 1.00])
    ax.set_xlabel(r'$z \: (\mu m)$')
    #ax.set_xlim([-10, 10])
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.legend(title=r'$\Delta z \: (\mu m)$')

    plt.tight_layout()

    if isinstance(spct_calib_id, (int, float)):
        spct_calib_id = True

    plt.savefig(path_results + '/compare_{}-self-similarity_by_z-step_calib-id-{}_DoF-fill.svg'.format(self_sim, spct_calib_id))
    plt.show()

    # ---

# ----------------------------------------------------------------------------------------------------------------------
# PLOT P2P SIMILARITY

plot_col_sim = False

if plot_col_sim:

    # simple
    plot_collections.plot_similarity_stats_simple(base_dir, min_percent_layers=0.5)

    # full analysis
    mean_min_dx = 25.5
    plot_collections.plot_similarity_analysis(base_dir, method='spct', mean_min_dx=mean_min_dx)


# ---


# ----------------------------------------------------------------------------------------------------------------------
# 03/28/23 - PLOT SELF SIMILARTY BY DZ-STEP
compare_p2p_and_self_sim = True

if compare_p2p_and_self_sim:

    # ------------------------------------------------------------------------------------------------------------------
    # FIRST, get p2p similarity

    path_similarity = join(base_dir, 'similarity')
    """path_results = join(base_dir, 'results', 'similarity-simple')

    if not os.path.exists(path_results):
        os.makedirs(path_results)"""
    min_particles_per_frame = 10

    # read
    dfs, dfsf, dfsm, dfas, dfcs = io.read_similarity(path_similarity)

    if dfcs is not None:
        dfpp = modify.groupby_stats(dfcs, group_by='frame', drop_columns=['image', 'template'])
        dfpp = dfpp[dfpp['z_counts'] > min_particles_per_frame]
        dfpp = dfpp.sort_values('z')

        fig, ax = plt.subplots()
        ax.plot(dfpp.z - 50, dfpp.cm, '-o', ms=1, color='k', label=r'$S_{wf}(z)$')
        """ax.errorbar(dfpp.z - 50, dfpp.cm, yerr=dfpp.cm_std, fmt='d', ms=3, color='black',
                    capsize=2, ecolor='silver', elinewidth=1, errorevery=5)
        ax.set_xlabel(r'$z_{calib.} \: (\mu m)$')
        ax.set_ylabel(r'$\overline{S}(i, N_{I})$')
        plt.tight_layout()
        plt.savefig(path_results + '/calib_per-frame_particle-to-particle-similarity.svg')
        plt.show()"""
    else:
        raise ValueError()

    # ------------------------------------------------------------------------------------------------------------------

    # setup
    self_sim = 'forward'
    min_percent_layers_per_particle = 87.5
    min_percent_layers = 0.01
    DoF = 6.5

    # plot
    z_steps = [1, 2, 3, 5]
    calib_ids = [None, None, None, None]  # [41, 42, 42, 42]

    # fig, ax = plt.subplots()  # figsize=(size_x_inches / 2, size_y_inches / 2)

    for dz, spct_calib_id in zip(z_steps, calib_ids):
        fp = base_dir + '/similarity/{}/calib_stacks_{}_self-similarity_z-step-{}.xlsx'.format(self_sim, self_sim, dz)
        dfs = pd.read_excel(fp)

        # setup
        zf = 50.0
        x = 'z_corr'
        y = 'cm'

        # filtering
        print("{} length: {}".format(dz, len(dfs)))
        dfs = dfs[dfs['layers'] > dfs['layers'].max() * min_percent_layers_per_particle / 100]
        print("{} filtered length: {}".format(dz, len(dfs)))

        # processing
        """dfg = dfs.groupby('id').mean().reset_index()
        passing_ids = dfg[dfg['cm'] > 0.975].id.values
        dfs = dfs[dfs['id'].isin(passing_ids)]"""

        # get calibration particle and center z on focal plane
        if spct_calib_id is None:
            dfs = dfs.groupby('z').mean().reset_index()
        else:
            dfs = dfs[dfs['id'] == spct_calib_id]
        dfs['z_corr'] = dfs['z'] / dz - zf
        dfs = dfs[dfs['z_corr'] < 51]

        # plot
        ax.plot(dfs[x], dfs[y], '-o', ms=1, label=dz, zorder=(33 - dz) / 30)


    # ax.fill_between([-DoF / 2, DoF / 2], 0.9, 1.1, color='red', ec='none', alpha=0.1, label='2X D.o.F.')

    if self_sim == 'forward':
        ax.set_ylabel(r'$S \left( z_{i}, z_{i+1} \right)$')
    else:
        ax.set_ylabel(r'$\overline{S}_{(i-1, i, i+1)}$')

    # ax.set_ylim(bottom=0.92, top=1.0025)
    # ax.set_ylim(bottom=0.965, top=1.00125)
    # ax.set_yticks([0.95, 1.00])
    ax.set_xlabel(r'$z \: (\mu m)$')
    #ax.set_xlim([-10, 10])
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.legend(loc='lower center', ncol=5, title=r'$\Delta z \: (\mu m)$', handlelength=0.75, handletextpad=0.4)

    plt.tight_layout()

    if isinstance(spct_calib_id, (int, float)):
        spct_calib_id = True

    plt.savefig(path_results + '/compare_p2p_with_self-sim_by_z-step_eval-by-SPCT.png')
    # plt.savefig(path_results + '/compare_{}-self-similarity_by_z-step_calib-id-{}_DoF-fill.svg'.format(self_sim, spct_calib_id))
    plt.show()

    # ---

# ----------------------------------------------------------------------------------------------------------------------

# ---

print("Analysis completed without errors.")