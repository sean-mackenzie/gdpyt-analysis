# test bin, analyze, and plot functions

from os.path import join

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd

import filter, analyze
from utils import io, plotting, modify

# description of code
"""
Purpose of this code:
    1. 

Process of this code:
    1. Setup
     
"""

# A note on SciencePlots colors
"""
Blue: #0C5DA5
Paler Blue: #0343DF
Azure: #069AF3
Green: #00B945
Red: #FF9500
Orange: #FF2C00
"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup figures
plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)
save_plots = True
show_plots = True

# ----------------------------------------------------------------------------------------------------------------------
# Average STACK-TO-TEST-PARTICLE similarity
#   Note, these c_m values come from the meta-assessment. This is the average of the maximum similarity value found by
#   applying a calibration collection (single particle for SPC; every particle for GDPyT) on itself.

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/'
read_path = join(base_dir, 'test_coords/similarity/stack-to-test-averages')
fig_path = join(base_dir, 'figs')
save_id = 'compare-gdpyt-spc-average-similarity-on-z-micrometer-v2'
filetype = '.xlsx'

savefig_filtype = '.png'
save_id_zcal = 'meta-assessment/meta-assessment_z-calibration' + savefig_filtype
save_id_meta = 'meta-assessment/meta-assessment_z-uncertainty' + savefig_filtype
save_id_ptop_similarity = 'similarity/image-averages/syn_exp_datasets_particle-to-particle_average_similarity' + savefig_filtype
save_id_self_similarity = 'similarity/self-similarity/calibration_self-similarity' + savefig_filtype


# ----------------------------------------------------------------------------------------------------------------------
# process data for average stack-to-test-particle similarity AND calibration plot (z_true vs. z_

gdpyt_path = join(read_path, 'test_id1_coords_meta_assessment' + filetype)
spc_path = join(read_path, 'test_id11_coords_meta_assessment' + filetype)

# read excels
dfgdpyt = pd.read_excel(gdpyt_path, dtype=float)
dfspc = pd.read_excel(spc_path, dtype=float)

# filter SPC for c_m > 0.9
dffspc = dfspc[dfspc['cm'] > 0.9]

# format plots
scatter_size = 2
scalex = 1
scaley = 1
colors = ['#00B945', '#0C5DA5']
lbl_cal = ['GDPT', 'GDPyT']


fig, ax = plt.subplots()

# plot
for i, df in enumerate([dffspc, dfgdpyt]):

    ax.scatter(df.z_true, df.z, s=scatter_size//2, color=colors[i], label=lbl_cal[i])

ax.set_xlabel(r'$z_{true}$')
ax.set_ylabel(r'$z$')
ax.legend()

plt.tight_layout()
if save_plots:
    plt.savefig(join(fig_path, save_id_zcal))
if show_plots:
    plt.show()


# get average similarity per true_z
dfgdpyt_cm = dfgdpyt.groupby('z_true').mean().cm
dfspc_cm = dfspc.groupby('z_true').mean().cm


# ----------------------------------------------------------------------------------------------------------------------
# process data for binned local z-uncertainty

test_sort_strings = ['test_id', '_coords_']
column_to_bin_and_assess = 'z_true'
bins = 21
round_z_to_decimal = 5
h = 105
z_range = [-0.001, 105.001]
min_cm = 0.5


# -----------------------------------------------------------------------------
# process data for meta-assessment (FULL COLLECTION) binned local uncertainties
dficts = io.read_files('df', read_path, test_sort_strings, filetype, startswith=test_sort_strings[0])

# filter to get cm > 0.9 for SPC
dficts = filter.dficts_filter(dficts, ['error'], [h/10], operations=['lessthan'], copy=True)
dficts = filter.dficts_filter(dficts, ['cm'], [0.75], operations=['greaterthan'], copy=True, only_keys=[11.0])

# bin by number and calculate local rmse_z
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range, round_z_to_decimal)

# split into dataframes to enable easier plotting customization
dfbgdpyt = dfbicts[1.0]
dfbspc = dfbicts[11.0]

# -----------------------------------------------------------------------------
# process data for meta-assessment (SINGLE PARTICLE) binned local uncertainties
dficts = io.read_files('df', read_path, test_sort_strings, filetype, startswith=test_sort_strings[0])

# filter to get particle ID == 0
dficts = filter.dficts_filter(dficts, ['id'], [0], operations=['equalto'], copy=True)

# bin by number and calculate local rmse_z
dfbicts = analyze.calculate_bin_local_rmse_z(dficts, column_to_bin_and_assess, bins, min_cm, z_range, round_z_to_decimal)

# split into dataframes to enable easier plotting customization
dfbgdpyt_id = dfbicts[1.0]
dfbspc_id = dfbicts[11.0]


# ----------------------------------------------------------------------------------------------------------------------
# plot figure

# setup

# general
colors = ['#0C5DA5', '#00B945', 'cornflowerblue', 'palegreen']
scatter_size = 2
scalex = 1
scaley = 1

# similarity
lbl_sim = ['GDPyT', 'GDPT']
ylim_sim = [0.5, 1.01]
ylabel_sim = r'$\langle S \left(p^{cal}_{i}, p^{test}_{i} \right) \rangle$'

# uncertainty
lbl_unc = [r'$p_{N}$', r'$p_{N}$', r'$p_{ID=0}$', r'$p_{ID=0}$']
ylim_unc = [-0.005, 15]
ylabel_unc = r'$\sigma_{z}\left(z\right)$'
xlabel_unc = r'$z/h$'


# plot
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * scalex, size_y_inches * scaley),
                               gridspec_kw={'height_ratios': [1, 1.5]})

for i, (dfsim, dfb, dfbid) in enumerate([[dfgdpyt_cm, dfbgdpyt, dfbgdpyt_id], [dfspc_cm, dfbspc, dfbspc_id]]):

    # top subplot: average similarity
    ax1.plot(dfsim.index, dfsim)
    ax1.scatter(dfsim.index, dfsim, s=scatter_size//2, label=lbl_sim[i])

    # bottom subplot: binned local uncertainty for collection AND particle ID == 0
    ax2.plot(dfb.index, dfb.rmse_z)
    ax2.scatter(dfb.index, dfb.rmse_z, s=scatter_size, label=lbl_unc[i])

    ax2.plot(dfbid.index, dfbid.rmse_z, color=colors[i+2])
    ax2.scatter(dfbid.index, dfbid.rmse_z, s=scatter_size, color=colors[i+2], label=lbl_unc[i+2])


ax1.set_ylim(ylim_sim)
ax1.set_ylabel(ylabel_sim)
ax1.legend(loc='upper left', bbox_to_anchor=(0.95, 1))

ax2.set_xlabel(xlabel_unc)
ax2.set_ylim(ylim_unc)
ax2.set_ylabel(ylabel_unc)
ax2.legend(loc='upper left', bbox_to_anchor=(0.95, 1))

plt.tight_layout()
if save_plots:
    plt.savefig(join(fig_path, save_id_meta))
if show_plots:
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Average PER-IMAGE particle-to-particle similarity - non-overlapping grid, random density 1e-4, z-micrometer-v2

# setup file paths
read_path = join(base_dir, 'test_coords/similarity/image-averages')
grid_no_overlap_path = join(read_path, 'image-average-similarity-grid-no-overlap' + filetype)
random_density_one_path = join(read_path, 'image-average-similarity-random-density-1e-4' + filetype)
random_density_two_path = join(read_path, 'image-average-similarity-random-density-2.5e-4' + filetype)
z_micrometer_path = join(read_path, 'image-average-similarity-z-micrometer-v2' + filetype)

# read excels
dfgrid = pd.read_excel(grid_no_overlap_path, names=['z', 'sim'], dtype=float)
dfrandone = pd.read_excel(random_density_one_path, names=['z', 'sim'], dtype=float)
dfrandtwo = pd.read_excel(random_density_two_path, names=['z', 'sim'], dtype=float)
dfmicro = pd.read_excel(z_micrometer_path, names=['z', 'sim'], dtype=float)

# adjust data to fit [0, 1] normalized measurement depth
dfgrid['z'] = (dfgrid['z'] + 40) / 80
dfrandone['z'] = (dfrandone['z'] + 40) / 80
dfrandtwo['z'] = (dfrandtwo['z'] + 40) / 80
dfmicro['z'] = (dfmicro['z'] - 1) / dfmicro['z'].max()

# collect data
dfs = [dfgrid, dfrandone, dfrandtwo, dfmicro]

# formatting
lbls = ['Synthetic: non-overlapping (Section 3.1)',
        r'Synthetic: random-distribution, $N_{s}$=1E-4 (Section 3.3)',
        r'Synthetic: random-distribution, $N_{s}$=2.5E-4 (Section 3.3)',
        r'Experimental: random-distribution, $N_{s}$=3E-4 (Section 4.1)']
scatter_size = 2
xlbl = r'$z/h$'
yblf = r'$\frac{\sum_{i, j}^n S \left(p_{i}, p_{j} \right)}{n^{2}}$'
yblm = r'$\langle S \left(z_{i}, z_{i+1} \right),\: S \left(z_{i}, z_{i-1} \right) \rangle$'
ylim = [0.0, 1.01]

# plot
fig, ax = plt.subplots()
for i, df in enumerate(dfs):
    ax.plot(df.z, df.sim)
    ax.scatter(df.z, df.sim, s=scatter_size, label=lbls[i])
ax.set_xlabel(xlbl)
ax.set_ylabel(yblf)
ax.set_ylim(ylim)
ax.legend(fontsize=6.5)
plt.tight_layout()
if save_plots:
    plt.savefig(join(fig_path, save_id_ptop_similarity + '.png'))
if show_plots:
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Calibration stacks self-similarity - z-micrometer-v2

# setup file paths
read_path = join(base_dir, 'test_coords/similarity/calibration-stack-self-similarity')
static_forward_path = join(read_path, 'static_avg3imgs_stack_forward_self_similarity' + filetype)
static_middle_path = join(read_path, 'static_avg3imgs_stack_middle_self_similarity' + filetype)
spc_forward_path = join(read_path, 'spc_avg3imgs_stackid0_forward_self_similarity' + filetype)
spc_middle_path = join(read_path, 'spc_avg3imgs_stackid0_middle_self_similarity' + filetype)

# read excels
sf = pd.read_excel(static_forward_path, dtype=float)
sm = pd.read_excel(static_middle_path, dtype=float)
spcf = pd.read_excel(spc_forward_path, dtype=float)
spcm = pd.read_excel(spc_middle_path, dtype=float)

# groupby z-coordinate for static
sf = sf.groupby('z').mean()
sm = sm.groupby('z').mean()

# collect data
fsims = [sf.cm, spcf.cm]
zf = [sf.index, spcf.index]
msims = [sm.cm, spcm.cm]
zm = [sm.index, spcm.index]

# formatting
lbls = ['GDPyT', 'GDPT']
scatter_size = 2
xlbl = r'$z (\mu m)$'
yblf = r'$S \left(z_{i}, z_{i+1} \right)$'
yblm = r'$\langle S \left(z_{i}, z_{i+1} \right),\: S \left(z_{i}, z_{i-1} \right) \rangle$'
ylim = [0.9, 1.01]

# plot
fig, ax = plt.subplots()
for i in range(len(fsims)):
    ax.plot(zf[i], fsims[i])
    ax.scatter(zf[i], fsims[i], s=scatter_size, label=lbls[i])
ax.set_xlabel(xlbl)
ax.set_ylabel(yblf)
ax.set_ylim(ylim)
ax.legend()
plt.tight_layout()
if save_plots:
    plt.savefig(join(base_dir, 'forward_similarity_plot.png'))
if show_plots:
    plt.show()

fig, ax = plt.subplots()
for i in range(len(msims)):
    ax.plot(zm[i], msims[i])
    ax.scatter(zm[i], msims[i], s=scatter_size, label=lbls[i])
ax.set_xlabel(xlbl)
ax.set_ylabel(yblm)
ax.set_ylim(ylim)
ax.legend()
plt.tight_layout()
if save_plots:
    plt.savefig(join(fig_path, save_id_self_similarity))
if show_plots:
    plt.show()



j=1