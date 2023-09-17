# test bin, analyze, and plot functions
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

import filter
import analyze
from correction import correct
from utils import fit, functions, bin, io, plotting, modify, plot_collections
from utils.plotting import lighten_color

# A note on SciencePlots colors
sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# file paths
# base_dir = "/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/results/spct/test-testset-['dynamic_neg_first', 11, 1]/test_coords_tc_neg_first', 11, 1]_cc_neg_first', 11, 1]_2022-10-13 16:37:54.029512.xlsx"
base_dir = "/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/results/spct/calibration-testset-['dynamic_neg_first', 11, 1]/calib_spct_stats_02.06.22_membrane_characterization_calib_['dynamic_neg_first', 11, 1].xlsx"
df = pd.read_excel(base_dir)

# plot setup
# x = 'frame'
x = 'z_corr'
y = 'id'

# processing
# df = df[df[x] > 39]
df['z_corr'] = df['z'] - 140
# dfdz_ids = df[df['z_corr'].abs() > 20].id.unique()
dfdz_ids = df[df['frame'] == 67].id.unique()
df = df[df['id'].isin(dfdz_ids)]
dfc = df.groupby(x).count().reset_index()
dfg = df.groupby(x).mean().reset_index()

# plot
"""fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.5),
                                    gridspec_kw={'height_ratios': [1, 1, 3]})

ax1.plot(dfc[x], dfc[y], '-o', ms=1, color='gray', alpha=0.5)
ax1.plot(dfc[x], dfc[y], 'o', ms=1.5, color='k')
ax1.set_ylabel(r"$N_{p}^{'}$")

ax2.plot(dfg[x], dfg['cm'], '-o', ms=1, color='gray', alpha=0.5)
ax2.plot(dfg[x], dfg['cm'], 'o', ms=1.5, color='k')
ax2.set_ylabel(r"$C_{m}$")

ax3.scatter(df[x], df['z_corr'], c=df['id'], s=1)
ax3.set_xlabel('Frame')
ax3.set_ylabel(r"$z \: (\mu m)$")"""

fig, ax = plt.subplots()
ax.plot(dfc[x], dfc[y], '-o', ms=1, color='gray', alpha=0.5)
ax.plot(dfc[x], dfc[y], 'o', ms=1.5, color='k')
ax.set_ylabel(r"$N_{p}^{'}$")
ax.set_xlabel(r"$z \: (\mu m)$")
ax.set_xticks([-100, 0, 100])

plt.tight_layout()
savepath = "/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/results/spct/test-testset-['dynamic_neg_first', 11, 1]/figs"
savepath = savepath + '/SPCT-number-calibration-collection.png'  # Cm-zcorr_for_zcorr-abs-greater-20.png'
plt.savefig(savepath)
plt.show()