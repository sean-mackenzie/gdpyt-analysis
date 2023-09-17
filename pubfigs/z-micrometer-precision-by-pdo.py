# test bin, analyze, and plot functions
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import analyze
from utils import io, bin, plotting, modify, functions
from utils.plotting import lighten_color
import filter


# ------------------------------------------------
# formatting
sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF9500'
sciorange = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# -----------------------

# ----------------------------------------------------------------------------------------------------------------------
# 0. SETUP DATASET DETAILS

# spct filepath
base_path = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/z-micrometer-v2_r-precision-by-pdo'
path_figs = join(base_path, 'figs')
save_fig_filetype = '.svg'

# experimental
microns_per_pixel = 1.6

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# PERCENT DIAMETER OVERLAP

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# 1. READ FILES

fpi = base_path + '/data/idpt/max-pdo-1.99_error-limit-4.5_exclude-dof-None_min-num-50/' \
                  'spct-stats_2d-precision_pdo-id/spct-stats_bin-pdo-id_weighted-average.xlsx'
fps = base_path + '/data/spct/max-pdo-1.99_error-limit-4.5_exclude-dof-None_min-num-50/' \
                  'spct-stats_2d-precision_pdo-id/spct-stats_bin-pdo-id_weighted-average.xlsx'

dfi = pd.read_excel(fpi)
dfs = pd.read_excel(fps)

# ----------------------------------------------------------------------------------------------------------------------
# 2. FILTERING
min_num_per_bin = 150
dfi = dfi[dfi['counts'] > min_num_per_bin]
dfs = dfs[dfs['counts'] > min_num_per_bin]

# ----------------------------------------------------------------------------------------------------------------------
# 2. PLOT R PRECISION BY PDO

save_plots = False
show_plots = True

# setup
x = 'percent_dx_diameter'
yi = 'rm'
ys = 'r'

# plot
fig, ax = plt.subplots()
ax.plot(dfi[x], dfi[yi] * microns_per_pixel, '-o', label='IDPT')
ax.plot(dfs[x], dfs[ys] * microns_per_pixel, '-o', label='SPCT')

ax.set_ylabel(r'$\sigma_{xy} \: (\mu m)$')
ax.set_xlabel(r'$\tilde{\varphi} \: (\%)$')
# ax2.set_ylabel(r'$N_{p}$')
ax.legend()

plt.tight_layout()
if save_plots:
    plt.savefig(path_figs + '/r-precision_by_pdo{}'.format(save_fig_filetype))
if show_plots:
    plt.show()
plt.close()

# ---
y2 = 'counts'

# plot lateral precision and number of particles
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(dfi[x], dfi[y2], '-o', label='IDPT')
ax1.plot(dfs[x], dfs[y2], '-o', label='SPCT')
ax2.plot(dfi[x], dfi[yi] * microns_per_pixel, '-o')
ax2.plot(dfs[x], dfs[ys] * microns_per_pixel, '-o')

ax1.set_ylabel(r'$N_{p}$')
ax1.legend()
ax2.set_ylabel(r'$\sigma_{xy} \: (\mu m)$')
ax2.set_xlabel(r'$\tilde{\varphi} \: (\%)$')

plt.tight_layout()
if save_plots:
    plt.savefig(path_figs + '/r-precision_and_counts_by_pdo{}'.format(save_fig_filetype))
if show_plots:
    plt.show()
plt.close()

# ---

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# MIN DX

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# 1. READ FILES

fpi = base_path + '/data/idpt/max-pdo-1.99_error-limit-4.5_exclude-dof-None_min-num-50/' \
                  'spct-stats_2d-precision_mindx-id/spct-stats_bin-mindx-id_weighted-average.xlsx'
fps = base_path + '/data/spct/max-pdo-1.99_error-limit-4.5_exclude-dof-None_min-num-50/' \
                  'spct-stats_2d-precision_mindx-id/spct-stats_bin-mindx-id_weighted-average.xlsx'

dfi = pd.read_excel(fpi)
dfs = pd.read_excel(fps)

# ----------------------------------------------------------------------------------------------------------------------
# 2. FILTERING
dfi = dfi[dfi['counts'] > min_num_per_bin]
dfs = dfs[dfs['counts'] > min_num_per_bin]

# ----------------------------------------------------------------------------------------------------------------------
# 2. PLOT R PRECISION BY PDO

# setup
x = 'min_dx'

# plot
fig, ax = plt.subplots()
ax.plot(dfi[x] * microns_per_pixel, dfi[yi] * microns_per_pixel, '-o', label='IDPT')
ax.plot(dfs[x] * microns_per_pixel, dfs[ys] * microns_per_pixel, '-o', label='SPCT')

ax.set_ylabel(r'$\sigma_{xy} \: (\mu m)$')
ax.set_xlabel(r'$\delta x_{min} \: (\mu m)$')
ax.legend()

plt.tight_layout()
if save_plots:
    plt.savefig(path_figs + '/r-precision_by_min-dx{}'.format(save_fig_filetype))
if show_plots:
    plt.show()
plt.close()

# ---
y2 = 'counts'

# plot lateral precision and number of particles
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(dfi[x] * microns_per_pixel, dfi[y2], '-o', label='IDPT')
ax1.plot(dfs[x] * microns_per_pixel, dfs[y2], '-o', label='SPCT')
ax2.plot(dfi[x] * microns_per_pixel, dfi[yi] * microns_per_pixel, '-o')
ax2.plot(dfs[x] * microns_per_pixel, dfs[ys] * microns_per_pixel, '-o')

ax1.set_ylabel(r'$N_{p}$')
ax1.legend()
ax2.set_ylabel(r'$\sigma_{xy} \: (\mu m)$')
ax2.set_xlabel(r'$\delta x_{min} \: (\mu m)$')

plt.tight_layout()
if save_plots:
    plt.savefig(path_figs + '/r-precision_and_counts_by_min-dx{}'.format(save_fig_filetype))
if show_plots:
    plt.show()
plt.close()