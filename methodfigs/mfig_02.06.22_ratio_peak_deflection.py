# 02.06.22 - ratio of peak deflections

# imports
import os
from os.path import join

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from utils import functions, bin, modify
from utils.plotting import lighten_color

# ---

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 0. Experimental Parameters

mag_eff = 5.0
numerical_aperture = 0.3
pixel_size = 16
depth_of_focus = functions.depth_of_field(mag_eff, numerical_aperture, 600e-9, 1.0, pixel_size=pixel_size * 1e-6) * 1e6
microns_per_pixel = 3.2
exposure_time = 40e-3
frame_rate = 24.444
time_per_frame = 1 / frame_rate
start_frame = 39
start_time = start_frame / frame_rate

# exclude outliers
pids_saturated = [12, 13, 18, 34, 39, 49, 66, 78]
excluded_pids = [39, 61]
bad_pids = [12, 13, 18, 34, 39, 49, 61, 66, 78]

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 1. FILES PATHS

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/' \
           'results-09.15.22_idpt-sweep-ttemp/results/dz1_ttemp13'
path_data = join(base_dir, 'id13_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx')
path_results = join(base_dir, 'fit-plate-theory')

# read
df = pd.read_excel(path_data)

# ----------------------------------------------------------------------------------------------------------------------
# 2. PLOT

# setup
df = df[df['time'] > start_time]
df['dz_ratio'] = df.rz_lr / df.rz_ul
avg_dz_ratio = df[(df['dz_ratio'] < 3) & (df['dz_ratio'] > 1.75)].dz_ratio.mean()
ms = 1

# plot
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

ax1.plot(df.time, df.rz_lr, 'o', ms=ms, label=r'$800$')
ax1.plot(df.time, df.rz_ul, 'o', ms=ms, label=r'$500$')
ax1.set_ylabel(r'$w_{o} \: (\mu m)$')
ax1.legend(title=r'$a \: (\mu m)$',
           markerscale=2, borderpad=0.2, borderaxespad=0.3, labelspacing=0.25, handletextpad=0.2)

ax2.plot(df.time, df.rz_lr / df.rz_ul, 'ko', ms=ms)
ax2.set_ylabel(r'$\frac{w_{o}(a=800 \: \mu m)} {w_{o}(a=500 \: \mu m)}$')
ax2.set_ylim([0.75, 2.75])
ax2.set_xlabel(r'$t \: (s)$')
plt.tight_layout()
plt.savefig(path_results + '/dz-lr-ul_and_dz-ratio_mean={}.png'.format(np.round(avg_dz_ratio, 2)))
plt.show()
plt.close()

# ---

# plot dz_ratio as a function of w0(r = 800)
dfp = df[df['rz_lr'] > 0]
dfn = df[df['rz_lr'] < 0]

fig, ax2 = plt.subplots(figsize=(size_x_inches, size_y_inches * 0.75))
ax2.plot(dfp.rz_lr, dfp.dz_ratio, 'ko', ms=1, label=r'$+w_{o}$')
ax2.plot(dfn.rz_lr.abs(), dfn.dz_ratio, 'ro', ms=1, label=r'$-w_{o}$')
ax2.set_ylabel(r'$\frac{w_{o}(a=800 \: \mu m)} {w_{o}(a=500 \: \mu m)}$')
ax2.set_ylim([0.8, 2.6])
ax2.set_yticks([1, 1.5, 2, 2.5])
ax2.set_xlabel(r'$w_{o}(a=800 \: \mu m)$')
ax2.legend(markerscale=2)
plt.tight_layout()
plt.savefig(path_results + '/dz-ratio_by_dz-lr.png')
plt.show()
plt.close()

# ---

# plot all together

# plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=False, figsize=(size_x_inches, size_y_inches * 1.25),
                                    gridspec_kw={'height_ratios': [1, 1, 1]})

ax1.plot(df.time, df.rz_lr, 'o', ms=ms, label=r'$800$')
ax1.plot(df.time, df.rz_ul, 'o', ms=ms, label=r'$500$')
ax1.set_ylabel(r'$w_{o} \: (\mu m)$')
ax1.set_xlabel(r'$t \: (s)$')
ax1.legend(title=r'$a \: (\mu m)$',
           loc='upper left', bbox_to_anchor=(1, 1),
           markerscale=2, borderpad=0.2, borderaxespad=0.3, labelspacing=0.25, handletextpad=0.2)

ax2.plot(df.time, df.rz_lr / df.rz_ul, 'o', ms=ms, color='gray')
ax2.set_ylabel(r'$\frac{w_{o}(a=800 \: \mu m)} {w_{o}(a=500 \: \mu m)}$')
ax2.set_ylim([0.8, 2.6])
ax2.set_yticks([1, 1.5, 2, 2.5])
ax2.set_xlabel(r'$t \: (s)$')

dfp = df[df['rz_lr'] > 0]
dfn = df[df['rz_lr'] < 0]

ax3.plot(dfp.rz_lr, dfp.dz_ratio, 'ko', ms=1, label=r'$+w_{o}$')
ax3.plot(dfn.rz_lr.abs(), dfn.dz_ratio, 'ro', ms=1, label=r'$-w_{o}$')
ax3.set_ylabel(r'$\frac{w_{o}(a=800 \: \mu m)} {w_{o}(a=500 \: \mu m)}$')
ax3.set_ylim([0.8, 2.6])
ax3.set_yticks([1, 1.5, 2, 2.5])
ax3.set_xlabel(r'$w_{o}(a=800 \: \mu m)$')
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1),
           markerscale=2, borderpad=0.2, borderaxespad=0.3, labelspacing=0.25, handletextpad=0.2)

ax1.tick_params(axis='both', which='minor',
                bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax2.tick_params(axis='both', which='minor',
                bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax3.tick_params(axis='both', which='minor',
                bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)

# plt.tight_layout()
plt.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.8, hspace=0.85)

plt.savefig(path_results + '/dz-ratio_all.png')
plt.show()
plt.close()