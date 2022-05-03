# test bin, analyze, and plot functions
import os
from os.path import join
from os import listdir

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import filter
import analyze
from correction import correct
from utils import fit, functions, bin, io, plotting, modify, plot_collections

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

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_0.87umNR/analyses/' \
           'results_04.17.22_spct-meta'

path_test_coords = join(base_dir, 'coords/test-coords')
path_calib_coords = join(base_dir, 'coords/calib-coords')
path_similarity = join(base_dir, 'similarity')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

method = 'spct'
microns_per_pixel = 0.8

# ----------------------------------------------------------------------------------------------------------------------
# 1. READ CALIB COORDS
read_calib_coords = False

if read_calib_coords:
    dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)


# ----------------------------------------------------------------------------------------------------------------------
# 2. EVALUATE DISTORTION-DEPENDENT LOCALIZATION ERRORS

analyze_distortion_errors = False

if analyze_distortion_errors:
    df = io.read_test_coords(path_test_coords)

    # get test coords stats
    i_num_rows = len(df)
    i_num_pids = len(df.id.unique())

    # setup
    df = df.dropna()
    df['r'] = functions.calculate_radius_at_xy(df.x, df.y, xc=256, yc=256)

    # filter error
    df = df[df.error.abs() < 5]

    # filter num frames
    dfg = df.groupby('id').count().reset_index()
    remove_ids = dfg[dfg['z'] < len(df.frame.unique()) * 0.6].id.values
    df = df[~df.id.isin(remove_ids)]

    # --- fit a line to each particle's z vs. z_true
    slopes = []
    intercepts = []
    radii = []
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches * 1.25))

    df = df.sort_values('r')
    inspect_pids = df.id.unique()

    for i, pid in enumerate(inspect_pids):
        dfpid = df[df.id == pid]

        popt, pcov = curve_fit(functions.line, dfpid.z_true, dfpid.z)
        slopes.append(popt[0])
        intercepts.append(popt[1])
        radii.append(int(dfpid.r.mean()))

        if i in np.arange(1, len(inspect_pids) + 1, (len(inspect_pids) + 1) // 7):
            ax.scatter(dfpid.z_true, dfpid.z, s=1, alpha=0.5)
            ax.plot(dfpid.z_true, functions.line(dfpid.z_true, *popt), linewidth=0.5,
                    label=r'$p_{ID}$' + '{} (r={}): {}'.format(pid, np.round(dfpid.r.mean(), 1), np.round(popt[0], 4)))

    ax.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(radii, slopes)
    ax.set_xlim([0, 350])
    plt.show()




    j = 1

# ----------------------------------------------------------------------------------------------------------------------
# 3. SPCT STATS

analyze_spct_stats = True

if analyze_spct_stats:

    # read
    plot_collections.plot_spct_stats(base_dir)