# test bin, analyze, and plot functions

# imports
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
from utils.plotting import lighten_color

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
scired = '#FF9500'
sciorange = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# 1. SETUP - BASE DIRECTORY

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.02.21-BPE_Pressure_Deflection_20X/analyses/'

# ----------------------------------------------------------------------------------------------------------------------
# 2. SETUP - IDPT

path_spct_calib1 = join(base_dir, 'results-04.26.22_spct_calib1_test-2-3')
path_test_coords = join(path_spct_calib1, 'coords/test-coords')
path_calib_coords = join(path_spct_calib1, 'coords/calib-coords')
path_similarity = join(path_spct_calib1, 'similarity')
path_results = join(path_spct_calib1, 'results')
path_figs = join(path_spct_calib1, 'figs')

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 3. ANALYSIS

method = 'spct'
microns_per_pixel = 0.8

# --- 3.1 SPCT STATS

analyze_spct_stats = False

if analyze_spct_stats:
    plot_collections.plot_spct_stats(base_dir=path_spct_calib1)

# --- 3.2 APPARENT PARTICLE POSITIONS


# --- 3.3 PARTICLE TO PARTICLE SIMILARITIES



j = 1

print("Analysis completed without errors.")