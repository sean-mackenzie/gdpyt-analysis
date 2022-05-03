# imports

import os
from os.path import join
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Akima1DInterpolator
from scipy.stats import norm
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version

from utils import bin, fit, functions, io, plotting
from correction import correct
import analyze

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import collections, colors, transforms

# formatting
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'
fontP = FontProperties()
fontP.set_size('medium')

plt.style.use(['science', 'ieee', 'std-colors'])
# plt.style.use(['science', 'scatter'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# Apparent lateral position as a function of actual axial position
"""
Accurate localization microscopy by intrinsic aberrations (2021)
Fig.2a
"""