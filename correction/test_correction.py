# test bin, analyze, and plot functions
import numpy as np
import pandas as pd
import random
import os
from os.path import join

from utils import io, plotting, bin, modify, fit
import filter, analyze
from tracking import plotting as trackplot

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee', 'std-colors'])

# read .xlsx files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/test/step'
sort_strings = ['test_id', '_coords_']
filetype = '.xlsx'
drop_columns = ['stack_id', 'z_true', 'max_sim', 'error']
dficts = io.read_dataframes(path_name, sort_strings, filetype, drop_columns=drop_columns)

# ----------------------------------------------------------------------------------------------------------------------
# filter dataframes
min_cm = 0.5

filters = True
if filters:
    keys = ['cm']
    values = [min_cm]
    operations = ['greaterthan']
    dficts = filter.dficts_filter(dficts, keys, values, operations)

dficts_i = filter.dficts_filter(dficts, keys=['frame'], values=[50], operations=['lessthan'], copy=True)
dficts_f = filter.dficts_filter(dficts, keys=['frame'], values=[50], operations=['greaterthan'], copy=True)
del dficts

dfi = modify.stack_dficts_by_key(dficts_i, drop_filename=False)
dff = modify.stack_dficts_by_key(dficts_f, drop_filename=False)
del dficts_i, dficts_f

dfi = dfi.drop(columns=['frame', 'x', 'y'])
dff = dff.drop(columns=['frame'])

dfi_mean = dfi.groupby(by=['filename', 'id']).mean()
dfi_mean = dfi_mean.rename(columns={'z': 'zi_mean', 'cm': 'cmi_mean'})
dfi_std = dfi.groupby(by=['filename', 'id']).std()
dfi_std = dfi_std.rename(columns={'z': 'zi_std', 'cm': 'cmi_std'})

dff_mean = dff.groupby(by=['filename', 'id']).mean()
dff_mean = dff_mean.rename(columns={'z': 'zf_mean', 'cm': 'cmf_mean'})
dff_std = dff.groupby(by=['filename', 'id']).std()
dff_std = dff_std.rename(columns={'z': 'zf_std', 'cm': 'cmf_std'})
dff_std = dff_std.drop(columns=['x', 'y'])

dfd = dff_mean.join([dfi_mean, dff_std, dfi_std])
dfd['dz_mean'] = dfd['zf_mean'] - dfd['zi_mean']
dfd['dz_std'] = dfd['zf_std'] + dfd['zi_std']

dfd = dfd.reset_index()
dfd['dzd_mean'] = (dfd['dz_mean'] - dfd['dz_mean'].mean())

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# plotting initial


def plot_err(dfd, xp, yp, errp, corrections):
    fig, ax = plt.subplots()
    ax.errorbar(dfd[xp], dfd[yp], yerr=2 * dfd[errp], fmt='o', elinewidth=1, capsize=2, alpha=0.5)
    ax.scatter(dfd[xp], dfd[yp], s=5)
    ax.set_xlabel(xp)
    ax.set_ylabel(yp)
    ax.set_title(
        '{} {} = {} +/- {}'.format(corrections, yp, np.round(dfd[yp].mean(), 2), 2 * np.round(dfd[errp].mean(), 2)))
    plt.tight_layout()
    plt.show()

def plot_distribution(corrections):

    xp, yp, errp = ['x', 'zi_mean', 'zi_std']
    plot_err(dfd, xp, yp, errp, corrections)

    xp, yp, errp = ['x', 'cmi_mean', 'cmi_std']
    plot_err(dfd, xp, yp, errp, corrections)

    xp, yp, errp = ['y', 'zi_mean', 'zi_std']
    plot_err(dfd, xp, yp, errp, corrections)

    xp, yp, errp = ['y', 'cmi_mean', 'cmi_std']
    plot_err(dfd, xp, yp, errp, corrections)

    xp, yp, errp = ['x', 'zf_mean', 'zf_std']
    plot_err(dfd, xp, yp, errp, corrections)

    xp, yp, errp = ['y', 'zf_mean', 'zf_std']
    plot_err(dfd, xp, yp, errp, corrections)

    xp, yp, errp = ['x', 'cmf_mean', 'cmf_std']
    plot_err(dfd, xp, yp, errp, corrections)

    xp, yp, errp = ['y', 'cmf_mean', 'cmf_std']
    plot_err(dfd, xp, yp, errp, corrections)

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# plotting initial

corrections = 'uncorrected'
plot_distribution(corrections)

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# correct: remove initial particles with stdev > 0.5 micron
corrections = 'correct initial'
dfd = dfd[dfd['zi_std'] < 0.5]
plot_distribution(corrections)

# ----------------------------------------------------------------------------------------------------------------------

# correct: remove final particles with stdev > 2 sigma
corrections = 'correct final'
dfd = dfd[dfd['zf_std'] < 1.25]
plot_distribution(corrections)