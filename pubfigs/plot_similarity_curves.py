# imports
from os.path import join
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import functions
from utils.bin import sample_array_at_intervals
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

# --- structure files

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_example/results/' \
           'single-particle-with-overlap-calibration-and-test/test-idpt-rand-1'
fn = 'particle_similarity_curves_tex_idpt_1_crand_test_idpt_nl2_sweep1_'
path_save = join(base_dir, 'similarity_curves')
filetype = '.xlsx'

df = pd.read_excel(join(base_dir, fn + filetype))

# setup
zs = [-40, -30, -20, -10, 0, 10]

# processing
df['cm'] = np.where(df['cm'] > 0, df['cm'].to_numpy(), 0)
df = df[df['z_cm'].isin(zs)]
df = df.sort_values('frame')


# setup
pid = 0
dfpid = df[df['id'] == pid]

if pid == 0:
    clr = 'firebrick'
    star = 'black' # 'red'
if pid == 12:
    clr = scigreen
    star = 'lime'

# plot
for fr in dfpid.frame.unique()[22:]:

    dfi = dfpid[dfpid['frame'] == fr].sort_values('z_cm').reset_index()

    # processing
    if dfi.cm.max() > 0.9:
        dfi['cm'] = dfi['cm'] * 0.9

    plot_vertical = False
    plot_sub_calibration_estimator = True

    if plot_vertical:
        fig, ax = plt.subplots(figsize=(1.05, 2.125))

        # similarity curve
        ax.plot(dfi.cm, dfi.z_cm, '-o', ms=2, color=clr, zorder=2)

        # peak correlation
        ax.scatter(dfi.cm.max(), dfi.iloc[dfi.cm.idxmax()].z_cm, s=30, marker='*', color=star, zorder=3)

        ax.set_ylim([-42, 12])
        ax.set_yticks(ticks=[-40, 10], labels=['0', r'$h$'], minor=False)
        ax.set_xlabel(r'$S$')
        ax.set_xlim([0, 1])
        ax.set_xticks(ticks=[0, 1], minor=False)

    elif plot_sub_calibration_estimator:

        idx_max = dfi.cm.idxmax()
        print(dfi.iloc[idx_max].z_cm)

        dfii = dfi.iloc[idx_max - 1:idx_max+2]

        # fit three-point parabolic estimator
        popt, pcov = curve_fit(functions.parabola, dfii.z_cm, dfii.cm)
        z_fit = np.linspace(dfii.z_cm.min(), dfii.z_cm.max())
        cm_fit = functions.parabola(z_fit, *popt)

        # plot horizontal (standard)
        fig, ax = plt.subplots(figsize=(1.75, 1.05))

        # similarity curve
        ax.plot(dfii.z_cm, dfii.cm, '-o', ms=2, color=clr, zorder=2)

        # three point estimator
        ax.plot(z_fit, cm_fit, '-', color='black', zorder=2.5)

        # estimate z value
        ax.axvline(z_fit[np.argmax(cm_fit)], 0, (np.max(cm_fit) - 0.6) / (1-0.6), ls='--', lw=0.5, color='silver', zorder=3)
        ax.scatter(z_fit[np.argmax(cm_fit)],  np.max(cm_fit), s=20, marker='d', color=scired, zorder=3)

        ax.set_xlim([dfii.z_cm.min() - 2.5, dfii.z_cm.max() + 2.5])
        ax.set_ylabel(r'$S$')
        ax.set_ylim([0.6, 1])
        ax.set_xticks(ticks=[-10, 0, 10], labels=[r'$z^{*}-1$', r'$z^{*}$', r'$z^{*}+1$'], minor=False)
        ax.set_yticks(ticks=[0.6, 1], labels=[], minor=False)

    else:
        # plot horizontal (standard)
        fig, ax = plt.subplots(figsize=(1.75, 1.05))

        # similarity curve
        ax.plot(dfi.z_cm, dfi.cm, '-o', ms=2, color=clr, zorder=2)

        # peak correlation
        ax.scatter(dfi.z_est.unique(), dfi.cm.max(), s=20, marker='^', color=star, zorder=3)

        # estimate z value
        """ax.axvline(dfi.z_est.unique(), 0, dfi.cm.max(), ls='--', lw=0.75, color='silver', zorder=3)
        ax.scatter(dfi.z_est.unique(), dfi.cm.max(), s=20, marker='^', color=star, zorder=3)
    
        # true z value
        ax.axvline(dfi.z_true.unique(), 0, dfi.cm.max(), ls='--', lw=0.75, color='black', zorder=3)
        ax.scatter(dfi.z_true.unique(), dfi.cm.max(), s=20, marker='*', color='black', zorder=3)"""

        ax.set_xlim([-42, 12])
        ax.set_ylabel(r'$S$')
        ax.set_ylim([0, 1])
        ax.set_xticks(ticks=[-40, 10], labels=['0', r'$h$'], minor=False)
        ax.set_yticks(ticks=[0, 1], minor=False)

    plt.tight_layout()
    plt.minorticks_off()
    plt.savefig(path_save + '/interp-sim-curves_pid{}_fr{}.svg'.format(pid, fr))
    plt.show()
    plt.close()
    # raise ValueError()
    j = 1



j = 1