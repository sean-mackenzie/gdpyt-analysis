# test bin, analyze, and plot functions
import numpy as np
import pandas as pd
from os.path import join
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import io, plotting, bin, modify, fit
import filter, analyze
from tracking import plotting as trackplot

import random

import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee', 'std-colors'])

# read .xlsx files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/test_coords/step'
sort_strings = ['test_id', '_SILPURAN_']
filetype = '.xlsx'
dficts = io.read_dataframes(path_name, sort_strings, filetype)

# scale the frames column to match the true z-coordinate
#dficts = modify.dficts_scale(dficts, ['frame'], multipliers=5)

# filter dataframes
min_cm = 0.95

filters = True
if filters:
    keys = ['cm']
    values = [min_cm]
    operations = ['greaterthan']
    dficts = filter.dficts_filter(dficts, keys, values, operations)

    dficts_i = filter.dficts_filter(dficts, keys=['frame'], values=[50], operations=['lessthan'], copy=True)
    dficts_f = filter.dficts_filter(dficts, keys=['frame'], values=[50], operations=['greaterthan'], copy=True)

"""# choose id to inspect
inspect_id = 1.0

# get inspectiond dataframe
dft = dficts[inspect_id]
particle_list = dft.id.unique()

# plot single particle
xparameter = 'frame'
yparameter = 'z'
z0 = 0
take_abs = False

pids = [int(p) for p in random.sample(set(particle_list), 20)]

fig, ax = trackplot.plot_scatter(dficts, pids=pids, xparameter='frame', yparameter='z', min_cm=min_cm, z0=0,
                                 take_abs=False, fit_data=False, fit_function='parabola')
plt.tight_layout()
plt.show()"""

initial_vals = analyze.calculate_mean_value(dficts_i, output_var='z', input_var='frame', span=(0, 49))
finals_vals = analyze.calculate_mean_value(dficts_f, output_var='z', input_var='frame', span=(52, 100))

z0 = np.mean(initial_vals[:, 1])
zf_std = np.mean(finals_vals[:, 2]) * 2

dfd = pd.DataFrame(finals_vals, columns=['id', 'dz', 'z_std'])
dfd.loc[:, 'dz'] = dfd.loc[:, 'dz'] - z0
dfd['z_true'] = np.array([-45, -30, 30, 45])

fig, ax = plt.subplots()
ax.errorbar(dfd.z_true, dfd.dz, yerr=dfd.z_std * 2, fmt='o', ms=4, ls='none', elinewidth=1, capsize=2, alpha=1)
# ax.scatter(dfd.z_true, dfd.dz, s=5)
# ax.plot(dfd.z_true, dfd.z_true, alpha=0.5, linewidth=2)

popt, pcov, fit_func = fit.fit(dfd.z_true, dfd.dz, fit_function=fit.line)
xfit = np.linspace(dfd.z_true.min(), dfd.z_true.max(), 100)
ax.plot(xfit, fit.line(xfit, *popt), color='black', linestyle='--', alpha=0.5)

perr = np.sqrt(np.diag(pcov))
print('1-sigma of fit parameters: {}'.format(perr))
ax.grid(alpha=0.25)
ax.set_xlabel(r'$z_{true}\: (\mu m)$')
ax.set_ylabel(r'$z_{measured}\: (\mu m)$')

ax.legend([r'Fit: $\sigma_{fit}=0.006$', r'GDPyT: $\sigma_{z}=0.2\: \mu m$'])

plt.tight_layout()
plt.savefig(join('/Users/mackenzie/Desktop', 'micrometer-v2-Barnkob-error2.png'))

plt.show()

dfd['error'] = (dfd['z_true'] - dfd['dz']) / np.sqrt(2)
print(dfd.abs().mean())



j=1