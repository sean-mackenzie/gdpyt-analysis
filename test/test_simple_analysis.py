# plot 2/7/22 membrane characterizations

from os.path import join

import matplotlib.pyplot as plt
# imports
import numpy as np
import pandas as pd

import filter
import analyze
from correction import correct
from utils import io, plotting, modify, details, functions, fit

# description of code
"""
Purpose of this code:
    1. 

Process of this code:
    1. Setup
        1.1 Setup file paths.
        1.2 Setup plotting style.
        1.3 Setup filters. 
        1.4 Read all test_coords.
        1.5 Filter particles by c_m (should always be 0.5)
    2. 
"""

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

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup



# setup file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/zipper'
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# experiment
path_name = join(base_dir, 'test_coords')
save_id = 'exp'


# test space

# zero values
fname = 'test_id0_coords_0V.xlsx'
dfzero = pd.read_excel(join(path_name, fname))
dfzeror = dfzero.round({'x': -1})
dfzerog = dfzeror.groupby('x').mean()
dfzerog = dfzerog[['y', 'z']]
dfzerog = dfzerog.rename(columns={"y": "y_i", "z": "z_i"})

zero_mean = dfzerog.z_i.mean()

heights = np.array([0, 60, 75])
max_z = []
mean_z = []
min_z = []



for i, h in enumerate(heights):
    fname = 'test_id{}_coords_{}V.xlsx'.format(i, h)
    df = pd.read_excel(join(path_name, fname), dtype=float)
    df = df.astype(float)
    # scatter
    """fig, ax = plotting.plot_scatter_3d(df, fig=None, ax=None, elev=5, azim=80, color=None, alpha=0.75)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vapp = {} V'.format(h), fontsize=18)
    plt.show()"""

    """dfr = df.copy()
    dfr = dfr.round({'x': -1})
    dfg = dfr.groupby('x').mean() 
    dfplot = pd.concat([dfg, dfzerog], axis=1).reindex(dfg.index)"""

    dfi = df[df['frame'] < 10]
    dff = df[df['frame'] > 10]

    dfim = dfi.z.mean()
    dffm = dff.z.mean()

    xi = dfi.x
    zi = dfi.z  # (dfplot.z - dfplot.z_i) * -1

    """fig, ax = plt.subplots()
    ax.scatter(xi, zi, s=5, label=r'$V_{i}$' + ': {}'.format(h))"""
    #ax.plot(xi, zi)

    xf = dff.x
    zf = dff.z  # (dfplot.z - dfplot.z_i) * -1

    """ax.scatter(xf, zf, s=5, label=r'$V_{f}$' + ': {}'.format(h))"""
    #ax.plot(xf, zf)

    print("height {}, deflection at z=420 = {}".format(h, np.max(zf)))
    zmax = np.max(zf) - 33.4
    #max_z.append(zmax)
    """min_z.append(dfplot.z.min())
    mean_z.append(dfplot.z.mean())"""

    """ax.axhline(y=zi.mean(), color='tab:blue', linestyle='--')
    ax.axhline(y=zf.mean(), color='tab:green', linestyle='--')"""

    if i == 0:
        zinit = 0.0
        print(zi.mean())
        max_z.append(zinit)
    else:
        print(zi.mean())
        print(zf.mean())
        max_z.append(zf.mean() - 33.4)

    """ax.set_xlabel('x (pixels)')
    ax.set_ylabel('z (microns)')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()"""

fig, ax = plt.subplots()
ax.plot(heights, max_z, '-o')
ax.set_xlabel(r'$V_{applied} (V)$')
ax.set_ylabel(r'$Deflection (\mu m)$')
plt.tight_layout()
plt.show()

raise ValueError('hah')

fig, ax = plt.subplots()
ax.plot(heights, mean_z, '-o')
ax.set_xlabel(r'$V_{applied} (V)$')
ax.set_ylabel(r'$Deflection (\mu m)$')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.plot(heights, min_z, '-o')
ax.set_xlabel(r'$V_{applied} (V)$')
ax.set_ylabel(r'$Deflection (\mu m)$')
plt.tight_layout()
plt.show()

raise ValueError('ha')

max_z = np.array([-1.245, -0.511, 1.161, 4.781, 8.4781, 12.209, 14.264])

fig, ax = plt.subplots()
ax.plot(heights, max_z)
ax.scatter(heights, max_z)
ax.set_xlabel(r'$Height \: (mm)$')
ax.set_ylabel(r'$z_{max} \: (\mu m)$')
plt.tight_layout()
plt.show()

radius = 400e-6  # m
E = 5e6  # Pa
h = 20e-6  # m
poisson = 0.5
D = E * h**3 / (12 * (1 - poisson**2))

rho = 1000  # kg/m**3
g = 9.81  # m/s**2
pressure = heights * rho * g * 1e-3

# plot theoretical
z_theory_clamped = pressure * radius**4 / (64 * D) * 1e6
z_theory_simply = pressure * radius**2 / (64 * D) * 1e6 * (((5 + poisson) / (1 + poisson)) * radius**2)

fig, ax = plt.subplots()
ax.plot(pressure, max_z, '-o', label=r'$Experiment$')
ax.plot(pressure, z_theory_clamped, '-o', label=r'$Theory_{Clamped}$')
ax.plot(pressure, z_theory_simply, '-o', label=r'$Theory_{Simply Supported}$')

ax.set_xlabel(r'$Pressure \: (Pa)$')
ax.set_ylabel(r'$z_{max} \: (\mu m)$')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# discard the first two points which aren't linear
pressure_fit = pressure[2:]
max_z_fit = max_z[2:]

# fit a line to find where max_z should be shifted up to coincide with P=0, z=0
fit_func_line = functions.line
popt, pcov, fit_func_line = fit.fit(pressure_fit, max_z_fit, fit_func_line)
line_slope = popt[0]
line_y_intercept = popt[1]
pressure_interp = np.linspace(0, np.max(pressure), 20)

fig, ax = plt.subplots()
ax.plot(pressure, max_z, '-o', label=r'$Experiment$')
ax.plot(pressure_interp, fit_func_line(pressure_interp, line_slope, line_y_intercept), color='black', label=r'$Fit$')
ax.set_xlabel(r'$Pressure \: (Pa)$')
ax.set_ylabel(r'$z_{max} \: (\mu m)$')
ax.set_title(r'$f(a={{{}}}, b={{{}}}) = a x + b$'.format(np.round(line_slope, 2), np.round(line_y_intercept, 2)))
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# adjustments
# adjust P values to be linear wrt P=0 @ x-intercept
pressure_fit = pressure_fit - 36
pressure = pressure - 36
pressure_interp = np.linspace(0, np.max(pressure), 20)

# adjust z values to be linear wrt y-intercept
max_z_orig_adj = max_z  # - line_y_intercept
max_z_adj = max_z_fit  # - line_y_intercept

# fit spherical uniformly loaded w/ clamped boundary conditions
inst = functions.fSphericalUniformLoad()
inst.r = radius
inst.h = h
inst.poisson = poisson

# --- fit clamped ---
fit_func_clamped = inst.spherical_uniformly_loaded_clamped_plate
bounds = None  # ([radius, h, 0, poisson], [radius+1e-7, h+1e-7, 20e6, poisson+1e-3])

popt_c, pcov_c, fit_func_clamped = fit.fit(pressure_fit, max_z_adj * 1e-6, fit_func_clamped)
E_c = popt_c[0]

fig, ax = plt.subplots()
ax.plot(pressure, max_z_orig_adj, '-o', label=r'$Experiment$')
ax.plot(pressure_interp, fit_func_clamped(pressure_interp, *popt_c) * 1e6, color='black', label=r'$Fit_{clamped}$')
ax.set_xlabel(r'$Pressure \: (Pa)$')
ax.set_ylabel(r'$z_{max} \: (\mu m)$')
ax.set_title(r'$f(E={{{}}} \: MPa) = $'.format(np.round(E_c * 1e-6, 2)) + r'$\frac{P r^4}{64Eh^3 \slash 12 (1-\upsilon)}$')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
# --- ---

# --- fit simply supported ---
fit_func_simply = inst.spherical_uniformly_loaded_simply_supported_plate
popt_ss, pcov_ss, fit_func_simply = fit.fit(pressure_fit, max_z_adj * 1e-6, fit_func_simply)
E_ss = popt_ss[0]

fig, ax = plt.subplots()
ax.plot(pressure, max_z_orig_adj, '-o', label=r'$Experiment$')
ax.plot(pressure_interp, fit_func_simply(pressure_interp, *popt_ss) * 1e6, color='black', label=r'$Fit_{simply supported}$')
ax.set_xlabel(r'$Pressure \: (Pa)$')
ax.set_ylabel(r'$z_{max} \: (\mu m)$')
ax.set_title(r'$f(E={{{}}} \: MPa) = $'.format(np.round(E_ss * 1e-6, 2)) + r'$\frac{P r^4}{64Eh^3 \slash 12 (1-\upsilon)}$')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()