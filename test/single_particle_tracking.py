# test bin, analyze, and plot functions
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import io, plotting, bin, modify, fit, functions
import filter, analyze
from tracking import plotting as trackplot

import random

import matplotlib.pyplot as plt

# read .xlsx files to dictionary
path_name = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/test_coords/rotate'
sort_strings = ['test_id', '_SILPURAN_']
filetype = '.xlsx'
dficts = io.read_dataframes(path_name, sort_strings, filetype)

# scale the frames column to match the true z-coordinate
#dficts = modify.dficts_scale(dficts, ['frame'], multipliers=5)

# filter dataframes
min_cm = 0.9

filters = True
if filters:
    keys = ['cm']
    values = [min_cm]
    operations = ['greaterthan']
    dficts = filter.dficts_filter(dficts, keys, values, operations)

    dficts_i = filter.dficts_filter(dficts, keys=['frame'], values=[50], operations=['lessthan'], copy=True)
    dficts_f = filter.dficts_filter(dficts, keys=['frame'], values=[50], operations=['greaterthan'], copy=True)

"""# choose id to inspect
inspect_id = 6.0

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
# finals_vals = analyze.calculate_mean_value(dficts, output_var='z', input_var='frame', span=(52, 100))


"""# plot 3D scatter
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
colors = iter(['red', 'blue'])
for name, df in dficts_i.items():
    c = next(colors)
    fig, ax = plotting.plot_scatter_3d(df, fig=fig, ax=ax, color=c)

colors = iter(['lime', 'magenta'])
for name, df in dficts_f.items():
    c = next(colors)
    fig, ax = plotting.plot_scatter_3d(df, fig=fig, ax=ax, color=c)
plt.show()"""

df = dficts_i[5]
df = df[df['z'] < 57]
df = df.groupby('id').mean()
points = np.stack((df.x, df.y, df.z)).T
px, py, pz = fit.fit_3d(points, fit_function='plane')

df2 = dficts_f[5]
df2 = df2[df2['z'] < 57]
df2 = df2.groupby('id').mean()
points2 = np.stack((df2.x, df2.y, df2.z)).T
px2, py2, pz2 = fit.fit_3d(points2, fit_function='plane')

a = np.dstack((px, py, pz))
b = np.dstack((px2, py2, pz2))
thetax, thetay = functions.calculate_angle_between_planes(a, b)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.x, df.y, df.z, color='cornflowerblue')
ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue')
ax.scatter(df2.x, df2.y, df2.z, color='indianred')
ax.plot_surface(px2, py2, pz2, alpha=0.4, color='red')
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
ax.view_init(5, 45)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.x, df.y, df.z, color='cornflowerblue')
ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue')
ax.scatter(df2.x, df2.y, df2.z, color='indianred')
ax.plot_surface(px2, py2, pz2, alpha=0.4, color='red')
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
ax.view_init(5, 0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.x, df.y, df.z, color='cornflowerblue')
ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue')
ax.scatter(df2.x, df2.y, df2.z, color='indianred')
ax.plot_surface(px2, py2, pz2, alpha=0.4, color='red')
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
ax.view_init(5, 90)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.x, df.y, df.z, color='cornflowerblue')
ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue')
ax.scatter(df2.x, df2.y, df2.z, color='indianred')
ax.plot_surface(px2, py2, pz2, alpha=0.4, color='red')
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
ax.view_init(90, 45)
plt.show()


fig, ax = plt.subplots()
data = ax.scatter(df.x, df.y, s=100, c=df.z, marker='*', label='i')
data2 = ax.scatter(df2.x, df2.y, s=25, c=df2.z, marker='o', label='f', alpha=0.75)
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.grid(alpha=0.125)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.5)
plt.colorbar(data2, cax=cax)
ax.legend(loc='upper right')
plt.show()


"""# plot the difference
data = ax.scatter(df.x, df.y, s=100, c=df.z, marker='*', label='i')
data2 = ax.scatter(df2.x, df2.y, s=25, c=df2.z, marker='o', label='f', alpha=0.75)

ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.grid(alpha=0.125)

# color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.5)
plt.colorbar(data2, cax=cax)

ax.legend(loc='upper right')

plt.show()"""

j=1