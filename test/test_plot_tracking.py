# test bin, analyze, and plot functions
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, CloughTocher2DInterpolator

import matplotlib.pyplot as plt

# experimental
fps = 24.44


fp = '/Users/mackenzie/Downloads/test_coords_particle_image_stats.xlsx'
df = pd.read_excel(fp)

dfc = df[df['z'] > 25]
df_counts = dfc.groupby('id').count().reset_index()
max_counts = df_counts.z.max()
passing_ids_counts = df_counts[df_counts.z > max_counts * 0.35].id.unique()

df = df[df.id.isin(passing_ids_counts)]

# time
df['t'] = df.frame / fps


fig, ax = plt.subplots()
ax.scatter(df.t, df.z, c=df.id)
plt.show()









print("Analysis completed without errors.")