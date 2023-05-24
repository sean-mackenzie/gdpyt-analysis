# calculate effective numerical aperture
"""
Reference: Rossi et al. (2012) On the effect of particle image intensity and image preprocessing on the depth of...
"""
import os
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, CloughTocher2DInterpolator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

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
scired = '#FF2C00'
sciorange = '#FF9500'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# TEST COORDS (FINAL)
"""
IDPT:
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/
results-07.29.22-idpt-tmg'

SPCT:
base_dir = ''
"""

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup

testset = 'today'

if testset == 'today':
    # today (8/14/22)
    title = '10.01X, 0.45NA, 2.15 um'
    # fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/results/calibration-spct_testset-spct-cal/calib_spct_stats_11.06.21_z-micrometer-v2_5umMS__spct-cal.xlsx'
    # df = pd.read_excel(fp)

    # file paths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_particle_image_ellipses'
    path_read = join(base_dir, 'data')
    fn = 'calib_spct_stats_11.06.21_z-micrometer-v2_1umSteps_pdf.xlsx'
    df = pd.read_excel(join(path_read, fn))
    df = df.dropna(subset=['gauss_sigma_x_y'])

    # optics
    mag_eff = 10.01
    na_nominal = 0.45
    na_eff = 0.35
    p_d = 2.15e-6
    wavelength = 600e-9
    n_0 = 1.0
    pixel_size = 16e-6
    microns_per_pixel = 1.6

elif testset == '20X_1X_mag_870nmNR':
    title = '20X, 0.45NA, 0.870 um'
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/intrinsic-aberrations-defocusing/20X_1Xmag_0.87umNR/results-04.24.22_spct-meta/coords/calib-coords/calib_spct_stats_20X_1Xmag_0.87umNR_calib_spct_meta-spct-cal.xlsx'
    df = pd.read_excel(fp)

    # optics
    mag_eff = 20
    na_nominal = 0.45
    na_eff = 0.45
    p_d = 0.87e-6
    wavelength = 600e-9
    n_0 = 1.0
    pixel_size = 16e-6
    microns_per_pixel = 0.8

elif testset == '10X_1X_mag_215umNR':
    title = '10X, 0.3NA, 2.15 um'
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/figure data/intrinsic-aberrations-defocusing/10X_1Xmag_2.15umNR/coords/calib-coords/calib_spct_stats_10X_1Xmag_2.15umNR_HighInt_0.12XHg_calib_SPC_spct-meta.xlsx'
    df = pd.read_excel(fp)

    # optics
    mag_eff = 10
    na_nominal = 0.3
    na_eff = 0.3
    p_d = 2.15e-6
    wavelength = 600e-9
    n_0 = 1.0
    pixel_size = 16e-6
    microns_per_pixel = 1.6

x = 'z_corr'
y1 = 'contour_diameter'
y2 = 'gauss_diameter'
# y2 = 'contour_diameter'

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 2. Plot a single particle image intensity by z

# get particle ID's with the most counts
dfc = df.groupby('id').count().reset_index()
max_counts = dfc[y1].max()
passing_ids = dfc[dfc[y1] > max_counts * 0.95].id.values

# filter
df = df[df['id'].isin(passing_ids)]
# df = df[(df['id'] > 35) & (df['id'] < 71)]

# ----------------------------------------------------------------------------------------------------------------------
# 3. Plot

# setup
width = 10  # defines the width of the parabola used to find z_f = z = 0

# optical model
fPI = functions.fParticleImageOpticalTheory(magnification=mag_eff,
                                            numerical_aperture=na_nominal,
                                            particle_diameter=p_d,
                                            wavelength=wavelength,
                                            pixel_size=pixel_size,
                                            n0=1.0)

# analytical particle image diameter
z = np.linspace(df[x].min(), df[x].max(), 100) * 1e-6  # units: meters
# dia_nominal_min = np.min(functions.particle_image_diameter_by_z(z, mag_eff, na_nominal, p_d, wavelength, n_0) * 1e6) / mag_eff

all_nas = []
for pid in df.id.unique():

    fit_nas = [pid]
    z_fit_lims = [50]

    for z_fit_lim in z_fit_lims:
        dfpid = df[df['id'] == pid].reset_index()

        dfpid_fit = dfpid[(dfpid[x] > -z_fit_lim) & (dfpid[x] < z_fit_lim)].copy()
        dfpid_fit = dfpid_fit.reset_index()
        idx_low = dfpid_fit[y2].idxmin() - width
        idx_high = dfpid_fit[y2].idxmin() + width

        if idx_low < 0:
            idx_low = 0

        x_min = dfpid_fit.iloc[idx_low:idx_high][x]
        y_min = dfpid_fit.iloc[idx_low:idx_high][y2] * microns_per_pixel
        poptt, pcovv = curve_fit(functions.quadratic_slide, x_min, y_min)
        x_min_fit = np.linspace(x_min.min(), x_min.max())
        x_offset = x_min_fit[np.argmin(functions.quadratic_slide(x_min_fit, *poptt))]
        y_offset = np.min(functions.quadratic_slide(x_min_fit, *poptt))

        """fig, ax = plt.subplots()
        ax.scatter(x_min, y_min, s=2, alpha=0.25)
        ax.plot(x_min_fit, functions.quadratic_slide(x_min_fit, *poptt))
        plt.show()"""

        # fit
        dia_min_offset = 0  # y_offset - dia_nominal_min - 0.5 #  * 1e-6 * microns_per_pixel * mag_eff - dia_nominal_min * 1e-6

        # y_data = dfpid[y2] * microns_per_pixel - dia_min_offset * 1e6

        x_fit = (dfpid_fit[x].to_numpy() - x_offset) * 1e-6
        y_fit = (dfpid_fit[y2].to_numpy() * microns_per_pixel - dia_min_offset) * 1e-6
        print(np.min(dfpid_fit[y2].to_numpy() * microns_per_pixel))
        print(np.min(y_fit * 1e6))

        xx_fit = x_fit * 1e6
        yy_fit = y_fit * 1e6

        popt, pcov = curve_fit(fPI.fit_effective_numerical_aperture,
                               x_fit,
                               y_fit,
                               # p0=[na_nominal],
                               bounds=(0.001, 1)
                               )

        fig, ax1 = plt.subplots()
        ax1.scatter(x_fit * 1e6, y_fit * 1e6, s=1, label=pid)
        ax1.scatter(x_fit * 1e6, fPI.fit_effective_numerical_aperture(x_fit, na_nominal) * 1e6, s=1, label=na_nominal)
        ax1.scatter(x_fit * 1e6, fPI.fit_effective_numerical_aperture(x_fit, *popt) * 1e6, s=1, label=popt[0])
        ax1.legend()
        plt.show()
        j = 1
        raise ValueError()

        fig, ax1 = plt.subplots()
        ax1.scatter(x_fit * 1e6, y_fit * 1e6, s=1, label=pid)
        dia_guess = functions.particle_image_diameter_by_z(z, mag_eff, popt[0], p_d, wavelength, n_0)
        ax1.plot(z * 1e6, dia_guess * 1e6,
                 color='black', linestyle='--', alpha=0.75, label='NA={}'.format(np.round(popt[0], 3)))
        ax1.legend()
        plt.show()
        j = 1

        raise ValueError()

        # particle image diameter + effective numerical aperture
        na_eff = np.round(popt[0], 2)
        if na_eff < 0.2:
            fit_nas.extend([np.nan])
            continue
        fit_nas.extend([na_eff])

        # plot
        # plot_figs = False
        if plot_figs:
            print("Fitted effective NA = {}".format(na_eff))
            dia_eff = functions.particle_image_diameter_by_z(z, mag_eff, na_eff, p_d, wavelength, n_0) * 1e6
            dia_nominal = functions.particle_image_diameter_by_z(z, mag_eff, na_nominal, p_d, wavelength, n_0) * 1e6

            # contributions to total diameter
            """dia_p = functions.particle_image_diameter_term_by_z('geometric', z, mag_eff, na_eff, p_d, wavelength, n_0) * 1e6
            dia_s = functions.particle_image_diameter_term_by_z('diffraction', z, mag_eff, na_eff, p_d, wavelength, n_0) * 1e6
            dia_f = functions.particle_image_diameter_term_by_z('defocused', z, mag_eff, na_eff, p_d, wavelength, n_0) * 1e6"""

            fig, ax1 = plt.subplots()

            # y_fit = (dfpid_fit[y2].to_numpy() * microns_per_pixel - dia_min_offset) * mag_eff * 1e-6
            y_data = dfpid[y2] * microns_per_pixel # - dia_min_offset
            ax1.scatter(dfpid[x] - x_offset, y_data, s=1, label=pid)
            ax1.plot(z * 1e6, dia_eff / mag_eff, color='black', linestyle='--', alpha=0.75, label='NA={}'.format(na_eff))
            ax1.plot(z * 1e6, dia_nominal / mag_eff, color='red', linestyle='-', alpha=0.25, label='NA={}'.format(na_nominal))

            print("Min dia data = {}".format(np.round(np.min(y_data), 2)))
            print("Min dia nominal = {}".format(np.round(np.min(dia_nominal / mag_eff), 2)))
            # ax1.plot(z * 1e6, dia_p / mag_eff, label=r'$d_{p}$')
            # ax1.plot(z * 1e6, dia_s / mag_eff, label=r'$d_{s}$')
            # ax1.plot(z * 1e6, dia_f / mag_eff, label=r'$d_{f}$')

            ax1.set_xlabel('z')
            ax1.set_ylabel(r'$d_{e} \: (\mu m)$')
            # ax1.set_ylim([-1, 21])
            ax1.legend()

            plt.title("{} ~ f(z = (-{}, {}))".format(y2, z_fit_lim, z_fit_lim))
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
            plt.close()

    # store
    all_nas.append(fit_nas)

# package
z_fit_lim_labels = ['z' + str(z) for z in z_fit_lims]
na_eff_columns = ['id'] + z_fit_lim_labels
data = np.array(all_nas)
dfna = pd.DataFrame(data, columns=na_eff_columns)
dfna['bin'] = 1

dfg = dfna.groupby('bin').mean()
dfstd = dfna.groupby('bin').std()

print(dfg)
print(dfstd)
j = 1

# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup


# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup


print("Analysis completed without errors.")