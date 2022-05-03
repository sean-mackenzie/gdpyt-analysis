# gdpyt-analysis: test.test_fit_3dsphere
"""
Notes
"""

# imports
from os.path import join
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from correction import correct
from utils import fit, plotting, functions


# read dataframe
fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.07.22_membrane_characterization/analysis/tests/compare-interior-particles-per-test/' \
     'df_id11.xlsx'
df = pd.read_excel(fp)

microns_per_pixel = 1.6
correctX = df.x.to_numpy()
correctY = df.y.to_numpy()
correctZ = df.z_corr.to_numpy()

raw_data = np.stack([correctX, correctY, correctZ]).T
xc = 498 * microns_per_pixel
yc = 253 * microns_per_pixel
zc = 3
r_edge = 500 * microns_per_pixel


# fit a sphere to 3D points
def fit_sphere(spX, spY, spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)

    A = np.zeros((len(spX), 4))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = spZ * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]


# fit a sphere to 3D points
def fit_spherexy(spX, spY, spZ, xc, yc):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)

    A = np.zeros((len(spX), 2))
    A[:, 0] = spZ * 2
    A[:, 1] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ) - (2 * spX * xc) - (2 * spY * yc) # + xc ** 2 + yc ** 2

    # least squares fit
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (xc**2) + (yc**2) + (C[0] * C[0]) + C[1]
    radius = math.sqrt(t)

    return radius, C[0]

def fit_ellipsoid_from_center(X, Y, Z, xc, yc, zc, r):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    f = np.zeros((len(X), 1))
    f[:, 0] = -1 * ((Z * Z) - (2 * zc * Z) + (zc * zc))

    A = np.zeros((len(X), 1))
    A[:, 0] = ((X * X) - (2 * xc * X) + (xc * xc) + (Y * Y) - (2 * yc * Y) + (yc * yc)) / (r * r) - 1

    # least squares fit
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    # solve for radius in z-dir.
    r_z = math.sqrt(C[0])

    return r_z


def calc_spherical_angle(r, xyz):
    """
    Given a point (x, y, z) approx. on a sphere of radius (r), return the angle phi and theta of that point.

    :param r:
    :param xyz:
    :return:
    """
    x, y, z = xyz[0], xyz[1], xyz[2]

    if np.abs(z) > r:
        return np.nan, np.nan
    else:
        phi = np.arccos(z / r)

        if x < 0 and y < 0:
            theta_half = np.arccos(x / (r * np.sin(phi)))
            theta_diff = np.pi - theta_half
            theta = np.pi + theta_diff
        else:
            theta = np.arccos(x / (r * np.sin(phi)))

    return phi, theta


# fit 3d ellipsoid
r_z = fit_ellipsoid_from_center(correctX, correctY, correctZ, xc, yc, zc, r_edge)

# general 3d sphere fit
rr, xx0, yy0, zz0 = fit_sphere(correctX, correctY, correctZ)

# custom 3d sphere fit
r, z0 = fit_spherexy(correctX, correctY, correctZ, xc, yc)
x0, y0 = xc, yc

phis = []
thetas = []
for i in range(raw_data.shape[0]):
    x, y, z, = raw_data[i, 0], raw_data[i, 1], raw_data[i, 2]
    dx = x - x0
    dy = y - y0
    dz = z - z0

    if x < x0 * 0.5:
        phi, theta = calc_spherical_angle(r, xyz=(dx, dy, dz))
        if any([np.isnan(phi), np.isnan(theta)]):
            continue
        else:
            # phis.append(phi)
            thetas.append(theta)

    if x < x0:
        phi, theta = calc_spherical_angle(r, xyz=(dx, dy, dz))
        if any([np.isnan(phi), np.isnan(theta)]):
            continue
        else:
            phis.append(phi)

phis = np.array(phis)
thetas = np.array(thetas)

# ----------------------------------- PLOTTING ELLIPSOID
custom_ellipsoid = True

if custom_ellipsoid:
    u = np.linspace(thetas.min(), thetas.max(), 20)
    v = np.linspace(0, np.pi/2, 20)
    u, v = np.meshgrid(u, v)

    xe = r_edge * np.cos(u) * np.sin(v)
    ye = r_edge * np.sin(u) * np.sin(v)
    ze = r_z * np.cos(v)

    xe = xe.flatten() + xc
    ye = ye.flatten() + yc
    ze = ze.flatten() + zc

    # --- plot sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(x, y, z, color="r")
    #ax.plot_surface(xe, ye, ze, cmap='coolwarm', alpha=0.5)
    ax.scatter(xe, ye, ze, zdir='z', s=20, c='r', rasterized=True)

    ax.scatter(correctX, correctY, correctZ, zdir='z', s=2, c='b', rasterized=True, alpha=0.25)

    ax.set_xlabel(r'$x \: (\mu m)$')
    ax.set_ylabel(r'$y \: (\mu m)$')
    ax.set_zlabel(r'$z \: (\mu m)$')
    ax.view_init(15, 255)
    plt.show()

raise ValueError('ah')


# ----------------------------------- PLOTTING SPHERES
gen_sphere, custom_sphere = True, True

# --- calculate points on sphere
if custom_sphere:
    u, v = np.mgrid[thetas.min():thetas.max():20j, 0:phis.max():20j]
    x=np.cos(u)*np.sin(v)*r
    y=np.sin(u)*np.sin(v)*r
    z=np.cos(v)*r
    x = x + x0
    y = y + y0
    z = z + z0

    # --- plot sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(x, y, z, color="r")
    ax.plot_surface(x, y, z, cmap='coolwarm', alpha=0.5)
    ax.scatter(correctX, correctY, correctZ, zdir='z', s=20, c='b', rasterized=True)
    ax.set_xlabel(r'$x \: (\mu m)$')
    ax.set_ylabel(r'$y \: (\mu m)$')
    ax.set_zlabel(r'$z \: (\mu m)$')
    ax.view_init(15, 255)
    plt.show()

    # plot sphere viewed from above
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='coolwarm', alpha=0.5)
    ax.scatter(correctX, correctY, correctZ, zdir='z', s=20, c='b', rasterized=True)
    ax.set_xlabel(r'$x \: (\mu m)$')
    ax.set_ylabel(r'$y \: (\mu m)$')
    ax.set_zlabel(r'$z \: (\mu m)$')
    ax.view_init(90, 255)
    plt.show()

if gen_sphere:
    x2 = np.cos(u) * np.sin(v) * rr
    y2 = np.sin(u) * np.sin(v) * rr
    z2 = np.cos(v) * rr
    x2 = x2 + xx0
    y2 = y2 + yy0
    z2 = z2 + zz0

    # plot spheres
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='coolwarm', alpha=0.5)
    ax.plot_surface(x2, y2, z2, cmap='cool', alpha=0.5)
    # ax.scatter(correctX, correctY, correctZ, zdir='z', s=20, c='b', rasterized=True)
    ax.set_xlabel(r'$x \: (\mu m)$')
    ax.set_ylabel(r'$y \: (\mu m)$')
    zlabel = ax.set_zlabel(r'$z \: (\mu m)$')
    ax.view_init(15, 255)
    plt.show()

j = 1