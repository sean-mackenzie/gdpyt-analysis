# gdpyt-analysis: test.test_fit_3dplane
"""
Notes
"""

# imports
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from correction import correct
from utils import fit, plotting, functions


# read dataframe
fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.07.22_membrane_characterization/analysis/boundary_points_in-focus.xlsx'
df = pd.read_excel(fp)

# get boundary particle ids
boundary_pids = df.id.unique()

# convert pixels to microns
microns_per_pixel = 1.6
df['x_um'] = df['x'] * microns_per_pixel
df['y_um'] = df['y'] * microns_per_pixel

# plot
"""fig, ax = plotting.plot_scatter_3d(df, fig=None, ax=None, elev=5, azim=-40, color=None, alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.show()"""

# ----------------------------------------------------------------------------------------------------------------------
# fit plane (units = microns)

points = np.stack((df.x_um, df.y_um, df.z_f_calc)).T
px, py, pz, popt = fit.fit_3d(points, fit_function='plane')
pz = np.round(pz, 9)

z1 = functions.calculate_z_of_3d_plane(px[0, 0], py[0, 0], popt)
z2 = functions.calculate_z_of_3d_plane(px[0, 1], py[0, 0], popt)
z3 = functions.calculate_z_of_3d_plane(px[1, 1], py[1, 1], popt)

d, normal = popt[3], popt[4]
# 0 = normal[0] * x + normal[1] * y + normal[2] * z + d
# z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.x_um, df.y_um, df.z_f_calc, color='cornflowerblue')
ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue')
ax.set_xlabel(r'$x \: (\mu m)$')
ax.set_ylabel(r'$y \: (\mu m)$')
ax.set_zlabel(r'$z \: (\mu m)$')
ax.view_init(5, 45)
ax.set_title(r"$0 = n_x x + n_y y + n_z z + d$" + "\n= {}x + {}y + {}z + {}".format(np.round(normal[0], 3),
                                                                                   np.round(normal[1], 3),
                                                                                   np.round(normal[2], 3),
                                                                                   np.round(d, 3)))
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# fit plane (x, y units = pixels; z units = microns)

points = np.stack((df.x, df.y, df.z_f_calc)).T
px, py, pz, popt = fit.fit_3d(points, fit_function='plane')
pz = np.round(pz, 9)

# ----------------------------------------------------------------------------------------------------------------------
# test correct.correct_z_by_fit_function(df, fit_func, popt, x_param='x', y_param='y', z_param='z')
correct_boundary_particles = False

if correct_boundary_particles:

    correct_by_plane = True
    if correct_by_plane:
        dfc = correct.correct_z_by_fit_function(df, fit_func=functions.calculate_z_of_3d_plane,
                                                popt=popt, x_param='x', y_param='y', z_param='z_f_calc')
    else:
        dfc = df.copy()
        dfc['z_corr'] = dfc['z'] + dfc['z'] - dfc['z_f_calc']

    # plot 3D scatter: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.x, df.y, df.z, color='black', s=2, marker='.', label='z')
    ax.scatter(df.x, df.y, df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
    ax.scatter(dfc.x, dfc.y, dfc.z_corr, color='red', s=4, marker='d', label=r'$z_{corrected}$')
    #ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue')
    ax.set_xlabel(r'$x \: (pix)$')
    ax.set_ylabel(r'$y \: (pix)$')
    ax.set_zlabel(r'$z \: (pix)$')
    ax.view_init(5, 45)
    ax.set_title("Corrected")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # plot 2D scatter along x-y axes: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
    x_line = np.linspace(df.x.min(), df.x.max())
    y_line = np.linspace(df.y.min(), df.y.max())
    zx_line = functions.calculate_z_of_3d_plane(x=x_line, y=np.zeros_like(x_line), popt=popt)
    zy_line = functions.calculate_z_of_3d_plane(x=np.zeros_like(y_line), y=y_line, popt=popt)
    zz_line = functions.calculate_z_of_3d_plane(x=x_line, y=y_line, popt=popt)

    fig, [ax1, ax2] = plt.subplots(nrows=2)
    ax1.scatter(df.x, df.z, color='black', s=2, marker='.', label='z')
    ax1.scatter(df.x, df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
    ax1.scatter(dfc.x, dfc.z_corr, color='red', s=4, marker='d', label=r'$z_{corrected}$')
    ax1.plot(x_line, zx_line, color='gray', alpha=0.25, linestyle='--', label=r'$z_{plane, x}$')
    ax1.plot(x_line, zz_line, color='gray', alpha=0.5, linestyle='--', label=r'$z_{plane, xy}$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.scatter(df.y, df.z, color='black', s=2, marker='.', label='z')
    ax2.scatter(df.y, df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
    ax2.scatter(dfc.y, dfc.z_corr, color='red', s=4, marker='d', label=r'$z_{corrected}$')
    ax2.plot(y_line, zy_line, color='gray', alpha=0.25, linestyle='--', label=r'$z_{plane, y}$')
    ax2.plot(y_line, zz_line, color='gray', alpha=0.5, linestyle='--', label=r'$z_{plane, xy}$')
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# test correct.correct_z_by_fit_function(df, fit_func, popt, x_param='x', y_param='y', z_param='z')
correct_all_particles = True

if correct_all_particles:

    # read dataframe
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.07.22_membrane_characterization/analysis/all_points_in-focus.xlsx'
    df = pd.read_excel(fp)

    dfc = correct.correct_z_by_fit_function(df, fit_func=functions.calculate_z_of_3d_plane,
                                            popt=popt, x_param='x', y_param='y', z_param='z_f_calc')

    # plot 3D scatter: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.x, df.y, df.z, color='black', s=2, marker='.', label='z')
    ax.scatter(df.x, df.y, df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
    ax.scatter(dfc.x, dfc.y, dfc.z_corr, color='red', s=4, marker='d', label=r'$z_{corrected}$')
    # ax.plot_surface(px, py, pz, alpha=0.4, color='mediumblue')
    ax.set_xlabel(r'$x \: (pix)$')
    ax.set_ylabel(r'$y \: (pix)$')
    ax.set_zlabel(r'$z \: (microns)$')
    ax.view_init(5, 45)
    ax.set_title("Corrected")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # plot 2D scatter along x-y axes: z (original; image acquisition), z_f_calc (in-focus), z_corr (corrected)
    x_line = np.linspace(df.x.min(), df.x.max())
    y_line = np.linspace(df.y.min(), df.y.max())
    zx_line = functions.calculate_z_of_3d_plane(x=x_line, y=np.zeros_like(x_line), popt=popt)
    zy_line = functions.calculate_z_of_3d_plane(x=np.zeros_like(y_line), y=y_line, popt=popt)
    zz_line = functions.calculate_z_of_3d_plane(x=x_line, y=y_line, popt=popt)

    fig, [ax1, ax2] = plt.subplots(nrows=2)
    ax1.scatter(df.x, df.z, color='black', s=2, marker='.', label='z')
    ax1.scatter(df.x, df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
    ax1.scatter(dfc.x, dfc.z_corr, color='red', s=4, marker='d', label=r'$z_{corrected}$')
    ax1.plot(x_line, zx_line, color='gray', alpha=0.25, linestyle='--', label=r'$z_{plane, x}$')
    ax1.plot(x_line, zz_line, color='gray', alpha=0.5, linestyle='--', label=r'$z_{plane, xy}$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.scatter(df.y, df.z, color='black', s=2, marker='.', label='z')
    ax2.scatter(df.y, df.z_f_calc, color='cornflowerblue', s=2, label=r'$z_f$')
    ax2.scatter(dfc.y, dfc.z_corr, color='red', s=4, marker='d', label=r'$z_{corrected}$')
    ax2.plot(y_line, zy_line, color='gray', alpha=0.25, linestyle='--', label=r'$z_{plane, y}$')
    ax2.plot(y_line, zz_line, color='gray', alpha=0.5, linestyle='--', label=r'$z_{plane, xy}$')
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    # plot 3D scatter with colorbar: z_corr (corrected) == the "true" z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(dfc.x, dfc.y, dfc.z_corr, c=dfc.z_corr, s=4)
    ax.set_xlabel(r'$x \: (pix)$')
    ax.set_ylabel(r'$y \: (pix)$')
    ax.set_zlabel(r'$z \: (microns)$')
    ax.view_init(5, 45)
    ax.set_title("True z")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # correct z such that z=0 is the in-focus position
    dfc['z_corr2'] = dfc['z_f_calc'] - dfc['z']

    # plot 2D scatter along x-y axes: z
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    ax1.scatter(dfc.x, dfc.z_corr2, s=3,)
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    ax2.scatter(dfc.y, dfc.z_corr2, s=3, )
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    plt.tight_layout()
    plt.show()

    # plot 3D scatter with colorbar: z_corr (corrected) == the "true" z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(dfc.x, dfc.y, dfc.z_corr2, c=dfc.z_corr2, s=4)
    ax.set_xlabel(r'$x \: (pix)$')
    ax.set_ylabel(r'$y \: (pix)$')
    ax.set_zlabel(r'$z \: (microns)$')
    ax.view_init(5, 45)
    ax.set_title("True z corrected to focal plane")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # translate z such that z=0 is the average of the boundary particles
    boundary_pids_mean_z = dfc[dfc['id'].isin(boundary_pids)].z_corr2.mean()
    dfc['z_corr3'] = dfc['z_corr2'] - boundary_pids_mean_z

    # plot 2D scatter along x-y axes: z
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    ax1.scatter(dfc.x, dfc.z_corr3, s=3,)
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    ax1.set_title(r'$\Delta z \: =$' + ' {} '.format(np.round(boundary_pids_mean_z, 2)) + r'$\mu m$')
    ax2.scatter(dfc.y, dfc.z_corr3, s=3, )
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    plt.tight_layout()
    plt.show()

    # plot 3D scatter with colorbar: z_corr (corrected) == the "true" z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(dfc.x, dfc.y, dfc.z_corr3, c=dfc.z_corr3, s=4)
    ax.set_xlabel(r'$x \: (pix)$')
    ax.set_ylabel(r'$y \: (pix)$')
    ax.set_zlabel(r'$z \: (microns)$')
    ax.view_init(5, 45)
    ax.set_title("True z corrected to z=0 to average boundary z")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Export to Excel
    save_path = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.07.22_membrane_characterization/analysis'
    save_corrections = join(save_path, 'per_particle_corrections.xlsx')
    save_correction_params = join(save_path, 'correction_dict.xlsx')

    # export per-particle corrections to Excel
    dfc['z_correct'] = dfc['z_corr3'] - dfc['z']
    df_per_particle_corrections = dfc[['id', 'x', 'y', 'z_correct']]
    df_per_particle_corrections.to_excel(save_corrections)


    # ------------------------------------------------------------------------------------------------------------------
    # export correction parameters to dictionary and save as Excel
    correction_dict = correct.package_correction_params(correction_type='tilt+offset',
                                                        function='calculate_z_of_3d_plane',
                                                        popt=popt,
                                                        z_offset=boundary_pids_mean_z,
                                                        save_path=save_correction_params)


j = 1