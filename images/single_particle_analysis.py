from images import processing
from utils import fit, functions, io
from images import implotting
from correction import correct

import os
import pandas as pd
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, griddata
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF9500'
sciorange = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# run
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
           'analyses/results-06.14.22_10X-spct'
base_results = base_dir + '/results'
image_path = base_dir + '/tracking/tiffs'
filetype = '.tiff'
split_strings = ('template_pid', '_z_true')
sort_strings = (split_strings[1], filetype)

# image data
particle_ids = [23, 22, 21, 16, 13, 14, 11, 0]
bkg_mean = 110
microns_per_pixels = 3.2

# per-particle from template.tiffs
per_particle_analysis = False
if per_particle_analysis:
    for particle_id in particle_ids:
        # create directory to save to
        # setup
        path_results = base_results + '/pid{}'.format(particle_id)
        if not os.path.exists(path_results):
            os.makedirs(path_results)

        # get files
        nums_and_files = processing.get_files(image_path, particle_id, split_strings, sort_strings, filetype='tiff')
        images = processing.read_images(image_path, nums_and_files, average_stack=True)

        data = []
        for fnum, img in images.items():
            # data.append(img)
            peak_intensity = np.max(img)
            """ hist, bin_edges = processing.histogram_image(img, density=False, show=True)"""
    
            popt, img_norm = processing.fit_2d_gaussian_on_image(img, normalize=True, bkg_mean=bkg_mean)
            A, xc, yc, sigmax, sigmay = popt
    
            # calculate the fit error
            XYZ, fZ, rmse, r_squared, residuals = processing.evaluate_fit_2d_gaussian_on_image(img_norm, popt)
    
            # reshape
            x, y, z, fz = processing.reshape_flattened(img, XYZ, fZ)
            fz_residuals = np.reshape(residuals, np.shape(img))
    
            # plot Gaussian
            """fig, ax = implotting.plot_gaussian_3d(x, y, fz)
            plt.show()
            plt.close()"""
    
            # plot image + Gaussian
            if fnum % 25 == 0:
                fig, (ax1, ax2) = implotting.plot_image_and_gaussian_3d(x, y, z, fz)
                plt.savefig(path_results + '/pid{}_z{}_img_and_gaussian_3d.svg'.format(particle_id, fnum))
                plt.close()
    
                fig, (ax1, ax2) = implotting.plot_image_and_gaussian_2d(z, fz_residuals)
                plt.savefig(path_results + '/pid{}_z{}_img_and_residuals_2d.svg'.format(particle_id, fnum))
                plt.close()
    
            # store data
            data.append([particle_id, fnum, r_squared, rmse, bkg_mean, peak_intensity, A, xc, yc, sigmax, sigmay])

        # dataframe
        df = pd.DataFrame(np.array(data), columns=['id', 'z', 'r_squared', 'rmse', 'bkg_mean', 'peak_int',
                                                   'A', 'xc', 'yc', 'sigmax', 'sigmay'])
        df.to_excel(path_results + '/pid{}_fit-gaussian.xlsx'.format(particle_id))
    
        # plot: sigmax, sigmay
        fig, (ax1, ax2) = plt.subplots(ncols=2)
    
        ax1.plot(df.z, df.sigmax * microns_per_pixels)
        ax1.set_xlabel(r'$z \: (\mu m)$')
        ax1.set_ylabel(r'$w_{x} \: (\mu m)$')
    
        ax2.plot(df.z, df.sigmay * microns_per_pixels)
        ax2.set_xlabel(r'$z \: (\mu m)$')
        ax2.set_ylabel(r'$w_{y} \: (\mu m)$')
    
        plt.tight_layout()
        plt.savefig(path_results + '/pid{}_gauss-sigma_by_z.svg'.format(particle_id))
        plt.close()
    
        # plot: normalized sigma x/y
        fig, ax = plt.subplots()
    
        ax.plot(df.z, df.sigmax / df.sigmay)
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel(r'$w_{x}/w_{y}$')
    
        plt.tight_layout()
        plt.savefig(path_results + '/pid{}_gauss-norm-sigma_by_z.svg'.format(particle_id))
        plt.close()
    
        # plot: peak intensity + gaussian amplitude
        fig, ax = plt.subplots()
        ax.plot(df.z, df.peak_int, label=r'$I_{max}$')
        ax.plot(df.z, df.A, label=r'$A$')
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_ylabel('Pixel Intensity (A.U.)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_results + '/pid{}_peak-int_and_gauss-A_by_z.svg'.format(particle_id))
        plt.close()
    
        # fit gaussian to intensity profile to get z_focus and plot residuals
        arr_z = df.z.to_numpy()
        arr_int = df.peak_int.to_numpy()
        popt, pcov, fit_function = fit.fit(arr_z, arr_int, fit_function=fit.gauss_1d_function)
        fit_z = np.linspace(arr_z.min(), arr_z.max(), 500)
        fA = fit.gauss_1d_function(fit_z, *popt)
        fAz = fit_z[np.argmax(fA)]
        residuals = functions.calculate_residuals(fit.gauss_1d_function(arr_z, *popt), arr_int)
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches),
                                       gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(df.z, df.peak_int)
        ax1.plot(fAz, np.max(arr_int), 'o', ms=3, color='blue', label=r'$z_{f}=$' +
                                                                      ' {} '.format(np.round(fAz, 2)) +
                                                                      r'$\mu m$')
        ax1.set_ylabel(r'$A$ (A.U.)')
        ax1.legend()
    
        ax2.plot(arr_z, residuals, '.', ms=1, color='black')
        ax2.set_xlabel(r'$z \: (\mu m)$')
        ax2.set_ylabel('Residual')
    
        plt.tight_layout()
        plt.savefig(path_results + '/pid{}_peak-int_and_residual_by_z.svg'.format(particle_id))
        plt.close()
    
    
        # plot: gaussian x and y
        # calculate apparent x and y from 'best focus'
        arr_x = df.xc.to_numpy()
        xf = arr_x[df.peak_int.idxmax()]
        arr_xa = arr_x - xf
        arr_y = df.yc.to_numpy()
        yf = arr_y[df.peak_int.idxmax()]
        arr_ya = arr_y - yf
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))
    
        ax1.plot(arr_z, arr_xa * microns_per_pixels, '.', ms=1, color='black')
        ax1.set_ylabel(r'$\Delta^{\prime}_{x} \: (\mu m)$')
    
        ax2.plot(arr_z, arr_ya * microns_per_pixels, '.', ms=1, color='black')
        ax2.set_ylabel(r'$\Delta^{\prime}_{y} \: (\mu m)$')
        ax2.set_xlabel(r'$z \: (\mu m)$')
    
        plt.tight_layout()
        plt.savefig(path_results + '/pid{}_apparent_x_y.svg'.format(particle_id))
        plt.close()

# ---

# spct stats
plot_spct_stats = False
if plot_spct_stats:
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/' \
         'results-06.14.22_10X-spct/coords/calib-coords/calib_spct_stats_02.06.22_membrane_characterization_calib_spct-cal.xlsx'
    df = pd.read_excel(fp)
    df = df.dropna()

    # plot sigma_xy as ellipses
    plot_ellipses = False
    if plot_ellipses:
        df = df[(df['gauss_sigma_x_y'] > 0.7) & (df['gauss_sigma_x_y'] < 1.3)]
        vmin = df.gauss_sigma_x_y.min()
        vmax = df.gauss_sigma_x_y.max()

        for zt in [2, 272, 128, 120, 124, 138, 136, 140, 150, 148, 152, 154, 88, 48, 8, 188, 238, 258]:
            dfz = df[df['z_true'] == zt]
            dfz = dfz[dfz['x'] > 220]
            dfz = dfz[(dfz['gauss_sigma_x_y'] > 0.7) & (dfz['gauss_sigma_x_y'] < 1.3)]

            x = dfz.x.to_numpy()
            y = dfz.y.to_numpy()
            wx = dfz.gauss_sigma_x.to_numpy() * 3
            wy = dfz.gauss_sigma_y.to_numpy() * 3
            wxy = dfz.gauss_sigma_x_y.to_numpy()

            # normalize
            w_scale = 30
            w_min = np.min([wx.min(), wy.min()])
            w_max = np.max([wx.max(), wy.max()])
            wx_norm = wx * w_scale / w_max  # (wx - w_min) / (w_max - w_min) * w_scale
            wy_norm = wy * w_scale / w_max  # (wy - w_min) / (w_max - w_min) * w_scale
            wxy_norm = (wxy - wxy.min()) / (wxy.max() - wxy.min())

            # color map
            RdBu = cm.get_cmap('RdBu', 8)
            cmap = mpl.cm.RdBu
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            # ellipses
            ells = [Ellipse(xy=(x[i], y[i]), width=wx_norm[i], height=wy_norm[i], angle=0) for i in range(len(x))]

            fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
            for ii, e in enumerate(ells):
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(1)
                e.set_facecolor(RdBu(wxy_norm[ii]))

            ax_lbls = [0, int(np.ceil(256 * microns_per_pixels)), int(np.ceil(512 * microns_per_pixels))]
            ax.set_xlim([225, 712])
            # ax.set_xticks([0, 256, 512], labels=ax_lbls)
            ax.set_xlabel(r'$x \: (\mu m)$')
            ax.set_ylim([300, 600])
            # ax.set_yticks([0, 256, 512], labels=ax_lbls)
            ax.set_ylabel(r'$y \: (\mu m)$')
            ax.set_title(r'$z = $' + ' {} '.format(zt) + r'$\mu m$')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=cax, orientation='vertical', label=r'$w_{x}/w_{y}$', extend='both')

            plt.tight_layout()
            plt.savefig(base_results + '/sigma_x_y_ellipses_at_z{}.png'.format(zt))
            plt.show()
            plt.close()

    # compare real to synthetic
    compare_real_to_synthetic = False
    if compare_real_to_synthetic:
        fps = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/grid-dz-overlap/tracking/noise-level-15/spct-no-dz/calibration-spct-get-spct-cal/calib_spct_stats_grid-dz_calib_nll2_spct_spct-cal.xlsx'
        dfs = pd.read_excel(fps)

        df = df[df['id'].isin([11, 22, 41, 46, 47, 73, 90])]
        dfg = df.groupby('frame').mean()
        dfgs = dfs.groupby('frame').mean()

        dfg = dfg[(dfg['z_corr'] > -50) & (dfg['z_corr'] < 50)]
        dfgs = dfgs[(dfgs['z_true'] > -50) & (dfgs['z_true'] < 50)]

        y1 = 'peak_int'
        ms = 3

        fig, (ax, axr) = plt.subplots(nrows=2, sharex=True)
        ax.plot(dfg.z_corr, dfg[y1], 'o', ms=ms, label='Experimental')
        ax.plot(dfgs.z_true + 4, dfgs[y1], 'o', ms=ms, label='Synthetic')
        ax.set_ylabel(r'$I_{max}$')
        ax.legend()

        axr.plot(dfg.z_corr, dfg.contour_area, 'o', ms=ms * 0.75, color=sciblue)
        axr.plot(dfgs.z_true + 4, dfgs.contour_area, 'o', ms=ms * 0.75, color=scigreen)
        axr.set_xlabel(r'$z \: (\mu m)$')
        axr.set_ylabel(r'$Area \: (pix.)$')
        plt.tight_layout()
        plt.savefig(base_results + '/compare_real_to_synthetic.svg')
        plt.show()
        plt.close()


# ---

# spct stats
plot_pid_focus_stats = False
if plot_pid_focus_stats:

    path_calib_coords = base_dir + '/calib-coords'
    path_figs = base_results
    path_results = base_results

    microns_per_pixel = 1.6
    img_xc, img_yc = 256, 256

    # 1. READ CALIB COORDS
    read_calib_coords = False

    if read_calib_coords:

        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method='spct')

        # ------------------------------------------------------------------------------------------------------------------
        # 1.1 COMPUTE INITIAL SURFACE CORRECTION
        compute_initial_surface_correction = True

        if compute_initial_surface_correction:
            param_zf = 'zf_from_peak_int'
            kx = 2
            ky = 2

            dfcpid = dfcpid[(dfcpid['zf_from_peak_int'] < 53) & (dfcpid['zf_from_peak_int'] > 48)]
            dfcpid = dfcpid[(dfcpid['zf_nearest_calib'] < 53) & (dfcpid['zf_nearest_calib'] > 48)]

            dict_fit_plane, dict_fit_plane_bspl_corrected, dfcal_field_curvature_corrected, bispl = \
                correct.fit_plane_correct_plane_fit_spline(dfcal=dfcpid,
                                                           param_zf=param_zf,
                                                           microns_per_pixel=microns_per_pixel,
                                                           img_xc=img_xc,
                                                           img_yc=img_yc,
                                                           kx=kx,
                                                           ky=ky,
                                                           path_figs=path_figs)

    else:
        fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/analyses/results-06.11.22_SPCT_1um-calib/results/calib_spct_pid_defocus_stats_field-curvature-corrected_xy.xlsx'
        df = pd.read_excel(fp)

    dfcpid = df

    plot_surface_2d = False
    plot_surface_3d = True

    zf_param = 'z_corr'
    num = 200
    vmin = -0.7
    vmax = 0.7

    # processing
    dfcpid = dfcpid[(dfcpid[zf_param] < vmax) & (dfcpid[zf_param] > vmin)]

    # get data arrays
    x = dfcpid.x.to_numpy()
    y = dfcpid.y.to_numpy()
    z = dfcpid[zf_param].to_numpy()

    if plot_surface_2d:

        fig, ax = plt.subplots()
        sc = ax.scatter(x, y, c=z, s=2, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        fig.colorbar(sc, cax=cax, orientation='vertical', label=r'$z^{f} \: (\mu m)$', extend='both')

        ax_lbls = [0, int(np.ceil(256 * microns_per_pixels)), int(np.ceil(512 * microns_per_pixels))]
        ax.set_xlim([0, 512])
        ax.set_xticks([0, 256, 512], labels=ax_lbls)
        ax.set_xlabel(r'$x \: (\mu m)$')
        ax.set_ylim([0, 512])
        ax.set_yticks([0, 256, 512], labels=ax_lbls)
        ax.set_ylabel(r'$y \: (\mu m)$')
        plt.tight_layout()

        plt.savefig(base_results + 'zf_2d-surface.png')
        plt.show()
        plt.close()

    # plot 3D
    if plot_surface_3d:
        xr = (x.min(), x.max())
        yr = (y.min(), y.max())
        grid_x, grid_y = np.mgrid[xr[0]:xr[1]:200j, yr[0]:yr[1]:200j]

        points = (x, y)
        values = z
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

        for view_height in [25, 90, 15, 0]:
            for view_azi in [-90, -70]:

                fig, ax = plt.subplots(figsize=(size_x_inches * 1.5, size_y_inches * 1.5),
                                       subplot_kw={"projection": "3d"})

                ax.scatter(x, y, z, s=5, marker='.', color='black', alpha=0.5)
                surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.coolwarm, alpha=0.95,
                                       linewidth=0, antialiased=False)

                fig.colorbar(surf, ax=ax, aspect=20, shrink=0.5, location='right', pad=-0.25, panchor=(0, -0.25))
                ax.set_zlim([-3, 3])

                ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.grid(False)

                for line in ax.xaxis.get_ticklines():
                    line.set_visible(False)
                for line in ax.yaxis.get_ticklines():
                    line.set_visible(False)
                for line in ax.zaxis.get_ticklines():
                    line.set_visible(False)

                ax.xaxis.set_ticks([], labels=[])
                ax.yaxis.set_ticks([], labels=[])
                ax.zaxis.set_ticks([], labels=[])

                ax.view_init(view_height, view_azi)

                plt.savefig(base_results + 'zf_3d-surface_height{}_angle{}.png'.format(view_height, view_azi))
                # plt.show()
                plt.close()

# ---

print("Analysis completed without errors.")