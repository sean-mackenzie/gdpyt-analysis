# Plot Ellipses

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

from utils import functions, bin

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'

sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

# file paths
base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_particle_image_ellipses'
path_read = join(base_dir, 'data')
path_save = join(base_dir, 'figs')

# file name
fn = 'calib_spct_stats_11.06.21_z-micrometer-v2_1umSteps_pdf.xlsx'

# setup
microns_per_pixels = 1.6
image_pixels = 512
image_padding = 5

# ---

# read file
df = pd.read_excel(join(path_read, fn))
df = df.dropna(subset=['gauss_sigma_x_y'])

# modifiers
normalize = True
plot_rotated = 'pdf'
filetype = '.svg'
dpi = 300

# plot variables
if plot_rotated == 'pdf':
    angle = -45
    px = 'pdf_sigma_x'
    py = 'pdf_sigma_y'
    pxy = 'pdf_x_y'
elif plot_rotated:
    angle = -45
    px = 'gauss_sigma_x_r'
    py = 'gauss_sigma_y_r'
    pxy = 'gauss_sigma_x_y_r'
else:
    angle = 0
    px = 'gauss_sigma_x'
    py = 'gauss_sigma_y'
    pxy = 'gauss_sigma_x_y'

# ---

# processing
max_pid_baseline = 87
min_num_frames = 50

# filters
df = df[df['id'] < max_pid_baseline]

# plot sigma_xy as ellipses
particle_ids_doublets = [6, 8, 17, 26, 35, 37, 58, 77]
particle_ids_bad = [25, 29, 36, 52, 59, 66, 87]
df = df[~df.id.isin([6, 8, 17, 25, 26, 29, 35, 36, 37, 52, 58, 59, 66, 77, 87])]

pxy = 'pdf_x_y'
if pxy == 'pdf_x_y':
    vmin = 0.875  # dfp[pxy].min()
    vmax = 1.125  # dfp[pxy].max()
    cbar_lbl = r'$w_{x}/w_{y}$'
elif pxy == 'pdf_rho':
    vmin = -0.15
    vmax = 0.15
    cbar_lbl = r'$\rho$'

# figure + color map
plot_labels = False
plot_colorbar = False
scale_ellipse_width = 75
raise_to_power = 3
cmap = mpl.cm.coolwarm
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# plot z positions
plot_dz = 5
# plot_zs = df.z_true.unique()[::plot_dz]
plot_zs = [20, 40, 50, 60, 80]

plot_ellipses_on_image = False
if plot_ellipses_on_image:
    for zt in plot_zs:
        dfz = df[df['z_true'] == zt]

        # get ellipse axes
        x = dfz.x.to_numpy()
        y = dfz.y.to_numpy()
        wx = dfz[px].to_numpy() * microns_per_pixels
        wy = dfz[py].to_numpy() * microns_per_pixels
        wxy = dfz[pxy].to_numpy()

        # collect stats
        num_particles = len(dfz)

        # normalize
        if normalize:
            magnitude_wx_wy = np.sqrt(wx ** 2 + wy ** 2)
            wx_norm = (wx / magnitude_wx_wy) ** raise_to_power * scale_ellipse_width
            wy_norm = (wy / magnitude_wx_wy) ** raise_to_power * scale_ellipse_width
            wxy_norm = wxy  # (wxy - wxy.min()) / (wxy.max() - wxy.min())

        # ellipses
        ells = [Ellipse(xy=(x[i], y[i]), width=wx_norm[i], height=wy_norm[i], angle=angle) for i in range(len(x))]

        fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5),
                               subplot_kw={'aspect': 'equal'}, frameon=False)
        if not plot_labels:
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False, top=False)

        for ii, e in enumerate(ells):
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor(cmap(norm(wxy_norm[ii])))  # RdBu(wxy_norm[ii])
            e.set_edgecolor('black')
            e.set_linewidth(0.25)

        ax.set_xlim([0 + image_padding, 512 + image_padding])
        ax.set_ylim([0 + image_padding, 512 + image_padding])

        if plot_labels:
            ax.set_xticks([0 + image_padding, 512 + image_padding], labels=[0, 512])
            ax.set_yticks([0 + image_padding, 512 + image_padding], labels=[0, 512])
            ax.set_xlabel(r'$X$')
            ax.set_ylabel(r'$Y$')
        # ax.set_title(r'$N_{p}, z = $' + '{}, {} '.format(num_particles, zt) + r'$\mu m$')

        plt.minorticks_off()

        if plot_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=cax, orientation='vertical', label=cbar_lbl, extend='both')

        plt.tight_layout()
        plt.savefig(path_save +
                    '/{}-power{}_ellipses_at_rotate{}deg_z{}'.format(pxy, raise_to_power, angle, zt) + filetype,
                    dpi=dpi, bbox_inches='tight')
        # plt.show()
        plt.close()

# ---

plot_single_ellipses = False
if plot_single_ellipses:
    plot_single_pids = [0, 7, 11, 39, 44, 54, 83, 84, 85]
    nz = len(plot_zs)

    # ---

    if pxy == 'pdf_x_y':
        scale_ellipse_width = 50
    elif pxy == 'pdf_rho':
        scale_ellipse_width = 75

    # ---

    def put_label(axes, xy, box_half_length, text):
        y = xy[1] - box_half_length / 7.5
        axes.text(xy[0], y, text, ha="center", family='serif', weight='black', size=10)

    # fig setup
    l_fig = 10.5

    # ---

    for pid in plot_single_pids:
        dfpid = df[df['id'] == pid]

        plot_colorbar = False
        if plot_colorbar:
            fig, ax = plt.subplots(ncols=nz,
                                   figsize=(size_x_inches * 1.125, size_y_inches),
                                   gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.405]},
                                   frameon=False)
        else:
            fig, ax = plt.subplots(ncols=nz,  figsize=(size_x_inches * 1.0875, size_y_inches * 0.3),
                                   frameon=False)
        plt.tick_params(axis='both', which='both',
                        left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False, top=False)

        for j, zt in enumerate(plot_zs):

            # get this 'z'
            dfz = dfpid[dfpid['z_true'] == zt]

            # get ellipse axes
            x = dfz.x.to_numpy()
            y = dfz.y.to_numpy()
            wx = dfz[px].to_numpy() * microns_per_pixels
            wy = dfz[py].to_numpy() * microns_per_pixels
            wxy = dfz[pxy].to_numpy()

            # normalize
            magnitude_wx_wy = np.sqrt(wx ** 2 + wy ** 2)
            wx_norm = (wx / magnitude_wx_wy) ** raise_to_power * scale_ellipse_width
            wy_norm = (wy / magnitude_wx_wy) ** raise_to_power * scale_ellipse_width
            wxy_norm = wxy

            # ellipses
            ells = [Ellipse(xy=(x[i], y[i]), width=wx_norm[i], height=wy_norm[i], angle=angle) for i in range(len(x))]

            for ii, e in enumerate(ells):
                ax[j].add_artist(e)
                e.set_clip_box(ax[j].bbox)
                e.set_alpha(1)
                e.set_facecolor(cmap(norm(wxy_norm[ii])))  # RdBu(wxy_norm[ii])
                e.set_edgecolor('black')
                e.set_linewidth(0.25)

                # put text
                put_text = True
                if put_text:
                    put_label(axes=ax[j], xy=e.center, box_half_length=l_fig, text=np.round(wxy_norm[ii], 2))

                put_title = True
                put_title_every = True
                if put_title:
                    if put_title_every:
                        ax[j].set_title(r'$z=$' + ' {} '.format(zt - 50) + r'$\mu m$', fontsize=8)
                    else:
                        if j == 2:
                            ax[j].set_title(r'$z=$' + ' {} '.format(zt - 50) + r'$\mu m$')
                        else:
                            ax[j].set_title('{}'.format(zt - 50))

                ax[j].set_xlim([e.center[0] - l_fig, e.center[0] + l_fig])
                ax[j].set_ylim([e.center[1] - l_fig, e.center[1] + l_fig])
                ax[j].set_aspect('equal', 'box')
                ax[j].tick_params(axis='both', which='both',
                                  bottom=False, top=False, left=False, right=False,
                                  labelbottom=False, labeltop=False, labelleft=False, labelright=False)

                plot_labels = False
                if plot_labels:
                    ax.set_xticks([0 + image_padding, 512 + image_padding], labels=[0, 512])
                    ax.set_yticks([0 + image_padding, 512 + image_padding], labels=[0, 512])
                    ax.set_xlabel(r'$X$')
                    ax.set_ylabel(r'$Y$')
                    ax.set_title(r'$N_{p}, z = $' + '{}, {} '.format(num_particles, zt) + r'$\mu m$')

        if plot_colorbar:
            divider = make_axes_locatable(ax[j])
            cax = divider.append_axes("right", size="10%", pad=0.15)
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                         orientation='vertical', aspect=2.5, label=cbar_lbl, extend='both')

        plt.tight_layout()
        plt.savefig(path_save +
                    '/{}-power{}_ellipses_pid{}_rotate{}deg_z-range'.format(pxy, raise_to_power, pid, angle) + filetype,
                    dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close()

# ---

# --- END ELLIPSE PLOTTING

# ---

plot_particle_image_stats = False
if plot_particle_image_stats:

    # plot raw + errorbars
    plot_raw_image_stats = False
    if plot_raw_image_stats:
        group_by = 'z_true'
        plot_by = 'z_corr'
        plot_columns = ['peak_int', 'mean_int', 'snr', 'gauss_A', 'pdf_A',
                        'contour_area', 'contour_diameter', 'gauss_diameter']
        plot_clr = 'id'

        # ---

        # plot raw
        plot_all = False
        if plot_all:
            for pc in plot_columns:
                fig, ax = plt.subplots()
                ax.scatter(df[plot_by], df[pc], c=df[plot_clr], s=1)
                ax.set_xlabel(plot_by)
                ax.set_ylabel(pc)
                plt.tight_layout()
                plt.show()
                plt.close()

        # ---

        # ---

        # plot group by z
        plot_grouped = False
        if plot_grouped:

            dfgm = df.groupby(group_by).mean()
            dfgs = df.groupby(group_by).std()

            for pc in plot_columns:
                fig, ax = plt.subplots()
                ax.errorbar(dfgm[plot_by], dfgm[pc], yerr=dfgs[pc], fmt='-o', capsize=1.5, elinewidth=0.5)
                ax.set_xlabel(plot_by)
                ax.set_ylabel(pc)
                plt.tight_layout()
                plt.show()
                plt.close()

    # ---

    # optical model
    fit_model_to_images = False
    if fit_model_to_images:

        # model
        mag_eff = 10.01
        na_nominal = 0.45
        p_d = 2.15e-6
        wavelength = 600e-9
        pixel_size = 16e-6
        bkg_mean = 100
        bkg_noise = 5
        n0 = 1.0

        fPI = functions.fParticleImageOpticalTheory(magnification=mag_eff,
                                                    numerical_aperture=na_nominal,
                                                    particle_diameter=p_d,
                                                    wavelength=wavelength,
                                                    pixel_size=pixel_size,
                                                    bkg_mean=bkg_mean,
                                                    bkg_noise=bkg_noise,
                                                    n0=n0,
                                                    )

        # data
        group_by = 'z_true'
        plot_by = 'z_corr'
        plot_columns = ['peak_int', 'mean_int', 'snr', 'gauss_A', 'pdf_A',
                        'contour_area', 'contour_diameter', 'gauss_diameter']
        plot_clr = 'id'
        z_range = [-50, 55]

        # dfg = df.groupby(group_by).mean().reset_index()
        dfg = df[df['id'] == 44]
        dfg = dfg[(dfg[plot_by] > z_range[0]) & (dfg[plot_by] < z_range[1])]

        # z space
        x_exp = dfg[plot_by].to_numpy()
        x_theory = np.linspace(x_exp.min(), x_exp.max(), 250)

        # diameter
        y_exp = dfg['gauss_diameter'].to_numpy() * microns_per_pixels / 2
        y_exp1 = dfg['pdf_sigma_x'].to_numpy() * microns_per_pixels * 2.72
        y_exp2 = dfg['pdf_sigma_y'].to_numpy() * microns_per_pixels * 2.72

        y_theory = fPI.particle_image_diameter_by_z(x_theory * 1e-6) * 1e6 / mag_eff
        y_theory2 = fPI.calculate_depth_dependent_stigmatic_diameter(x_theory) * 1e6 / mag_eff

        # intensity
        int_exp = dfg['peak_int'].to_numpy() - bkg_mean
        int_exp_fit = dfg['pdf_A'].to_numpy() - bkg_mean

        int_theory = fPI.calculate_depth_dependent_stigmatic_peak_intensity(x_theory) * dfg['pdf_A'].max() - bkg_mean

        # ---

        # plot setup
        clr_theory = sciblack
        clr_exp = sciblue

        # plot
        fig, (ax0, ax) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))

        ax0.plot(x_theory, int_theory, '-', color=clr_theory, zorder=3.1,
                 label='Theory \n' + r'$(NA_{eff}=0.4)$')
        # ax0.plot(x_exp + 0.4, int_exp, 'o', ms=1, color='red', label='Exp.', alpha=0.25)
        ax0.plot(x_exp + 0.4, int_exp_fit, 'o', ms=1.25, color=clr_exp, zorder=3.0, label='Exp.', alpha=1)

        ax0.set_ylabel(r'$A \: (A.U.)$')
        ax0.set_yticks([0, 5000])
        ax0.legend(markerscale=1.5, handlelength=0.7)

        ax.plot(x_theory, y_theory, color=clr_theory, zorder=3.1, label='Theory')
        ax.scatter(x_exp + 0.4, y_exp, s=1, marker='o', color=clr_exp, zorder=3.0, label='Exp.')
        # ax.scatter(x_exp + 0.4, y_exp1, s=3, marker='d', color='red', label=r'$d_{e}(x)$')
        # ax.scatter(x_exp + 0.4, y_exp2, s=3, marker='o', color='blue', label=r'$d_{e}(y)$')

        # ax.legend(markerscale=2, handlelength=0.6)
        ax.set_ylabel(r'$d_{e} \: (\mu m)$', labelpad=10)
        ax.set_ylim([0, 50])
        ax.set_yticks([0, 25, 50])
        ax.set_xlabel(r'$z \: (\mu m)$')
        ax.set_xlim([-53.5, 57.5])
        ax.set_xticks([-50, 0, 50])

        plt.tight_layout()
        # plt.savefig(path_save + '/compare_A_dia_exp_theory_spct-calib-pid_{}'.format(clr_exp) + filetype)
        plt.show()
        plt.close()

# ---

# ---

# --- CUSTOM PLOT --- CUSTOM PLOT --- CUSTOM PLOT --- CUSTOM PLOT --- CUSTOM PLOT --- CUSTOM PLOT --- CUSTOM PLOT
"""
The purpose of this plot is to compare the variance between:
    (1) wx, wy, A --> which we compute and plot here,
    (2) Swf (particle-to-particle similarity) --> which is computed and plotted elsewhere. 
    
NOTE:
    * we reload the dataframe for this analysis so it may include different particles from the ellipse analysis above.
    
"""

# ---

plot_particle_image_depth_dependent_variance = True
if plot_particle_image_depth_dependent_variance:

    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/results/' \
         'calibration-spct_including_pdf_rmse/calib_spct_stats_11.06.21_z-micrometer-v2_1umSteps__i.a..xlsx'
    df = pd.read_excel(fp)
    df = df.dropna(subset=['pdf_x_y'])  # NOTE: this is different from above
    df = df[df['id'] < max_pid_baseline]
    df = df[~df.id.isin([6, 8, 17, 25, 26, 29, 35, 36, 37, 52, 58, 59, 66, 77, 86, 87])]  # NOTE: "86" is added
    df = df[df['z_corr'].abs() < 50]

    # ---

    # plot raw + errorbars
    plot_raw_image_stats = True
    if plot_raw_image_stats:
        group_by = 'z_true'
        plot_by = 'z_corr'
        plot_columns = ['snr']  # ['pdf_rmse', 'pdf_r_squared', 'pdf_A', 'pdf_sigma_x', 'pdf_sigma_y', 'pdf_x_y', 'pdf_rho']
        plot_clr = 'id'

        # ---

        # bin by radial coordinate
        plot_radial_binning = False
        if plot_radial_binning:
            columns_to_bin = ['z', 'r']
            bin_z = df['z'].unique()
            column_to_count = 'id'
            round_to_decimals = [0, 1]
            min_num_bin = 5
            return_groupby = True

            # sweep different radial bins to evaluate radial dependence on parameters (vs. randomness)
            for bin_r_num in [3, 6, 7, 8]:

                bin_r = np.round(np.linspace(0, 362, bin_r_num), 0).astype(int)
                bins = [bin_z, bin_r]

                dfm, dfstd = bin.bin_generic_2d(df,
                                                columns_to_bin,
                                                column_to_count,
                                                bins,
                                                round_to_decimals,
                                                min_num_bin,
                                                return_groupby
                                                )

                for pc in plot_columns:
                    fig, ax = plt.subplots()
                    for br in bin_r:
                        dfbr = dfm[dfm['bin_ll'] == br]
                        rel_ecc = np.round(np.mean(np.abs(dfbr[pc] - 1)), 3)

                        ax.plot(dfbr[plot_by], dfbr[pc], '-o', ms=1, label='{}, {}'.format(br, rel_ecc))
                    ax.set_xlabel(plot_by)
                    ax.set_ylabel(pc)
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                              borderpad=0.2, labelspacing=0.35, handlelength=1.25,
                              title=r'$r, \: wxy$')
                    plt.tight_layout()
                    plt.show()
                    plt.close()

            # ---

        # ---

        # plot raw
        plot_all = False
        if plot_all:
            for pc in plot_columns:
                fig, ax = plt.subplots()
                ax.scatter(df[plot_by], df[pc], c=df[plot_clr], s=1)
                ax.set_xlabel(plot_by)
                ax.set_ylabel(pc)
                plt.tight_layout()
                plt.show()
                plt.close()

        # ---

        plot_per_pid = False
        if plot_per_pid:
            for pc in plot_columns:
                fig, ax = plt.subplots()
                for pid in df.id.unique():
                    dfpid = df[df['id'] == pid]

                    ax.scatter(dfpid[plot_by], dfpid[pc], s=1, label=pid)

                ax.set_xlabel(plot_by)
                ax.set_ylabel(pc)
                ax.legend()
                plt.tight_layout()
                plt.show()
                plt.close()

        # ---

        # plot group by z
        plot_grouped = True
        if plot_grouped:

            dfgm = df.groupby(group_by).mean()
            dfgs = df.groupby(group_by).std()

            for pc in plot_columns:
                fig, ax = plt.subplots()
                ax.errorbar(dfgm[plot_by], dfgm[pc], yerr=dfgs[pc],
                            fmt='-o', ms=2, capsize=0.15, elinewidth=0.05)
                ax.errorbar(dfgm[plot_by], np.flip(dfgm[pc].to_numpy()), yerr=np.flip(dfgs[pc].to_numpy()),
                            fmt='-o', ms=2, capsize=0.15, elinewidth=0.05)
                ax.set_xlabel(plot_by)
                ax.set_ylabel(pc)
                plt.tight_layout()
                plt.show()
                plt.close()

        # ---
        raise ValueError()

        # plot coefficient of variation for group by z
        plot_coefficient_of_variation = False
        if plot_coefficient_of_variation:

            # collection similarity
            sim_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/particle-similarities-in-image/data'
            dfsg1 = pd.read_excel(sim_dir + '/col_avg_11.06.21_filtered_mean.xlsx')
            dfsg1_std = pd.read_excel(sim_dir + '/col_avg_11.06.21_filtered_std.xlsx')

            # read self similarity
            fpi = 'calib_stacks_forward_self-similarity_11.06.21.xlsx'
            dfss = pd.read_excel(join(sim_dir, fpi))
            dfss_m = dfss.groupby('z').mean().reset_index()
            dfss_std = dfss.groupby('z').std().reset_index()

            # bivariate Gaussian models
            df['pdf_area'] = df['pdf_sigma_x'] * df['pdf_sigma_y'] * np.pi
            dfgm = df.groupby(group_by).mean()
            dfgs = df.groupby(group_by).std()
            dfgc = df.groupby(group_by).count()

            # ---

            # compute coefficient of variation
            dfsg1['cov'] = dfsg1_std.cm / dfsg1.cm
            dfss_m['cov'] = dfss_std.cm / dfss_m.cm
            dfgm['cov_A'] = dfgs['pdf_A'] / dfgm['pdf_A']
            dfgm['cov_area'] = dfgs['pdf_area'] / dfgm['pdf_area']
            dfgm['cov_x_y'] = dfgs['pdf_x_y'] / dfgm['pdf_x_y']
            dfgm['cov_rmse'] = dfgs['pdf_rmse'] / dfgm['pdf_rmse']
            dfgm['cov_r_squared'] = dfgs['pdf_r_squared'] / dfgm['pdf_r_squared']
            dfgm['cov_cov'] = dfgm['cov_A'] + dfgm['cov_area'] + dfgm['cov_x_y'] - 0.5

            # ---

            # setup
            dof = 6.5
            plot_columns = ['pdf_A', 'pdf_area', 'pdf_sigma_x', 'pdf_sigma_y', 'pdf_x_y']
            plot_lbls = [r'$I_{o}$', r'$A_{p}$', r'$\sigma_{x}$', r'$\sigma_{y}$', r'$\sigma_{x}/\sigma_{y}$']

            # plot_columns = ['cov_cov']
            # plot_lbls = [r'$I_{o} \cdot A_{p}$']

            # ---

            # plot
            fig, ax = plt.subplots()
            for pc, lbl in zip(plot_columns, plot_lbls):
                ax.plot(dfgm[plot_by], dfgs[pc] / dfgm[pc], '-o', ms=1, label=lbl)
                # ax.plot(dfgm[plot_by], dfgm[pc], '-o', ms=2, label=lbl)

            # ax.plot(dfgm[plot_by], dfgm['cov_cov'], '-o', ms=1, label='3 CoV')

            ax.plot(dfsg1['z'] - 50, dfsg1['cov'], '-o', ms=1, label=r'$S_{wf}$')
            ax.plot(dfss_m['z'] - 50, dfss_m['cov'], '-o', ms=1, label=r'$S_{ss}$')

            # plot normalized number of particles per frame
            ax.plot(dfgm[plot_by], dfgc['id'] / (dfgc['id'].max() * 2.5), 'o', ms=1, color='b', alpha=0.25,
                    label=r'$N_{p}=$' + ' {}'.format(dfgc['id'].max()))

            ax.axvline(dof * -2, linestyle='--', linewidth=0.5, color='k', alpha=0.25, label='2 DoF')
            ax.axvline(dof * 2, linestyle='--', linewidth=0.5, color='k', alpha=0.25)

            ax.set_xlabel(plot_by)
            ax.set_xlim([-42.5, 42.5])
            ax.set_ylabel('CoV')
            ax.set_ylim([-0.02, 0.42])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderpad=0.2, labelspacing=0.35, handlelength=1.25)
            plt.tight_layout()
            plt.show()
            plt.close()

print("Analysis completed without errors.")