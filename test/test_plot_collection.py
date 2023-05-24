from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import analyze
from utils import io
from utils.plotting import lighten_color
from utils.plot_collections import plot_spct_stats, plot_similarity_stats_simple, plot_similarity_analysis, plot_rigid_displacement_test

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)



# ---------------------------------------------- PLOT SOMETHING QUICKLY ------------------------------------------------

# spct stats plots
quick_plot = True

if quick_plot:

    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_exposure_displacement/data/df_reconstruction_local_displacement_dz-exposure.xlsx'
    df = pd.read_excel(fp)
    dfg = df.groupby('frame').mean()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

    ax1.plot(dfg.index, dfg.z_corr, 'o', ms=1, color='b', label='Exp.', zorder=3.2)
    ax1.plot(dfg.index, dfg.z_poly, '-', color='k', label='Fit', zorder=3.1)
    ax1.set_ylabel(r'$z \: (\mu m)$')
    ax1.legend(loc='upper left', handlelength=0.4, handletextpad=0.2, borderpad=0.2, labelspacing=0.25)

    ax2.plot(dfg.index, dfg.dz_poly_exposure, 'o', ms=1, color='k')
    ax2.set_ylabel(r'$\Delta z_{exposure}^{fit} \: (\mu m)$')

    ax3.plot(dfg.index, dfg.cm, '-o', ms=1, color='b')
    ax3.set_ylabel(r'$C_{m}$')
    ax3.set_xlabel(r'Frame')

    for fr_discontinuous in [71, 97.5, 128.5, 155, 184.5]:
        ax1.axvline(x=fr_discontinuous, color='r', linestyle='--', linewidth=0.5)
        ax2.axvline(x=fr_discontinuous, color='r', linestyle='--', linewidth=0.5)
        ax3.axvline(x=fr_discontinuous, color='r', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    sp = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/02.06.22_exposure_displacement/results/z/exposure-displacement'
    plt.savefig(sp + '/grouped-by-frame_plot_z-dz-exposure-Cm_by_time.png')
    plt.show()




# ------------------------------------------------- PLOT SPCT STATS ----------------------------------------------------

# spct stats plots
spct_stats = False

if spct_stats:

    # filpaths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
               'analyses/shared-results/10.09.22_spct-calibration'

    # read
    plot_spct_stats(base_dir, method='spct')

    plot_similarity_stats_simple(base_dir, min_percent_layers=0.75)

    plot_similarity_analysis(base_dir)

# ---

# ------------------------------------------------- PLOT IDPT STATS ----------------------------------------------------

# IDPT stats plots
idpt_stats = False

if idpt_stats:

    # filpaths
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/' \
               'analyses/results-07.05.22_idpt-calibration'

    # read
    # plot_spct_stats(base_dir, method='idpt')

    # plot_similarity_stats_simple(base_dir, min_percent_layers=0.75)

    plot_similarity_analysis(base_dir, method='idpt')



# ---

# ---------------------------------------------- FIT CONTOUR DIAMETER --------------------------------------------------

# spct stats plots
fit_diameter = False

if fit_diameter:

    path_calib_spct_stats = '/Users/mackenzie/Desktop/gdpyt-characterization/datasets/synthetic_overlap_noise-level2/' \
                            'grid-dz/results/calibration-spct-no-dz-no-dz/' \
                            'calib_spct_stats_grid-dz_calib_nll2_spct_no-dz.xlsx'
    # test overlap
    popt_contour, fig, ax = analyze.fit_contour_diameter(path_calib_spct_stats, fit_z_dist=50, show_plot=True)
    plt.show()


# --------------------------------------------- PLOT TEST TRAJECTORIES -------------------------------------------------

# spct stats plots
plot_pair_trajectories = False

if plot_pair_trajectories:
    path_figs = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/discussion/results-FINAL-05.03.22_10X-idpt/filtered/test_id1/overlapping_pids'

    # slope of fitted sphere
    """fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/discussion/results-FINAL-05.03.22_10X-idpt/filtered/test_id1/id1_rz_lr-ul_fitted_sphere_for_frames-of-interest.xlsx'
    dft = pd.read_excel(fp)
    frame_rate = 24.444
    dft['t'] = dft['frame'] / frame_rate"""

    # test coords
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.06.22_membrane_characterization/analyses/results-05.03.22_10X-idpt-FINAL/coords/test-coords/filtered/test_coords_id1_corrected.xlsx'
    df = pd.read_excel(fp)
    df = df.sort_values('id')

    pids = df.id.unique()
    num_pids = len(pids)
    num_plots = 10
    num_figs = int(np.ceil(num_pids/num_plots))
    # measure precision
    res = []
    for i in range(num_figs):
        these_pids = pids[i * num_plots: (i + 1) * num_plots]
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))
        for pid in these_pids:
            dfpid = df[df['id'] == pid]
            dfpid = dfpid.sort_values('frame')
            x = dfpid.frame.to_numpy()
            y = dfpid.rm.to_numpy()
            z = dfpid.z_corr.to_numpy()

            # r-precision
            pcoeff1, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg=12, full=True)
            pf1 = np.poly1d(pcoeff1)
            y_model = pf1(x)
            y_error = y - y_model
            y_precision = np.mean(np.std(y_error))

            # z-precision
            pcoeff2, residuals, rank, singular_values, rcond = np.polyfit(x, z, deg=12, full=True)
            pf2 = np.poly1d(pcoeff2)
            z_model = pf2(x)
            z_error = z - z_model
            z_precision = np.mean(np.std(z_error))

            res.append([pid, np.mean(z), y_precision, z_precision])

            p1, = ax1.plot(x, y, 'o', ms=0.5, label=pid)
            ax1.plot(x, y_model, '--', linewidth=0.5, color=lighten_color(p1.get_color(), 1.15))

            p1, = ax2.plot(x, z, 'o', ms=0.5)
            ax2.plot(x, z_model, '--', linewidth=0.5, color=lighten_color(p1.get_color(), 1.15))

        ax1.set_ylabel('r')
        ax2.set_ylabel('z')
        ax2.set_xlabel('frame')
        plt.tight_layout()
        plt.savefig(path_figs + '/localization-error-group{}.svg'.format(i))
        plt.close()

    dfr = pd.DataFrame(np.array(res), columns=['id', 'mz', 'pr', 'pz'])
    dfr.to_excel(path_figs + '/localization-precision-all.xlsx')

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches, size_y_inches))
    ax1.plot(dfr.mz, dfr.pr, 'o', ms=0.5)
    ax2.plot(dfr.mz, dfr.pz, 'o', ms=0.5)
    ax1.set_ylabel('r precision')
    ax2.set_ylabel('z precision')
    ax2.set_xlabel('z')
    plt.tight_layout()
    plt.show()
    # plt.savefig(path_figs + '/localization-error_by_theta_all-pids-lr.svg')
    j = 1
    raise ValueError()





    df = df[df['t'] > 1.575]
    microns_per_pixels = 3.2

    # --- processing: map theta to test coords
    map_frames = dft['frame'].to_numpy()
    map_theta = dft['theta_lr_deg'].to_numpy()
    mapping_dict = {map_frames[i]: map_theta[i] for i in range(len(map_frames))}
    df['theta_lr_deg'] = df['frame']
    df['theta_lr_deg'] = df['theta_lr_deg'].map(mapping_dict)
    df['bintoo'] = df['frame'] - df['theta_lr_deg']
    df = df.dropna()

    # other columns to plot:
    # 'drm' - displacement in the r direction
    # 'cm' - similarity

    # get theta > 2 degrees
    df = df[df['theta_lr_deg'] > 4]

    # all particle ID's on the lower right membrane
    particle_ids_lr = [43, 44, 47, 49, 51, 52, 53, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 70, 76, 77, 79,
                       82,
                       83, 84, 88, 89]
    df = df[df.id.isin(particle_ids_lr)]

    # plot group statistics by measuring each error and then plotting the collection of errors
    res = []
    pn_thetas = []
    pn_errors = []
    for pid in df.id.unique():

        dfpid = df[df['id'] == pid]
        param_spatial = 'z_corr'

        # ---

        # positive
        dfpid_pos = dfpid[dfpid['z_corr'] > 0]
        if len(dfpid_pos) > 3:

            dfpid_pos = dfpid_pos.sort_values('theta_lr_deg')
            x1 = dfpid_pos.theta_lr_deg.to_numpy()
            y1 = dfpid_pos.z_corr.to_numpy()

            # fit 2nd order polynomial
            pcoeff1, residuals, rank, singular_values, rcond = np.polyfit(x1, y1, deg=2, full=True)
            pf1 = np.poly1d(pcoeff1)

            # error assessment
            y1_model = pf1(x1)
            y1_error = y1 - y1_model
            y1_precision = np.mean(np.std(y1_error))
            y1_rmse = np.sqrt(np.mean(y1_error ** 2))

            pn_thetas.append(x1)
            pn_errors.append(y1_error)
        else:
            y1_precision, y1_rmse = np.nan, np.nan

        # ---

        # negative
        dfpid_neg = dfpid[dfpid['z_corr'] < 0]
        if len(dfpid_neg) > 3:
            dfpid_neg = dfpid_neg.sort_values('theta_lr_deg')
            x2 = dfpid_neg.theta_lr_deg.to_numpy()
            y2 = dfpid_neg.z_corr.to_numpy()

            # fit 2nd order polynomial
            pcoeff2, residuals, rank, singular_values, rcond = np.polyfit(x2, y2, deg=2, full=True)
            pf2 = np.poly1d(pcoeff2)

            # error assessment
            y2_model = pf2(x2)
            y2_error = y2 - y2_model
            y2_precision = np.mean(np.std(y2_error))
            y2_rmse = np.sqrt(np.mean(y2_error ** 2))

            pn_thetas.append(x2)
            pn_errors.append(y2_error)

        else:
            y2_precision, y2_rmse = np.nan, np.nan

            """# combined errors
            yh_thetas = np.hstack((x1, x2))
            pn_thetas.append(yh_thetas)
            yh_errors = np.hstack((y1_error, y2_error))
            pn_errors.append(yh_errors)"""

        # results
        res.append([pid, y1_precision, y1_rmse, y2_precision, y2_rmse])

    dfr = pd.DataFrame(np.array(res), columns=['id', 'zp_precision', 'zp_rmse', 'zn_precision', 'zn_rmse'])
    dfr.to_excel(path_figs + '/localization-error_lower_right.xlsx')

    # histogram - total
    flat_list = [item for sublist in pn_errors for item in sublist]
    yh_errors = np.array(flat_list)
    xh_binwidth = 0.5
    bandwidth = 0.5
    kde_space = 1
    xh_lim_low = (int(np.min(yh_errors) / xh_binwidth) - 1) * xh_binwidth  # + binwidth_x
    xh_lim_high = (int(np.max(yh_errors) / xh_binwidth) + 1) * xh_binwidth
    xh_bins = np.arange(xh_lim_low, xh_lim_high + xh_binwidth, xh_binwidth)

    # kernel density estimation
    max_val = np.max(np.abs(np.array(np.min(yh_errors), np.max(yh_errors))))
    yh_errors = yh_errors[:, np.newaxis]
    xh_plot = np.linspace(-max_val - kde_space, max_val + kde_space,
                          100)  # np.linspace(np.min(y1_error) - bandwidth, np.max(y1_error) + bandwidth * 2, 500)
    xh_plot = xh_plot[:, np.newaxis]
    kde_x = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(yh_errors)
    log_dens_y1_error = kde_x.score_samples(xh_plot)

    # bottom: histogram of residuals
    fig, ax3 = plt.subplots(figsize=(size_x_inches, size_y_inches / 2))
    nx, binsx, patchesx = ax3.hist(yh_errors, bins=xh_bins, zorder=2.5, color='gray')
    p1 = ax3.fill_between(xh_plot[:, 0], 0,
                          np.exp(log_dens_y1_error) * np.max(nx) / np.max(np.exp(log_dens_y1_error)),
                          fc="None", ec=scired, zorder=2.5)

    p1.set_linewidth(0.5)
    ax3.set_xlim([-7.65, 7.65])
    ax3.set_xlabel(r'error $(\mu m)$')
    ax3.set_ylabel('counts')

    plt.tight_layout()
    plt.savefig(path_figs + '/localization-error_by_theta_all-pids-lr.svg')
    plt.show()
    plt.close()

    # plot error by theta
    flat_list = [item for sublist in pn_thetas for item in sublist]
    yh_thetas = np.array(flat_list)
    fig, ax = plt.subplots()
    ax.scatter(yh_thetas, yh_errors)
    ax.set_xlabel('theta')
    ax.set_ylabel('error')
    plt.show()
    raise ValueError()

    # ---

    # ---

    # plot per particle or groups of nearby particles
    plot_per_particle_or_per_group = False
    if plot_per_particle_or_per_group:

        pid_pairs = [[68, 76], [58, 59, 62], [47, 49, 52]]
        for pp in pid_pairs:

            dfp = df[df.id.isin(pp)]


            for pid in dfp.id.unique():
                for param_spatial in ['z_corr']:  # , 'r_drm'
                    dfpid = dfp[dfp['id'] == pid]

                    # ---

                    # positive
                    dfpid_pos = dfpid[dfpid['z_corr'] > 0]
                    dfpid_pos = dfpid_pos.sort_values('theta_lr_deg')
                    x1 = dfpid_pos.theta_lr_deg.to_numpy()
                    y1 = dfpid_pos.z_corr.to_numpy()

                    # fit 2nd order polynomial
                    pcoeff1, residuals, rank, singular_values, rcond = np.polyfit(x1, y1, deg=2, full=True)
                    pf1 = np.poly1d(pcoeff1)

                    # error assessment
                    y1_model = pf1(x1)
                    y1_error = y1 - y1_model
                    y1_precision = np.mean(np.std(y1_error))
                    y1_rmse = np.sqrt(np.mean(y1_error ** 2))

                    # ---

                    # negative
                    dfpid_neg = dfpid[dfpid['z_corr'] < 0]
                    dfpid_neg = dfpid_neg.sort_values('theta_lr_deg')
                    x2 = dfpid_neg.theta_lr_deg.to_numpy()
                    y2 = dfpid_neg.z_corr.to_numpy()

                    # fit 2nd order polynomial
                    pcoeff2, residuals, rank, singular_values, rcond = np.polyfit(x2, y2, deg=2, full=True)
                    pf2 = np.poly1d(pcoeff2)

                    # error assessment
                    y2_model = pf2(x2)
                    y2_error = y2 - y2_model
                    y2_precision = np.mean(np.std(y2_error))
                    y2_rmse = np.sqrt(np.mean(y2_error ** 2))


                    # histogram
                    yh_errors = np.hstack((y1_error, y2_error))
                    xh_binwidth = 0.5
                    bandwidth = 0.5
                    kde_space = 1.5
                    xh_lim_low = (int(np.min(yh_errors) / xh_binwidth) - 1) * xh_binwidth  # + binwidth_x
                    xh_lim_high = (int(np.max(yh_errors) / xh_binwidth) + 1) * xh_binwidth
                    xh_bins = np.arange(xh_lim_low, xh_lim_high + xh_binwidth, xh_binwidth)

                    # kernel density estimation
                    max_val = np.max(np.abs(np.array(np.min(yh_errors), np.max(yh_errors))))
                    yh_errors = yh_errors[:, np.newaxis]
                    xh_plot = np.linspace(-max_val - kde_space, max_val + kde_space,
                                          100)  # np.linspace(np.min(y1_error) - bandwidth, np.max(y1_error) + bandwidth * 2, 500)
                    xh_plot = xh_plot[:, np.newaxis]
                    kde_x = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(yh_errors)
                    log_dens_y1_error = kde_x.score_samples(xh_plot)

                    # plot
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(size_x_inches, size_y_inches * 1.25),
                                                        gridspec_kw={'height_ratios': [2.5, 1, 1]})
                    """fig = plt.figure(constrained_layout=True, figsize=(size_x_inches, size_y_inches * 1.25))
                    subfigs = fig.subfigures(2, 1, height_ratios=[2, 1.])
                    (ax1, ax2) = subfigs[0].subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                    ax3 = subfigs[1].subplots()"""

                    # top: z + z_fit
                    pi1, = ax1.plot(x1, y1, 'o', ms=1, zorder=3.1, label=pid)
                    fi1, = ax1.plot(x1, y1_model, '--', color='black', linewidth=0.5, zorder=2.1, label='Fit')
                    pi2, = ax1.plot(x2, y2, 'o', ms=1, zorder=3.1, label=pid)
                    ax1.plot(x2, y2_model, '--', color='black', linewidth=0.5, zorder=2.1)

                    # middle: residuals
                    ax2.plot(x1, y1_error, 'o', ms=1, linewidth=1)
                    ax2.plot(x2, y2_error, 'o', ms=1, linewidth=1)

                    # bottom: histogram of residuals
                    nx, binsx, patchesx = ax3.hist(yh_errors, bins=xh_bins, zorder=2.5, color='gray')
                    p1 = ax3.fill_between(xh_plot[:, 0], 0,
                                          np.exp(log_dens_y1_error) * np.max(nx) / np.max(np.exp(log_dens_y1_error)),
                                          fc="None", ec=scired, zorder=2.5)

                    # format
                    ax1.set_xticks([4, 5, 6, 7, 8], labels=[])  # ax1.set_xlabel(r'$time \: (s)$')
                    ax1.set_ylabel(r'$z \: (\mu m)$')
                    """ax1.legend(loc='upper left',
                               markerscale=1.5, borderpad=0.2, labelspacing=0.25, handletextpad=0.5, borderaxespad=0.35)"""
                    l1 = ax1.legend([(pi1, pi2), (fi1)], [pid, 'Fit'], numpoints=1,
                                   handler_map={tuple: HandlerTuple(ndivide=None)},
                                    loc='lower right'
                                   )

                    ax2.set_xticks([4, 5, 6, 7, 8])
                    ax2.set_xlabel(r'$\theta \: (^{\circ})$', labelpad=-1.25)
                    ax2.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')

                    p1.set_linewidth(0.5)
                    ax3.set_xlabel(r'error $(\mu m)$')
                    ax3.set_ylabel('counts')

                    plt.tight_layout()
                    fig.subplots_adjust(hspace=0.55)  # bottom=0.1, left=0.1)  # adjust space between axes
                    plt.savefig(path_figs + '/localization-error_by_theta_{}_{}.svg'.format(pid, param_spatial))
                    plt.show()
                    plt.close()



                    plot_fit_and_hist = False
                    if plot_fit_and_hist:

                        if param_spatial == 'z_corr':
                            xh_binwidth = 5
                            bandwidth = 5
                            kde_space = 5
                        else:
                            xh_binwidth = 1
                            bandwidth = 1
                            kde_space = 2

                        # data
                        x = dfpid.t.to_numpy()
                        y1 = dfpid[param_spatial].to_numpy()
                        # y2 = dfpid.r_drm * microns_per_pixels

                        # fit 12-th order polynomial
                        pcoeff, residuals, rank, singular_values, rcond = np.polyfit(x, y1, deg=12, full=True)
                        pf = np.poly1d(pcoeff)

                        # error assessment
                        y1_model = pf(x)
                        y1_error = y1 - y1_model
                        y1_precision = np.mean(np.std(y1_error))
                        y1_rmse = np.sqrt(np.mean(y1_error ** 2))

                        # histogram
                        xh_lim_low = (int(np.min(y1_error) / xh_binwidth) - 1) * xh_binwidth  # + binwidth_x
                        xh_lim_high = (int(np.max(y1_error) / xh_binwidth) + 1) * xh_binwidth
                        xh_bins = np.arange(xh_lim_low, xh_lim_high + xh_binwidth, xh_binwidth)

                        # kernel density estimation
                        max_val = np.max(np.abs(np.array(np.min(y1_error), np.max(y1_error))))
                        y1_error = y1_error[:, np.newaxis]
                        xh_plot = np.linspace(-max_val - kde_space, max_val + kde_space, 100)  # np.linspace(np.min(y1_error) - bandwidth, np.max(y1_error) + bandwidth * 2, 500)
                        xh_plot = xh_plot[:, np.newaxis]
                        kde_x = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(y1_error)
                        log_dens_y1_error = kde_x.score_samples(xh_plot)

                        # ---

                        # plot
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(size_x_inches, size_y_inches * 1.25),
                                                            gridspec_kw={'height_ratios': [2.5, 1, 1]})
                        """fig = plt.figure(constrained_layout=True, figsize=(size_x_inches, size_y_inches * 1.25))
                        subfigs = fig.subfigures(2, 1, height_ratios=[2, 1.])
                        (ax1, ax2) = subfigs[0].subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                        ax3 = subfigs[1].subplots()"""

                        # top: z + z_fit
                        ax1.plot(x, y1, 'o', ms=1, zorder=3.1, label=pid)
                        ax1.plot(x, y1_model, '--', color='black', linewidth=0.5, zorder=2.1, label='Fit')

                        # middle: residuals
                        ax2.plot(x, y1_error, '-', ms=1, linewidth=1)

                        # bottom: histogram of residuals
                        nx, binsx, patchesx = ax3.hist(y1_error, bins=xh_bins, zorder=2.5, color='gray')
                        p1 = ax3.fill_between(xh_plot[:, 0], 0, np.exp(log_dens_y1_error) * np.max(nx) / np.max(np.exp(log_dens_y1_error)),
                                              fc="None", ec=scired, zorder=2.5)

                        # format
                        ax1.set_xticks([2, 4, 6, 8], labels=[])  # ax1.set_xlabel(r'$time \: (s)$')
                        ax1.set_ylabel(r'$z \: (\mu m)$')
                        ax1.legend(loc='upper left',
                                   markerscale=1.5, borderpad=0.2, labelspacing=0.25, handletextpad=0.5, borderaxespad=0.35)

                        ax2.set_xticks([2, 4, 6, 8])
                        ax2.set_xlabel(r'$time \: (s)$', labelpad=-5)
                        ax2.set_ylabel(r'$\epsilon_{z} \: (\mu m)$')

                        p1.set_linewidth(0.5)
                        ax3.set_xlabel(r'error $(\mu m)$')
                        ax3.set_ylabel('counts')

                        # plt.tight_layout()
                        fig.subplots_adjust(hspace=0.35)  # bottom=0.1, left=0.1)  # adjust space between axes
                        plt.savefig(path_figs + '/localization-error_and_theta_{}_{}.svg'.format(pid, param_spatial))
                        plt.show()
        raise ValueError()
        j = 1

        """fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        for pid in dfp.id.unique():
            dfpid = dfp[dfp['id'] == pid]
            ax1.plot(dfpid.t, dfpid.cm, '-o', ms=0.5)
            ax2.plot(dfpid.t, dfpid.z_corr, 'o', ms=1, label=pid)
        ax1.set_ylabel(r'$c_{m}$')
        ax2.set_xlabel(r'$time \: (s)$')
        ax2.set_ylabel(r'$z \: (\mu m)$')
        ax2.legend(markerscale=1.5, borderpad=0.2, labelspacing=0.25, handletextpad=0.5, borderaxespad=0.35)
        plt.tight_layout()
        plt.savefig(path_figs + '/pdo_and_cm_{}.svg'.format(pp))
        plt.show()
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        for pid in dfp.id.unique():
            dfpid = dfp[dfp['id'] == pid]
            ax1.plot(dfpid.t, dfpid.drm * microns_per_pixels, '-o', ms=0.5)
            ax2.plot(dfpid.t, dfpid.z_corr, 'o', ms=1, label=pid)
        ax1.set_ylabel(r'$\Delta r_{m} \: (\mu m)$')
        ax2.set_xlabel(r'$time \: (s)$')
        ax2.set_ylabel(r'$z \: (\mu m)$')
        ax2.legend(markerscale=1.5, borderpad=0.2, labelspacing=0.25, handletextpad=0.5, borderaxespad=0.35)
        plt.tight_layout()
        plt.savefig(path_figs + '/pdo_and_drm_{}.svg'.format(pp))
        plt.show()
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        for pid in dfp.id.unique():
            dfpid = dfp[dfp['id'] == pid]
            ax1.plot(dfpid.t, dfpid.r_drm * microns_per_pixels, '-o', ms=0.5)
            ax2.plot(dfpid.t, dfpid.z_corr, 'o', ms=1, label=pid)
        ax1.set_ylabel(r'$r_{m} \: (\mu m)$')
        ax2.set_xlabel(r'$time \: (s)$')
        ax2.set_ylabel(r'$z \: (\mu m)$')
        ax2.legend(markerscale=1.5, borderpad=0.2, labelspacing=0.25, handletextpad=0.5, borderaxespad=0.35)
        plt.tight_layout()
        plt.savefig(path_figs + '/pdo_and_r_drm_{}.svg'.format(pp))
        plt.show()
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        for pid in dfp.id.unique():
            dfpid = dfp[dfp['id'] == pid]
            ax1.plot(dfpid.t, dfpid.r * microns_per_pixels, '-o', ms=0.5)
            ax2.plot(dfpid.t, dfpid.z_corr, 'o', ms=1, label=pid)
        ax1.set_ylabel(r'$r \: (\mu m)$')
        ax2.set_xlabel(r'$time \: (s)$')
        ax2.set_ylabel(r'$z \: (\mu m)$')
        ax2.legend(markerscale=1.5, borderpad=0.2, labelspacing=0.25, handletextpad=0.5, borderaxespad=0.35)
        plt.tight_layout()
        plt.savefig(path_figs + '/pdo_and_r_{}.svg'.format(pp))
        plt.show()"""
        j = 1


# ------------------------------------------------- PLOT PRECISION -----------------------------------------------------


# idpt - displacement
# analyze precision of all tests
analyze_precision = False

if analyze_precision:
    fp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/test/step/test_id2_coords_30micron_step_towards.xlsx'

    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/test_coords/test/step_signed'
    filetype = '.xlsx'

    sort_ids = ['test_id', '_coords_']
    sort_dzs = ['_coords_', 'micron_step_']

    files = [f for f in listdir(base_dir) if f.endswith(filetype)]
    files = sorted(files, key=lambda x: float(x.split(sort_ids[0])[-1].split(sort_ids[1])[0]))
    names = [float(f.split(sort_ids[0])[-1].split(sort_ids[1])[0]) for f in files]
    dzs = [float(f.split(sort_dzs[0])[-1].split(sort_dzs[1])[0]) for f in files]

    data_spct = []
    data_idpt = []
    for fp, name, dz in zip(files, names, dzs):

        df = pd.read_excel(join(base_dir, fp))
        mdp, mdm, mdmp, rmse = analyze.evaluate_displacement_precision(df,
                                                                       group_by='id',
                                                                       split_by='frame',
                                                                       split_value=50.5,
                                                                       precision_columns='z',
                                                                       true_dz=dz)
        if name < 10:
            data_idpt.append([dz, mdm, mdp, mdmp, rmse])
        else:
            data_spct.append([dz, mdm, mdp, mdmp, rmse])

    dfp_idpt = pd.DataFrame(np.array(data_idpt), columns=['true_dz', 'dz', 'pz', 'pdz', 'rmse'])
    dfp_spct = pd.DataFrame(np.array(data_spct), columns=['true_dz', 'dz', 'pz', 'pdz', 'rmse'])

    # plot rms error
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True)

    ax1.scatter(dfp_spct.index + 1, dfp_spct.pz, label='SPCT')
    ax1.scatter(dfp_idpt.index + 1, dfp_idpt.pz, label='IDPT')
    ax1.set_ylabel(r'$\sigma_{i} + \sigma_{f} \: (\mu m)$')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax2.errorbar(dfp_spct.index + 1, dfp_spct.dz, yerr=dfp_spct.pdz, fmt='o', elinewidth=3, capsize=4)
    ax2.errorbar(dfp_idpt.index + 1, dfp_idpt.dz, yerr=dfp_idpt.pdz, fmt='o', elinewidth=3, capsize=4)
    ax2.set_ylabel(r'$\overline{\Delta z} \pm \sigma_{\Delta z} \: (\mu m)$')

    ax3.scatter(dfp_spct.index + 1, dfp_spct.rmse)
    ax3.scatter(dfp_idpt.index + 1, dfp_idpt.rmse)
    ax3.set_ylabel(r'$\Delta z$ r.m.s. error $(\mu m)$')
    ax3.set_xticks(ticks=[y + 1 for y in range(len(dfp_idpt.true_dz.unique()))], labels=dfp_idpt.true_dz.unique())
    ax3.set_xlabel(r'$\Delta z_{true} \: (\mu m)$')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------- PLOT RIGID DISPLACEMENT -----------------------------------------------


# gdpyt
test_rigid = False
if test_rigid:
    # idpt - meta assessment
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/results/idpt-meta-assessment-avg-2calib-1test/calibration-idpt-[17, 13]/'
    fn = 'calib_idpt_stats_11.06.21_z-micrometer-v2_cSILPURAN_17.xlsx'

    fp = '/Users/mackenzie/Desktop/dummy_coords_30micron_step_towards.xlsx'
    plot_rigid_displacement_test(test_coords_path=fp, spct_stats_path=base_dir + fn)


# ------------------------------------------------- PLOT SPCT STATS ----------------------------------------------------


# single particle calibration
spct = False
if spct:
    # base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/calibration/20X_1Xmag_2.15umNR_HighInt_0.03XHg/results/meta-assessment/calibration-plott-gen_cal/'
    # fn = 'calib_spct_stats_20X_1Xmag_2.15umNR_HighInt_0.03XHg_cGlass_g.xlsx'

    # idpt - meta assessment
    base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/11.06.21_z-micrometer-v2/results/idpt-meta-assessment-avg-2calib-1test/calibration-idpt-[17, 13]/'
    fn = 'calib_idpt_stats_11.06.21_z-micrometer-v2_cSILPURAN_17.xlsx'

    plot_spct_stats(spct_stats_path=base_dir + fn)

j = 1