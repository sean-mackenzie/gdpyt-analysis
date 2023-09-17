# test bin, analyze, and plot functions
import itertools
import os
from os.path import join

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from utils import iterative_closest_point as icp
from sklearn.neighbors import NearestNeighbors, KernelDensity

import matplotlib.pyplot as plt

from utils import bin, functions

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

plt.style.use(['science', 'ieee', 'std-colors'])  # 'ieee', 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]

def pci_best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def pci_nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def pci_icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = pci_nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = pci_best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = pci_best_fit_transform(A, src[:m, :].T)

    return T, distances, i, indices


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# 0. Setup - General

# subject to change
true_num_particles_per_frame = 88
baseline_frame = 39
baseline_frames = [39, 40, 41]

# experimental
microns_per_pixel = 1.6
num_pixels = 512
padding = 5
img_xc, img_yc = num_pixels / 2 + padding, num_pixels / 2 + padding

# processing
z_range = [-50, 55]
measurement_depth = z_range[1] - z_range[0]
h = measurement_depth
num_frames_per_step = 3
filter_barnkob = measurement_depth / 10

# ---

# set limits for all analyses

# --- FILTERS: error relative calib particle
z_error_limit = 5  # microns
in_plane_distance_threshold = np.round(2 * microns_per_pixel, 1)  # (units: microns) in-plane distance threshold
min_counts = 1
min_counts_bin_z = 20
min_counts_bin_r = 20
min_counts_bin_rz = 5

# --- FILTERS: calc rigid body displacements
filter_step_size = z_error_limit  # error threshold
min_num_particles_for_icp = 5  # number of particles per frame threshold
# note: SPCT(Cmin=0.9) has some z-positions with 9 and 19 particles so 5 and 15 are good natural limits, respectively.

# ---


# ----------------------------------------------------------------------------------------------------------------------
# 1. Setup - File paths

base_dir = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_calc_rigid_body_displacement'
path_coords = join(base_dir, 'coords')
path_results = join(base_dir, 'results')
path_figs = join(base_dir, 'figs')

# ----------------------------------------------------------------------------------------------------------------------
# 2. BEGIN PROCESSING

# ---
# export the test coords of the single calibration particle for IDPT and SPCT
export_calib_particle_coords = False  # True False
if export_calib_particle_coords:
    fpi = 'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
    idpt_calib_id_from_testset = 42
    fp = join(path_coords, fpi)
    df = pd.read_excel(fp)
    dfic = df[df['id'] == idpt_calib_id_from_testset]
    dfic.to_excel(join(path_coords, 'test_coords_idpt_calib_particle_only.xlsx'))

    fps = 'test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'  # z is tilt-corrected
    spct_calib_id_from_testset = 92
    df = pd.read_excel(join(path_coords, fps))
    dfsc = df[df['id'] == spct_calib_id_from_testset]
    dfsc.to_excel(join(path_coords, 'test_coords_spct_calib_particle_only.xlsx'))

    # plot
    ms = 2
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.25, size_y_inches * 1.5))
    ax1.plot(dfic.z_true, dfic.z - dfic.z_true, '-o', ms=ms,
             label='z({}, {})'.format(np.round(dfic.z.min(), 1), np.round(dfic.z.max(), 1)))
    # ax1.plot(dfic.z_true, dfic.z_no_corr - dfic.z_true, '-o', ms=ms, label='z no corr')
    # ax1.plot(dfic.z_true, dfic.z_corr_tilt - dfic.z_true, '-o', ms=ms, label='z')
    # ax1.plot(dfic.z_true, dfic.z_corr_tilt_fc - dfic.z_true, '-o', ms=ms, label='z')

    ax2.plot(dfsc.z_true, dfsc.z_no_corr - dfsc.z_true, '-o', ms=ms, label='z no corr')
    ax2.plot(dfsc.z_true, dfsc.z_corr_tilt - dfsc.z_true, '-o', ms=ms, label='z corr tilt')
    ax2.plot(dfsc.z_true, dfsc.z_corr_tilt_fc - dfsc.z_true, '-o', ms=ms, label='z corr tilt fc')
    ax2.plot(dfsc.z_true, dfsc.z_no_corr - dfsc.z_true, '-o', ms=ms,
             label='z({}, {})'.format(np.round(dfsc.z_no_corr.min(), 1), np.round(dfsc.z_no_corr.max(), 1)))

    # ---
    # groupby: mean
    dficgm = dfic.groupby('z_true').mean().reset_index()
    dfscgm = dfsc.groupby('z_true').mean().reset_index()

    # groupby: std
    dficgstd = dfic.groupby('z_true').std().reset_index()
    dfscgstd = dfsc.groupby('z_true').std().reset_index()

    ax3.errorbar(dficgm.z_true, dficgm.z - dficgm.z_true, yerr=dficgstd.z,
                 fmt='-o', capsize=2, elinewidth=1, ms=ms, label='IDPT z')
    ax3.errorbar(dfscgm.z_true, dfscgm.z - dfscgm.z_true, yerr=dfscgstd.z,
                 fmt='-o', capsize=2, elinewidth=1, ms=ms, label='SPCT z')

    ax1.set_ylim([-1, 1.5])
    ax2.set_ylim([-1, 1.5])
    ax3.set_ylim([-0.7, 1.2])
    ax1.legend(fontsize='small', title='IDPT')  # loc='upper left', bbox_to_anchor=(1, 1),
    ax2.legend(fontsize='small', title='SPCT')
    ax3.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

# ---

# ---------------------------------------------------------------------
# 3. CALCULATE RIGID TRANSFORMATIONS VIA ICP - WHERE THE PARTICLE POSITIONS ARE EVALUATED EACH FRAME (NOT AVERAGED)

calculate_rigid_transforms_by_frame = False  # True False
if calculate_rigid_transforms_by_frame:

    # read coords

    use_raw = False
    use_meta = False
    use_focus = True
    min_cm = 0.9  # similarity threshold

    if use_raw:
        # OPTION 1: "RAW" COORDINATES
        fpi = '/test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
        fps = '/test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'  # z is tilt-corrected
        path_read_err_rel_calib = None
    elif use_raw is False and use_meta is True:
        fpi = None
        fps = '/spct_meta_cmin0.5_test_coords_post-processed.xlsx'  # z is raw
        path_read_err_rel_calib = None
    else:
        # OPTION 2: POST-PROCESSED COORDINATES FROM "ERROR RELATIVE CALIBRATION PARTICLE"
        path_read_err_rel_calib = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/' \
                                  '11.06.21_error_relative_calib_particle/results/' \
                                  'relative-to-tilt-corr-calib-particle/' \
                                  'zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts) + \
                                  '/ztrue_is_fit-plane-xyzc'
        fpi = 'idpt_error_relative_calib_particle_zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts)
        fps = 'spct_error_relative_calib_particle_zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts)

    # ---
    if use_focus:

        path_results_true_positions = join(path_figs, 'idpt', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'true_xy')
        if not os.path.exists(path_results_true_positions):
            os.makedirs(path_results_true_positions)

        fp_xy_at_zf = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_calc_rigid_body_displacement/' \
                      'coords/true_positions_fiji.xlsx'
        dfxyzf = pd.read_excel(fp_xy_at_zf)

        # shift locations to match IDPT + padding
        padding_rel_true_x = 0
        padding_rel_true_y = 0
        dfxyzf['x'] = dfxyzf['x'] + padding_rel_true_x
        dfxyzf['y'] = dfxyzf['y'] + padding_rel_true_y
        dfxyzf['gauss_xc'] = dfxyzf['gauss_xc'] + padding_rel_true_x
        dfxyzf['gauss_yc'] = dfxyzf['gauss_yc'] + padding_rel_true_y

        dfxyzf['z'] = 0
        dfxyzf['gauss_rc'] = np.sqrt(dfxyzf['gauss_xc'] ** 2 + dfxyzf['gauss_yc'] ** 2)

        for pix2microns in ['gauss_xc', 'gauss_yc', 'gauss_rc']:
            dfxyzf[pix2microns] = dfxyzf[pix2microns] * microns_per_pixel

        dfxyzf['x'] = dfxyzf['gauss_xc']
        dfxyzf['y'] = dfxyzf['gauss_yc']
        dfxyzf['r'] = dfxyzf['gauss_rc']

        """fig, ax = plt.subplots()
        ax.scatter(dfxyzf.x, dfxyzf.y, s=dfxyzf.contour_area, c=dfxyzf.id)
        ax.invert_yaxis()
        ax.set_title('CoM positions from FIJI')
        plt.savefig(path_results_true_positions + '/fiji_positions.png')
        plt.show()"""

    else:
        dfxyzf = None

    # ---

    # -

    """
    NOTE TO SELF:
        1. the "initial" point cloud should (could?) be the positions of best focus at z=0.
            a. for SPCT, you might not even need the tilt or field curvature corrections. 
        2. then extrapolated outwards in both directions (+z and -z).
    """


    # function from ICP
    def nearest_neighbor(src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()


    def distances_xyz(AR, BR, TR):
        """ xdists, ydists, zdists = distances_xyz(AR=, BR=) """

        # get number of dimensions
        m = AR.shape[1]
        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, AR.shape[0]))
        dst = np.ones((m + 1, BR.shape[0]))
        src[:m, :] = np.copy(AR.T)
        dst[:m, :] = np.copy(BR.T)

        # update the current source
        src_transformed = np.dot(TR, src)
        src_transformed = src_transformed.T

        # calculate x, y, and z distances (errors)
        x_distances = src_transformed[:, 0] - BR[:, 0]
        y_distances = src_transformed[:, 1] - BR[:, 1]
        z_distances = src_transformed[:, 2] - BR[:, 2]

        return x_distances, y_distances, z_distances


    # ---

    # iterative closest point algorithm
    for fp, mtd in zip([fpi, fps], ['idpt', 'spct']):
        # for fp, mtd in zip([fps, fpi], ['spct', 'idpt']):

        if not os.path.exists(join(path_figs, mtd)):
            os.makedirs(join(path_figs, mtd))

        if use_raw:
            path_scatter_yz = join(path_figs, mtd, 'scatter_yz_A2B')
            path_results_mtd = join(path_results, mtd, 'raw')
        elif use_raw is False and use_meta is True:
            path_scatter_yz = join(path_figs, mtd, 'scatter_yz_A2B_meta')
            path_results_mtd = join(path_results, mtd, 'meta')
        elif use_focus:
            path_scatter_yz = join(path_figs, mtd, 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'scatter_yz_A2B_rel-calib')
            path_results_mtd = join(path_results, mtd, 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'rel-calib')
        else:
            path_scatter_yz = join(path_figs, mtd, 'ztrue_is_fit-plane-xyzc', 'scatter_yz_A2B_rel-calib')
            path_results_mtd = join(path_results, mtd, 'ztrue_is_fit-plane-xyzc', 'rel-calib')

        if not os.path.exists(path_scatter_yz):
            os.makedirs(path_scatter_yz)
        if not os.path.exists(path_results_mtd):
            os.makedirs(path_results_mtd)

        if use_raw or use_meta:
            if mtd == 'spct':
                xsub, ysub, rsub = 'gauss_xc', 'gauss_yc', 'gauss_rc'
                # continue
            else:
                xsub, ysub, rsub = 'xg', 'yg', 'rg'
                continue

            # read coords
            dft = pd.read_excel(path_coords + fp)
            dft['gauss_rc'] = np.sqrt(dft['gauss_xc'] ** 2 + dft['gauss_yc'] ** 2)
            dft['error'] = dft['error_z']

            ############# THESE FUNCTIONS "MAY" ONLY APPLY TO "USE_META"... I'm not sure
            dft['z'] = dft['z'] - 49.91
            dft['z_true'] = dft['z_true'] - 49.91
            dft['z_calib'] = dft['z_true']
            dft = dft.dropna()
            #############

            # get only useful columns
            dft = dft[['frame', 'id', xsub, ysub, rsub, 'z_true', 'z', 'error', 'cm', 'x', 'y']]

            # filters
            dft = dft[dft['error'].abs() < filter_step_size]
            dft = dft[dft['cm'] > min_cm]

            fig, ax = plt.subplots()
            ax.scatter(dft.z_true, dft.z - dft.z_true, c=dft.id, s=1)
            plt.show()

            # adjust units from pixels to microns
            for pix2microns in ['x', 'y', xsub, ysub, rsub]:
                dft[pix2microns] = dft[pix2microns] * microns_per_pixel

        else:
            if mtd == 'spct':
                pass  # continue pass
            else:
                if min_cm == 0.9:
                    continue  # continue pass
                else:
                    pass  # continue pass

            xsub, ysub, rsub = 'x', 'y', 'r'  # these are already the sub-pixel positions

            # read coords
            dft = pd.read_excel(join(path_read_err_rel_calib, fp + '.xlsx'))

            # get only useful columns
            dft = dft[['frame', 'id', 'cm', 'x', 'y', 'r', 'z_true', 'z_no_corr', 'z', 'z_calib', 'error_rel_p_calib']]

            dft['z_nominal'] = dft['z_true']
            dft['error'] = dft['error_rel_p_calib']

            # filters
            dft = dft[dft['error'].abs() < filter_step_size]
            dft = dft[dft['cm'] > min_cm]

            # adjust units from pixels to microns
            for pix2microns in ['x', 'y', 'r']:
                dft[pix2microns] = dft[pix2microns] * microns_per_pixel

        # ---

        # ---

        # evaluate axial displacement (~5 microns) between the 3 frames per z
        eval_between_dz_frames = True
        if eval_between_dz_frames:

            # iterate through z-directions
            data_ = []
            for z_direction in ['neg2pos']:

                # get z_true values and sort
                zts = dft.z_true.sort_values(ascending=True).unique()
                if use_focus:
                    dz_normalize = 0
                else:
                    dz_normalize = -1  # -5

                data = []
                ddx, ddy, ddz = [], [], []
                z_distances = []
                dfBB_icps = []

                if use_focus:
                    z_range_rt = len(zts)
                    """
                    if not os.path.exists(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx'):
                        frames_and_ztrues = dft.sort_values('frame')[['frame', 'z_true']].values.tolist()
                        frames_and_ztrues = np.unique(np.array(frames_and_ztrues), axis=0)
                        temp = []
                        for ft, zt in frames_and_ztrues:
                            df_true['z_true'] = zt
                            df_true['z'] = zt
                            df_true['frame'] = ft
                            temp.append(df_true.copy())
                        df_true = pd.concat(temp)
                        df_true.to_excel(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx', index=False)

                    else:
                        df_true = pd.read_excel(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx')
                    """
                else:
                    z_range_rt = len(zts) - 1

                data_z = []
                for ii in range(z_range_rt):

                    z_frames = dft[dft['z_true'] == zts[ii]].frame.unique()
                    for fr in z_frames:

                        if use_focus:
                            dfA = dfxyzf.copy()
                            dfB = dft[(dft['z_true'] == zts[ii]) & (dft['frame'] == fr)].reset_index()
                            dfA['z'] = 0  # zts[ii]

                        else:
                            dfA = dft[(dft['z_true'] == zts[ii]) & (dft['frame'] == fr)].reset_index()
                            dfB = dft[(dft['z_true'] == zts[ii]) & (dft['frame'] == fr)].reset_index()

                        # -

                        # scatter plot points
                        plot_inputs_scatter = False
                        if plot_inputs_scatter:
                            fig, ax = plt.subplots()
                            if use_raw or use_meta:
                                ax.scatter(dfA['y'], dfA['z'] - zts[ii], label='A')
                                ax.scatter(dfB['y'], dfB['z'] - zts[ii + 1], label='B')
                                save_id_scatter_yz = 'scatter-y_z-from'
                            else:
                                ax.scatter(dfA['y'], dfA['z'] - dfA['z_calib'], label='A')
                                ax.scatter(dfB['y'], dfB['z'] - dfB['z_calib'], label='B')
                                save_id_scatter_yz = 'scatter-y_z-rel-calib-from'

                            ax.set_xlabel('y (microns)')
                            ax.set_ylabel('z')
                            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            plt.tight_layout()
                            plt.savefig(join(path_scatter_yz,
                                             '{}_{}_{}_to_{}.png'.format(mtd, save_id_scatter_yz,
                                                                         np.round(zts[ii], 1),
                                                                         np.round(zts[ii + 1], 1))))

                            continue

                        # -

                        A = dfA[['x', 'y', 'z']].to_numpy()
                        B = dfB[['x', 'y', 'z']].to_numpy()

                        i_num_A = len(A)
                        i_num_B = len(B)

                        # minimum number of particles per frame threshold
                        if np.min([i_num_A, i_num_B]) < min_num_particles_for_icp:
                            continue

                        # match particle positions between A and B using NearestNeighbors
                        if len(A) > len(B):
                            ground_truth_xy = A[:, :2]
                            ground_truth_pids = dfA.id.to_numpy()
                            locations = B[:, :2]

                            fit_to_pids = dfB.id.to_numpy()
                            A_longer = True

                        else:
                            ground_truth_xy = B[:, :2]
                            ground_truth_pids = dfB.id.to_numpy()
                            locations = A[:, :2]

                            fit_to_pids = dfA.id.to_numpy()
                            A_longer = False

                        # calcualte distance using NearestNeighbors
                        nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_xy)
                        distances, indices = nneigh.kneighbors(locations)

                        ############## make True/False array
                        ################ 6/14/2023
                        use_new_icp_pairing = True
                        if use_new_icp_pairing:
                            if A_longer:
                                idx_locations = np.arange(len(locations))
                                idx_locations = idx_locations[:, np.newaxis]
                                sorted_uniq_pids = np.where(distances < in_plane_distance_threshold, indices, np.nan)
                                sorted_uniq_pids = flatten_list_of_lists(sorted_uniq_pids)

                                sorted_fit_uniq_pids = np.where(distances < in_plane_distance_threshold, idx_locations, np.nan)
                                sorted_fit_uniq_pids = flatten_list_of_lists(sorted_fit_uniq_pids)
                            else:
                                idx_locations = np.arange(len(locations))
                                idx_locations = idx_locations[:, np.newaxis]
                                sorted_uniq_pids = np.where(distances < in_plane_distance_threshold, idx_locations, np.nan)
                                sorted_uniq_pids = flatten_list_of_lists(sorted_uniq_pids)

                                sorted_fit_uniq_pids = np.where(distances < in_plane_distance_threshold, indices, np.nan)
                                sorted_fit_uniq_pids = flatten_list_of_lists(sorted_fit_uniq_pids)

                            # remove NaNs from array: x = x[~numpy.isnan(x)]
                            sorted_uniq_pids = np.array(sorted_uniq_pids)
                            sorted_fit_uniq_pids = np.array(sorted_fit_uniq_pids)

                            sorted_uniq_pids = sorted_uniq_pids[~np.isnan(sorted_uniq_pids)]
                            sorted_fit_uniq_pids = sorted_fit_uniq_pids[~np.isnan(sorted_fit_uniq_pids)]

                            ##############

                        else:

                            distances = np.where(distances < in_plane_distance_threshold, distances, np.nan)

                            # these are the p_ID's from "ground_truth_xy"
                            nearest_pid = ground_truth_pids[indices]
                            nearest_pid = np.where(distances < in_plane_distance_threshold, nearest_pid, np.nan)
                            nearest_pid = nearest_pid[~np.isnan(nearest_pid)]
                            uniq_pids = np.unique(nearest_pid)

                            # these are the p_ID's from "locations"
                            fit_to_pids = np.where(distances < in_plane_distance_threshold, fit_to_pids, np.nan)
                            fit_to_pids = fit_to_pids[~np.isnan(fit_to_pids)]
                            fit_uniq_pids = np.unique(fit_to_pids)

                        ################ 6/14/2023
                        use_new_icp_pairing = True
                        if use_new_icp_pairing:
                            if A_longer:  # meaning: A is "ground truth"
                                # dfAA = dfA[dfA.id.isin(uniq_pids)]
                                # dfBB = dfB[dfB.id.isin(fit_uniq_pids)]

                                dfAA = dfA.iloc[sorted_uniq_pids]
                                dfBB = dfB.iloc[sorted_fit_uniq_pids]

                            else:  # meaning: B is "ground truth"
                                # dfAA = dfA[dfA.id.isin(fit_uniq_pids)]
                                # dfBB = dfB[dfB.id.isin(uniq_pids)]

                                dfAA = dfA.iloc[sorted_fit_uniq_pids]
                                dfBB = dfB.iloc[sorted_uniq_pids]

                        else:
                            # iterative closest point algorithm
                            dfAA = dfA[dfA.id.isin(uniq_pids)]
                            dfBB = dfB[dfB.id.isin(uniq_pids)]

                            if len(dfAA) > len(dfBB):
                                uniq_pids = dfBB.id.unique()
                                dfAA = dfAA[dfAA.id.isin(uniq_pids)]
                            elif len(dfAA) < len(dfBB):
                                uniq_pids = dfAA.id.unique()
                                dfBB = dfBB[dfBB.id.isin(uniq_pids)]
                        ################

                        """fig, ax = plt.subplots()
                        ax.scatter(dfAA['x'], dfAA['y'], color='r', label='AA')
                        ax.scatter(dfBB['x'], dfBB['y'], color='b', label='BB')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.legend()
                        plt.show()
                        raise ValueError()"""

                        A = dfAA[[xsub, ysub, 'z']].to_numpy()
                        B = dfBB[[xsub, ysub, 'z']].to_numpy()

                        # print(len(A))
                        # print(len(B))

                        N = len(A)
                        # T, distances, iterations = icp.icp(A, B, tolerance=0.000001)  # originally: icp.icp(B, A,...)
                        T, distances, iterations, dst_indices = pci_icp(A, B, tolerance=0.000001)  # returns indices

                        # ------------ distance by direction ------------
                        xdists, ydists, zdists = distances_xyz(AR=A, BR=B, TR=T)
                        rmse_xdist = np.sqrt(np.mean(np.square(xdists)))
                        rmse_ydist = np.sqrt(np.mean(np.square(ydists)))
                        rmse_zdist = np.sqrt(np.mean(np.square(zdists)))

                        # get matching indices from dfBB
                        dfBB_icp = dfBB.iloc[dst_indices]
                        dfBB_icp['errx'] = xdists
                        dfBB_icp['erry'] = ydists
                        dfBB_icp['errxy'] = np.sqrt(dfBB_icp['errx'] ** 2 + dfBB_icp['erry'] ** 2)
                        dfBB_icp['errz'] = zdists

                        # if you want dfBB and dfAA (below), or, if you only want dfBB (below, below)
                        include_AA = True
                        if include_AA:
                            # dfAA should already be the correct indices
                            dfAA_icp = dfAA[['id', xsub, ysub, 'z']]
                            dfAA_icp = dfAA_icp.rename(
                                columns={'id': 'a_id', xsub: 'a_' + xsub, ysub: 'a_' + ysub, 'z': 'a_z'})

                            # reset the natural index of both dfAA and dfBB
                            dfAA_icp = dfAA_icp.reset_index(drop=True)
                            dfBB_icpAB = dfBB_icp.reset_index(drop=True)

                            # concat
                            dfBB_icpAB = pd.concat([dfBB_icpAB, dfAA_icp], axis=1)
                            dfBB_icpAB['ab_errx'] = dfBB_icpAB['a_' + xsub] - dfBB_icpAB[xsub]
                            dfBB_icpAB['ab_erry'] = dfBB_icpAB['a_' + ysub] - dfBB_icpAB[ysub]
                            dfBB_icpAB['ab_errxy'] = np.sqrt(dfBB_icpAB['ab_errx'] ** 2 + dfBB_icpAB['ab_erry'] ** 2)
                            dfBB_icpAB['ab_errz'] = dfBB_icpAB['a_z'] - dfBB_icpAB['z']

                            # remove rows with x-y errors that exceed limit
                            dfBB_icpAB = dfBB_icpAB[(dfBB_icpAB['errx'].abs() < in_plane_distance_threshold) &
                                                    (dfBB_icpAB['erry'].abs() < in_plane_distance_threshold)]

                            dfBB_icps.append(dfBB_icpAB)

                        else:
                            # remove rows with x-y errors that exceed limit
                            dfBB_icp = dfBB_icp[(dfBB_icp['errx'].abs() < in_plane_distance_threshold) &
                                                (dfBB_icp['erry'].abs() < in_plane_distance_threshold)]

                            dfBB_icps.append(dfBB_icp)

                        # Make C a homogeneous representation of B
                        C = np.ones((N, 4))
                        C[:, 0:3] = np.copy(B)

                        # Transform C
                        C = np.dot(T, C.T).T

                        # evaluate transformation results
                        # ddx.append(T[0, 3])
                        # ddy.append(T[1, 3])
                        # ddz.append(T[2, 3])
                        deltax, deltay, deltaz = T[0, 3], T[1, 3], T[2, 3]
                        # z_distances.append(distances.tolist())
                        precision_dist = np.std(distances)
                        rmse_dist = np.sqrt(np.mean(np.square(distances)))
                        if use_focus:
                            data.append([fr, 0, zts[ii], zts[ii],
                                         precision_dist, rmse_dist,
                                         deltax, deltay, deltaz,
                                         rmse_xdist, rmse_ydist, rmse_zdist,
                                         len(distances), i_num_A, i_num_B])
                        else:
                            data.append([fr, zts[ii], zts[ii + 1], zts[ii + 1],
                                         precision_dist, rmse_dist,
                                         deltax, deltay, deltaz,
                                         rmse_xdist, rmse_ydist, rmse_zdist,
                                         len(distances), i_num_A, i_num_B])

                # package
                # ------------------
                dfBB_icps = pd.concat(dfBB_icps)
                dfBB_icps.to_excel(
                    path_results_mtd +
                    '/{}_icp-dist-xyz_{}_errlim{}_inplanedist{}_Cmin{}_microns_joinAB.xlsx'.format(mtd,
                                                                                                   z_direction,
                                                                                                   filter_step_size,
                                                                                                   in_plane_distance_threshold,
                                                                                                   min_cm)
                )

                # compute the true mean rmse (bins = 1)
                dfBB_icps_mean = dfBB_icps[['errx', 'erry', 'errxy', 'errz']]
                dfBB_icps_mean['bin'] = 1
                dfBB_icps_mean['rmse_errx'] = dfBB_icps_mean['errx'] ** 2
                dfBB_icps_mean['rmse_erry'] = dfBB_icps_mean['erry'] ** 2
                dfBB_icps_mean['rmse_errxy'] = dfBB_icps_mean['errxy'] ** 2
                dfBB_icps_mean['rmse_errz'] = dfBB_icps_mean['errz'] ** 2
                dfBB_icps_mean['rmse_errxyz'] = dfBB_icps_mean['errx'] ** 2 + dfBB_icps_mean['erry'] ** 2 + dfBB_icps_mean['errz'] ** 2
                dfBB_icps_mean = dfBB_icps_mean.groupby('bin').mean().reset_index()
                dfBB_icps_mean['rmse_errx'] = np.sqrt(dfBB_icps_mean['rmse_errx'])
                dfBB_icps_mean['rmse_erry'] = np.sqrt(dfBB_icps_mean['rmse_erry'])
                dfBB_icps_mean['rmse_errxy'] = np.sqrt(dfBB_icps_mean['rmse_errxy'])
                dfBB_icps_mean['rmse_errz'] = np.sqrt(dfBB_icps_mean['rmse_errz'])
                dfBB_icps_mean['rmse_errxyz'] = np.sqrt(dfBB_icps_mean['rmse_errxyz'])
                dfBB_icps_mean = dfBB_icps_mean[['bin', 'rmse_errx', 'rmse_erry', 'rmse_errxy', 'rmse_errz', 'rmse_errxyz']]
                dfBB_icps_mean.to_excel(
                    path_results_mtd +
                    '/{}_icp-mean-rmse-xyz-1bin_{}_errlim{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                                      z_direction,
                                                                                                      filter_step_size,
                                                                                                      in_plane_distance_threshold,
                                                                                                      min_cm)
                )

                # -------------------
                # bin by radial and axial position

                # 2d-bin by r and z
                bin_r_z = True  # True False
                if bin_r_z:

                    dftt = dfBB_icps.copy()
                    z_trues = dftt.z_true.unique()

                    pzs = ['errx', 'erry', 'errz']
                    nzs = ['rmse_x', 'rmse_y', 'rmse_z']
                    columns_to_bin = [rsub, 'z_true']
                    column_to_count = 'id'
                    bins = [3, z_trues]
                    round_to_decimals = [1, 3]
                    min_num_bin = 5
                    return_groupby = True
                    plot_fit = False

                    for pz, nz in zip(pzs, nzs):

                        dft = dftt.copy()

                        plot_rmse = True
                        if plot_rmse:
                            save_id = nz
                            dft[nz] = dftt[pz] ** 2
                            pz = nz
                        else:
                            save_id = nz

                        dfm, dfstd = bin.bin_generic_2d(dft,
                                                        columns_to_bin,
                                                        column_to_count,
                                                        bins,
                                                        round_to_decimals,
                                                        min_num_bin,
                                                        return_groupby
                                                        )

                        # resolve floating point bin selecting
                        dfm = dfm.round({'bin_tl': 0, 'bin_ll': 2})
                        dfstd = dfstd.round({'bin_tl': 0, 'bin_ll': 2})

                        dfm = dfm.sort_values(['bin_tl', 'bin_ll'])
                        dfstd = dfstd.sort_values(['bin_tl', 'bin_ll'])

                        mean_rmse_z = np.round(np.mean(np.sqrt(dfm[pz])), 3)

                        # plot
                        fig, ax = plt.subplots()

                        for i, bin_r in enumerate(dfm.bin_tl.unique()):
                            dfbr = dfm[dfm['bin_tl'] == bin_r]
                            dfbr_std = dfstd[dfstd['bin_tl'] == bin_r]

                            if plot_rmse:
                                ax.plot(dfbr.bin_ll, np.sqrt(dfbr[pz]), '-o', ms=2.5,
                                        label='{}, {}'.format(int(np.round(bin_r, 0)),
                                                              np.round(np.mean(np.sqrt(dfbr[pz])), 3)))
                                ylbl = r'$\sigma_{z}^{\delta} \: (\mu m)$'
                            else:
                                # scatter: mean +/- std
                                ax.errorbar(dfbr.bin_ll, dfbr[pz], yerr=dfbr_std[pz],
                                            fmt='-o', ms=2, elinewidth=0.5, capsize=1, label=int(np.round(bin_r, 0)))
                                ylbl = r'$\epsilon_{z} \: (\mu m)$'

                        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                        # ax.set_xlim([10, 90])
                        # ax.set_xticks(ticks=xyticks, labels=xyticks)
                        # ax.set_ylim([0, 3.5])
                        # ax.set_yticks(ticks=yerr_ticks, labels=yerr_ticks)
                        ax.legend(loc='upper left', title=r'$r_{bin}, \overline{\sigma_{z}}$',
                                  borderpad=0.2, handletextpad=0.6, borderaxespad=0.25, markerscale=0.75)
                        ax.set_ylabel(ylbl)
                        ax.set_title('mean {} = {} microns'.format(pz, mean_rmse_z))
                        plt.savefig(path_results_mtd + '/bin-r-z_{}-{}_by_z.png'.format(save_id, pz))
                        plt.tight_layout()
                        plt.show()
                        plt.close()

                del dfBB_icps

                # -------------------

                df_icp = pd.DataFrame(np.array(data), columns=['frame', 'zA', 'zB', 'z',
                                                               'precision', 'rmse',
                                                               'dx', 'dy', 'dz',
                                                               'rmse_x', 'rmse_y', 'rmse_z',
                                                               'num_icp', 'numA', 'numB'])
                df_icp.to_excel(
                    path_results_mtd +
                    '/{}_dfdz-per-frame_icp_{}_zerror{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                        z_direction,
                                                                                        filter_step_size,
                                                                                        in_plane_distance_threshold,
                                                                                        min_cm,
                                                                                        )
                )
                data_.append(df_icp)

                # bin z
                df_icp = df_icp.groupby('z').mean().reset_index()
                df_icp.to_excel(
                    path_results_mtd +
                    '/{}_dfdz_icp_{}_zerror{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                        z_direction,
                                                                                        filter_step_size,
                                                                                        in_plane_distance_threshold,
                                                                                        min_cm,
                                                                                        )
                )

                # -

                # plot accuracy of rigid transformations
                plot_rt_accuracy = True  # True False
                if plot_rt_accuracy:

                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                        figsize=(size_x_inches, size_y_inches * 1.5))

                    ax1.plot(df_icp.z, df_icp.rmse, '-o', label='rmse(R.T.)')
                    ax1.plot(df_icp.z, df_icp.precision, '-o', label='precision(R.T.)')
                    ax1.set_ylabel('Transform')
                    ax1.legend()

                    ax2.plot(df_icp.z, df_icp.dx, '-o', label='dx')
                    ax2.plot(df_icp.z, df_icp.dy, '-o', label='dy')
                    ax2.plot(df_icp.z, df_icp.dz + dz_normalize, '-o', label='|dz|-5')
                    ax2.set_ylabel('displacement (um)')
                    ax2.legend()

                    ax3.plot(df_icp.z, df_icp.rmse_x, '-o', label='x')
                    ax3.plot(df_icp.z, df_icp.rmse_y, '-o', label='y')
                    ax3.plot(df_icp.z, df_icp.rmse_z, '-o', label='z')
                    ax3.set_ylabel('r.m.s. error (um)')
                    ax3.legend()
                    plt.tight_layout()
                    plt.savefig(
                        join(path_results_mtd,
                             'RT_accuracy_{}_zerror{}_inplanedist{}_Cmin{}_microns.png'.format(z_direction,
                                                                                               filter_step_size,
                                                                                               in_plane_distance_threshold,
                                                                                               min_cm,
                                                                                               )
                             )
                    )
                    plt.show()

                # ---

            # plot accuracy of rigid transformations
            plot_rt_accuracy_avg_dirs = False
            if plot_rt_accuracy_avg_dirs:
                df_icp_ = pd.concat(data_)
                df_icp_['dx'] = df_icp_['dx'].abs()
                df_icp_['dy'] = df_icp_['dy'].abs()
                df_icp_['dz'] = df_icp_['dz'].abs()
                df_icp_.to_excel(
                    path_results + '/' + mtd +
                    '/{}_dfdz_icp_both-dirs_zerror{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                               z_direction,
                                                                                               filter_step_size,
                                                                                               in_plane_distance_threshold,
                                                                                               min_cm,
                                                                                               )
                )
                df_icp = df_icp_.groupby('z').mean().reset_index()
                df_icp.to_excel(
                    path_results + '/' + mtd +
                    '/{}_dfdz_icp_avg-dirs_zerror{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                              filter_step_size,
                                                                                              in_plane_distance_threshold,
                                                                                              min_cm)
                )

                # -

                # plot
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))

                ax1.plot(df_icp.z, df_icp.rmse, '-o', label='rmse(R.T.)')
                ax1.plot(df_icp.z, df_icp.precision, '-o', label='precision(R.T.)')
                ax1.set_ylabel('Transform')
                ax1.legend()

                ax2.plot(df_icp.z, df_icp.dx, '-o', label='dx')
                ax2.plot(df_icp.z, df_icp.dy, '-o', label='dy')
                ax2.plot(df_icp.z, df_icp.dz - 5, '-o', label='dz-5')
                ax2.set_ylabel('displacement (microns)')
                ax2.legend()

                ax3.plot(df_icp.z, df_icp.rmse_x, '-o', label='x')
                ax3.plot(df_icp.z, df_icp.rmse_y, '-o', label='y')
                ax3.plot(df_icp.z, df_icp.rmse_z, '-o', label='z')
                ax3.set_ylabel('r.m.s. error (microns)')
                ax3.legend()
                plt.tight_layout()
                plt.savefig(join(path_figs, mtd,
                                 'RT_accuracy_avg-dirs_z-error-{}_in-plane-dist-{}_units=microns.png'.format(
                                     filter_step_size,
                                     in_plane_distance_threshold)))
                plt.show()

            else:
                del data_
        # ---

        # evaluate between the 3 frames per z
        eval_between_3_frames = False
        """
        NOTE: I haven't updated the below 'eval_between_3_frames' code since ~10/2022. So, it needs updating if I want
        to use it all (maybe not necessary).  
        """
        if eval_between_3_frames:

            # iterate
            zts = dft.z_true.unique()
            data = []
            for zt in zts:
                dfzt = dft[dft['z_true'] == zt]

                frs = dfzt.frame.unique()
                frs.sort()

                ddx, ddy, ddz = [], [], []
                z_distances = []
                for i, fr in enumerate(frs):
                    if i == len(frs) - 1:
                        ii = 0
                    else:
                        ii = i + 1
                    dfA = dfzt[dfzt['frame'] == fr]
                    dfB = dfzt[dfzt['frame'] == frs[ii]]

                    A = dfA[['x', 'y', 'z']].to_numpy()
                    B = dfB[['x', 'y', 'z']].to_numpy()

                    if len(A) > len(B):
                        ground_truth_xy = A[:, :2]
                        ground_truth_pids = dfA.id.to_numpy()
                        locations = B[:, :2]
                    else:
                        ground_truth_xy = B[:, :2]
                        ground_truth_pids = dfB.id.to_numpy()
                        locations = A[:, :2]

                    # calcualte distance using NearestNeighbors
                    nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_xy)
                    distances, indices = nneigh.kneighbors(locations)

                    distances = np.where(distances < in_plane_distance_threshold, distances, np.nan)

                    nearest_pid = ground_truth_pids[indices]
                    nearest_pid = np.where(distances < in_plane_distance_threshold, nearest_pid, np.nan)
                    nearest_pid = nearest_pid[~np.isnan(nearest_pid)]

                    # nearest_pid = np.ravel(ground_truth_pids[indices])

                    uniq_pids = np.unique(nearest_pid)

                    # iterative closest point algorithm
                    dfAA = dfA[dfA.id.isin(uniq_pids)]
                    dfBB = dfB[dfB.id.isin(uniq_pids)]

                    if len(dfAA) > len(dfBB):
                        uniq_pids = dfBB.id.unique()
                        dfAA = dfAA[dfAA.id.isin(uniq_pids)]
                    elif len(dfAA) < len(dfBB):
                        uniq_pids = dfAA.id.unique()
                        dfBB = dfBB[dfBB.id.isin(uniq_pids)]

                    A = dfAA[['xg', 'yg', 'z']].to_numpy()
                    B = dfBB[['xg', 'yg', 'z']].to_numpy()

                    print(len(A))
                    print(len(B))

                    N = len(A)
                    T, distances, iterations = icp.icp(B, A, tolerance=0.000001)

                    # Make C a homogeneous representation of B
                    C = np.ones((N, 4))
                    C[:, 0:3] = np.copy(B)

                    # Transform C
                    C = np.dot(T, C.T).T

                    # evaluate transformation results
                    ddx.append(T[0, 3])
                    ddy.append(T[1, 3])
                    ddz.append(T[2, 3])
                    # deltax, deltay, deltaz = T[0, 3], T[1, 3], T[2, 3]
                    z_distances.append(distances.tolist())

                # error analysis
                z_distances = np.array(list(itertools.chain(*z_distances)))
                precision_dist = np.std(z_distances)
                rmse_dist = np.sqrt(np.mean(np.square(z_distances)))
                std_dx = np.std(ddx)
                std_dy = np.std(ddy)
                std_dz = np.std(ddz)

                data.append([zt, precision_dist, rmse_dist, std_dx, std_dy, std_dz])

            df_icp = pd.DataFrame(np.array(data),
                                  columns=['z', 'precision', 'rmse', 'std_dx', 'std_dy', 'std_dz'])
            df_icp.to_excel(path_results +
                            '/df_icp_z-error-{}_in-plane-dist-{}.xlsx'.format(filter_step_size,
                                                                              in_plane_distance_threshold))

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            ax1.plot(df_icp.z, df_icp.rmse, label='rmse')
            ax1.plot(df_icp.z, df_icp.precision, label='precision')
            ax1.legend()
            ax2.plot(df_icp.z, df_icp.std_dx, label='x')
            ax2.plot(df_icp.z, df_icp.std_dy, label='y')
            ax2.plot(df_icp.z, df_icp.std_dz, label='z')
            ax2.legend()
            plt.show()

    # ---

# ---

# ---------------------------------------------------------------------
# 3. CALCULATE RIGID TRANSFORMATIONS VIA ICP -- WHERE THE PARTICLE POSISTIONS ARE AVERAGED AT EACH Z-POSITION

""" THIS IS THE OLDER WAY """

calculate_rigid_transforms = False  # True False
if calculate_rigid_transforms:

    # read coords

    use_raw = False
    use_meta = False
    use_focus = True
    min_cm = 0.5  # similarity threshold

    if use_raw:
        # OPTION 1: "RAW" COORDINATES
        fpi = '/test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
        fps = '/test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'  # z is tilt-corrected
        path_read_err_rel_calib = None
    elif use_raw is False and use_meta is True:
        fpi = None
        fps = '/spct_meta_cmin0.5_test_coords_post-processed.xlsx'  # z is raw
        path_read_err_rel_calib = None
    else:
        # OPTION 2: POST-PROCESSED COORDINATES FROM "ERROR RELATIVE CALIBRATION PARTICLE"
        path_read_err_rel_calib = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/' \
                                  '11.06.21_error_relative_calib_particle/results/' \
                                  'relative-to-tilt-corr-calib-particle/' \
                                  'zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts) + \
                                  '/ztrue_is_fit-plane-xyzc'
        fpi = 'idpt_error_relative_calib_particle_zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts)
        fps = 'spct_error_relative_calib_particle_zerrlim{}_cmin{}_mincountsallframes{}'.format(z_error_limit, min_cm, min_counts)

    # ---
    if use_focus:

        path_results_true_positions = join(path_figs, 'idpt', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'true_xy')
        if not os.path.exists(path_results_true_positions):
            os.makedirs(path_results_true_positions)

        fp_xy_at_zf = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/11.06.21_calc_rigid_body_displacement/' \
                      'coords/true_positions_fiji.xlsx'
        dfxyzf = pd.read_excel(fp_xy_at_zf)

        # shift locations to match IDPT + padding
        padding_rel_true_x = 0
        padding_rel_true_y = 0
        dfxyzf['x'] = dfxyzf['x'] + padding_rel_true_x
        dfxyzf['y'] = dfxyzf['y'] + padding_rel_true_y
        dfxyzf['gauss_xc'] = dfxyzf['gauss_xc'] + padding_rel_true_x
        dfxyzf['gauss_yc'] = dfxyzf['gauss_yc'] + padding_rel_true_y

        dfxyzf['z'] = 0
        dfxyzf['gauss_rc'] = np.sqrt(dfxyzf['gauss_xc'] ** 2 + dfxyzf['gauss_yc'] ** 2)

        for pix2microns in ['gauss_xc', 'gauss_yc', 'gauss_rc']:
            dfxyzf[pix2microns] = dfxyzf[pix2microns] * microns_per_pixel

        dfxyzf['x'] = dfxyzf['gauss_xc']
        dfxyzf['y'] = dfxyzf['gauss_yc']
        dfxyzf['r'] = dfxyzf['gauss_rc']

        """fig, ax = plt.subplots()
        ax.scatter(dfxyzf.x, dfxyzf.y, s=dfxyzf.contour_area, c=dfxyzf.id)
        ax.invert_yaxis()
        ax.set_title('CoM positions from FIJI')
        plt.savefig(path_results_true_positions + '/fiji_positions.png')
        plt.show()"""

    else:
        dfxyzf = None

    # ---

    # -

    """
    NOTE TO SELF:
        1. the "initial" point cloud should (could?) be the positions of best focus at z=0.
            a. for SPCT, you might not even need the tilt or field curvature corrections. 
        2. then extrapolated outwards in both directions (+z and -z).
    """


    # function from ICP
    def nearest_neighbor(src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()


    def distances_xyz(AR, BR, TR):
        """ xdists, ydists, zdists = distances_xyz(AR=, BR=) """

        # get number of dimensions
        m = AR.shape[1]
        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, AR.shape[0]))
        dst = np.ones((m + 1, BR.shape[0]))
        src[:m, :] = np.copy(AR.T)
        dst[:m, :] = np.copy(BR.T)

        # update the current source
        src_transformed = np.dot(TR, src)
        src_transformed = src_transformed.T

        # calculate x, y, and z distances (errors)
        x_distances = src_transformed[:, 0] - BR[:, 0]
        y_distances = src_transformed[:, 1] - BR[:, 1]
        z_distances = src_transformed[:, 2] - BR[:, 2]

        return x_distances, y_distances, z_distances


    # ---

    # iterative closest point algorithm
    for fp, mtd in zip([fpi, fps], ['idpt', 'spct']):
        # for fp, mtd in zip([fps, fpi], ['spct', 'idpt']):

        if not os.path.exists(join(path_figs, mtd)):
            os.makedirs(join(path_figs, mtd))

        if use_raw:
            path_scatter_yz = join(path_figs, mtd, 'scatter_yz_A2B')
            path_results_mtd = join(path_results, mtd, 'raw')
        elif use_raw is False and use_meta is True:
            path_scatter_yz = join(path_figs, mtd, 'scatter_yz_A2B_meta')
            path_results_mtd = join(path_results, mtd, 'meta')
        elif use_focus:
            path_scatter_yz = join(path_figs, mtd, 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'scatter_yz_A2B_rel-calib')
            path_results_mtd = join(path_results, mtd, 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'rel-calib')
        else:
            path_scatter_yz = join(path_figs, mtd, 'ztrue_is_fit-plane-xyzc', 'scatter_yz_A2B_rel-calib')
            path_results_mtd = join(path_results, mtd, 'ztrue_is_fit-plane-xyzc', 'rel-calib')

        if not os.path.exists(path_scatter_yz):
            os.makedirs(path_scatter_yz)
        if not os.path.exists(path_results_mtd):
            os.makedirs(path_results_mtd)

        if use_raw or use_meta:
            if mtd == 'spct':
                xsub, ysub, rsub = 'gauss_xc', 'gauss_yc', 'gauss_rc'
                # continue
            else:
                xsub, ysub, rsub = 'xg', 'yg', 'rg'
                continue

            # read coords
            dft = pd.read_excel(path_coords + fp)
            dft['gauss_rc'] = np.sqrt(dft['gauss_xc'] ** 2 + dft['gauss_yc'] ** 2)
            dft['error'] = dft['error_z']

            ############# THESE FUNCTIONS "MAY" ONLY APPLY TO "USE_META"... I'm not sure
            dft['z'] = dft['z'] - 49.91
            dft['z_true'] = dft['z_true'] - 49.91
            dft['z_calib'] = dft['z_true']
            dft = dft.dropna()
            #############

            # get only useful columns
            dft = dft[['frame', 'id', xsub, ysub, rsub, 'z_true', 'z', 'error', 'cm', 'x', 'y']]

            # filters
            dft = dft[dft['error'].abs() < filter_step_size]
            dft = dft[dft['cm'] > min_cm]

            fig, ax = plt.subplots()
            ax.scatter(dft.z_true, dft.z - dft.z_true, c=dft.id, s=1)
            plt.show()

            # adjust units from pixels to microns
            for pix2microns in ['x', 'y', xsub, ysub, rsub]:
                dft[pix2microns] = dft[pix2microns] * microns_per_pixel

        else:
            if mtd == 'spct':
                pass  # continue pass
            else:
                if min_cm == 0.9:
                    continue  # continue pass
                else:
                    pass  # continue pass

            xsub, ysub, rsub = 'x', 'y', 'r'  # these are already the sub-pixel positions

            # read coords
            dft = pd.read_excel(join(path_read_err_rel_calib, fp + '.xlsx'))

            # get only useful columns
            dft = dft[['frame', 'id', 'cm', 'x', 'y', 'r', 'z_true', 'z_no_corr', 'z', 'z_calib', 'error_rel_p_calib']]

            dft['z_nominal'] = dft['z_true']
            dft['error'] = dft['error_rel_p_calib']

            # filters
            dft = dft[dft['error'].abs() < filter_step_size]
            dft = dft[dft['cm'] > min_cm]

            # adjust units from pixels to microns
            for pix2microns in ['x', 'y', 'r']:
                dft[pix2microns] = dft[pix2microns] * microns_per_pixel

        # ---

        # ---

        # evaluate axial displacement (~5 microns) between the 3 frames per z
        eval_between_dz_frames = True
        if eval_between_dz_frames:

            # iterate through z-directions
            data_ = []
            for z_direction in ['neg2pos']:

                # get z_true values and sort
                zts = dft.z_true.sort_values(ascending=True).unique()
                if use_focus:
                    dz_normalize = 0
                else:
                    dz_normalize = -1  # -5

                data = []
                ddx, ddy, ddz = [], [], []
                z_distances = []
                dfBB_icps = []

                if use_focus:
                    z_range_rt = len(zts)
                    """
                    if not os.path.exists(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx'):
                        frames_and_ztrues = dft.sort_values('frame')[['frame', 'z_true']].values.tolist()
                        frames_and_ztrues = np.unique(np.array(frames_and_ztrues), axis=0)
                        temp = []
                        for ft, zt in frames_and_ztrues:
                            df_true['z_true'] = zt
                            df_true['z'] = zt
                            df_true['frame'] = ft
                            temp.append(df_true.copy())
                        df_true = pd.concat(temp)
                        df_true.to_excel(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx', index=False)

                    else:
                        df_true = pd.read_excel(path_results_true_coords + '/true_fiji_in-focus_test_by_frames.xlsx')
                    """
                else:
                    z_range_rt = len(zts) - 1

                for ii in range(z_range_rt):

                    if use_focus:
                        dfA = dfxyzf.copy()
                        dfB = dft[dft['z_true'] == zts[ii]].groupby('id').mean().reset_index()
                        dfA['z'] = 0  # zts[ii]

                        # dfA = dft[dft['z_true'] == zts[ii]].groupby('id').mean().reset_index()
                        # dfB = dft[dft['z_true'] == zts[ii + 1]].groupby('id').mean().reset_index()

                    else:
                        dfA = dft[dft['z_true'] == zts[ii]].groupby('id').mean().reset_index()
                        dfB = dft[dft['z_true'] == zts[ii + 1]].groupby('id').mean().reset_index()

                    # -

                    # scatter plot points
                    plot_inputs_scatter = False
                    if plot_inputs_scatter:
                        fig, ax = plt.subplots()
                        if use_raw or use_meta:
                            ax.scatter(dfA['y'], dfA['z'] - zts[ii], label='A')
                            ax.scatter(dfB['y'], dfB['z'] - zts[ii + 1], label='B')
                            save_id_scatter_yz = 'scatter-y_z-from'
                        else:
                            ax.scatter(dfA['y'], dfA['z'] - dfA['z_calib'], label='A')
                            ax.scatter(dfB['y'], dfB['z'] - dfB['z_calib'], label='B')
                            save_id_scatter_yz = 'scatter-y_z-rel-calib-from'

                        ax.set_xlabel('y (microns)')
                        ax.set_ylabel('z')
                        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        plt.savefig(join(path_scatter_yz,
                                         '{}_{}_{}_to_{}.png'.format(mtd, save_id_scatter_yz,
                                                                     np.round(zts[ii], 1),
                                                                     np.round(zts[ii + 1], 1))))

                        continue

                    # -

                    A = dfA[['x', 'y', 'z']].to_numpy()
                    B = dfB[['x', 'y', 'z']].to_numpy()

                    i_num_A = len(A)
                    i_num_B = len(B)

                    # minimum number of particles per frame threshold
                    if np.min([i_num_A, i_num_B]) < min_num_particles_for_icp:
                        continue

                    # match particle positions between A and B using NearestNeighbors
                    if len(A) > len(B):
                        ground_truth_xy = A[:, :2]
                        ground_truth_pids = dfA.id.to_numpy()
                        locations = B[:, :2]

                        fit_to_pids = dfB.id.to_numpy()
                        A_longer = True

                    else:
                        ground_truth_xy = B[:, :2]
                        ground_truth_pids = dfB.id.to_numpy()
                        locations = A[:, :2]

                        fit_to_pids = dfA.id.to_numpy()
                        A_longer = False

                    # calcualte distance using NearestNeighbors
                    nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_xy)
                    distances, indices = nneigh.kneighbors(locations)

                    ############## make True/False array
                    ################ 6/14/2023
                    use_new_icp_pairing = True
                    if use_new_icp_pairing:
                        if A_longer:
                            idx_locations = np.arange(len(locations))
                            idx_locations = idx_locations[:, np.newaxis]
                            sorted_uniq_pids = np.where(distances < in_plane_distance_threshold, indices, np.nan)
                            sorted_uniq_pids = flatten_list_of_lists(sorted_uniq_pids)

                            sorted_fit_uniq_pids = np.where(distances < in_plane_distance_threshold, idx_locations, np.nan)
                            sorted_fit_uniq_pids = flatten_list_of_lists(sorted_fit_uniq_pids)
                        else:
                            idx_locations = np.arange(len(locations))
                            idx_locations = idx_locations[:, np.newaxis]
                            sorted_uniq_pids = np.where(distances < in_plane_distance_threshold, idx_locations, np.nan)
                            sorted_uniq_pids = flatten_list_of_lists(sorted_uniq_pids)

                            sorted_fit_uniq_pids = np.where(distances < in_plane_distance_threshold, indices, np.nan)
                            sorted_fit_uniq_pids = flatten_list_of_lists(sorted_fit_uniq_pids)

                        # remove NaNs from array: x = x[~numpy.isnan(x)]
                        sorted_uniq_pids = np.array(sorted_uniq_pids)
                        sorted_fit_uniq_pids = np.array(sorted_fit_uniq_pids)

                        sorted_uniq_pids = sorted_uniq_pids[~np.isnan(sorted_uniq_pids)]
                        sorted_fit_uniq_pids = sorted_fit_uniq_pids[~np.isnan(sorted_fit_uniq_pids)]

                        ##############

                    else:

                        distances = np.where(distances < in_plane_distance_threshold, distances, np.nan)

                        # these are the p_ID's from "ground_truth_xy"
                        nearest_pid = ground_truth_pids[indices]
                        nearest_pid = np.where(distances < in_plane_distance_threshold, nearest_pid, np.nan)
                        nearest_pid = nearest_pid[~np.isnan(nearest_pid)]
                        uniq_pids = np.unique(nearest_pid)

                        # these are the p_ID's from "locations"
                        fit_to_pids = np.where(distances < in_plane_distance_threshold, fit_to_pids, np.nan)
                        fit_to_pids = fit_to_pids[~np.isnan(fit_to_pids)]
                        fit_uniq_pids = np.unique(fit_to_pids)

                    ################ 6/14/2023
                    use_new_icp_pairing = True
                    if use_new_icp_pairing:
                        if A_longer:  # meaning: A is "ground truth"
                            # dfAA = dfA[dfA.id.isin(uniq_pids)]
                            # dfBB = dfB[dfB.id.isin(fit_uniq_pids)]

                            dfAA = dfA.iloc[sorted_uniq_pids]
                            dfBB = dfB.iloc[sorted_fit_uniq_pids]

                        else:  # meaning: B is "ground truth"
                            # dfAA = dfA[dfA.id.isin(fit_uniq_pids)]
                            # dfBB = dfB[dfB.id.isin(uniq_pids)]

                            dfAA = dfA.iloc[sorted_fit_uniq_pids]
                            dfBB = dfB.iloc[sorted_uniq_pids]

                    else:
                        # iterative closest point algorithm
                        dfAA = dfA[dfA.id.isin(uniq_pids)]
                        dfBB = dfB[dfB.id.isin(uniq_pids)]

                        if len(dfAA) > len(dfBB):
                            uniq_pids = dfBB.id.unique()
                            dfAA = dfAA[dfAA.id.isin(uniq_pids)]
                        elif len(dfAA) < len(dfBB):
                            uniq_pids = dfAA.id.unique()
                            dfBB = dfBB[dfBB.id.isin(uniq_pids)]
                    ################

                    """fig, ax = plt.subplots()
                    ax.scatter(dfAA['x'], dfAA['y'], color='r', label='AA')
                    ax.scatter(dfBB['x'], dfBB['y'], color='b', label='BB')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.legend()
                    plt.show()
                    raise ValueError()"""

                    A = dfAA[[xsub, ysub, 'z']].to_numpy()
                    B = dfBB[[xsub, ysub, 'z']].to_numpy()

                    # print(len(A))
                    # print(len(B))

                    N = len(A)
                    # T, distances, iterations = icp.icp(A, B, tolerance=0.000001)  # originally: icp.icp(B, A,...)
                    T, distances, iterations, dst_indices = pci_icp(A, B, tolerance=0.000001)  # returns indices

                    # ------------ distance by direction ------------
                    xdists, ydists, zdists = distances_xyz(AR=A, BR=B, TR=T)
                    rmse_xdist = np.sqrt(np.mean(np.square(xdists)))
                    rmse_ydist = np.sqrt(np.mean(np.square(ydists)))
                    rmse_zdist = np.sqrt(np.mean(np.square(zdists)))

                    # get matching indices from dfBB
                    dfBB_icp = dfBB.iloc[dst_indices]
                    dfBB_icp['errx'] = xdists
                    dfBB_icp['erry'] = ydists
                    dfBB_icp['errxy'] = np.sqrt(dfBB_icp['errx'] ** 2 + dfBB_icp['erry'] ** 2)
                    dfBB_icp['errz'] = zdists

                    # if you want dfBB and dfAA (below), or, if you only want dfBB (below, below)
                    include_AA = True
                    if include_AA:
                        # dfAA should already be the correct indices
                        dfAA_icp = dfAA[['id', xsub, ysub, 'z']]
                        dfAA_icp = dfAA_icp.rename(
                            columns={'id': 'a_id', xsub: 'a_' + xsub, ysub: 'a_' + ysub, 'z': 'a_z'})

                        # reset the natural index of both dfAA and dfBB
                        dfAA_icp = dfAA_icp.reset_index(drop=True)
                        dfBB_icpAB = dfBB_icp.reset_index(drop=True)

                        # concat
                        dfBB_icpAB = pd.concat([dfBB_icpAB, dfAA_icp], axis=1)
                        dfBB_icpAB['ab_errx'] = dfBB_icpAB['a_' + xsub] - dfBB_icpAB[xsub]
                        dfBB_icpAB['ab_erry'] = dfBB_icpAB['a_' + ysub] - dfBB_icpAB[ysub]
                        dfBB_icpAB['ab_errxy'] = np.sqrt(dfBB_icpAB['ab_errx'] ** 2 + dfBB_icpAB['ab_erry'] ** 2)
                        dfBB_icpAB['ab_errz'] = dfBB_icpAB['a_z'] - dfBB_icpAB['z']

                        # remove rows with x-y errors that exceed limit
                        dfBB_icpAB = dfBB_icpAB[(dfBB_icpAB['errx'].abs() < in_plane_distance_threshold) &
                                                (dfBB_icpAB['erry'].abs() < in_plane_distance_threshold)]

                        dfBB_icps.append(dfBB_icpAB)

                    else:
                        # remove rows with x-y errors that exceed limit
                        dfBB_icp = dfBB_icp[(dfBB_icp['errx'].abs() < in_plane_distance_threshold) &
                                            (dfBB_icp['erry'].abs() < in_plane_distance_threshold)]

                        dfBB_icps.append(dfBB_icp)

                    # Make C a homogeneous representation of B
                    C = np.ones((N, 4))
                    C[:, 0:3] = np.copy(B)

                    # Transform C
                    C = np.dot(T, C.T).T

                    # evaluate transformation results
                    # ddx.append(T[0, 3])
                    # ddy.append(T[1, 3])
                    # ddz.append(T[2, 3])
                    deltax, deltay, deltaz = T[0, 3], T[1, 3], T[2, 3]
                    # z_distances.append(distances.tolist())
                    precision_dist = np.std(distances)
                    rmse_dist = np.sqrt(np.mean(np.square(distances)))
                    if use_focus:
                        data.append([0, zts[ii], zts[ii], precision_dist, rmse_dist, deltax, deltay, deltaz,
                                     rmse_xdist, rmse_ydist, rmse_zdist, len(distances), i_num_A, i_num_B])
                    else:
                        data.append([zts[ii], zts[ii + 1], zts[ii + 1], precision_dist, rmse_dist, deltax, deltay, deltaz,
                                     rmse_xdist, rmse_ydist, rmse_zdist, len(distances), i_num_A, i_num_B])

                # package
                # ------------------
                dfBB_icps = pd.concat(dfBB_icps)
                dfBB_icps.to_excel(
                    path_results_mtd +
                    '/{}_icp-dist-xyz_{}_errlim{}_inplanedist{}_Cmin{}_microns_joinAB.xlsx'.format(mtd,
                                                                                                   z_direction,
                                                                                                   filter_step_size,
                                                                                                   in_plane_distance_threshold,
                                                                                                   min_cm)
                )

                # compute the true mean rmse (bins = 1)
                dfBB_icps_mean = dfBB_icps[['errx', 'erry', 'errxy', 'errz']]
                dfBB_icps_mean['bin'] = 1
                dfBB_icps_mean['rmse_errx'] = dfBB_icps_mean['errx'] ** 2
                dfBB_icps_mean['rmse_erry'] = dfBB_icps_mean['erry'] ** 2
                dfBB_icps_mean['rmse_errxy'] = dfBB_icps_mean['errxy'] ** 2
                dfBB_icps_mean['rmse_errz'] = dfBB_icps_mean['errz'] ** 2
                dfBB_icps_mean = dfBB_icps_mean.groupby('bin').mean().reset_index()
                dfBB_icps_mean['rmse_errx'] = np.sqrt(dfBB_icps_mean['rmse_errx'])
                dfBB_icps_mean['rmse_erry'] = np.sqrt(dfBB_icps_mean['rmse_erry'])
                dfBB_icps_mean['rmse_errxy'] = np.sqrt(dfBB_icps_mean['rmse_errxy'])
                dfBB_icps_mean['rmse_errz'] = np.sqrt(dfBB_icps_mean['rmse_errz'])
                dfBB_icps_mean = dfBB_icps_mean[['bin', 'rmse_errx', 'rmse_erry', 'rmse_errxy', 'rmse_errz']]
                dfBB_icps_mean.to_excel(
                    path_results_mtd +
                    '/{}_icp-mean-rmse-xyz-1bin_{}_errlim{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                                      z_direction,
                                                                                                      filter_step_size,
                                                                                                      in_plane_distance_threshold,
                                                                                                      min_cm)
                )

                # -------------------
                # bin by radial and axial position

                # 2d-bin by r and z
                bin_r_z = True  # True False
                if bin_r_z:

                    dftt = dfBB_icps.copy()
                    z_trues = dftt.z_true.unique()

                    pzs = ['errx', 'erry', 'errz']
                    nzs = ['rmse_x', 'rmse_y', 'rmse_z']
                    columns_to_bin = [rsub, 'z_true']
                    column_to_count = 'id'
                    bins = [3, z_trues]
                    round_to_decimals = [1, 3]
                    min_num_bin = 5
                    return_groupby = True
                    plot_fit = False

                    for pz, nz in zip(pzs, nzs):

                        dft = dftt.copy()

                        plot_rmse = True
                        if plot_rmse:
                            save_id = nz
                            dft[nz] = dftt[pz] ** 2
                            pz = nz
                        else:
                            save_id = nz

                        dfm, dfstd = bin.bin_generic_2d(dft,
                                                        columns_to_bin,
                                                        column_to_count,
                                                        bins,
                                                        round_to_decimals,
                                                        min_num_bin,
                                                        return_groupby
                                                        )

                        # resolve floating point bin selecting
                        dfm = dfm.round({'bin_tl': 0, 'bin_ll': 2})
                        dfstd = dfstd.round({'bin_tl': 0, 'bin_ll': 2})

                        dfm = dfm.sort_values(['bin_tl', 'bin_ll'])
                        dfstd = dfstd.sort_values(['bin_tl', 'bin_ll'])

                        mean_rmse_z = np.round(np.mean(np.sqrt(dfm[pz])), 3)

                        # plot
                        fig, ax = plt.subplots()

                        for i, bin_r in enumerate(dfm.bin_tl.unique()):
                            dfbr = dfm[dfm['bin_tl'] == bin_r]
                            dfbr_std = dfstd[dfstd['bin_tl'] == bin_r]

                            if plot_rmse:
                                ax.plot(dfbr.bin_ll, np.sqrt(dfbr[pz]), '-o', ms=2.5,
                                        label='{}, {}'.format(int(np.round(bin_r, 0)),
                                                              np.round(np.mean(np.sqrt(dfbr[pz])), 3)))
                                ylbl = r'$\sigma_{z}^{\delta} \: (\mu m)$'
                            else:
                                # scatter: mean +/- std
                                ax.errorbar(dfbr.bin_ll, dfbr[pz], yerr=dfbr_std[pz],
                                            fmt='-o', ms=2, elinewidth=0.5, capsize=1, label=int(np.round(bin_r, 0)))
                                ylbl = r'$\epsilon_{z} \: (\mu m)$'

                        ax.set_xlabel(r'$z_{true} \: (\mu m)$')
                        # ax.set_xlim([10, 90])
                        # ax.set_xticks(ticks=xyticks, labels=xyticks)
                        # ax.set_ylim([0, 3.5])
                        # ax.set_yticks(ticks=yerr_ticks, labels=yerr_ticks)
                        ax.legend(loc='upper left', title=r'$r_{bin}, \overline{\sigma_{z}}$',
                                  borderpad=0.2, handletextpad=0.6, borderaxespad=0.25, markerscale=0.75)
                        ax.set_ylabel(ylbl)
                        ax.set_title('mean {} = {} microns'.format(pz, mean_rmse_z))
                        plt.savefig(path_results_mtd + '/bin-r-z_{}-{}_by_z.png'.format(save_id, pz))
                        plt.tight_layout()
                        plt.show()
                        plt.close()

                del dfBB_icps

                # -------------------

                df_icp = pd.DataFrame(np.array(data), columns=['zA', 'zB', 'z', 'precision', 'rmse', 'dx', 'dy', 'dz',
                                                               'rmse_x', 'rmse_y', 'rmse_z', 'num_icp', 'numA', 'numB'])
                df_icp.to_excel(
                    path_results_mtd +
                    '/{}_dfdz_icp_{}_zerror{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                        z_direction,
                                                                                        filter_step_size,
                                                                                        in_plane_distance_threshold,
                                                                                        min_cm,
                                                                                        )
                )
                data_.append(df_icp)

                # -

                # plot accuracy of rigid transformations
                plot_rt_accuracy = True  # True False
                if plot_rt_accuracy:

                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True,
                                                        figsize=(size_x_inches, size_y_inches * 1.5))

                    ax1.plot(df_icp.z, df_icp.rmse, '-o', label='rmse(R.T.)')
                    ax1.plot(df_icp.z, df_icp.precision, '-o', label='precision(R.T.)')
                    ax1.set_ylabel('Transform')
                    ax1.legend()

                    ax2.plot(df_icp.z, df_icp.dx, '-o', label='dx')
                    ax2.plot(df_icp.z, df_icp.dy, '-o', label='dy')
                    ax2.plot(df_icp.z, df_icp.dz + dz_normalize, '-o', label='|dz|-5')
                    ax2.set_ylabel('displacement (um)')
                    ax2.legend()

                    ax3.plot(df_icp.z, df_icp.rmse_x, '-o', label='x')
                    ax3.plot(df_icp.z, df_icp.rmse_y, '-o', label='y')
                    ax3.plot(df_icp.z, df_icp.rmse_z, '-o', label='z')
                    ax3.set_ylabel('r.m.s. error (um)')
                    ax3.legend()
                    plt.tight_layout()
                    plt.savefig(
                        join(path_results_mtd,
                             'RT_accuracy_{}_zerror{}_inplanedist{}_Cmin{}_microns.png'.format(z_direction,
                                                                                               filter_step_size,
                                                                                               in_plane_distance_threshold,
                                                                                               min_cm,
                                                                                               )
                             )
                    )
                    plt.show()

                # ---

            # plot accuracy of rigid transformations
            plot_rt_accuracy_avg_dirs = False
            if plot_rt_accuracy_avg_dirs:
                df_icp_ = pd.concat(data_)
                df_icp_['dx'] = df_icp_['dx'].abs()
                df_icp_['dy'] = df_icp_['dy'].abs()
                df_icp_['dz'] = df_icp_['dz'].abs()
                df_icp_.to_excel(
                    path_results + '/' + mtd +
                    '/{}_dfdz_icp_both-dirs_zerror{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                               z_direction,
                                                                                               filter_step_size,
                                                                                               in_plane_distance_threshold,
                                                                                               min_cm,
                                                                                               )
                )
                df_icp = df_icp_.groupby('z').mean().reset_index()
                df_icp.to_excel(
                    path_results + '/' + mtd +
                    '/{}_dfdz_icp_avg-dirs_zerror{}_inplanedist{}_Cmin{}_microns.xlsx'.format(mtd,
                                                                                              filter_step_size,
                                                                                              in_plane_distance_threshold,
                                                                                              min_cm)
                )

                # -

                # plot
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))

                ax1.plot(df_icp.z, df_icp.rmse, '-o', label='rmse(R.T.)')
                ax1.plot(df_icp.z, df_icp.precision, '-o', label='precision(R.T.)')
                ax1.set_ylabel('Transform')
                ax1.legend()

                ax2.plot(df_icp.z, df_icp.dx, '-o', label='dx')
                ax2.plot(df_icp.z, df_icp.dy, '-o', label='dy')
                ax2.plot(df_icp.z, df_icp.dz - 5, '-o', label='dz-5')
                ax2.set_ylabel('displacement (microns)')
                ax2.legend()

                ax3.plot(df_icp.z, df_icp.rmse_x, '-o', label='x')
                ax3.plot(df_icp.z, df_icp.rmse_y, '-o', label='y')
                ax3.plot(df_icp.z, df_icp.rmse_z, '-o', label='z')
                ax3.set_ylabel('r.m.s. error (microns)')
                ax3.legend()
                plt.tight_layout()
                plt.savefig(join(path_figs, mtd,
                                 'RT_accuracy_avg-dirs_z-error-{}_in-plane-dist-{}_units=microns.png'.format(
                                     filter_step_size,
                                     in_plane_distance_threshold)))
                plt.show()

            else:
                del data_
        # ---

        # evaluate between the 3 frames per z
        eval_between_3_frames = False
        """
        NOTE: I haven't updated the below 'eval_between_3_frames' code since ~10/2022. So, it needs updating if I want
        to use it all (maybe not necessary).  
        """
        if eval_between_3_frames:

            # iterate
            zts = dft.z_true.unique()
            data = []
            for zt in zts:
                dfzt = dft[dft['z_true'] == zt]

                frs = dfzt.frame.unique()
                frs.sort()

                ddx, ddy, ddz = [], [], []
                z_distances = []
                for i, fr in enumerate(frs):
                    if i == len(frs) - 1:
                        ii = 0
                    else:
                        ii = i + 1
                    dfA = dfzt[dfzt['frame'] == fr]
                    dfB = dfzt[dfzt['frame'] == frs[ii]]

                    A = dfA[['x', 'y', 'z']].to_numpy()
                    B = dfB[['x', 'y', 'z']].to_numpy()

                    if len(A) > len(B):
                        ground_truth_xy = A[:, :2]
                        ground_truth_pids = dfA.id.to_numpy()
                        locations = B[:, :2]
                    else:
                        ground_truth_xy = B[:, :2]
                        ground_truth_pids = dfB.id.to_numpy()
                        locations = A[:, :2]

                    # calcualte distance using NearestNeighbors
                    nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_xy)
                    distances, indices = nneigh.kneighbors(locations)

                    distances = np.where(distances < in_plane_distance_threshold, distances, np.nan)

                    nearest_pid = ground_truth_pids[indices]
                    nearest_pid = np.where(distances < in_plane_distance_threshold, nearest_pid, np.nan)
                    nearest_pid = nearest_pid[~np.isnan(nearest_pid)]

                    # nearest_pid = np.ravel(ground_truth_pids[indices])

                    uniq_pids = np.unique(nearest_pid)

                    # iterative closest point algorithm
                    dfAA = dfA[dfA.id.isin(uniq_pids)]
                    dfBB = dfB[dfB.id.isin(uniq_pids)]

                    if len(dfAA) > len(dfBB):
                        uniq_pids = dfBB.id.unique()
                        dfAA = dfAA[dfAA.id.isin(uniq_pids)]
                    elif len(dfAA) < len(dfBB):
                        uniq_pids = dfAA.id.unique()
                        dfBB = dfBB[dfBB.id.isin(uniq_pids)]

                    A = dfAA[['xg', 'yg', 'z']].to_numpy()
                    B = dfBB[['xg', 'yg', 'z']].to_numpy()

                    print(len(A))
                    print(len(B))

                    N = len(A)
                    T, distances, iterations = icp.icp(B, A, tolerance=0.000001)

                    # Make C a homogeneous representation of B
                    C = np.ones((N, 4))
                    C[:, 0:3] = np.copy(B)

                    # Transform C
                    C = np.dot(T, C.T).T

                    # evaluate transformation results
                    ddx.append(T[0, 3])
                    ddy.append(T[1, 3])
                    ddz.append(T[2, 3])
                    # deltax, deltay, deltaz = T[0, 3], T[1, 3], T[2, 3]
                    z_distances.append(distances.tolist())

                # error analysis
                z_distances = np.array(list(itertools.chain(*z_distances)))
                precision_dist = np.std(z_distances)
                rmse_dist = np.sqrt(np.mean(np.square(z_distances)))
                std_dx = np.std(ddx)
                std_dy = np.std(ddy)
                std_dz = np.std(ddz)

                data.append([zt, precision_dist, rmse_dist, std_dx, std_dy, std_dz])

            df_icp = pd.DataFrame(np.array(data),
                                  columns=['z', 'precision', 'rmse', 'std_dx', 'std_dy', 'std_dz'])
            df_icp.to_excel(path_results +
                            '/df_icp_z-error-{}_in-plane-dist-{}.xlsx'.format(filter_step_size,
                                                                              in_plane_distance_threshold))

            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            ax1.plot(df_icp.z, df_icp.rmse, label='rmse')
            ax1.plot(df_icp.z, df_icp.precision, label='precision')
            ax1.legend()
            ax2.plot(df_icp.z, df_icp.std_dx, label='x')
            ax2.plot(df_icp.z, df_icp.std_dy, label='y')
            ax2.plot(df_icp.z, df_icp.std_dz, label='z')
            ax2.legend()
            plt.show()

    # ---

# ---

# ---------------------------------------------------------------------
# 4. COMPARE RESULTS

compare_rigid_transforms = False  # True False
if compare_rigid_transforms:

    # save paths
    path_results_compare = join(path_results, 'compare', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc',
                                'z-error-limit_{}'.format(z_error_limit))
    path_figs_compare = join(path_figs, 'compare', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc',
                             'zerrlim{}_inplanedist{}'.format(z_error_limit, in_plane_distance_threshold))
    if not os.path.exists(path_results_compare):
        os.makedirs(path_results_compare)
    if not os.path.exists(path_figs_compare):
        os.makedirs(path_figs_compare)

    # read paths
    if z_error_limit == 4:
        fpi = 'idpt_dfdz_icp_neg2pos_zerror4_inplanedist{}_Cmin0.5_microns.xlsx'.format(in_plane_distance_threshold)
        fps = 'spct_dfdz_icp_neg2pos_zerror4_inplanedist{}_Cmin0.5_microns.xlsx'.format(in_plane_distance_threshold)
        fpss = 'spct_dfdz_icp_neg2pos_zerror4_inplanedist{}_Cmin0.9_microns.xlsx'.format(in_plane_distance_threshold)

        dfi = pd.read_excel(join(path_results_compare, fpi))
        dfs = pd.read_excel(join(path_results_compare, fps))
        dfss = pd.read_excel(join(path_results_compare, fpss))

    elif z_error_limit == 10:
        fpi = 'idpt_cm0.5_dfdz_icp_neg2pos_z-error-10_in-plane-dist-5_units=microns.xlsx'
        fps = 'spct_cm0.5_dfdz_icp_neg2pos_z-error-10_in-plane-dist-5_units=microns.xlsx'
        fpss = 'spct_cm0.9_dfdz_icp_neg2pos_z-error-10_in-plane-dist-5_units=microns.xlsx'

        dfi = pd.read_excel(join(path_results_compare, fpi))
        dfs = pd.read_excel(join(path_results_compare, fps))
        dfss = pd.read_excel(join(path_results_compare, fpss))

    else:
        fpi = 'idpt_dfdz_icp_neg2pos_zerror{}_inplanedist{}_Cmin0.5_microns.xlsx'.format(z_error_limit,
                                                                                         in_plane_distance_threshold)
        fps = 'spct_dfdz_icp_neg2pos_zerror{}_inplanedist{}_Cmin0.5_microns.xlsx'.format(z_error_limit,
                                                                                         in_plane_distance_threshold)
        fpss = 'spct_dfdz_icp_neg2pos_zerror{}_inplanedist{}_Cmin0.9_microns.xlsx'.format(z_error_limit,
                                                                                          in_plane_distance_threshold)

        path_results_idpt = join(path_results, 'idpt', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'rel-calib')
        path_results_spct = join(path_results, 'spct', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'rel-calib')

        dfi = pd.read_excel(join(path_results_idpt, fpi))
        dfs = pd.read_excel(join(path_results_spct, fps))
        dfss = pd.read_excel(join(path_results_spct, fpss))

    # ---

    # plot axial position as determined by rigid transformations
    plot_tracking = False  # True False
    if plot_tracking:

        # read calibration particle coords
        dfic = pd.read_excel(join(path_coords, 'test_coords_idpt_calib_particle_only.xlsx'))
        dfsc = pd.read_excel(join(path_coords, 'test_coords_spct_calib_particle_only.xlsx'))

        dfic = dfic.groupby('z_true').mean().reset_index()
        dfsc = dfic.groupby('z_true').mean().reset_index()

        dfic['z_assert_true'] = dfic['z']
        # dfic.loc[dfic['z_assert_true'] == -3.10000, 'z_assert_true'] = -3.6
        dfic.iloc[9, -1] = -3.6

        dfsc['z_assert_true'] = dfsc['z']
        # dfsc.loc[dfsc['z_assert_true'] == -3.10000, 'z_assert_true'] = -3.6
        dfsc.iloc[9, -1] = -3.6

        z_tracking = []
        for df in [dfi, dfs, dfss]:
            z_nominal = np.append(df['zA'].to_numpy(), df['zB'].to_numpy()[-1])
            dz_rigids = df['dz'].to_numpy()

            z_rigid = z_nominal[0]
            z_rigids = [z_rigid]
            for dz in dz_rigids:
                z_rigid += dz
                z_rigids.append(z_rigid)

            z_tracking.append(np.vstack([z_nominal, z_rigids]).T)

        # plot
        ms = 1
        xlim, ylim = 55, 55
        xticks = [-50, -25, 0, 25, 50]

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches, size_y_inches * 1.25))

        ax1.plot(z_tracking[0][:, 0], z_tracking[0][:, 0], '-', color='k', linewidth=0.75, label=r'$z_{nominal}$')
        ax1.plot(z_tracking[0][:, 0], z_tracking[0][:, 1], '-o', ms=ms, color=sciblue, label='IDPT')

        ax2.plot(z_tracking[0][:, 0], z_tracking[0][:, 0], '-', color='k', linewidth=0.75)
        ax2.plot(z_tracking[1][:, 0], z_tracking[1][:, 1], '-o', ms=ms, color=scigreen,
                 label='SPCT' + r'$(C_{m,min}=0.5)$')

        ax3.plot(z_tracking[0][:, 0], z_tracking[0][:, 0], '-', color='k', linewidth=0.75)
        ax3.plot(z_tracking[2][:, 0], z_tracking[2][:, 1], '-o', ms=ms, color=sciorange,
                 label='SPCT' + r'$(C_{m,min}=0.9)$')

        ax1.set_ylabel(r'$z_{R.T.} \: (\mu m)$')
        ax1.set_ylim([-ylim, ylim])
        ax1.legend()
        ax2.set_ylabel(r'$z_{R.T.} \: (\mu m)$')
        ax2.set_ylim([-ylim, ylim])
        ax2.legend()
        ax3.set_ylabel(r'$z_{R.T.} \: (\mu m)$')
        ax3.set_ylim([-ylim, ylim])
        ax3.set_xlabel(r'$z_{nominal} \: (\mu m)$')
        ax3.set_xlim([-xlim, xlim])
        ax3.set_xticks(xticks)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(join(path_figs_compare, 'compare_z-RT_by_z-nominal_3plots.png'))
        plt.show()
        plt.close()

        # -

        # plot the difference between z-nominal and z-rigid at each position

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1, size_y_inches * 1.5))

        # ax1.plot(z_tracking[0][:, 0], z_tracking[0][:, 0] - z_tracking[0][:, 0], '-', color='k', linewidth=0.75, label=r'$z_{nominal}$')
        ax1.plot(z_tracking[0][:, 0], z_tracking[0][:, 1] - z_tracking[0][:, 0], '-o', ms=ms, color=sciblue,
                 label='IDPT')

        # ax2.plot(z_tracking[1][:, 0], z_tracking[1][:, 0] - z_tracking[1][:, 0], '-', color='k', linewidth=0.75)
        ax2.plot(z_tracking[1][:, 0], z_tracking[1][:, 1] - z_tracking[1][:, 0], '-o', ms=ms, color=scigreen,
                 label='SPCT' + r'$(C_{m,min}=0.5)$')

        # ax3.plot(z_tracking[2][:, 0], z_tracking[2][:, 0] - z_tracking[2][:, 0], '-', color='k', linewidth=0.75)
        ax3.plot(z_tracking[2][:, 0], z_tracking[2][:, 1] - z_tracking[2][:, 0], '-o', ms=ms, color=sciorange,
                 label='SPCT' + r'$(C_{m,min}=0.9)$')

        ax1.set_ylabel(r'$z_{R.T.} - z_{nom.} \: (\mu m)$')
        # ax1.set_ylim([-ylim, ylim])
        ax1.grid(alpha=0.25)
        ax1.legend()
        ax2.set_ylabel(r'$z_{R.T.} - z_{nom.} \: (\mu m)$')
        # ax2.set_ylim([-ylim, ylim])
        ax2.grid(alpha=0.25)
        ax2.legend()
        ax3.set_ylabel(r'$z_{R.T.} - z_{nom.} \: (\mu m)$')
        # ax3.set_ylim([-ylim, ylim])
        ax3.set_xlabel(r'$z_{nominal} \: (\mu m)$')
        ax3.set_xlim([-xlim, xlim])
        ax3.set_xticks(xticks)
        ax3.grid(alpha=0.25)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(join(path_figs_compare, 'compare_diff-z-RT-nominal_by_z-nominal_3plots.png'))
        plt.show()
        plt.close()

        # ---

        # plot (i) rigid transform displacement minus 5, and (ii) single calibration particle

        # plot
        ylim = 10

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.5, size_y_inches * 1.5))

        ax1.plot(z_tracking[0][:, 0], z_tracking[0][:, 1] - z_tracking[0][:, 0], '-o', ms=ms, color=sciblue,
                 label='IDPT')
        ax1.plot(dfic.z_true, dfic.z - dfic.z_true, '-D', ms=ms * 2, color='k', linewidth=1,
                 label='IDPT' + r'$(I_{o}^{c})$')
        ax1.plot(dfic.z_true, dfic.z_assert_true - dfic.z_true, '-o', ms=ms, color='r', linewidth=0.7,
                 label='IDPT' + r'$(I_{o}^{c})(z_{assert})$')

        ax2.plot(z_tracking[1][:, 0], z_tracking[1][:, 1] - z_tracking[1][:, 0], '-o', ms=ms, color=scigreen,
                 label='SPCT' + r'$(C_{m,min}=0.5)$')
        ax2.plot(dfsc.z_true, dfsc.z - dfsc.z_true, '-D', ms=ms * 2, color='k', linewidth=1,
                 label='SPCT' + r'$(I_{o}^{c})$')
        ax2.plot(dfsc.z_true, dfsc.z_assert_true - dfsc.z_true, '-o', ms=ms, color='r', linewidth=0.7,
                 label='SPCT' + r'$(I_{o}^{c})(z_{assert})$')

        ax3.plot(z_tracking[2][:, 0], z_tracking[2][:, 1] - z_tracking[2][:, 0], '-o', ms=ms, color=sciorange,
                 label='SPCT' + r'$(C_{m,min}=0.9)$')
        ax3.plot(dfsc.z_true, dfsc.z - dfsc.z_true, '-D', ms=ms * 2, color='k', linewidth=1,
                 label='SPCT' + r'$(I_{o}^{c})$')
        ax3.plot(dfsc.z_true, dfsc.z_assert_true - dfsc.z_true, '-o', ms=ms, color='r', linewidth=0.7,
                 label='SPCT' + r'$(I_{o}^{c})(z_{assert})$')

        ax1.set_ylabel(r'$z_{R.T.} - z_{nom.} \: (\mu m)$')
        ax1.set_ylim([-ylim, ylim])
        ax1.grid(alpha=0.25)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.set_ylabel(r'$z_{R.T.} - z_{nom.} \: (\mu m)$')
        ax2.set_ylim([-ylim, ylim])
        ax2.grid(alpha=0.25)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax3.set_ylabel(r'$z_{R.T.} - z_{nom.} \: (\mu m)$')
        ax3.set_ylim([-ylim, ylim])
        ax3.set_xlabel(r'$z_{nominal} \: (\mu m)$')
        ax3.set_xlim([-xlim, xlim])
        ax3.set_xticks(xticks)
        ax3.grid(alpha=0.25)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(join(path_figs_compare, 'compare_calib-and-diff-z-RT-nominal_by_z-nominal_3plots.png'))
        plt.show()
        plt.close()

        # -

        # plot the same but on a single figure
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 1))

        ax.plot(z_tracking[0][:, 0], z_tracking[0][:, 1] - z_tracking[0][:, 0], '-o', ms=ms, color=sciblue,
                label='IDPT')
        ax.plot(z_tracking[1][:, 0], z_tracking[1][:, 1] - z_tracking[1][:, 0], '-o', ms=ms, color=scigreen,
                label='SPCT' + r'$(C_{m,min}=0.5)$')
        ax.plot(z_tracking[2][:, 0], z_tracking[2][:, 1] - z_tracking[2][:, 0], '-o', ms=ms, color=sciorange,
                label='SPCT' + r'$(C_{m,min}=0.9)$')

        ax.plot(dfic.z_true, dfic.z - dfic.z_true, '-D', ms=ms * 3, color='k', linewidth=1,
                label='IDPT' + r'$(I_{o}^{c})$')
        ax.plot(dfsc.z_true, dfsc.z - dfsc.z_true, '-s', ms=ms * 2.25, color='r', linewidth=0.8,
                label='SPCT' + r'$(I_{o}^{c})$')

        ax.plot(dfic.z_true, dfic.z_assert_true - dfic.z_true, '-o', ms=ms * 1.5, color='b', linewidth=0.6,
                label='IDPT' + r'$(I_{o}^{c})(z_{assert})$')
        ax.plot(dfsc.z_true, dfsc.z_assert_true - dfsc.z_true, '-o', ms=ms, color='pink', linewidth=0.4,
                label='SPCT' + r'$(I_{o}^{c})(z_{assert})$')

        ax.set_ylabel(r'$z_{R.T.} - z_{nom.} \: (\mu m)$')
        ax.set_ylim([-10, 7.5])
        ax.set_xlabel(r'$z_{nominal} \: (\mu m)$')
        ax.set_xlim([-xlim, xlim])
        ax.set_xticks(xticks)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(join(path_figs_compare, 'compare_calib-and-diff-z-RT-nominal_by_z-nominal_1plot.png'))
        plt.show()
        plt.close()

        # ---

    # ---

    # plot displacement between frames

    # setup
    ms = 2

    px = 'z'
    py1 = 'dx'
    py2 = 'dy'
    py3 = 'dz'

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.125, size_y_inches * 1.25))

    ax1.plot(dfi[px], dfi[py1], '-o', ms=ms, label='IDPT')
    ax1.plot(dfs[px], dfs[py1], '-o', ms=ms, label='SPCT(0.5)')
    ax1.plot(dfss[px], dfss[py1], '-o', ms=ms, label='SPCT(0.9)')

    ax2.plot(dfi[px], dfi[py2], '-o', ms=ms, label='IDPT')
    ax2.plot(dfs[px], dfs[py2], '-o', ms=ms, label='SPCT(0.5)')
    ax2.plot(dfss[px], dfss[py2], '-o', ms=ms, label='SPCT(0.9)')

    ax3.plot(dfi[px], dfi[py3], '-o', ms=ms, label='IDPT')
    ax3.plot(dfs[px], dfs[py3], '-o', ms=ms, label='SPCT(0.5)')
    ax3.plot(dfss[px], dfss[py3], '-o', ms=ms, label='SPCT(0.9)')

    ax1.set_ylabel(r'$\Delta_{x} \: (\mu m)$')
    ax2.set_ylabel(r'$\Delta_{y} \: (\mu m)$')
    ax2.legend(ncol=3)
    ax3.set_ylabel(r'$\Delta_{z} \: (\mu m)$')
    ax3.set_xlabel(r'$z \: (\mu m)$')

    plt.tight_layout()
    plt.savefig(join(path_figs_compare, 'compare_displacement-x-y-z.png'))
    plt.show()
    plt.close()

    # ---

    # plot z-displacement minus z-nominal
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))

    ax.plot(dfi[px], dfi[py3] - dfi['z'], '-o', ms=ms, label='IDPT')
    ax.plot(dfs[px], dfs[py3] - dfs['z'], '-o', ms=ms, label='SPCT(0.5)')
    ax.plot(dfss[px], dfss[py3] - dfss['z'], '-o', ms=ms, label='SPCT(0.9)')

    ax.set_ylabel(r'$z_{meas} - z_{nominal} \: (\mu m)$')
    ax.set_xlabel(r'$z \: (\mu m)$')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(join(path_figs_compare, 'compare_z-measure_to_z-nominal.png'))
    plt.show()
    plt.close()

    # ---

    # plot number of ICP particles per z

    px = 'z'
    py1 = 'num_icp'

    # plot
    fig, ax1 = plt.subplots(figsize=(size_x_inches * 1.125, size_y_inches * 0.75))

    ax1.plot(dfi[px], dfi[py1] / true_num_particles_per_frame, '-o', ms=ms, label='IDPT')
    ax1.plot(dfs[px], dfs[py1] / true_num_particles_per_frame, '-o', ms=ms, label='SPCT(0.5)')
    ax1.plot(dfss[px], dfss[py1] / true_num_particles_per_frame, '-o', ms=ms, label='SPCT(0.9)')

    ax1.set_ylabel(r'$\phi_{ICP}$')
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel(r'$z \: (\mu m)$')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(join(path_figs_compare, 'compare_num_icp_particles.png'))
    plt.show()
    plt.close()

    # ---

    # plot fit accuracy of rigid transformations

    px = 'z'
    py1 = 'precision'
    py2 = 'rmse'

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.125, size_y_inches * 1.25))

    ax1.plot(dfi[px], dfi[py1], '-o', ms=ms, label='IDPT')
    ax1.plot(dfs[px], dfs[py1], '-o', ms=ms, label='SPCT(0.5)')
    ax1.plot(dfss[px], dfss[py1], '-o', ms=ms, label='SPCT(0.9)')

    ax2.plot(dfi[px], dfi[py2], '-o', ms=ms, label='IDPT')
    ax2.plot(dfs[px], dfs[py2], '-o', ms=ms, label='SPCT(0.5)')
    ax2.plot(dfss[px], dfss[py2], '-o', ms=ms, label='SPCT(0.9)')

    ax1.set_ylabel(r'$precision \: (\mu m)$')
    ax1.legend()
    ax2.set_ylabel(r'$r.m.s. error \: (\mu m)$')
    ax2.set_xlabel(r'$z \: (\mu m)$')

    plt.tight_layout()
    plt.savefig(join(path_figs_compare, 'compare_RT_precision_rmse.png'))
    plt.show()
    plt.close()

    # ---

    # plot rmse x, y, z

    px = 'z'
    py1 = 'rmse_x'
    py2 = 'rmse_y'
    py3 = 'rmse_z'

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(size_x_inches * 1.125, size_y_inches * 1.25))

    ax1.plot(dfi[px], dfi[py1], '-o', ms=ms, label='IDPT')
    ax1.plot(dfs[px], dfs[py1], '-o', ms=ms, label='SPCT(0.5)')
    ax1.plot(dfss[px], dfss[py1], '-o', ms=ms, label='SPCT(0.9)')

    ax2.plot(dfi[px], dfi[py2], '-o', ms=ms, label='IDPT')
    ax2.plot(dfs[px], dfs[py2], '-o', ms=ms, label='SPCT(0.5)')
    ax2.plot(dfss[px], dfss[py2], '-o', ms=ms, label='SPCT(0.9)')

    ax3.plot(dfi[px], dfi[py3], '-o', ms=ms, label='IDPT')
    ax3.plot(dfs[px], dfs[py3], '-o', ms=ms, label='SPCT(0.5)')
    ax3.plot(dfss[px], dfss[py3], '-o', ms=ms, label='SPCT(0.9)')

    ax1.set_ylabel(r'$\sigma_{x} \: (\mu m)$')
    ax2.set_ylabel(r'$\sigma_{y} \: (\mu m)$')
    ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax3.set_xlabel(r'$z \: (\mu m)$')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(join(path_figs_compare, 'compare_rmse-x-y-z.png'))
    plt.show()
    plt.close()

    # ---

    # plot rmse x, y, z

    px = 'z'
    py1 = 'rmse_x'
    py2 = 'rmse_y'
    py3 = 'rmse_z'

    # plot
    fig, (ax1, ax3) = plt.subplots(nrows=2, sharex=True, figsize=(size_x_inches * 1.125, size_y_inches * 1.125))

    ax1.plot(dfi[px], np.sqrt(dfi[py1] ** 2 + dfi[py2] ** 2), '-o', ms=ms, label='IDPT')
    ax1.plot(dfs[px], np.sqrt(dfs[py1] ** 2 + dfs[py2] ** 2), '-o', ms=ms, label='SPCT(0.5)')
    ax1.plot(dfss[px], np.sqrt(dfss[py1] ** 2 + dfss[py2] ** 2), '-o', ms=ms, label='SPCT(0.9)')

    ax3.plot(dfi[px], dfi[py3], '-o', ms=ms, label='IDPT')
    ax3.plot(dfs[px], dfs[py3], '-o', ms=ms, label='SPCT(0.5)')
    ax3.plot(dfss[px], dfss[py3], '-o', ms=ms, label='SPCT(0.9)')

    ax1.set_ylabel(r'$\sigma_{xy} \: (\mu m)$')
    ax3.set_ylabel(r'$\sigma_{z} \: (\mu m)$')
    ax3.set_xlabel(r'$z \: (\mu m)$')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(join(path_figs_compare, 'compare_rmse-xy-z.png'))
    plt.show()
    plt.close()

    # ---

# ---

# plot x-y scatter of rmse_z per particle
plot_histogram_of_errors = False  # True False
if plot_histogram_of_errors:

    # plot histogram for a single array
    # plot kernel density estimation
    def scatter_and_kde_y(y, binwidth_y=1, kde=True, bandwidth_y=0.5, xlbl='residual', ylim_top=525, yticks=[],
                          save_path=None):

        fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches / 1.5))

        # y
        ylim_low = (int(np.min(y) / binwidth_y) - 1) * binwidth_y  # + binwidth_y
        ylim_high = (int(np.max(y) / binwidth_y) + 1) * binwidth_y - binwidth_y
        ybins = np.arange(ylim_low, ylim_high + binwidth_y, binwidth_y)
        ny, binsy, patchesy = ax.hist(y, bins=ybins, orientation='vertical', color='gray', zorder=2.5)

        # kernel density estimation
        if kde:
            ymin, ymax = np.min(y), np.max(y)
            y_range = ymax - ymin
            y_plot = np.linspace(ymin - y_range / 5, ymax + y_range / 5, 250)

            y = y[:, np.newaxis]
            y_plot = y_plot[:, np.newaxis]

            kde_y = KernelDensity(kernel="gaussian", bandwidth=bandwidth_y).fit(y)
            log_dens_y = kde_y.score_samples(y_plot)
            scale_to_max = np.max(ny) / np.max(np.exp(log_dens_y))

            # p2 = ax.fill_betweenx(y_plot[:, 0], 0, np.exp(log_dens_y) * scale_to_max, fc="None", ec=scired, zorder=2.5)
            # p2.set_linewidth(0.5)
            ax.plot(y_plot[:, 0], np.exp(log_dens_y) * scale_to_max, linewidth=0.75, linestyle='-', color=scired)

        ax.set_xlabel(xlbl + r'$(\mu m)$')
        ax.set_xlim([-3, 3])
        ax.set_ylabel('Counts')
        ax.set_ylim([0, ylim_top])
        ax.set_yticks(yticks)
        ax.grid(alpha=0.25)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()


    # -
    mtd = 'spct'
    cmin = 0.9

    fpi = '{}_icp-dist-xyz_neg2pos_errlim{}_inplanedist{}_Cmin{}_microns_joinAB.xlsx'.format(mtd,
                                                                                             z_error_limit,
                                                                                             in_plane_distance_threshold,
                                                                                             cmin)
    dfi = pd.read_excel(join(path_results, mtd, 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'rel-calib', fpi))
    # iter1: dfi = pd.read_excel(join(path_results, mtd, 'min_cm_{}'.format(cmin), 'z-error-limit_{}'.format(z_error_limit), fpi))

    # filter z-errors
    dfi = dfi[dfi['errz'].abs() < z_error_limit]

    # filter xy-errors
    dfi = dfi[dfi['errx'].abs() < in_plane_distance_threshold]
    dfi = dfi[dfi['erry'].abs() < in_plane_distance_threshold]

    # plot formatting
    if mtd == 'idpt':
        ylim_top = 1050
        yticks = [0, 500, 1000]
    else:
        ylim_top = 450
        yticks = [0, 200, 400]

    # plot histogram
    err_cols = ['errx', 'erry', 'errz']
    err_lbls = ['x residual ', 'y residual ', 'z residual ']

    for ycol, xlbl in zip(err_cols, err_lbls):
        y = dfi[ycol].to_numpy()
        save_dir = join(path_figs, mtd,
                        'cmin{}'.format(cmin) + '_zerrlim{}'.format(z_error_limit) + '_inplanedist{}'.format(in_plane_distance_threshold))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = join(save_dir, 'hist_{}.png'.format(ycol))
        scatter_and_kde_y(y, binwidth_y=0.1, kde=False, bandwidth_y=0.25,
                          xlbl=xlbl, ylim_top=ylim_top, yticks=yticks,
                          save_path=save_path)

    # ---

# ---

# ----------------------------------------------------------------------------------------------------------------------
# 5. Calculate single particle precision

# ---
# calculate the x, y, z, and Cm precision (standard deviation of 3 frames per z) of IDPT and SPCT
calculate_single_particle_precision = False  # True False
if calculate_single_particle_precision:

    use_raw = True

    if use_raw:
        # OPTION 1: "RAW" COORDINATES
        fpi = 'test_coords_particle_image_stats_tm16_cm19_dzf-post-processed.xlsx'
        fps = 'test_coords_particle_image_stats_spct-1_dzf-post-processed.xlsx'  # z is tilt-corrected
        dfi = pd.read_excel(join(path_coords, fpi))
        dfs = pd.read_excel(join(path_coords, fps))

        path_precision = join(path_results, 'precision', 'raw')

    else:
        # OPTION 2: POST-PROCESSED COORDINATES FROM "ERROR RELATIVE CALIBRATION PARTICLE"
        path_read_err_rel_calib = '/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/' \
                                  '11.06.21_error_relative_calib_particle/results/' \
                                  'relative-to-tilt-corr-calib-particle/zerrlim5_cmin0.5_mincountsallframes1'
        fpi = 'idpt_error_relative_calib_particle_zerrlim5_cmin0.5_mincountsallframes1.xlsx'
        fps = 'spct_error_relative_calib_particle_zerrlim5_cmin0.5_mincountsallframes1.xlsx'
        dfi = pd.read_excel(join(path_read_err_rel_calib, fpi))
        dfs = pd.read_excel(join(path_read_err_rel_calib, fps))

        dfi = dfi.rename(columns={'x': 'xg', 'y': 'yg', 'r': 'rg'})
        dfs = dfs.rename(columns={'x': 'gauss_xc', 'y': 'gauss_yc', 'r': 'gauss_rc'})

        path_precision = join(path_results, 'precision', 'rel-calib')

    if not os.path.exists(path_precision):
        os.makedirs(path_precision)

    # dfi = dfi[dfi['cm'] > 0.9]
    # dfs = dfs[dfs['cm'] > 0.9]

    # ---

    z_trues = dfi.z_true.unique()

    pi = []
    ps = []
    for z_true in z_trues:
        dfiz = dfi[dfi['z_true'] == z_true][['frame', 'id', 'z_true', 'z', 'xg', 'yg', 'rg', 'cm']]
        dfsz = dfs[dfs['z_true'] == z_true][['frame', 'id', 'z_true', 'z', 'gauss_xc', 'gauss_yc', 'gauss_rc', 'cm']]

        dfizg = dfiz.groupby('id').std()
        dfszg = dfsz.groupby('id').std()

        pi.append([z_true,
                   dfizg.z.mean(),
                   dfizg.xg.mean() * microns_per_pixel,
                   dfizg.yg.mean() * microns_per_pixel,
                   dfizg.rg.mean() * microns_per_pixel,
                   dfizg.cm.mean()])
        ps.append([z_true,
                   dfszg.z.mean(),
                   dfszg.gauss_xc.mean() * microns_per_pixel,
                   dfszg.gauss_yc.mean() * microns_per_pixel,
                   dfszg.gauss_rc.mean() * microns_per_pixel,
                   dfszg.cm.mean()])

    dfpi = pd.DataFrame(np.array(pi), columns=['z_true', 'pz', 'px', 'py', 'pr', 'pcm'])
    dfps = pd.DataFrame(np.array(ps), columns=['z_true', 'pz', 'px', 'py', 'pr', 'pcm'])

    dfpi.to_excel(join(path_precision, 'idpt_single_particle_precision_by_z.xlsx'))
    dfps.to_excel(join(path_precision, 'spct_single_particle_precision_by_z.xlsx'))

    dfpi['bin'] = 1
    dfps['bin'] = 1
    dfpig = dfpi.groupby('bin').mean()
    dfpsg = dfps.groupby('bin').mean()
    dfpig.to_excel(join(path_precision, 'idpt_single_particle_precision_mean.xlsx'))
    dfpsg.to_excel(join(path_precision, 'spct_single_particle_precision_mean.xlsx'))

    # plot
    ms = 3
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, sharex=True,
                                                  figsize=(size_x_inches * 1.25, size_y_inches * 1.75))

    ax0.plot(dfpi.z_true, dfpi.pcm, '-o', ms=ms, label='IDPT={}'.format(np.round(dfpi.pcm.mean(), 3)))
    ax0.plot(dfps.z_true, dfps.pcm, '-o', ms=ms, label='SPCT={}'.format(np.round(dfps.pcm.mean(), 3)))
    ax0.set_ylabel(r'$p_{cm} \: (\mu m)$')
    ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.plot(dfpi.z_true, dfpi.px, '-o', ms=ms, label=np.round(dfpi.px.mean(), 3))
    ax1.plot(dfps.z_true, dfps.px, '-o', ms=ms, label=np.round(dfps.px.mean(), 3))
    ax1.set_ylabel(r'$p_{x} \: (\mu m)$')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax2.plot(dfpi.z_true, dfpi.py, '-o', ms=ms, label=np.round(dfpi.py.mean(), 3))
    ax2.plot(dfps.z_true, dfps.py, '-o', ms=ms, label=np.round(dfps.py.mean(), 3))
    ax2.set_ylabel(r'$p_{y} \: (\mu m)$')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax3.plot(dfpi.z_true, dfpi.pr, '-o', ms=ms, label=np.round(dfpi.pr.mean(), 3))
    ax3.plot(dfps.z_true, dfps.pr, '-o', ms=ms, label=np.round(dfps.pr.mean(), 3))
    ax3.set_ylabel(r'$p_{r} \: (\mu m)$')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax4.plot(dfpi.z_true, dfpi.pz, '-o', ms=ms, label=np.round(dfpi.pz.mean(), 3))
    ax4.plot(dfps.z_true, dfps.pz, '-o', ms=ms, label=np.round(dfps.pz.mean(), 3))
    ax4.set_ylabel(r'$p_{z} \: (\mu m)$')
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(join(path_precision, 'compare_single_particle_precision_by_z.png'))
    plt.show()

# ---

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 6. Publication figures - putting it all together into one plot

# ---

plot_pubfigs = False  # True False
if plot_pubfigs:

    include_cmin_zero_nine = False  # True False

    # file paths

    # save data
    path_pubfigs = join(
        path_figs,
        'pubfigs',
        'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc',
        'zerrlim{}_inplanedist{}_minnumicp{}'.format(z_error_limit,
                                                     in_plane_distance_threshold,
                                                     min_num_particles_for_icp),
    )
    if not os.path.exists(path_pubfigs):
        os.makedirs(path_pubfigs)

    # -

    # read ICP results
    path_results_compare = join(path_results, 'compare', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc',
                                'z-error-limit_{}'.format(z_error_limit))

    if z_error_limit == 4:
        fpirt = 'idpt_dfdz_icp_neg2pos_zerror4_inplanedist{}_Cmin0.5_microns.xlsx'.format(in_plane_distance_threshold)
        fpsrt = 'spct_dfdz_icp_neg2pos_zerror4_inplanedist{}_Cmin0.5_microns.xlsx'.format(in_plane_distance_threshold)
        fpssrt = 'spct_dfdz_icp_neg2pos_zerror4_inplanedist{}_Cmin0.9_microns.xlsx'.format(in_plane_distance_threshold)

        dfirt = pd.read_excel(join(path_results_compare, fpirt))
        dfsrt = pd.read_excel(join(path_results_compare, fpsrt))
        dfssrt = pd.read_excel(join(path_results_compare, fpssrt))
    elif z_error_limit == 10:
        fpirt = 'idpt_cm0.5_dfdz_icp_neg2pos_z-error-10_in-plane-dist-5_units=microns.xlsx'
        fpsrt = 'spct_cm0.5_dfdz_icp_neg2pos_z-error-10_in-plane-dist-5_units=microns.xlsx'
        fpssrt = 'spct_cm0.9_dfdz_icp_neg2pos_z-error-10_in-plane-dist-5_units=microns.xlsx'

        dfirt = pd.read_excel(join(path_results_compare, fpirt))
        dfsrt = pd.read_excel(join(path_results_compare, fpsrt))
        dfssrt = pd.read_excel(join(path_results_compare, fpssrt))
    else:
        fpirt = 'idpt_dfdz_icp_neg2pos_zerror{}_inplanedist{}_Cmin0.5_microns.xlsx'.format(z_error_limit, in_plane_distance_threshold)
        fpsrt = 'spct_dfdz_icp_neg2pos_zerror{}_inplanedist{}_Cmin0.5_microns.xlsx'.format(z_error_limit, in_plane_distance_threshold)
        fpssrt = 'spct_dfdz_icp_neg2pos_zerror{}_inplanedist{}_Cmin0.9_microns.xlsx'.format(z_error_limit, in_plane_distance_threshold)

        path_read_idpt = join(path_results, 'idpt', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'rel-calib')
        path_read_spct = join(path_results, 'spct', 'dfAxy_at_zf', 'ztrue_is_fit-plane-xyzc', 'rel-calib')

        dfirt = pd.read_excel(join(path_read_idpt, fpirt))
        dfsrt = pd.read_excel(join(path_read_spct, fpsrt))
        dfssrt = pd.read_excel(join(path_read_spct, fpssrt))

    # ---

    # read: results from error relative to calibration particle
    path_read_err_rel_p_cal = join('/Users/mackenzie/Desktop/gdpyt-characterization/methodfigs/'
                                   '11.06.21_error_relative_calib_particle',
                                   'results',
                                   'relative-to-tilt-corr-calib-particle_08.06.23_raw-original',
                                   'spct-is-corr-fc',
                                   'zerrlim{}_cmin0.5_mincountsallframes{}'.format(z_error_limit, min_counts),
                                   'ztrue_is_fit-plane-xyzc',
                                   'bin-z_' + 'zerrlim{}_mincountsallframes{}'.format(z_error_limit, min_counts))

    dfim = pd.read_excel(join(path_read_err_rel_p_cal, 'idpt_cm0.5_bin-z_rmse-z.xlsx'))
    dfsm = pd.read_excel(join(path_read_err_rel_p_cal, 'spct_cm0.5_bin-z_rmse-z.xlsx'))
    dfssm = pd.read_excel(join(path_read_err_rel_p_cal, 'spct_cm0.9_bin-z_rmse-z.xlsx'))

    # filter before plotting
    dfim = dfim[dfim['count_id'] > min_counts_bin_z]
    dfsm = dfsm[dfsm['count_id'] > min_counts_bin_z]
    dfssm = dfssm[dfssm['count_id'] > min_counts_bin_z]

    # ---

    # plot local correlation coefficient

    # setup - general
    clr_i = sciblue
    clr_s = scigreen
    clr_ss = sciorange
    if include_cmin_zero_nine:
        lgnd_i = 'IDPT' + r'$(C_{m,min}=0.5)$'
        lgnd_s = 'SPCT' + r'$(C_{m,min}=0.5)$'
        lgnd_ss = 'SPCT' + r'$(C_{m,min}=0.9)$'
    else:
        lgnd_i = 'IDPT'
        lgnd_s = 'SPCT'
        lgnd_ss = 'SPCT'
    zorder_i, zorder_s, zorder_ss = 3.5, 3.3, 3.4

    ms = 4
    xlbl = r'$z \: (\mu m)$'
    xticks = [-50, -25, 0, 25, 50]

    # -

    # setup plot

    # variables: error relative calib particle
    px = 'bin'
    py = 'cm'
    pyb = 'percent_meas'
    py4 = 'rmse_z'

    # variables: rigid transformations
    px1 = 'z'
    py1 = 'rmse_x'
    py2 = 'rmse_y'

    ylbl_cm = r'$C_{m}^{\delta}$'
    ylim_cm = [0.71, 1.02]  # data range: [0.7, 1.0]
    yticks_cm = [0.8, 0.9, 1.0]  # data ticks: np.arange(0.75, 1.01, 0.05)

    ylbl_phi = r'$\phi^{\delta}$'
    ylim_phi = [0, 1.1]
    yticks_phi = [0, 0.5, 1]

    ylbl_rmse_xy = r'$\sigma_{xy}^{\delta} \: (\mu m)$'
    ylim_rmse_xy = [0, 1]
    yticks_rmse_xy = [0, 0.5, 1]

    ylbl_rmse_z = r'$\sigma_{z}^{\delta} \: (\mu m)$'
    ylim_rmse_z = [0, 2.6]
    yticks_rmse_z = [0, 1, 2]

    # plot
    if include_cmin_zero_nine:
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(size_x_inches * 2, size_y_inches * 1.25))
    else:
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(size_x_inches * 2, size_y_inches * 1.05))

    ax2, ax3, ax1, ax4 = axs.ravel()

    ax1.plot(dfim[px], dfim[py], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
    ax1.plot(dfsm[px], dfsm[py], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)

    if include_cmin_zero_nine:
        ax1.plot(dfssm[px], dfssm[py], '-o', ms=ms, color=clr_ss, label=lgnd_ss, zorder=zorder_ss)

    ax1.set_xlabel(xlbl)
    ax1.set_xticks(xticks)
    ax1.set_ylabel(ylbl_cm)
    ax1.set_ylim(ylim_cm)
    ax1.set_yticks(yticks_cm)
    # ax1.legend(loc='lower center')  # loc='upper left', bbox_to_anchor=(1, 1))

    # -

    ax2.plot(dfim[px], dfim[pyb], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
    ax2.plot(dfsm[px], dfsm[pyb], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)

    if include_cmin_zero_nine:
        ax2.plot(dfssm[px], dfssm[pyb], '-o', ms=ms, color=clr_ss, label=lgnd_ss, zorder=zorder_ss)

    # ax2.set_xlabel(xlbl)
    # ax2.set_xticks(xticks)
    ax2.set_ylabel(ylbl_phi)
    ax2.set_ylim(ylim_phi)
    ax2.set_yticks(yticks_phi)

    if include_cmin_zero_nine:
        ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.05))
    else:
        ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)  # loc='upper left', bbox_to_anchor=(1, 1)) , ncol=2

    # -

    ax3.plot(dfirt[px1], np.sqrt(dfirt[py1] ** 2 + dfirt[py2] ** 2), '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
    ax3.plot(dfsrt[px1], np.sqrt(dfsrt[py1] ** 2 + dfsrt[py2] ** 2), '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)

    if include_cmin_zero_nine:
        ax3.plot(dfssrt[px1], np.sqrt(dfssrt[py1] ** 2 + dfssrt[py2] ** 2), '-o', ms=ms, color=clr_ss, label=lgnd_ss, zorder=zorder_ss)

    # ax3.set_xlabel(xlbl)
    # ax3.set_xticks(xticks)
    ax3.set_ylabel(ylbl_rmse_xy)
    # ax3.set_ylim(ylim_rmse_xy)
    # ax3.set_yticks(yticks_rmse_xy)
    # ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # -

    ax4.plot(dfim[px], dfim[py4], '-o', ms=ms, color=clr_i, label=lgnd_i, zorder=zorder_i)
    ax4.plot(dfsm[px], dfsm[py4], '-o', ms=ms, color=clr_s, label=lgnd_s, zorder=zorder_s)

    if include_cmin_zero_nine:
        ax4.plot(dfssm[px], dfssm[py4], '-o', ms=ms, color=clr_ss, label=lgnd_ss, zorder=zorder_ss)

    ax4.set_xlabel(xlbl)
    ax4.set_xticks(xticks)
    ax4.set_ylabel(ylbl_rmse_z)
    # ax4.set_ylim(ylim_rmse_z)
    # ax4.set_yticks(yticks_rmse_z)
    # ax4.legend(loc='upper left')  # , borderpad=0.25, handletextpad=0.6, borderaxespad=0.3, markerscale=0.75)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # hspace=0.175, wspace=0.25

    if include_cmin_zero_nine:
        plt.savefig(join(path_pubfigs, 'compare_local_Cm-phi-rmsexyz_by_z_all_auto-ticks.png'))
    else:
        plt.savefig(join(path_pubfigs, 'compare_local_Cm-phi-rmsexyz_by_z_alt-legend.png'))
    plt.show()
    plt.close()

    # ---

# ---

print("Analysis completed without errors.")