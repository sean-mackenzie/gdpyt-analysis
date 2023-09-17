# gdpyt-analysis: utils.boundary
"""
Notes
"""

# imports
from os.path import join, exists
from os import makedirs

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
from matplotlib import pyplot as plt
from skimage.draw import polygon, disk, circle_perimeter, ellipse

from utils import io

# scripts


def compute_boundary_mask(mask_dict):

    # unpack dict
    path_mask_boundary = mask_dict['path_mask_boundary']
    path_image_to_mask = mask_dict['path_image_to_mask']
    padding_during_gdpyt = mask_dict['padding_during_gdpyt']
    acceptance_boundary_pixels = mask_dict['acceptance_boundary_pixels']
    save_mask_boundary = mask_dict['save_mask_boundary']
    save_boundary_images = mask_dict['save_boundary_images']
    show_boundary_images = mask_dict['show_boundary_images']

    # inherent file paths
    if not exists(path_mask_boundary):
        makedirs(path_mask_boundary)
    if not exists(path_mask_boundary + '/mask'):
        makedirs(path_mask_boundary + '/mask')

    fp_mask_boundary = join(path_mask_boundary, 'mask', 'mask_boundary')
    fp_mask_edge = join(path_mask_boundary, 'mask', 'mask_edge')

    # ensure compatibility
    circle_coords = mask_dict['circle_coords']
    if len(np.shape(circle_coords)) == 1:
        circle_coords = [circle_coords]

    # if mask already exists, do not recompute
    if exists(fp_mask_boundary + '.npy'):
        mask_boundary = np.load(fp_mask_boundary + '.npy')
    else:
        mask_boundary = None

    if exists(fp_mask_edge + '.npy'):
        mask_edge = np.load(fp_mask_edge + '.npy')
    else:
        mask_edge = None

    # if no mask is present, we must compute the mask
    if mask_boundary is None or mask_edge is None:

        # read image
        img = imread(path_image_to_mask)

        # pad the image as it was during gdpyt analysis
        print("Padding mask with {} pixels!".format(padding_during_gdpyt))
        img = np.pad(img, pad_width=padding_during_gdpyt, mode='minimum')

        # if image stack, take average
        if len(img.shape) > 2:
            img = np.mean(img, axis=0)

        # create the full mask by stacking individual regions
        mask_edge = np.zeros_like(img)
        mask_boundary = np.zeros_like(img)

        for cc in circle_coords:
            xc, yc, r_edge = cc

            # define the "true" boundary
            xc, yc, r_edge = xc + padding_during_gdpyt, yc + padding_during_gdpyt, r_edge
            mask_edge += draw_boundary_circle_perimeter(img, xc, yc, r_edge)

            # define the "acceptance" boundary
            r_boundary = r_edge + acceptance_boundary_pixels
            mask_boundary += draw_boundary_circle(img, xc, yc, r_boundary)

        # ------------------------------------------------------------------------------------------------------------------
        if save_mask_boundary:
            # save boundary mask
            np.save(fp_mask_boundary + '.npy', mask_boundary)
            imsave(fp_mask_boundary + '.tif', mask_boundary)

            # save edge mask
            np.save(fp_mask_edge + '.npy', mask_edge)
            imsave(fp_mask_edge + '.tif', mask_edge)

        # ------------------------------------------------------------------------------------------------------------------
        if show_boundary_images:
            # rescale to draw nicely
            mask_edge_rescaled = rescale_intensity(mask_edge, out_range=np.uint16)
            mask_boundary_rescaled = rescale_intensity(mask_boundary, out_range=np.uint16)

            # draw boundaries on image
            img_mask = img + mask_edge_rescaled // 10 + mask_boundary_rescaled // 75
            fig, ax = plt.subplots()
            ax.imshow(img_mask)
            ax.set_xticks([1, np.shape(img_mask)[1] // 2, np.shape(img_mask)[1]])
            ax.set_yticks([1, np.shape(img_mask)[0] // 2, np.shape(img_mask)[0]])
            ax.set_aspect('equal', adjustable='box')
            plt.title('boundary edge(xc, yc, r) = ({}, {}, {})'.format(xc, yc, r_edge))
            plt.tight_layout()
            if save_boundary_images:
                plt.savefig(join(path_mask_boundary, 'boundaries_on_image.png'))
            if show_boundary_images:
                plt.show()
            plt.close()

            # mask out all non-boundary particles by applying mask to image
            mask_circle_inverted = np.logical_not(mask_boundary).astype(int)
            img_masked = img * mask_circle_inverted
            fig, ax = plt.subplots()
            ax.imshow(img_masked)
            ax.set_xticks([1, np.shape(img_mask)[1] // 2, np.shape(img_mask)[1]])
            ax.set_yticks([1, np.shape(img_mask)[0] // 2, np.shape(img_mask)[0]])
            ax.set_aspect('equal', adjustable='box')
            plt.title('boundary(xc, yc, r) = ({}, {}, {})'.format(xc, yc, r_boundary))
            plt.tight_layout()
            if save_boundary_images:
                plt.savefig(join(path_mask_boundary, 'boundaries_masked_on_image.png'))
            if show_boundary_images:
                plt.show()
            plt.close()

    # export results
    dfict_mask_dict = pd.DataFrame.from_dict(mask_dict, orient='index', columns=['value'])
    dfict_mask_dict.to_excel(path_mask_boundary + '/mask_boundary_dict.xlsx')

    mask_dict.update({
        'mask_boundary': mask_boundary,
        'mask_edge': mask_edge
    })

    return mask_dict


def get_boundary_particles(mask, df, return_interior_particles, flip_xy_coords,
                           flip_xy_coords_minus_mask_size=False):
    """
    Using an image mask, return particles from df whose location is on the "boundary".
    Optionally, also return interior particles as a separate list.

    :param mask:
    :param df:
    :param return_interior_particles:
    :return:
    """

    # apply mask to image to get pixel coordinates where particles would be on the boundary
    mask_inverted = np.logical_not(mask).astype(int)
    mask_size_x, mask_size_y = np.shape(mask_inverted)
    boundary_indices = np.argwhere(mask_inverted > 0)

    # dfg = df.groupby(by='id').mean().astype(int).reset_index()
    dfg = df.groupby(by='id').mean().reset_index()
    dfg_particle_ids = dfg.id.unique()

    boundary_ids = []
    for pid in dfg_particle_ids:
        x_arr = dfg[dfg.id == pid].x.to_numpy()
        y_arr = dfg[dfg.id == pid].y.to_numpy()

        if flip_xy_coords:
            p_loc = [y_arr[0], x_arr[0]]
        elif flip_xy_coords_minus_mask_size:
            p_loc = [mask_size_x - y_arr[0], x_arr[0]]
        else:
            p_loc = [x_arr[0], y_arr[0]]

        for b in boundary_indices:
            if all(b == p_loc):
                boundary_ids.append(pid)

    if return_interior_particles:
        interior_ids = list(set(dfg_particle_ids).difference(boundary_ids))
        return boundary_ids, interior_ids
    else:
        return boundary_ids


def get_pids_on_features(df, path_results, path_boundary, export_pids_per_membrane):
    """

    dflr, dful, dfll, dfmm, dfbd, lr_pids, ul_pids, ll_pids, mm_pids, boundary_pids =
    get_pids_on_features(df, path_results, path_boundary, export_pids_per_membrane=True)

    :param df:
    :param path_results:
    :param path_boundary:
    :param export_pids_per_membrane:
    :return:
    """

    # --- DEFINE LOWER, UPPER, LEFT, MIDDLE MEMBRANES
    x_bounds_lr, y_bounds_lr = 200, 240
    y_bounds_ul = 210
    x_bounds_ll, y_bounds_ll = 100, 300
    x_bounds_mm, y_bounds_mm = [140, 200], [220, 300]

    boundary_pids = io.read_txt_file_to_list(join(path_boundary, 'boundary_pids.txt'), data_type='int')
    interior_pids = io.read_txt_file_to_list(join(path_boundary, 'interior_pids.txt'), data_type='int')

    boundary_pids.sort()
    interior_pids.sort()

    # get boundary particles (i.e., particles not on a feature)
    dfbd = df[df['id'].isin(boundary_pids)]

    # split interior pids into (1) lower right membrane and (2) upper left membrane
    df_interior_pids = df[df['id'].isin(interior_pids)]

    dflr = df_interior_pids[(df_interior_pids['x'] > x_bounds_lr) & (df_interior_pids['y'] > y_bounds_lr)]
    dful = df_interior_pids[df_interior_pids['y'] < y_bounds_ul]
    dfll = df_interior_pids[(df_interior_pids['x'] < x_bounds_ll) & (df_interior_pids['y'] > y_bounds_ll)]
    dfmm = df_interior_pids[(df_interior_pids['x'] > x_bounds_mm[0]) &
                            (df_interior_pids['x'] < x_bounds_mm[1]) &
                            (df_interior_pids['y'] > y_bounds_mm[0]) &
                            (df_interior_pids['y'] < y_bounds_mm[1])]

    lr_pids = sorted(dflr.id.unique())
    ul_pids = sorted(dful.id.unique())
    ll_pids = sorted(dfll.id.unique())
    mm_pids = sorted(dfmm.id.unique())

    # ---

    # export particle ID's for each membrane
    if export_pids_per_membrane:
        dict_memb_pids = {'lr': lr_pids,
                          'ul': ul_pids,
                          'll': ll_pids,
                          'mm': mm_pids,
                          'boundary': boundary_pids,
                          }

        export_memb_pids = pd.DataFrame.from_dict(data=dict_memb_pids, orient='index')
        export_memb_pids = export_memb_pids.rename(columns={0: 'pids'})
        export_memb_pids.to_excel(path_results + '/df_pids_per_membrane.xlsx')

    # ---

    return dflr, dful, dfll, dfmm, dfbd, lr_pids, ul_pids, ll_pids, mm_pids, boundary_pids


def draw_boundary_points(img, x, y, r=4, mask=None, draw_mask=True):
    """

    :param img: image to draw points to define boundary mask
    :param x: center; x-coord
    :param y: center; y-coord
    :return:
    """
    if mask is None:
        mask = np.zeros_like(img)
    else:
        print("adding on top of mask.")

    rr, cc = disk((y, x), r, shape=mask.shape)
    mask[rr, cc] = 1

    mask = np.array(mask, dtype=bool)

    if draw_mask:
        img[rr, cc] = np.max(img)

    return img, mask


def draw_boundary_rectangle(img, tlx, tly, brx, bry):
    """

    :param img: image to fit boundary mask
    :param tlx: top left x-coord
    :param tly: top left y-coord
    :param brx: bottom right x-coord
    :param bry: bottom right y-coord
    :return:
    """
    mask = np.zeros_like(img)

    rect = np.array((
        (tly, tlx),
        (tly, brx),
        (bry, brx),
        (bry, tlx)
    ))

    rr, cc = polygon(rect[:, 0], rect[:, 1], mask.shape)
    mask[rr, cc] = 1

    mask = np.array(mask, dtype=bool)

    return mask


def draw_boundary_circle(img, xc, yc, r):
    """

    :param img: image to fit circular boundary mask
    :param xc: center; x-coord
    :param yc: center; y-coord
    :param r: radius
    :return:
    """
    mask = np.zeros_like(img)

    rr, cc = disk((yc, xc), r, shape=mask.shape)
    mask[rr, cc, ] = 1

    mask = np.array(mask, dtype=bool)

    return mask


def draw_boundary_circle_perimeter(img, xc, yc, r):
    """

    :param img:
    :param xc:
    :param yc:
    :param r:
    :return:
    """

    mask = np.zeros_like(img)

    rr, cc = circle_perimeter(yc, xc, r, shape=mask.shape)
    mask[rr, cc, ] = 1

    mask = np.array(mask, dtype=bool)

    return mask