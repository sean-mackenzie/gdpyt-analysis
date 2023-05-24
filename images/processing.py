from os import listdir
from os.path import join
from collections import OrderedDict

import numpy as np
import numpy.ma as ma
import pandas as pd
from skimage.io import imread
from scipy.optimize import curve_fit
from scipy.ndimage import rotate

import matplotlib.pyplot as plt

from utils import fit, functions


# ---

# --- functions


def gauss_2d_function(xy, a, x0, y0, sigmax, sigmay):
    return a * np.exp(-((xy[:, 0] - x0)**2 / (2 * sigmax**2) + (xy[:, 1] - y0)**2 / (2 * sigmay**2)))


def bivariate_gaussian_pdf_bkg(xy, a, x0, y0, sigmax, sigmay, rho, bkg):
    return a * np.exp(
        -((1 / (2 * (1 - rho ** 2))) * ((xy[:, 0] - x0) ** 2 / sigmax ** 2 - 2 * rho * (xy[:, 0] - x0) * (
                    xy[:, 1] - y0) / (sigmax * sigmay) +
          (xy[:, 1] - y0) ** 2 / sigmay ** 2)
          )
    ) + bkg


# ---

# --- analysis


def get_background(image, threshold):
    particle_mask = image > threshold
    bkg = ma.masked_array(image, mask=particle_mask)
    return bkg


def normalize_image(img, bkg_mean):
    if bkg_mean is None:
        bkg_mean = np.percentile(img, 15)

    norm_img = img - bkg_mean

    norm_img = np.where(norm_img > 1, norm_img, 1)

    return norm_img


def fit_2d_gaussian_on_image(img, normalize, bkg_mean):
    """ popt, img_norm = fit_2d_gaussian_on_image(img, normalize, bkg_mean) """

    img_norm, XYZ = flatten_image(img, normalize, bkg_mean)

    yc, xc = np.shape(img_norm)
    xc, yc = xc // 2, yc // 2

    guess_A, guess_c, guess_sigma = get_amplitude_center_sigma(x_space=None, y_profile=None, img=img_norm, y_slice=yc)

    # fit 2D gaussian
    guess = [guess_A, xc, yc, guess_sigma, guess_sigma]
    try:
        popt, pcov = curve_fit(gauss_2d_function, XYZ[:, :2], XYZ[:, 2], guess)
    except RuntimeError:
        popt = None

    return popt, img_norm


def fit_2d_pdf_on_image(img, rotate_degrees, maintain_original_image=False):
    if rotate_degrees != 0:
        img = rotate(img, angle=rotate_degrees,
                     reshape=maintain_original_image, mode='grid-constant', cval=np.percentile(img, 13.5))
        # NOTE: the "13.5" percentile was chosen because the ratio of the area of a circle circumscribed in a square
        # of equal diameter/width is 78.5%. So, even if the defocused particle image extends to the extents of the
        # template, taking the 13.5 percentile should get the middle of the noise values.

    y, x = np.shape(img)
    xc, yc = x // 2, y // 2
    guess_A, guess_c, guess_sigma = get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=img)

    # make grid
    X = np.arange(np.shape(img)[1])
    Y = np.arange(np.shape(img)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = img.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    try:
        guess = [guess_A, xc, yc, guess_sigma, guess_sigma, 0, 100]
        popt, pcov = curve_fit(bivariate_gaussian_pdf_bkg, XYZ[:, :2], XYZ[:, 2],
                               guess,
                               bounds=([0, 0, 0, 0, 0, -0.99, 0],
                                       [2**16, 512, 512, 100, 100, 0.99, 2**16])
                               )
    except RuntimeError:
        popt = None

    return popt, img


def evaluate_fit_2d_gaussian_on_image(img_norm, fit_func, popt):
    """ XYZ, fZ, rmse, r_squared = evaluate_fit_2d_gaussian_on_image(img_norm, popt) """

    img_norm, XYZ = flatten_image(img_norm, normalize=False, bkg_mean=None)

    # 2D Gaussian from fit
    fZ = fit_func(XYZ[:, :2], *popt)

    # data used for fitting
    img_arr = XYZ[:, 2]

    # get residuals
    residuals = functions.calculate_residuals(fit_results=fZ, data_fit_to=img_arr)

    # get goodness of fit
    rmse, r_squared = fit.calculate_fit_error(fit_results=fZ, data_fit_to=img_arr)

    return XYZ, fZ, rmse, r_squared, residuals


def calculate_intensity_uniformity(img, thresh_val, grid_size, thresh_multiplier=3.5, show_image=False):
    """

    :param img:
    :param thresh_val: should be bkg_max(img) + 2 * bkg_noise(img) of raw image
    :param grid_size: suggested = (16, 16)
    :param thresh_multiplier: suggested = 3; raises the threshold value after calculating noise of first mask
    :return:
    """
    # user defined threshold
    bkg = get_background(img, thresh_val)
    bkg_max = np.max(bkg)
    bkg_mean = np.mean(bkg)
    bkg_noise = np.std(bkg)

    if show_image:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        ax1.imshow(bkg, interpolation='none')
        ax1.set_title("thresh_val={}, measured: mean={}, noise={}".format(int(thresh_val),
                                                                          int(bkg_mean),
                                                                          int(bkg_noise),
                                                                          ))

    # refine threshold
    thresh_val_refined = bkg_mean + bkg_noise * thresh_multiplier
    bkg = get_background(img, thresh_val_refined)
    bkg_max = np.max(bkg)
    bkg_mean = np.mean(bkg)
    bkg_noise = np.std(bkg)

    if show_image:
        ax2.imshow(bkg, interpolation='none')
        ax2.set_title("thresh_val_refined={}, measured: mean={}, noise={}".format(int(thresh_val_refined),
                                                                                  int(bkg_mean),
                                                                                  int(bkg_noise),
                                                                                  ))
        plt.show()

    # define grid
    dimx, dimy = np.shape(bkg)
    sizex = dimx / grid_size
    sizey = dimy / grid_size

    # analyze image across grid
    df = analyze_image_grid(bkg, grid_size, sizex, sizey)

    return df, bkg


def analyze_image_grid(image, grid_size, sizex, sizey, print_res=False):
    bkg = image.copy()
    data = []

    for i in range(grid_size):
        yl, yr = int(i * sizex), int((i + 1) * sizex)

        for j in range(grid_size):
            xl, xr = int(j * sizex), int((j + 1) * sizex)

            # get sub-image
            sub_image = bkg[yl:yr, xl:xr]

            # calculate mean + std
            sub_image_mean = np.mean(sub_image)
            sub_image_noise = np.std(sub_image)

            # append data
            data.append([i, j, yl, yr, xl, xr, sub_image_mean, sub_image_noise])

            # print results
            if print_res:
                print("({}, {}) = [{}:{}, {}:{}] = {}".format(i, j, yl, yr, xl, xr, np.mean(sub_image)))

    df = pd.DataFrame(np.array(data),
                      columns=['i', 'j', 'yl', 'yr', 'xl', 'xr', 'sub_mean', 'sub_std'],
                      )

    return df

# ---

# --- io


def get_files(image_path, image_id, split_strings, sort_strings, filetype='tif'):
    """ nums_and_files = get_files(image_path, image_id, split_strings, sort_strings, filetype='tif') """

    all_files = [f for f in listdir(image_path) if f.endswith(filetype)]

    if isinstance(image_id, (int, float)):
        get_files = [f for f in all_files if
                     float(f.split(split_strings[0])[-1].split(split_strings[1])[0]) == image_id]
    elif isinstance(image_id, (list, np.ndarray)):
        get_files = [f for f in all_files if
                     float(f.split(split_strings[0])[-1].split(split_strings[1])[0]) in image_id]
    else:
        get_files = [f for f in all_files]

    get_nums = [filename2float(f, sort_strings) for f in get_files]
    nums_and_files = sorted(list(zip(get_nums, get_files)), key=lambda x: x[0])

    return nums_and_files


def read_image(file, average_stack=True):
    img = imread(file, plugin='tifffile')
    if len(np.shape(img)) > 2:
            if average_stack:
                if np.shape(img)[2] == 3:
                    img = np.rint(np.mean(img, axis=2, dtype=float)).astype(np.int16)
                else:
                    img = np.rint(np.mean(img, axis=0, dtype=float)).astype(np.int16)
            else:
                raise ValueError("Image stack splitting is not implemented.")
    return img


def read_images(image_path, nums_and_files, average_stack=True):
    images = OrderedDict()
    for n, f in nums_and_files:
        img = read_image(join(image_path, f), average_stack=average_stack)
        images.update({n: img})
    return images


# ---

# --- plotting


def histogram_image(img, density, show=True):
    """ hist, bin_edges = histogram_image(img, density, show=True) """

    hist, bin_edges = np.histogram(img, density=density)

    if show:
        plt.hist(img.ravel(), fc='k', ec='k')
        plt.show()

    return hist, bin_edges


def plot_image_non_uniformity(df, grid_size):
    sub_mean = df.sub_mean.to_numpy()
    sub_noise = df.sub_std.to_numpy()

    res_mean = np.reshape(sub_mean, (grid_size, grid_size), order='C')
    res_noise = np.reshape(sub_noise, (grid_size, grid_size), order='C')

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

    # mean
    p1 = ax[0].imshow(res_mean, interpolation='nearest', origin='upper')
    ax[0].set_title(r"Bkg $(\overline{I_{bkg}} \pm \sigma_{I_{bkg}})=$" +
                    "{}+/-{}".format(np.round(np.mean(res_mean), 2), np.round(np.std(res_mean), 2)))

    cbar1 = fig.colorbar(p1, ax=ax[0], extend='both')
    cbar1.minorticks_on()

    # noise
    p2 = ax[1].imshow(res_noise, interpolation='nearest', origin='upper')
    ax[1].set_title(r"Noise $(\overline{\sigma_{I}} \pm \sigma_{\sigma_{I}})=$" +
                    "{}+/-{}".format(np.round(np.mean(res_noise), 2), np.round(np.std(res_noise), 2)))

    cbar2 = fig.colorbar(p2, ax=ax[1], extend='both')
    cbar2.minorticks_on()

    plt.suptitle(r"Non-uniformity($\frac{\sigma_{I_{bkg}}}{\overline{I_{bkg}}})=$" +
                 " {}\%".format(np.round(np.std(res_mean) / np.mean(res_mean) * 100, 4)),
                 fontsize=14)

    plt.tight_layout()

    return fig, ax


# ---

# --- helper functions


def filename2float(f, sort_strings):
    return float(f.split(sort_strings[0])[-1].split(sort_strings[1])[0])


def reshape_flattened(img, XYZ, fZ):
    """ x, y, z, fz = reshape_flattened(img, XYZ, fZ) """
    x = np.reshape(XYZ[:, 0], np.shape(img))
    y = np.reshape(XYZ[:, 1], np.shape(img))
    z = np.reshape(XYZ[:, 2], np.shape(img))
    fz = np.reshape(fZ, np.shape(img))
    return x, y, z, fz


def flatten_image(img, normalize=True, bkg_mean=None):
    """ img_norm, XYZ = flatten_image(img, normalize=True, bkg_mean=None) """
    if normalize:
        img = normalize_image(img, bkg_mean)

    # make grid
    X = np.arange(np.shape(img)[1])
    Y = np.arange(np.shape(img)[0])
    X, Y = np.meshgrid(X, Y)

    # flatten arrays
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = img.flatten()

    # stack for gaussian curve fitting
    XYZ = np.stack([Xf.flatten(), Yf.flatten(), Zf.flatten()]).T

    return img, XYZ


def get_amplitude_center_sigma(x_space=None, y_profile=None, img=None, y_slice=None):

    if y_profile is None:
        # get sub-image slice
        y_profile = img[y_slice, :]

    if x_space is None:
        x_space = np.arange(len(y_profile))

    # get amplitude
    raw_amplitude = y_profile.max() - y_profile.min()

    # get center
    raw_c = x_space[np.argmax(y_profile)]

    # get sigma
    raw_profile_diff = np.diff(y_profile)
    diff_peaks = np.argpartition(np.abs(raw_profile_diff), -2)[-2:]
    diff_width = np.abs(x_space[diff_peaks[1]] - x_space[diff_peaks[0]])
    raw_sigma = diff_width / 2

    return raw_amplitude, raw_c, raw_sigma


def get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=None):
    """raw_amplitude, raw_c, raw_sigma = get_amplitude_center_sigma_improved(x_space=None, y_profile=None, img=None)"""
    if y_profile is None:
        y_slice = np.unravel_index(np.argmax(img, axis=None), img.shape)[0]
        y_profile = img[y_slice, :]

    if x_space is None:
        x_space = np.arange(len(y_profile))

    # get amplitude
    raw_amplitude = y_profile.max() - y_profile.min()

    # get center
    raw_c = x_space[np.argmax(y_profile)]

    # get sigma
    y_pl_zero = len(np.where(y_profile[:np.argmax(y_profile)] - np.mean(y_profile) < 0)[0])
    y_pr_zero = len(np.where(y_profile[np.argmax(y_profile):] - np.mean(y_profile) < 0)[0])
    raw_sigma = np.mean([y_pl_zero, y_pr_zero])

    return raw_amplitude, raw_c, raw_sigma


def pad_image(img, pad_width, pad_value):
    img_padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=pad_value)
    return img_padded


def unrotate_and_crop_to_original_size(arr, rotate_degrees, original_shape):
    """ arr = processing.unrotate_and_crop_to_original_size(arr, rotate_degrees, original_shape) """

    # un-rotate
    if rotate_degrees != 0:
        arr = rotate(arr, angle=-rotate_degrees, reshape=False, mode='grid-constant', cval=0)

    # calculate half-size difference
    halfsize_x = int(np.ceil((np.shape(arr)[0] - original_shape[0]) / 2))
    halfsize_y = int(np.ceil((np.shape(arr)[1] - original_shape[1]) / 2))

    # use slice indexing to crop to original size
    arr = arr[halfsize_x:-halfsize_x, halfsize_y:-halfsize_y]

    return arr