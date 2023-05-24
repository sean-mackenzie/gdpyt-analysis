# gdpyt-analysis: analyze
"""
Notes
"""

# imports
import os
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import itertools

from utils import functions, fit, io, models, plotting, bin
from utils.functions import fNonDimensionalNonlinearSphericalUniformLoad
from utils.bin import *

# other
# A note on SciencePlots colors
"""
Blue: #0C5DA5
Green: #00B945
Red: #FF2C00
Orange: #FF9500

Other Colors:
Light Blue: #7BC8F6
Paler Blue: #0343DF
Azure: #069AF3
Dark Green: #054907
"""

sciblue = '#0C5DA5'
scigreen = '#00B945'
sciorange = '#FF9500'
scired = '#FF2C00'

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# ---


# ------------------------------------ PRECISION FUNCTIONS (DATAFRAMES) ------------------------------------------------


# ---

def evaluate_precision_from_polyfit(x, y, poly_deg):
    pcoeff, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg=poly_deg, full=True)
    pf = np.poly1d(pcoeff)

    # error assessment
    y_model = pf(x)
    y_residuals = y_model - y
    y_precision = np.mean(np.std(y_residuals))

    return y_precision


# ---


def evaluate_1d_static_precision(df, column_to_bin, precision_columns, bins, round_to_decimal=4, min_num_samples=3):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) precision of 'precision_columns' for each 'bin' in 'column_to_bin' and (2) mean precision thereof.
    """
    # ensure list for iterating
    if not isinstance(precision_columns, list):
        precision_columns = [precision_columns]

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # returns an identical dataframe but adds a column named "bin"
    if (isinstance(bins, int) or isinstance(bins, float)):
        df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)

    elif isinstance(bins, (list, tuple, np.ndarray)):
        df = bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    # bin list
    df = df.sort_values('bin')
    bin_list = df.bin.unique()

    # calculate static lateral precision
    data = []
    for bn in df.bin.unique():
        dfb = df[df['bin'] == bn]

        if len(dfb) < min_num_samples:
            continue

        temp_precision = []
        temp_mean = []
        temp_counts = len(dfb)

        # evaluate the precision for each column
        for pcol in precision_columns:
            temp_precision.append(calculate_precision(dfb[pcol]))
            temp_mean.append(dfb[pcol].mean())

        data.append([bn] + [temp_counts] + temp_precision + temp_mean)

    data = np.array(data)

    mean_columns = [c + '_m' for c in precision_columns]
    columns = [column_to_bin] + ['counts'] + precision_columns + mean_columns

    dfp = pd.DataFrame(data, columns=columns)

    dfpm = dfp[precision_columns].mean()

    return dfp, dfpm


# ---


def evaluate_2d_static_precision(df, column_to_bin, precision_columns, bins=25, round_to_decimal=4):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # bin - top level (usually a spatial parameter: x, y, z, r)
    # old method: df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)

    # returns an identical dataframe but adds a column named "bin"
    if (isinstance(bins, int) or isinstance(bins, float)):
        df = bin_by_column(df, column_to_bin=column_to_bin, number_of_bins=bins, round_to_decimal=round_to_decimal)

    elif isinstance(bins, (list, tuple, np.ndarray)):
        df = bin_by_list(df, column_to_bin=column_to_bin, bins=bins, round_to_decimal=round_to_decimal)

    data = []

    # for each bin (x, y, z)
    for bn in df.bin.unique():

        # get the dataframe for this bin only
        dfb = df[df['bin'] == bn]

        # get particles in this bin
        dfb_pids = dfb.id.unique()

        # for each particle in this bin
        for b_pid in dfb_pids:

            # get the dataframe for this particle in this bin only
            dfbpid = dfb[dfb['id'] == b_pid]

            temp_precision = []
            temp_mean = []
            temp_counts = len(dfbpid)

            # evaluate the precision for each column
            for pcol in precision_columns:
                temp_precision.append(calculate_precision(dfbpid[pcol]))
                temp_mean.append(dfbpid[pcol].mean())

            data.append([bn] + [b_pid] + temp_precision + temp_mean + [temp_counts])

    data = np.array(data)

    mean_columns = [c + '_m' for c in precision_columns]
    columns = [column_to_bin] + ['id'] + precision_columns + mean_columns + ['counts']

    # dataframe(bin, id)
    df_bin_id = pd.DataFrame(data, columns=columns)

    # dataframe(bin)
    # 1. weighted average
    series_bin = df_bin_id.groupby(column_to_bin).apply(lambda x: np.average(x, axis=0, weights=x.counts)).to_numpy()
    series_data = []
    for sb in series_bin:
        series_data.append(sb)
    series_data_arr = np.array(series_data)
    df_bin = pd.DataFrame(series_data_arr, columns=columns)

    # 2. count number of particles + frames in each bin
    bin_counts = df_bin_id.groupby(column_to_bin).sum().counts
    df_bin = df_bin.drop(columns=['id', 'counts'])
    df_bin['counts'] = bin_counts.to_numpy()

    return df_bin_id, df_bin


# ---


def evaluate_3d_static_precision(df, columns_to_bin, precision_columns, bins, round_to_decimals):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    column_to_bin_top_level = columns_to_bin[0]
    column_to_bin_low_level = columns_to_bin[1]

    bins_top_level = bins[0]
    bins_low_level = bins[1]

    round_to_decimals_top_level = round_to_decimals[0]
    round_to_decimals_low_level = round_to_decimals[1]

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin_top_level, column_to_bin_low_level])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # bin - top level (usually an axial spatial parameter: z)
    if (isinstance(bins_top_level, int) or isinstance(bins_top_level, float)):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_top_level,
                           number_of_bins=bins_top_level,
                           round_to_decimal=round_to_decimals_top_level)

    elif isinstance(bins_top_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_top_level,
                         bins=bins_top_level,
                         round_to_decimal=round_to_decimals_top_level)

    df = df.rename(columns={'bin': 'bin_tl'})

    # bin - low level (usually a lateral spatial parameter: x, y, r, dx, percent overlap diameter)
    if isinstance(bins_low_level, (int, float)):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_low_level,
                           number_of_bins=bins_low_level,
                           round_to_decimal=round_to_decimals_low_level)

    elif isinstance(bins_low_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_low_level,
                         bins=bins_low_level,
                         round_to_decimal=round_to_decimals_low_level)

    df = df.rename(columns={'bin': 'bin_ll'})

    data = []

    # for each bin (z)
    for bntl in df.bin_tl.unique():

        # get the dataframe for this bin only
        dfbtl = df[df['bin_tl'] == bntl]

        # for each bin (x, y, r)
        for bnll in dfbtl.bin_ll.unique():
            # get the dataframe for this bin only
            dfbll = dfbtl[dfbtl['bin_ll'] == bnll]

            # get particles in this bin
            dfb_pids = dfbll.id.unique()

            # for each particle in this bin
            for b_pid in dfb_pids:

                # get the dataframe for this particle in this bin only
                dfbpid = dfbll[dfbll['id'] == b_pid]

                temp_precision = []
                temp_mean = []
                temp_counts = len(dfbpid)

                # evaluate the precision for each column
                for pcol in precision_columns:
                    temp_precision.append(calculate_precision(dfbpid[pcol]))
                    temp_mean.append(dfbpid[pcol].mean())

                data.append([bntl, bnll, b_pid] + [temp_counts] + temp_precision + temp_mean)

    data = np.array(data)

    mean_columns = [c + '_m' for c in precision_columns]
    columns = columns_to_bin + ['id', 'counts'] + precision_columns + mean_columns

    # dataframe(bin, id)
    df_bin_id = pd.DataFrame(data, columns=columns)

    # dataframe(bin)
    # 1. weighted average
    series_bin = df_bin_id.groupby(columns_to_bin).apply(lambda x: np.average(x, axis=0, weights=x.counts)).to_numpy()
    series_data = []
    for sb in series_bin:
        series_data.append(sb)
    series_data_arr = np.array(series_data)
    df_bin = pd.DataFrame(series_data_arr, columns=columns)

    # 2. count number of particles + frames in each bin
    bin_counts = df_bin_id.groupby(columns_to_bin).sum().counts
    df_bin = df_bin.drop(columns=['id', 'counts'])
    df_bin['counts'] = bin_counts.to_numpy()

    return df_bin_id, df_bin


# ---


# ------------------------------------ MODEL ERROR ASSESSMENT FUNCTIONS ------------------------------------------------


# evaluate model

def evaluate_reference_model_by_bin(model,
                                    dict_model,
                                    df,
                                    column_to_bin,
                                    column_to_fit,
                                    xy_cols,
                                    path_results,
                                    std_filter,
                                    distance_filter,
                                    save_figs,
                                    save_every_x_fig=5,
                                    ):
    if model == 'plane':
        refModel = models.ReferencePlane3D(dict_model)
    elif model == 'bispl':
        refModel = models.ReferenceSmoothBivariateSpline(dict_model)
    else:
        raise ValueError("Implemented models are 'plane' and 'bispl'.")

    # setup
    df = df.sort_values(column_to_bin)
    bins = df[column_to_bin].unique()
    data = []

    # iterate
    for i, bin in enumerate(bins):
        # plot option
        if save_figs is True and i % save_every_x_fig == 0:
            sf = True
        else:
            sf = False

        # 1. get dataframe for this bin
        dfb = df[df[column_to_bin] == bin]
        num_counts_raw = len(dfb)

        # 2. apply filters
        if std_filter is not None:
            dfb = dfb[np.abs(dfb[column_to_fit] - dfb[column_to_fit].mean()) < dfb[column_to_fit].std() * std_filter]

        if distance_filter is not None:
            dfb = dfb[np.abs(dfb[column_to_fit] - dfb[column_to_fit].mean()) < distance_filter]

        num_counts_filtered = len(dfb)

        # 2. apply model
        xy_data = dfb[xy_cols].to_numpy()
        z_data = dfb[column_to_fit].to_numpy()
        fig, ax, dz, dz_ref, rmse, r_squared = refModel.fit_evaluate_plot_data_on_surface(xy_data, z_data, sf)

        # 3. save figure
        if fig is not None:
            fig.savefig(
                path_results + '/fit-spline_{}={}_rmse={}.png'.format(column_to_bin,
                                                                      np.round(bin, 2),
                                                                      np.round(rmse, 2),
                                                                      ),
            )
            plt.close(fig)

        # 4. store results
        data.append([bin, num_counts_raw, num_counts_filtered, dz, dz_ref, rmse, r_squared])

    # package
    dfres = pd.DataFrame(np.array(data),
                         columns=[column_to_bin, 'num_raw', 'num_filtered', 'dz', 'dz_ref', 'rmse', 'r_squared'])

    return dfres


# ---


# Calculate error of points relative to a fitted plane measured on a different dataset

class calculatePlaneError:

    def __init__(self, dict_fit_plane):
        # inherent values
        self.dict_fit_plane = dict_fit_plane

        popt_plane = dict_fit_plane['popt_pixels']
        self.a = popt_plane[0]
        self.b = popt_plane[1]
        self.c = popt_plane[2]
        self.d = popt_plane[3]
        self.normal = popt_plane[4]

        self.px = dict_fit_plane['px']
        self.py = dict_fit_plane['py']
        self.pz = dict_fit_plane['pz']

        self.zf_img_xyc = dict_fit_plane['z_f_fit_plane_image_center']

        # derived values
        self.popt_dz = None
        self.pz_dz = None
        self.rmse = None
        self.r_square = None

    def calculate_surface_by_xy(self, x, y):
        """ calculate_z_of_3d_plane """
        z = (-self.normal[0] * x - self.normal[1] * y - self.d) * 1. / self.normal[2]
        return z

    def function_fit_surface_by_dz(self, xy, z_fit):
        return (-self.normal[0] * xy[:, 0] - self.normal[1] * xy[:, 1] - self.d) * 1. / self.normal[2] + z_fit

    def fit_data_to_surface(self, xy_data, z_data):
        """
        :param xy_data: [N x 2] array (e.g., df[xy_cols].to_numpy() where xy_cols = ['x', 'y']).
        :param z_data: [N x 1] array (e.g., df[z_data].to_numpy() where z_data = 'z').
        :return:
        """
        popt, pcov = curve_fit(self.function_fit_surface_by_dz, xy_data, z_data)

        self.popt_dz = popt[0]
        self.pz_dz = self.pz + self.popt_dz
        self.dz = self.popt_dz + self.zf_img_xyc

        return popt[0]

    def evaluate_rmse_from_surface(self, xy_data, z_data):

        popt = self.fit_data_to_surface(self, xy_data, z_data)

        z_surface = self.function_fit_surface_by_dz(self, xy=xy_data, z_fit=popt)

        # rmse and r-squared
        rmse, r_squared = fit.calculate_fit_error(fit_results=z_surface, data_fit_to=z_data)

        self.rmse = rmse
        self.r_square = r_squared

        return rmse, r_squared

    def plot_data_and_surface(self, xy_data, z_data):
        # plot
        fig = plt.figure(figsize=(6.5, 5))

        for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):

            ax = fig.add_subplot(2, 2, i, projection='3d')
            sc = ax.scatter(xy_data[:, 0], xy_data[:, 1], z_data, c=z_data, s=1)
            ax.plot_surface(self.px, self.py, self.pz_dz, alpha=0.4, color='red')
            ax.view_init(5, v)
            ax.patch.set_alpha(0.0)
            if i == 2:
                plt.colorbar(sc, shrink=0.5)
                ax.get_xaxis().set_ticks([])
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            elif i == 4:
                ax.get_yaxis().set_ticks([])
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_zlabel(r'$z \: (\mu m)$')
            else:
                ax.set_xlabel(r'$x \: (pixels)$')
                ax.set_ylabel(r'$y \: (pixels)$')
                ax.get_zaxis().set_ticklabels([])

        # title
        plt.suptitle('RMSE: {}, '.format(np.round(self.rmse, 3)) +
                     r'$R^2$' + ': {}'.format(np.round(self.r_squared, 3))
                     )

        plt.subplots_adjust(hspace=-0.1, wspace=0.15)

        return fig, ax

    def fit_evaluate_plot_data_on_surface(self, xy_data, z_data):
        rmse, r_squared = self.evaluate_rmse_from_surface(self, xy_data, z_data)

        fig, ax = self.plot_data_and_surface(self, xy_data, z_data)

        return fig, ax, self.dz, self.rmse, self.r_squared


# ---


class calculateBiSplineError:

    def __init__(self, bispl, dict_fit_plane):
        # inherent values
        self.bispl = bispl
        self.zf_img_xyc = dict_fit_plane['z_f_fit_plane_image_center']

        # derived values
        self.popt_dz = None
        self.pz_dz = None
        self.rmse = None
        self.r_square = None

    def calculate_surface_by_xy(self, x, y):
        z = self.bispl.ev(x, y)
        return z

    def function_fit_surface_by_dz(self, xy, z_fit):
        return self.bispl.ev(xy[:, 0], xy[:, 1]) + z_fit

    def fit_data_to_surface(self, xy_data, z_data):
        """
        :param xy_data: [N x 2] array (e.g., df[xy_cols].to_numpy() where xy_cols = ['x', 'y']).
        :param z_data: [N x 1] array (e.g., df[z_data].to_numpy() where z_data = 'z').
        :return:
        """
        popt, pcov = curve_fit(self.function_fit_surface_by_dz, xy_data, z_data)

        self.popt_dz = popt[0]
        self.dz = self.popt_dz + self.zf_img_xyc

        return popt[0]

    def evaluate_rmse_from_surface(self, xy_data, z_data):

        popt = self.fit_data_to_surface(self, xy_data, z_data)

        z_surface = self.function_fit_surface_by_dz(self, xy=xy_data, z_fit=popt)

        # rmse and r-squared
        rmse, r_squared = fit.calculate_fit_error(fit_results=z_surface, data_fit_to=z_data)

        if np.round(rmse, 1) != np.round(np.sqrt(self.bispl.get_residual() / len(z_data)), 1):
            raise ValueError("Custom r.m.s.e. != bispl r.m.s.e.")

        self.rmse = rmse
        self.r_square = r_squared

        return rmse, r_squared

    def plot_data_and_surface(self, xy_data, z_data, grid_resolution=20):

        fig, ax = plotting.scatter_3d_and_spline(x=xy_data[:, 0],
                                                 y=xy_data[:, 1],
                                                 z=z_data,
                                                 bispl=self.bispl,
                                                 cmap='RdBu',
                                                 grid_resolution=grid_resolution,
                                                 view='multi')

        if self.rmse is not None:
            plt.suptitle('fit RMSE={}, R^2={}'.format(np.round(self.rmse, 3), np.round(self.r_squared, 3)))

        return fig, ax

    def fit_evaluate_plot_data_on_surface(self, xy_data, z_data):
        rmse, r_squared = self.evaluate_rmse_from_surface(self, xy_data, z_data)

        fig = self.plot_data_and_surface(self, xy_data, z_data)

        return fig, self.dz, self.rmse, self.r_squared


# ---

def initialize_plate_model(df_test, df_results, membrane_radius, p_col, k_col, dict_params,
                           nonlinear_only=False, exclude_outside_membrane_radius=False):
    """
    dft = initialize_plate_model(df_test, membrane_radius, df_results, p_col, k_col)

    """

    # define parameters
    microns_per_pixel = dict_params['microns_per_pixel']
    t_membrane = dict_params['t_membrane']
    E_silpuran = dict_params['E_silpuran']
    poisson = dict_params['poisson']

    # derived parameters
    a_membrane = membrane_radius * microns_per_pixel
    t_membrane_norm = t_membrane * 1e6

    # ---

    # setup model
    fND = fNonDimensionalNonlinearSphericalUniformLoad(r=a_membrane * 1e-6,
                                                       h=t_membrane,
                                                       youngs_modulus=E_silpuran,
                                                       poisson=poisson)

    # 2. calculate non-dimensional pressure and pre-tension
    nd_P, nd_k = fND.non_dimensionalize_p_k(d_p0=df_results[p_col].to_numpy(),
                                            d_n0=df_results[k_col].to_numpy()
                                            )
    df_results['nd_p'] = nd_P
    df_results['nd_k'] = nd_k

    # 4. Append nd_P, nd_k columns to 'dft'

    # 4.1 - columns to be mapped
    df_test['d_p'] = df_test['frame']
    df_test['d_k'] = df_test['frame']
    df_test['nd_p'] = df_test['frame']
    df_test['nd_k'] = df_test['frame']

    # 4.2 - create mapping dict
    mapper_dict = df_results[['frame', p_col, k_col, 'nd_p', 'nd_k']].set_index('frame').to_dict()

    # 4.3 - map nd_P, nd_k to 'dft' by 'frame'
    df_test = df_test.replace({'d_p': mapper_dict[p_col]})
    df_test = df_test.replace({'d_k': mapper_dict[k_col]})
    df_test = df_test.replace({'nd_p': mapper_dict['nd_p']})
    df_test = df_test.replace({'nd_k': mapper_dict['nd_k']})

    # ---

    # 5. COMPUTE NON-LINEAR NON-DIMENSIONAL

    # 5.1 - Calculate nd_z, nd_slope, nd_curvature using nd_P, nd_k.

    nd_P = df_test['nd_p'].to_numpy()
    nd_k = df_test['nd_k'].to_numpy()

    nd_r = df_test['r'].to_numpy() * microns_per_pixel / a_membrane
    nd_z = fND.nd_nonlinear_clamped_plate_p_k(nd_r, nd_P, nd_k)

    nd_theta = fND.nd_nonlinear_theta(nd_r, nd_P, nd_k)
    nd_curve = np.where(nd_k > 0.0,
                        fND.nd_nonlinear_curvature_lva(nd_r, nd_P, nd_k),
                        fND.nd_nonlinear_theta_plate(nd_r, nd_P),
                        )

    df_test['nd_r'] = nd_r
    df_test['nd_rg'] = df_test['rg'].to_numpy() * microns_per_pixel / a_membrane
    df_test['nd_dr'] = df_test['drg'] * microns_per_pixel / t_membrane_norm
    df_test['nd_dz'] = nd_z
    df_test['nd_dz_corr'] = (df_test['z_corr'] + df_test['z_offset']) / t_membrane_norm
    df_test['d_dz_corr'] = df_test['z_corr'] + df_test['z_offset']
    df_test['d_dz'] = nd_z * t_membrane_norm
    df_test['nd_theta'] = nd_theta
    df_test['nd_curve'] = nd_curve

    # 7. Calculate error: (z_corr - d_z); and squared error for rmse
    df_test['dz_error'] = df_test['d_dz_corr'] - df_test['d_dz']
    df_test['z_rmse'] = df_test['dz_error'] ** 2

    # -

    if nonlinear_only:
        df_test = df_test[df_test['nd_k'] > 0.0001]

    if exclude_outside_membrane_radius:
        df_test = df_test[df_test['r'] < membrane_radius]

    # -

    return df_test, fND


# ---


# ---------------------------------------------------- BREAK -----------------------------------------------------------

# ----------------------------------------------------       -----------------------------------------------------------


# ---------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------


def scatter_xy_on_plane_by_bin(df, column_to_bin, column_to_fit, column_to_color, xy_cols, dict_plane,
                               path_results, save_id, scatter_size=5, plane_alpha=0.2,
                               relative=False, cx=None, cy=None, flip_correction=False):
    path_figs_dist_xy = path_results + '/figs_xy_by_{}_{}'.format(column_to_bin, save_id)
    if not os.path.exists(path_figs_dist_xy):
        os.makedirs(path_figs_dist_xy)

    # processing
    xmin, xmax = df[xy_cols[0]].min(), df[xy_cols[0]].max()
    ymin, ymax = df[xy_cols[1]].min(), df[xy_cols[1]].max()

    x_data = np.array([xmin, xmax, xmax, xmin, xmin])
    y_data = np.array([ymin, ymin, ymax, ymax, ymin])

    if relative:
        popt_plane = dict_plane['popt_pixels']
        z_relative_zero = functions.calculate_z_of_3d_plane(cx, cy, popt=popt_plane)
        z_data = functions.calculate_z_of_3d_plane(x_data, y_data, popt_plane) - z_relative_zero

        if flip_correction:
            z_data = z_data * -1

    else:
        popt_plane = dict_plane['popt_pixels']
        z_data = functions.calculate_z_of_3d_plane(x_data, y_data, popt_plane)

    # iterate
    for cb in df[column_to_bin].unique():
        dfb = df[df[column_to_bin] == cb]

        # plane z-offset
        z_points_mean = dfb[column_to_fit].mean()
        z_plane_mean = np.mean(z_data)
        z_offset = z_plane_mean - z_points_mean

        fig, (axx, axy) = plt.subplots(ncols=2, sharey=True, figsize=(size_x_inches * 1.2, size_y_inches * 0.8))

        axx.scatter(dfb[xy_cols[0]], dfb[column_to_fit], c=dfb[column_to_color], s=scatter_size)
        axx.plot(x_data, z_data - z_offset, color='red', alpha=plane_alpha)

        axy.scatter(dfb[xy_cols[1]], dfb[column_to_fit], c=dfb[column_to_color], s=scatter_size)
        axy.plot(y_data, z_data - z_offset, color='red', alpha=plane_alpha)

        axx.set_xlabel(xy_cols[0])
        axx.set_ylabel(column_to_fit)
        axy.set_xlabel(xy_cols[1])
        plt.tight_layout()
        plt.savefig(path_figs_dist_xy + '/scatter-xy_bin-{}={}.png'.format(column_to_bin, np.round(cb, 2)))
        plt.close()


# ---


def evaluate_error_from_fitted_plane(df, dict_fit_plane, xy_cols, eval_col):
    """
    NOTE:
        > columns are added to the dataframe.
        > the name of the column depends on the name of 'eval_col'.
        > an example is provided below for 'eval_col' = 'z':

            'z_plane': evaluated plane height at particle position (x, y)
            'error_z_plane': error of particle position (z) relative to plane height
            'dz_plane': mean position (z) of plane
                        > i.e., @ image center: dict_fit_plane['z_f_fit_plane_image_center']
    """

    if not isinstance(dict_fit_plane, dict):
        raise ValueError("dict_fit_plane must be a 'dict'.")

    # Instantiate the 'Plane' class
    fPE = calculatePlaneError(dict_fit_plane)
    xy_data = df[xy_cols].to_numpy()
    z_data = df[eval_col].to_numpy()

    # Find the 'z' of best fit (between data and plane)
    popt_z = fPE.fit_data_to_surface(xy_data, z_data)

    # calculate 'z' for each point
    df[eval_col + '_plane'] = fPE.function_fit_surface_by_dz(xy_data, popt_z)
    df['error_' + eval_col + '_plane'] = df[eval_col] - df[eval_col + '_plane']
    df['d' + eval_col + '_plane'] = popt_z + dict_fit_plane['z_f_fit_plane_image_center']

    # rmse and r-squared
    rmse, r_squared = fit.calculate_fit_error(fit_results=df[eval_col + '_plane'].to_numpy(),
                                              data_fit_to=z_data)

    return df, rmse, r_squared


def helper_plot_fitted_plane_and_points(df, dict_fit_plane, xy_cols, eval_col, rmse, r_squared):
    # data
    px, py, pz = dict_fit_plane['px'], dict_fit_plane['py'], dict_fit_plane['pz']
    pz_adj = pz + df['d' + eval_col + '_plane'].mean() - dict_fit_plane['z_f_fit_plane_image_center']

    # plot
    fig = plt.figure(figsize=(6.5, 5))

    for i, v in zip(np.arange(1, 5), [45, 0, 315, 270]):

        ax = fig.add_subplot(2, 2, i, projection='3d')
        sc = ax.scatter(df[xy_cols[0]], df[xy_cols[1]], df[eval_col], c=df[eval_col], s=1)
        ax.plot_surface(px, py, pz_adj, alpha=0.4, color='red')
        ax.view_init(5, v)
        ax.patch.set_alpha(0.0)
        if i == 2:
            plt.colorbar(sc, shrink=0.5)
            ax.get_xaxis().set_ticks([])
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        elif i == 4:
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_zlabel(r'$z \: (\mu m)$')
        else:
            ax.set_xlabel(r'$x \: (pixels)$')
            ax.set_ylabel(r'$y \: (pixels)$')
            ax.get_zaxis().set_ticklabels([])

    # title
    plt.suptitle('RMSE: {}, '.format(np.round(rmse, 3)) +
                 r'$R^2$' + ': {}'.format(np.round(r_squared, 3))
                 )

    plt.subplots_adjust(hspace=-0.1, wspace=0.15)

    return fig


# ---


def evaluate_error_from_fitted_bispl(df, column_to_bin, column_to_fit, xy_cols, kx, ky, path_results,
                                     std_filter, min_num_counts, save_figs):
    df = df.sort_values(column_to_bin)
    bins = df[column_to_bin].unique()
    rmses = []
    for bin in bins:
        # 1. get dataframe for this bin
        dfb = df[df[column_to_bin] == bin]

        # 2. apply std filter
        dfb = dfb[np.abs(dfb[column_to_fit] - dfb[column_to_fit].mean()) < dfb[column_to_fit].std() * std_filter]

        # 3. apply min_num_counts filter
        if len(dfb) < min_num_counts:
            continue

        # 2. fit bivariate spline to points
        bispl, rmse = fit.fit_3d_spline(x=dfb[xy_cols[0]],
                                        y=dfb[xy_cols[1]],
                                        z=dfb[column_to_fit],
                                        kx=kx,
                                        ky=ky)

        # 3. store results
        rmses.append(rmse)

        # 4. plot
        if save_figs:
            fig, ax = plotting.scatter_3d_and_spline(x=dfb[xy_cols[0]],
                                                     y=dfb[xy_cols[1]],
                                                     z=dfb[column_to_fit],
                                                     bispl=bispl,
                                                     cmap='RdBu',
                                                     grid_resolution=20,
                                                     view='multi')
            ax.set_xlabel('{} (pixels)'.format(xy_cols[0]))
            ax.set_ylabel('{} (pixels)'.format(xy_cols[1]))
            ax.set_zlabel('{}'.format(column_to_fit))
            plt.suptitle('fit RMSE = {}'.format(np.round(rmse, 3)))
            plt.savefig(
                path_results + '/fit-spline_{}={}_rmse={}_kx{}_ky{}.png'.format(column_to_bin,
                                                                                np.round(bin, 2),
                                                                                np.round(rmse, 2),
                                                                                kx,
                                                                                ky),
            )
            plt.close()

    # plot rmse by bin
    fig, ax = plt.subplots()
    ax.plot(bins, rmses, 'o')
    ax.set_xlabel(column_to_bin)
    ax.set_ylabel('r.m.s. error ({})'.format(column_to_fit))
    ax.set_title('Mean r.m.s. error = {}'.format(np.round(np.mean(rmses), 3)))
    plt.tight_layout()
    plt.savefig(path_results + '/fit-spline_rmse-by-{}.png'.format(column_to_bin))
    plt.close()


# ---


def cov_1d_localization_error(df, column_to_bin, column_to_count, bins, round_to_decimal, column_to_error):
    """
    Coefficient of variation of localization error with respect to a single variable.

    Assumptions:
        * Variations in localization error are due ONLY to variations in 'column_to_bin'.
            ** Requires that variations due to other factors are effectively averaged out.
    """

    if 'sq_error' not in df.columns:
        df['sq_error'] = df[column_to_error] ** 2

    dfm, dfstd = bin_generic(df,
                             column_to_bin=column_to_bin,
                             column_to_count=column_to_count,
                             bins=bins,
                             round_to_decimal=round_to_decimal,
                             return_groupby=True,
                             )

    coefficient_of_variation = np.mean(dfstd.sq_error.to_numpy() / dfm.sq_error.to_numpy())

    return coefficient_of_variation


def cov_2d_localization_error(df, columns_to_bin, column_to_count, bins, round_to_decimals, column_to_error,
                              min_num_bin):
    """
    Coefficient of variation of localization error with respect to two variables.

    Assumptions:
        * Variations in localization error are due ONLY to variations in 'columns_to_bin'.
            ** Requires that variations due to other factors are effectively averaged out.
    """

    raise ValueError("It is not currently understood how this function is physically meaningful or how to properly "
                     "implement it.")


# ------------------------------------ DISPLACEMENT MEASUREMENT (DATAFRAMES) -------------------------------------------


def disc_calculate_local_displacement(df, xy_cols, r_cols, disc_coords, start_time, start_time_col,
                                      min_num_measurements, disc_id, disc_id_col, z_offset):
    """
    df = analyze.disc_calculate_local_displacement(df, xy_cols, r_cols, disc_coords, start_time, start_time_col,
    min_num_measurements)

    Calculates the in-plane displacement relative to initial in-plane position.
        > initial in-plane position: the "average" position of all frames/times prior to start time.
        > initial radial position: computed from 'xy_cols' and 'disc_coords'
        > in-plane and radial displacement: new columns are added using 'xy_cols' and 'r_cols' names

    Example new columns:
        > in-plane (xy) initial position and displacement: ['xg0', 'yg0', 'dxg', 'dyg']
        > in-plane (r) initial position and displacement: ['rg0', 'drg']

    Example parameters:
    xy_cols = [['x', 'y'], ['xg', 'yg'], ['gauss_xc', 'gauss_yc'], ['pdf_xc', 'pdf_yc']]
    r_cols = ['r', 'rg', 'gauss_r', 'pdf_r']

    disc_xc, disc_yc, disc_radius = 423, 502, 252
    disc_coords = [disc_xc, disc_yc, disc_radius]

    start_time = 1.2
    start_time_col = 't'

    min_num_measurements = 3
    disc_id = 1

    :param df:
    :param xy_cols:
    :param r_cols:
    :param disc_coords:
    :param start_time:
    :param min_num_measurements:
    :param disc_id:
    :param disc_id_col:

    :return:
    """
    # insert membrane id
    if disc_id_col not in df.columns:
        df.insert(loc=0, column=disc_id_col, value=disc_id)

    # get disc coordinates
    disc_xc, disc_yc, disc_radius = disc_coords
    if 'memb_radius' not in df.columns:
        df['memb_radius'] = disc_radius

    # add z-offset column (the 'z = 0' position for this feature)
    df['z_offset'] = z_offset

    # calculate radial position in each frame
    for r_col, xy_col in zip(r_cols, xy_cols):
        df[r_col] = functions.calculate_radius_at_xy(df[xy_col[0]], df[xy_col[1]], xc=disc_xc, yc=disc_yc)

    # get pre-deflection dataframe
    dfst = df[df[start_time_col] < start_time].groupby('id').mean().reset_index()

    df_temps = []
    for pid in df.id.unique():
        # get dataframe per particle
        dfpid = df[df['id'] == pid]
        dfpid_im = dfst[dfst['id'] == pid].reset_index()

        # discard particles with < 3 measurements
        if len(dfpid) < min_num_measurements:
            continue

        if len(dfpid_im) < 1:
            continue

        # add initial positions
        for r_col, xy_col in zip(r_cols, xy_cols):
            # add initial positions
            dfpid[xy_col[0] + '0'] = dfpid_im.iloc[0][xy_col[0]]
            dfpid[xy_col[1] + '0'] = dfpid_im.iloc[0][xy_col[1]]
            dfpid[r_col + '0'] = dfpid_im.iloc[0][r_col]

            # calculate relative displacement
            dfpid['d' + xy_col[0]] = dfpid[xy_col[0] + '0'] - dfpid[xy_col[0]]  # positive is right
            dfpid['d' + xy_col[1]] = dfpid[xy_col[1]] - dfpid[xy_col[1] + '0']  # positive is up
            dfpid['d' + r_col] = dfpid[r_col] - dfpid[r_col + '0']

        # store
        df_temps.append(dfpid)

    # package
    df = pd.concat(df_temps)

    return df


# ---


def disc_bin_plot_local_displacement(df,
                                     columns_to_bin, column_to_count,
                                     dz_col, z_offset,
                                     dr_cols,
                                     norm_r_bins, norm_z_r, norm_cols_str,
                                     memb_radius, memb_pids_list, memb_id, memb_id_col,
                                     export_results, path_results,
                                     show_plots=False, save_plots=False, microns_per_pixel=None, units_pixels=True,
                                     ):
    """
    dfm = analyze.disc_bin_plot_local_displacement(df,
                                     dz_col, z_offset,
                                     dr_cols,
                                     norm_r_bins, norm_z_r, norm_cols_str,
                                     memb_radius, memb_pids_list, memb_id, memb_id_col,
                                     export_results, path_results,
                                     return_binned_df=True,
                                     show_plots=False, save_plots=False, microns_per_pixel=None, units_pixels=True
                                     )

    :param df:
    :param dz_col:
    :param z_offset:
    :param dr_cols:
    :param norm_r_bins:
    :param norm_z_r:
    :param norm_cols_str:
    :param memb_radius:
    :param memb_pids_list:
    :param memb_id:
    :param memb_id_col:
    :param export_results:
    :param path_results:
    :param return_binned_df:
    :param show_plots:
    :param save_plots:
    :param microns_per_pixel:
    :param units_pixels:
    :return:
    """

    # ---

    # get initial stats
    i_num_rows = len(df)
    i_num_pids = len(df.id.unique())

    # get dataframe of particles on this membrane
    if memb_id_col is not None:
        df = df[df[memb_id_col] == memb_id]
        print("Evaluating {}/{} pids ({}/{} rows) on feature[{}] = {}".format(len(df.id.unique()), i_num_pids,
                                                                              len(df), i_num_rows,
                                                                              memb_id_col, memb_id))
    elif memb_pids_list is not None:
        df = df[df.id.isin(memb_pids_list)]
        print("Evaluating {}/{} pids ({}/{} rows)".format(len(df.id.unique()), i_num_pids, len(df), i_num_rows))
    else:
        print("Evaluating the full dataframe ({} rows) as a single feature.".format(i_num_rows))

    # 2d binning
    bin_frames = df[columns_to_bin[0]].unique()
    bin_r = np.round(np.array(norm_r_bins) * memb_radius, 0)
    bins = [bin_frames, bin_r]
    round_to_decimals = [0, 1]
    min_num_bin = 1
    return_groupby = True

    dfm, dfstd = bin.bin_generic_2d(df,
                                    columns_to_bin,
                                    column_to_count,
                                    bins,
                                    round_to_decimals,
                                    min_num_bin,
                                    return_groupby
                                    )
    # normalize columns
    dfm['norm_dz_value'] = norm_z_r[0]
    dfm['norm_dr_value'] = norm_z_r[1]

    # normalized displacement handles
    dz_col_offset = dz_col + '_offset'
    dz_col_norm = dz_col + '_offset' + norm_cols_str

    # normalize z-displacement
    dfm[dz_col_offset] = dfm[dz_col] + z_offset
    dfm[dz_col_norm] = dfm[dz_col_offset] / norm_z_r[0]

    # normalize r-displacement
    dr_cols_norm = [r_col + norm_cols_str for r_col in dr_cols]
    for norm_r_col, r_col in zip(dr_cols_norm, dr_cols):
        dfm[norm_r_col] = dfm[r_col] / norm_z_r[1]

    # ---

    dfm = dfm.sort_values(['bin_tl', 'bin_ll'])
    dfstd = dfstd.sort_values(['bin_tl', 'bin_ll'])

    if export_results:
        dfm.to_excel(path_results +
                     '/2d-bin-local-disp_disc-id{}_units=pixels_{}bins_mean.xlsx'.format(memb_id, len(bin_r)))
        dfstd.to_excel(path_results +
                       '/2d-bin-local-disp_disc-id{}_units=pixels_{}bins_std.xlsx'.format(memb_id, len(bin_r)))

    # ---

    # PLOT DIMENSIONAL Z- AND R-DISPLACEMENT BY TIME
    if show_plots or save_plots:

        plotting.plot_local_z_r_displacement(dfm,
                                             dz_col=dz_col,
                                             dr_cols=dr_cols,
                                             bin_r=bin_r,
                                             bin_r_lbls=norm_r_bins,
                                             microns_per_pixel=microns_per_pixel,
                                             units_pixels=units_pixels,
                                             clr_map=None,
                                             z_orders=None,
                                             show_plots=show_plots,
                                             save_plots=save_plots,
                                             save_id=memb_id,
                                             path_results=path_results,
                                             )

    # ---

    return dfm

# ---


def evaluate_displacement_precision(df, group_by, split_by, split_value, precision_columns, true_dz=None,
                                    std_filter=None):
    # split dataframe into initial and final
    dfi = df[df[split_by] < split_value]
    dff = df[df[split_by] > split_value]

    # count number of measurements
    initial_bins = dfi[group_by].unique()
    i_num = len(dfi)
    f_num = len(dff)

    # filter out measurements > 2 * standard deviations from the the mean
    if std_filter is not None:
        dfi = dfi[np.abs(dfi[precision_columns].mean() - dfi[precision_columns]) <
                  np.max([dfi[precision_columns].std() * std_filter, 20])]
        dff = dff[np.abs(dff[precision_columns].mean() - dff[precision_columns]) <
                  np.max([dff[precision_columns].std() * std_filter, 20])]

    # kernel density estimation plot
    """
    fig = plt.figure()
    yscatter = dff.z.to_numpy()
    xscatter = dfi.z.to_numpy()[:len(yscatter)]
    yscatter = yscatter[:len(xscatter)]
    dist = np.max([dff[precision_columns].std() * 1, 7.5])

    fig, ax, ax_histx, ax_histy = plotting.scatter_hist(xscatter, yscatter, fig,
                                               color=None,
                                               colormap='coolwarm',
                                               scatter_size=0.25,
                                               kde=True,
                                               distance_from_mean=dist)

    ax.set_xlim([np.mean(xscatter) - dist, np.mean(xscatter) + dist])
    ax.set_ylim([np.mean(yscatter) - dist, np.mean(yscatter) + dist])
    ax.set_xlabel(r'$z_{initial} \: (\mu m)$')
    ax.set_ylabel(r'$z_{final} \: (\mu m)$')
    ax_histx.set_title(r'$\Delta z_{true}=$' + ' {} '.format(true_dz) + r'$\mu m$')
    sp = '/Users/mackenzie/Desktop/gdpyt-characterization/publication data/iteration 5/experiment validation/figs/test_displacement_precision'
    #plt.savefig(sp + '/kde_truedz{}_mean_zf{}.png'.format(true_dz, np.round(np.mean(yscatter), 2)))
    plt.show()
    """

    # count number of measurements after filtering
    i_num_filt = len(dfi)
    f_num_filt = len(dff)

    # evaluate precision: initial
    dfip, dfipm = evaluate_1d_static_precision(dfi,
                                               column_to_bin=group_by,
                                               precision_columns=precision_columns,
                                               bins=initial_bins,
                                               round_to_decimal=0)
    dfip = dfip.set_index(group_by)
    cols = dfip.columns
    for c in cols:
        dfip = dfip.rename(columns={c: c + '_i'})

    # evaluate precision: final
    final_bins = dff[group_by].unique()
    dffp, dffpm = evaluate_1d_static_precision(dff,
                                               column_to_bin=group_by,
                                               precision_columns=precision_columns,
                                               bins=final_bins,
                                               round_to_decimal=0)
    dffp = dffp.set_index(group_by)
    cols = dffp.columns
    for c in cols:
        dffp = dffp.rename(columns={c: c + '_f'})

    # merge dataframes
    dfp = pd.concat([dfip, dffp], axis=1, join='inner', sort=False)

    # calculate displacement precision and magnitude
    dfp['p_dz'] = dfp.z_i + dfp.z_f
    dfp['m_dz'] = dfp.z_m_f - dfp.z_m_i

    mean_displacement_precision = dfp.p_dz.mean()
    mean_displacement_magnitude = dfp.m_dz.mean()
    mean_displacement_magnitude_precision = functions.calculate_precision(dfp.m_dz)
    num_pids_measured = len(dfp.m_dz)
    percent_pids_measured = len(dfp.m_dz) / len(initial_bins)
    percent_pid_frames_measured = (i_num_filt + f_num_filt) / (i_num + f_num)

    # calculate root mean squared error
    if true_dz:
        # calculate error
        dfp['error_dz'] = true_dz - dfp.m_dz

        rmse = np.sqrt(np.sum(dfp.error_dz.to_numpy() ** 2) / len(dfp.error_dz.to_numpy()))

        return mean_displacement_precision, mean_displacement_magnitude, mean_displacement_magnitude_precision, \
               percent_pids_measured, percent_pid_frames_measured, rmse, num_pids_measured, i_num_filt, f_num_filt

    else:
        return mean_displacement_precision, mean_displacement_magnitude, mean_displacement_magnitude_precision, \
               percent_pids_measured, percent_pid_frames_measured, num_pids_measured, i_num_filt, f_num_filt


# ---------------------------------------- RMSE FUNCTIONS (BELOW) ------------------------------------------------------


def calculate_bin_local_rmse_z(dficts, column_to_bin='z_true', bins=20, min_cm=0.5, z_range=None, round_to_decimal=4,
                               dficts_ground_truth=None):
    """
    Calculate the local rmse_z uncertainty for every dataframe in a dictionary and return the binned dataframe.

    :param dficts:
    :param column_to_bin:
    :param bins:
    :param min_cm:
    :param z_range:
    :param round_to_decimal:
    :param true_num_particles:
    :return:
    """

    dfbs = {}
    for name, df in dficts.items():

        if dficts_ground_truth is not None:
            df_ground_truth = dficts_ground_truth[name]
        else:
            df_ground_truth = None

        # calculate the local rmse_z uncertainty
        dfb = bin_local_rmse_z(df, column_to_bin=column_to_bin, bins=bins, min_cm=min_cm, z_range=z_range,
                               round_to_decimal=round_to_decimal, df_ground_truth=df_ground_truth)

        # update dictionary
        dfbs.update({name: dfb})

    return dfbs


# ---


def evaluate_2d_bin_local_rmse_z(df, columns_to_bin, bins, round_to_decimals, min_cm=0.5, equal_bins=None,
                                 error_column=None, include_xy=False):
    """
    Bin dataframe 'df' into 'bins' # of bins on column 'column_to_bin' after rounding to 'round_to_decimal' places.

    Return: (1) mean, (2) standard deviation, (3) and counts for a single column
    """
    column_to_bin_top_level = columns_to_bin[0]
    column_to_bin_low_level = columns_to_bin[1]

    if equal_bins is not None:
        if equal_bins[0] is True:
            out, eqbins = pd.qcut(df[column_to_bin_top_level], bins[0], retbins=True, duplicates='drop')
            eqbins = pd.Series(eqbins).rolling(2).mean()
            bins_top_level = eqbins.dropna().to_numpy()
        else:
            bins_top_level = bins[0]

        if equal_bins[1] is True:
            out, eqbins = pd.qcut(df[column_to_bin_low_level], bins[1], retbins=True, duplicates='drop')
            eqbins = pd.Series(eqbins).rolling(2).mean()
            bins_low_level = eqbins.dropna().to_numpy()
        else:
            bins_low_level = bins[1]
    else:
        bins_top_level = bins[0]
        bins_low_level = bins[1]

    round_to_decimals_top_level = round_to_decimals[0]
    round_to_decimals_low_level = round_to_decimals[1]

    # drop NaNs in 'column_to_bin' which cause an Empty DataFrame
    raw_length = len(df)
    df = df.dropna(subset=[column_to_bin_top_level, column_to_bin_low_level])
    dropna_length = len(df)
    if raw_length > dropna_length:
        print("Dropped {} rows with NaN.".format(raw_length - dropna_length))

    # bin - top level (usually an axial spatial parameter: z)
    if (isinstance(bins_top_level, int) or isinstance(bins_top_level, float)):
        df = bin_by_column(df,
                           column_to_bin=column_to_bin_top_level,
                           number_of_bins=bins_top_level,
                           round_to_decimal=round_to_decimals_top_level)

    elif isinstance(bins_top_level, (list, tuple, np.ndarray)):
        df = bin_by_list(df,
                         column_to_bin=column_to_bin_top_level,
                         bins=bins_top_level,
                         round_to_decimal=round_to_decimals_top_level)

    df = df.rename(columns={'bin': 'bin_tl'})
    df = df.sort_values('bin_tl')

    data = {}

    # for each bin (z)
    for bntl in df.bin_tl.unique():
        # bin - low level
        # get the dataframe for this bin only (usually a spatial parameter: x, y, r, dx, percent overlap diameter)
        dfbtl = df[df['bin_tl'] == bntl]

        # evaluate the local rmse_z uncertainty
        dfb = bin_local_rmse_z(dfbtl,
                               column_to_bin=column_to_bin_low_level,
                               bins=bins_low_level,
                               min_cm=min_cm,
                               round_to_decimal=round_to_decimals_low_level,
                               df_ground_truth=None,
                               error_column=error_column,
                               include_xy=include_xy,
                               )

        dfb = dfb.reset_index()

        data.update({bntl: dfb})

    return data


# ---


# ---------------------------------------- BIN FUNCTIONS (ABOVE) -------------------------------------------------------


# ---


# -------------------------------------- IN-PLANE DISTANCE FUNCTIONS (BELOW) -------------------------------------------


# ---


def calculate_distance_from_baseline_positions(df, xy_columns,
                                               df_ground_truth, ground_truth_xy_columns,
                                               name_dist_columns,
                                               error_threshold=512,
                                               drop_na=False,
                                               drop_na_cols=None,
                                               ):
    """ df = analyze.calculate_distance_from_baseline_positions(df, xy_columns,
    df_ground_truth, ground_truth_xy_columns, name_dist_columns, error_threshold=50)

    error_threshold: the minimum value should probably be 1/2 the maximum particle image diameter (i.e. radius)
    """
    max_n_neighbors = 1

    # true coords dataframe
    if len(df_ground_truth) > len(df_ground_truth.id.unique()):
        df_ground_truth = df_ground_truth.groupby('id').mean().reset_index()
    ground_truth_xy = df_ground_truth[[ground_truth_xy_columns[0], ground_truth_xy_columns[1]]].values

    # other "true" columns to use
    ground_truth_xs = df_ground_truth[ground_truth_xy_columns[0]].to_numpy()
    ground_truth_ys = df_ground_truth[ground_truth_xy_columns[1]].to_numpy()
    ground_truth_pids = df_ground_truth.id.to_numpy()

    # test coords dataframe
    df = df.sort_values(by=['frame', 'id'])

    data = []
    for fr in df.frame.unique():

        # get all the particles in this frame
        dfr = df[df['frame'] == fr]

        for xyl, ndc in zip(xy_columns, name_dist_columns):
            x_loc, y_loc = xyl[0], xyl[1]

            # fill NaNs with 1000 so they can be removed later
            dfr[x_loc] = dfr[x_loc].fillna(2000)
            dfr[y_loc] = dfr[y_loc].fillna(2000)

            # get locations
            locations = np.array([dfr[x_loc].values, dfr[y_loc].values]).T

            if len(locations) < 1:
                continue

            # ---

            # calcualte distance using NearestNeighbors
            nneigh = NearestNeighbors(n_neighbors=max_n_neighbors,
                                      algorithm='ball_tree',
                                      ).fit(ground_truth_xy)
            distances, indices = nneigh.kneighbors(locations)

            # add columns for distance + ID of nearest "true" particle
            distances = np.where(distances < error_threshold, distances, np.nan)

            nearest_pid = ground_truth_pids[indices]
            nearest_pid = np.where(distances < error_threshold, nearest_pid, np.nan)

            dfr['x_true'] = ground_truth_xs[indices]
            dfr['y_true'] = ground_truth_ys[indices]
            dfr['x_error_' + ndc] = dfr[x_loc] - dfr['x_true']
            dfr['y_error_' + ndc] = dfr[y_loc] - dfr['y_true']
            dfr['dist_' + ndc] = distances
            dfr['nid_' + ndc] = nearest_pid

            # replace 2000's with NaNs again
            dfr[x_loc] = dfr[x_loc].where(dfr[x_loc] < 1000, np.nan)
            dfr[y_loc] = dfr[y_loc].where(dfr[y_loc] < 1000, np.nan)

        # store
        data.append(dfr)

    # concat results
    df_res = pd.concat(data)

    if drop_na:
        if drop_na_cols is None:
            raise ValueError("Must define which columns to seek out NaNs.")
        df_res = df_res.dropna(subset=drop_na_cols)

    return df_res


# ---


def calculate_particle_to_particle_spacing(test_coords_path,
                                           theoretical_diameter_params_path,
                                           mag_eff,
                                           z_param='z',
                                           zf_at_zero=False,
                                           zf_param='zf_from_nsv',
                                           max_n_neighbors=10,
                                           true_coords_path=None,
                                           maximum_allowable_diameter=None,
                                           popt_contour=None,
                                           param_percent_diameter_overlap='gauss_diameter',
                                           microns_per_pixels=None,
                                           path_save_true_coords_overlap=None,
                                           id_save_true_coords_overlap=None,
                                           ):
    """
    1. Reads test_coords.xlsx and computes theoretical Gaussian diameter given parameters.
    2. Computes the percent diameter overlap of all particles in all frame (frame by frame).

    :param param_percent_diameter_overlap: options = ('gauss_diameter', 'contour_diameter')
    """

    # setup
    append_cols = ['frame', 'id', 'mean_dx', 'min_dx', 'num_dx', 'mean_dxo', 'num_dxo', 'percent_dx_diameter']

    # ---

    # read test coords
    if isinstance(test_coords_path, str):
        df = pd.read_excel(test_coords_path)
    elif isinstance(test_coords_path, pd.DataFrame):
        df = test_coords_path
    else:
        raise ValueError('test_coords_path should be a filepath or pd.DataFrame.')

    if 'frame' in df.columns:
        df = df.sort_values(by=['frame', 'id'])

    # read true coords (if available)
    if true_coords_path is not None:
        if isinstance(true_coords_path, str):
            if true_coords_path == 'from-focus':
                z_true_nearest_focus = df.iloc[df.z_true.abs().idxmin()].z_true
                df_focus = df[df['z_true'] == z_true_nearest_focus]
                df_focus = df_focus.groupby('id').mean()
                ground_truth_xy = df_focus[['x', 'y']].values
            elif true_coords_path.endswith('.txt'):
                ground_truth = np.loadtxt(true_coords_path)
                ground_truth_xy = ground_truth[:, 0:2]
            elif true_coords_path.endswith('.xlsx'):
                ground_truth = pd.read_excel(true_coords_path)
                ground_truth_xy = ground_truth[['x', 'y']].values
        elif isinstance(true_coords_path, pd.DataFrame):
            if 'filename' in true_coords_path.columns:
                filename = true_coords_path.filename.unique()[0]
                ground_truth_xy = true_coords_path[true_coords_path['filename'] == filename][['x', 'y']].values
            else:
                if len(true_coords_path) > len(true_coords_path.id.unique()):
                    true_coords_path = true_coords_path.groupby('id').mean().reset_index()
                ground_truth_xy = true_coords_path[['x', 'y']].values
        else:
            raise ValueError('true_coords_path should be a filepath or a pd.DataFrame.')

        # check if x-y units are in pixels or microns
        if df.x.max() - df.x.min() > 512:
            print("Detecting that x-y units of TEST COORDS are NOT in PIXELS!")

            if ground_truth_xy[:, 0].max() - ground_truth_xy[:, 0].min() < 512:
                print("Detecting that x-y units of GROUND TRUTH are YES in PIXELS!")

                if microns_per_pixels is not None:
                    if isinstance(microns_per_pixels, list):
                        if microns_per_pixels[0] == 'invert':
                            df['x'] = df['x'] / microns_per_pixels[1]
                            df['y'] = df['y'] / microns_per_pixels[1]
                            print("Multiplying x-y values of TEST COORDS by 1 / {} MICRONS_PER_PIXELS!".format(
                                microns_per_pixels[1]))
                            units = r'$(pix.)$'
                        else:
                            raise ValueError("List element 0 should be invert to apply to test coords.")
                    else:
                        ground_truth_xy = ground_truth_xy * microns_per_pixels
                        print("Multiplying x-y values of GROUND TRUTH by {} MICRONS_PER_PIXELS!".format(
                            microns_per_pixels))
                        units = r'$(\mu m)$'

                    # confirm x-y range of test coords and ground truth
                    print("TEST COORDS X-RANGE: ({}, {})".format(np.round(df.x.min(), 1), np.round(df.x.max(), 1)))
                    print("GROUND TRUTH X-RANGE: ({}, {})".format(np.round(ground_truth_xy[:, 0].min(), 1),
                                                                  np.round(ground_truth_xy[:, 0].max(), 1)))
                    print("TEST COORDS Y-RANGE: ({}, {})".format(np.round(df.y.min(), 1), np.round(df.y.max(), 1)))
                    print("GROUND TRUTH Y-RANGE: ({}, {})".format(np.round(ground_truth_xy[:, 1].min(), 1),
                                                                  np.round(ground_truth_xy[:, 1].max(), 1)))

                else:
                    raise ValueError("Need to confirm if units in TEST COORDS and GROUND TRUTH are matching.")
        else:
            print("Detecting that x-y units of TEST COORDS are YES in PIXELS!")
            units = r'$(pixels)$'

    else:
        ground_truth_xy = None

    # --- CHOICE OF PERCENT OVERLAP PARAMETER

    # plot true_coords under test_coords
    if ground_truth_xy is not None:
        fig, ax = plt.subplots(figsize=(size_x_inches * 1.1, size_y_inches))
        ax.scatter(ground_truth_xy[:, 0], ground_truth_xy[:, 1], s=3, color='lightskyblue', alpha=0.5,
                   label='ground truth')
        ax.scatter(df.x, df.y, s=1, marker='x', color='red', alpha=0.85, label='test')
        ax.set_xlabel(r'$x$' + ' ' + units)
        ax.set_ylabel(r'$y$' + ' ' + units)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        if path_save_true_coords_overlap is not None:
            plt.savefig(join(path_save_true_coords_overlap,
                             'ground-truth-overlap-coords_{}.png'.format(id_save_true_coords_overlap)))
        else:
            plt.show()
        plt.close()

    if param_percent_diameter_overlap == 'gauss_diameter':
        # --- GAUSSIAN DIAMETER
        diameter_params = pd.read_excel(theoretical_diameter_params_path, index_col=0)
        if zf_at_zero is True:
            zf = 0
        else:
            zf = diameter_params.loc[[zf_param]]['mean'].values[0]

        if 'pop_c1' in diameter_params.columns:
            c1 = diameter_params.loc[['pop_c1']]['mean'].values[0]
            c2 = diameter_params.loc[['pop_c2']]['mean'].values[0]
            mag_eff = diameter_params.loc[['mag_eff']]['mean'].values[0]
        else:
            c1 = diameter_params.loc[['c1']]['mean'].values[0]
            c2 = diameter_params.loc[['c2']]['mean'].values[0]

        if mag_eff is None:
            mag_eff, = io.read_pop_gauss_diameter_properties(theoretical_diameter_params_path)

        def theoretical_diameter_function(z):
            return mag_eff * np.sqrt(c1 ** 2 * (z - zf) ** 2 + c2 ** 2)

        df['gauss_diameter'] = theoretical_diameter_function(df[z_param])

        if maximum_allowable_diameter is not None:
            df['gauss_diameter'] = df['gauss_diameter'].where(df['gauss_diameter'] < maximum_allowable_diameter,
                                                              maximum_allowable_diameter)

    elif param_percent_diameter_overlap == 'contour_diameter':
        # --- CONTOUR DIAMETER
        if popt_contour is not None:
            if zf_at_zero is True:
                zf = 0
            elif isinstance(zf_at_zero, (int, float)):
                zf = zf_at_zero
            else:
                raise ValueError("Need to assign zf_at_zero to a 'int' or 'float' if not at zero.")

            df['contour_diameter'] = functions.general_gaussian_diameter(df[z_param] + zf, *popt_contour)

    else:
        append_cols = ['frame', 'id', 'mean_dx', 'min_dx', 'num_dx']

    data = []
    for fr in df.frame.unique():

        # get all the particles in this frame
        dfr = df[df['frame'] == fr]

        # get ids and locations of all particles in this frame
        pids = dfr.id.values
        locations = np.array([dfr.x.values, dfr.y.values]).T

        if len(locations) < 2:
            continue
        elif len(locations) < max_n_neighbors + 1:
            temp_max_n_neighbors = len(locations)
        else:
            temp_max_n_neighbors = max_n_neighbors + 1

        if ground_truth_xy is not None:
            nneigh = NearestNeighbors(n_neighbors=temp_max_n_neighbors, algorithm='ball_tree').fit(ground_truth_xy)
            distances, indices = nneigh.kneighbors(locations)
            distance_to_others = distances[:, 1:]
        else:
            nneigh = NearestNeighbors(n_neighbors=temp_max_n_neighbors, algorithm='ball_tree').fit(locations)
            distances, indices = nneigh.kneighbors(locations)
            distance_to_others = distances[:, 1:]

        for distance, pid in zip(distance_to_others, pids):

            # get series of just this particle id
            dfpid = dfr[dfr['id'] == pid]

            # calculate minimum distance
            mean_dx_all = np.mean(distance)
            min_dx_all = np.min(distance)
            num_dx_all = temp_max_n_neighbors - 1

            # calculate overlap
            if len(append_cols) > 5:
                diameter = dfpid[param_percent_diameter_overlap].values[0]
                percent_dx_diameter = diameter / min_dx_all

                overlapping_dists = distance[distance < diameter]
                if len(overlapping_dists) == 0:
                    mean_dxo = min_dx_all
                    num_dxo = 0
                elif len(overlapping_dists) == 1:
                    mean_dxo = overlapping_dists[0]
                    num_dxo = 1
                else:
                    mean_dxo = np.mean(overlapping_dists)
                    num_dxo = len(overlapping_dists)

                pid_to_particle_spacing = [fr, pid,
                                           mean_dx_all, min_dx_all, num_dx_all,
                                           mean_dxo, num_dxo, percent_dx_diameter,
                                           ]
            else:
                pid_to_particle_spacing = [fr, pid,
                                           mean_dx_all, min_dx_all, num_dx_all,
                                           ]

            data.append(pid_to_particle_spacing)

    # overlap dataframe
    df_dxo = pd.DataFrame(data, columns=append_cols)

    # merge original dataframe and overlap dataframe
    dfo = pd.merge(left=df, right=df_dxo, on=['frame', 'id'])

    return dfo


# ---


def calculate_dict_particle_spacing(dficts, theoretical_diameter_params_path, mag_eff,
                                    max_n_neighbors=10, dficts_ground_truth=None, maximum_allowable_diameter=None):
    dfos = {}
    for (name, df), (tname, tdf) in zip(dficts.items(), dficts_ground_truth.items()):
        # calculate percent diameter overlap
        dfo = calculate_particle_to_particle_spacing(df, theoretical_diameter_params_path, mag_eff,
                                                     max_n_neighbors=10, true_coords_path=tdf,
                                                     maximum_allowable_diameter=maximum_allowable_diameter)
        # update dictionary
        dfos.update({name: dfo})

    return dfos


# ---


def fit_contour_diameter(path_calib_spct_stats, fit_z_dist=30, show_plot=False, scale_diameter_by=None):
    df = pd.read_excel(path_calib_spct_stats)
    df = df.dropna()

    if 'contour_diameter' in df.columns:
        param_dia = 'contour_diameter'
    elif 'diameter_contour' in df.columns:
        param_dia = 'diameter_contour'
    else:
        raise ValueError('calib_spct_stats does not have a contour_diameter or diameter_contour column.')

    if scale_diameter_by is not None:
        df[param_dia] = df[param_dia] * scale_diameter_by
        print("Scaling contour diameter by {}.".format(scale_diameter_by))

    # plot raw prior to groupby
    if show_plot:
        fig, ax = plt.subplots()
        ax.scatter(df.z_corr, df[param_dia], s=1, color='gray', alpha=0.1, label='raw')

    # groupby should make fitting easier by reducing outliers
    df = df.groupby('z_true').mean().reset_index()
    df = df[(df.z_corr > -fit_z_dist) & (df.z_corr < fit_z_dist)]

    popt_contour, pcov_contour = curve_fit(functions.general_gaussian_diameter, df.z_corr, df[param_dia])

    if show_plot:
        ax.scatter(df.z_corr, df[param_dia], s=2.5, color=scired, label='avg.')
        ax.plot(df.z_corr, functions.general_gaussian_diameter(df.z_corr, *popt_contour),
                color='black', alpha=0.85, label='fit')
        ax.set_xlabel(r'$z_{corr} \: (\mu m)$')
        ax.set_ylabel(r'$d_{e, contour} \: (pixels)$')
        ax.set_title('zf={}, zf_corr={}'.format(np.round(df.iloc[df[param_dia].idxmin()].z, 2),
                                                np.round(df.iloc[df[param_dia].idxmin()].z_corr, 2)))
        ax.legend()

        return popt_contour, fig, ax

    else:
        return popt_contour


# --------------------------------------------------- END --------------------------------------------------------------


# ---------------------------------------- ONE-OFF FUNCTIONS (BELOW) ---------------------------------------------------


def evaluate_intrinsic_aberrations(df, z_f, min_cm, param_z_true='z_true', param_z_cm='z_cm', shift_z_by_z_f=True):
    """
    If dataframe (df) is already zero-centered, z_f should equal 0.
    :param df:
    :param z_f:
    :param param_z_true:
    :return:
    """
    ids = df.id.unique()
    num_ids = len(ids)
    num_frames = len(df.frame.unique())

    data_id = []
    dataz = []
    datacm = []

    for i in ids:

        dfp = df[(df['id'] == i)]
        frames = dfp.frame.unique()
        p_ids = []
        dzs = []
        asyms = []

        for f in frames:
            dfpf = dfp[dfp['frame'] == f]
            asym = eval_focal_asymmetry(dfpf, z_f, min_cm=min_cm, param_z_cm=param_z_cm)

            if asym != np.nan:
                if shift_z_by_z_f:
                    dzs.append(dfpf[param_z_true].unique() - z_f)
                else:
                    dzs.append(dfpf[param_z_true].unique())

                p_ids.append([i])
                asyms.append([asym])

        data_id.append(p_ids)
        dataz.append(dzs)
        datacm.append(asyms)

    data_id = list(itertools.chain(*data_id))
    dataz = list(itertools.chain(*dataz))
    datacm = list(itertools.chain(*datacm))

    dfai = np.hstack([data_id, dataz, datacm])
    dfai = pd.DataFrame(dfai, columns=['id', 'zs', 'cms'])
    dfai = dfai.dropna()

    # ---

    # add spatial data (x, y, r)

    # add columns to be mapped
    dfai['x'] = dfai['id']
    dfai['y'] = dfai['id']
    dfai['r'] = dfai['id']

    # create mapping dict
    dfg = df[['id', 'x', 'y', 'r']].groupby('id').mean()
    mapper_dict = dfg.to_dict()

    # map x, y, and r positions to particles by 'id'
    dfai = dfai.replace({'x': mapper_dict['x']})
    dfai = dfai.replace({'y': mapper_dict['y']})
    dfai = dfai.replace({'r': mapper_dict['r']})

    dict_intrinsic_aberrations = {'zf': z_f,
                                  'num_pids': num_ids,
                                  'num_frames': num_frames,
                                  'dfai': dfai,
                                  }

    return dict_intrinsic_aberrations


def eval_focal_asymmetry(df, zf, min_cm, param_z_cm='z_cm'):
    cml = df[df[param_z_cm] < zf].cm.max()
    cmr = df[df[param_z_cm] > zf].cm.max()
    if np.min([cml, cmr]) < min_cm:
        asym = np.nan
    else:
        asym = cml / cmr - 1
    return asym


def fit_intrinsic_aberrations(dict_intrinsic_aberrations):
    zs = dict_intrinsic_aberrations['dfai'].zs
    cms = dict_intrinsic_aberrations['dfai'].cms

    zfit = np.linspace(np.min(zs), np.max(zs), 200)

    cpopt, cpcov = curve_fit(functions.cubic, zs, cms)
    qpopt, qpcov = curve_fit(functions.quartic, zs, cms)

    cmfit_cubic = functions.cubic(zfit, *cpopt)
    cmfit_quartic = functions.quartic(zfit, *qpopt)

    dict_intrinsic_aberrations.update({'zfit': zfit,
                                       'cpopt': cpopt,
                                       'cmfit_cubic': cmfit_cubic,
                                       'qpopt': qpopt,
                                       'cmfit_quartic': cmfit_quartic
                                       })

    return dict_intrinsic_aberrations


def evaluate_ia_sensitivity(df):
    """
    z_dependent_sensitivity, global_sensitivity = analyze.evaluate_ia_sensitivity(df)

    The focal plane (z_f) must be at z = 0.

    :param df:
    :return:
    """

    # initialize dataframe for mapping replacement
    df_cms_mean_by_zs = df.groupby('zs').mean()
    df_cms_std_by_zs = df.groupby('zs').std()
    df_sensitivity = df[['zs', 'cms']].groupby('zs').count()
    df_sensitivity['mean_cms'] = df_cms_mean_by_zs['cms']
    df_sensitivity['std_cms'] = df_cms_std_by_zs['cms']
    df_sensitivity['num_meas'] = df_sensitivity['cms']
    df_sensitivity['true_positives'] = 0
    df_sensitivity['false_negatives'] = 0

    # left half
    dfn = df[df['zs'] < 0][['zs', 'cms']]
    dfn_true_positive = dfn[dfn['cms'] > 0]
    dfn_false_negatives = dfn[dfn['cms'] < 0]
    dfnc_true_positive = dfn_true_positive.groupby('zs').count()
    dfnc_false_negatives = dfn_false_negatives.groupby('zs').count()

    # right half
    dfp = df[df['zs'] > 0][['zs', 'cms']]
    dfp_true_positive = dfp[dfp['cms'] < 0]
    dfp_false_negatives = dfp[dfp['cms'] > 0]
    dfpc_true_positive = dfp_true_positive.groupby('zs').count()
    dfpc_false_negatives = dfp_false_negatives.groupby('zs').count()

    # rename
    dfnc_true_positive = dfnc_true_positive.rename(columns={'cms': 'ntrue_positives'})
    dfnc_false_negatives = dfnc_false_negatives.rename(columns={'cms': 'nfalse_negatives'})
    dfpc_true_positive = dfpc_true_positive.rename(columns={'cms': 'ptrue_positives'})
    dfpc_false_negatives = dfpc_false_negatives.rename(columns={'cms': 'pfalse_negatives'})

    # concat
    df_sensitivity = pd.concat([df_sensitivity, dfnc_true_positive],
                               axis=1)  # replace({'true_positives': dfnc_true_positive['cms']})
    df_sensitivity = pd.concat([df_sensitivity, dfpc_true_positive],
                               axis=1)  # .replace({'true_positives': dfpc_true_positive['cms']})
    df_sensitivity = pd.concat([df_sensitivity, dfnc_false_negatives],
                               axis=1)  # .replace({'false_negatives': dfnc_false_negatives['cms']})
    df_sensitivity = pd.concat([df_sensitivity, dfpc_false_negatives],
                               axis=1)  # .replace({'false_negatives': dfpc_false_negatives['cms']})

    # fill na with zeros
    df_sensitivity = df_sensitivity.fillna(0)

    # local
    df_sensitivity['true_positives'] = df_sensitivity['ntrue_positives'] + df_sensitivity['ptrue_positives']
    df_sensitivity['false_negatives'] = df_sensitivity['nfalse_negatives'] + df_sensitivity['pfalse_negatives']

    # drop columns
    df_sensitivity = df_sensitivity.drop(columns=['cms',
                                                  'ntrue_positives', 'ptrue_positives',
                                                  'nfalse_negatives', 'pfalse_negatives',
                                                  ],
                                         )

    df_sensitivity['true_positive_rate'] = df_sensitivity['true_positives'] / \
                                           (df_sensitivity['true_positives'] + df_sensitivity['false_negatives'])

    df_sensitivity['sensitivity'] = df_sensitivity['true_positives'].sum() / \
                                    (df_sensitivity['true_positives'].sum() + df_sensitivity['false_negatives'].sum())

    return df_sensitivity


def evaluate_cm_gradient(df):
    pids = []
    z_trues = []
    dcmdzs = []

    for pid in df.id.unique():

        dfpid = df[df['id'] == pid]

        for fr in dfpid.frame.unique():

            dfr = dfpid[dfpid['frame'] == fr]

            dfr = dfr.reset_index(drop=True)
            z_cm_idx = dfr.cm.idxmax()

            if (z_cm_idx == len(dfr) - 1) or (z_cm_idx == 0):
                continue
            else:
                pids.append(pid)
                z_trues.append(dfr.z_true.unique()[0])
                dcmdzs.append(compute_cm_gradient(dfr, z_cm_idx))

    df_dcmdz = pd.DataFrame(np.vstack([pids, z_trues, dcmdzs]).T, columns=['id', 'z_true', 'dcmdz'])

    return df_dcmdz


def evaluate_self_similarity_gradient(dft, z_f, dcm_threshold=None, path_figs=False, z_range=None):
    # get only particle ID's present in all frames
    dft = dft[dft['layers'] == dft['layers'].max()]

    # center z on focal plane
    dft['z'] = dft['z'] - z_f

    # filter z_range
    if z_range is not None:
        dft = dft[(dft['z'] > -z_range) & (dft['z'] < z_range)]

    # compute gradient for each particle ID
    zs = []
    dcms = []
    pids = []
    for pid in dft.id.unique():
        dfpid = dft[dft['id'] == pid]
        zs.append(dfpid.z.to_numpy())
        dcms.append(dfpid.cm.diff().to_numpy())
        pids.append(dfpid.id.to_numpy())

    # store as dataframe and drop NaNs
    df = pd.DataFrame(np.vstack([np.array(pids).flatten(),
                                 np.array(zs).flatten(),
                                 np.array(dcms).flatten()]).T,
                      columns=['id', 'z', 'dcm'])
    df = df.dropna()

    # filter dcm/dz
    if dcm_threshold is not None:
        df = df[df['dcm'].abs() < dcm_threshold]

    # fit parabola
    popt, pcov = curve_fit(functions.quadratic, df.z, df.dcm.abs())
    z_fit = np.linspace(df.z.min(), df.z.max())
    dcm_fit = functions.quadratic(z_fit, *popt)

    # fit sliding parabola
    popt_slide, pcov = curve_fit(functions.quadratic_slide, df.z, df.dcm.abs())
    dcm_fit_slide = functions.quadratic_slide(z_fit, *popt_slide)

    # groupby z (for concise plotting)
    dfg = df.groupby('z').mean().reset_index()

    # plot
    if path_figs:
        fig, ax = plt.subplots(nrows=3, figsize=(size_x_inches, size_y_inches * 2))

        # raw data
        ax[0].scatter(dft.z, dft.cm, c=dft.id, s=1, alpha=0.25)
        ax[0].set_xlabel('z')
        ax[0].set_ylabel('cm')

        # raw gradient data + fit
        ax[1].scatter(df.z, df.dcm.abs(), c=df.id, s=1, alpha=0.25)
        ax[1].plot(z_fit, dcm_fit, color='black')
        ax[1].plot(z_fit, dcm_fit_slide, color='black', linestyle='--')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('|dcm/dz|')
        """ax[1].set_title(r'Fit: {}x^2 + {}x + {}'.format(np.format_float_scientific(popt[0], precision=3, exp_digits=2),
                                                        np.format_float_scientific(popt[1], precision=3, exp_digits=2),
                                                        np.round(popt[2], 3),
                                                        )
                        )"""

        # groupby-z gradient + fit
        ax[2].scatter(dfg.z, dfg.dcm.abs(), s=3)
        ax[2].plot(z_fit, dcm_fit, color='black')
        ax[2].plot(z_fit, dcm_fit_slide, color='black', linestyle='--')
        ax[2].set_xlabel('z')
        ax[2].set_ylabel('|dcm/dz|')
        """ax[2].set_title(
            r'Fit: {}(x+{})^2 + {}x + {}'.format(np.format_float_scientific(popt_slide[0], precision=3, exp_digits=2),
                                                 np.round(popt_slide[1], 3),
                                                 np.format_float_scientific(popt_slide[2], precision=3, exp_digits=2),
                                                 np.round(popt_slide[3], 3),
                                                 )
            )"""

        plt.tight_layout()
        plt.savefig(path_figs + '/self_similarity_gradient.png')
        plt.show()

    return df


def calculate_plane_tilt_angle(df, microns_per_pixel, z, x='x', y='y'):
    # fit plane (x, y, z units: microns)
    points_microns = np.stack((df[x] * microns_per_pixel, df[y] * microns_per_pixel, df[z])).T

    px_microns, py_microns, pz_microns, popt_microns = fit.fit_3d(points_microns, fit_function='plane')

    # tilt angle (degrees)
    tilt_x = np.rad2deg(np.arctan((pz_microns[0, 1] - pz_microns[0, 0]) / (px_microns[0, 1] - px_microns[0, 0])))
    tilt_y = np.rad2deg(np.arctan((pz_microns[1, 0] - pz_microns[0, 0]) / (py_microns[1, 0] - py_microns[0, 0])))
    print("x-tilt = {} degrees".format(np.round(tilt_x, 3)))
    print("y-tilt = {} degrees".format(np.round(tilt_y, 3)))

    return tilt_x, tilt_y


# --------------------------------------------------- END --------------------------------------------------------------


# ----------------------------------------- HELPER FUNCTIONS (BELOW) ---------------------------------------------------


def compute_cm_gradient(df, z_cm_idx):
    # compute the left and right derivative
    dcmdz_l = (df.iloc[z_cm_idx].cm - df.iloc[z_cm_idx - 1].cm) / (df.iloc[z_cm_idx].z_cm - df.iloc[z_cm_idx - 1].z_cm)
    dcmdz_r = (df.iloc[z_cm_idx].cm - df.iloc[z_cm_idx + 1].cm) / (df.iloc[z_cm_idx].z_cm - df.iloc[z_cm_idx + 1].z_cm)

    # average derivatives
    mean_dcmdz = np.mean([np.abs(dcmdz_l), np.abs(dcmdz_r)])

    return mean_dcmdz


# ---

# ---

# --------------------------------------------------- END --------------------------------------------------------------


# ----------------------------------------- BELOW TO BE DEPRECATED -----------------------------------------------------


def df_calc_mean_std_count(df, by, keep_cols, std=True, count=True, return_original_df=False):
    """
    Calculate the dataframe mean, stdev, and counts by groupby keys, 'by'. Optional: stack these values onto the
    original dataframe as new columns.

    :param df:
    :param by:
    :param keep_cols:
    :param mean:
    :param std:
    :param count:
    :return:
    """

    dfm = df.groupby(by=by).mean()
    for col in keep_cols:
        dfm = dfm.rename(columns={col: '{}_mean'.format(col)})

    if std:
        dfstd = df.groupby(by=by).std()
        for col in keep_cols:
            dfstd = dfstd.rename(columns={col: '{}_std'.format(col)})
    else:
        dfstd = None

    if count:
        dfc = df.groupby(by=by).count()
        for col in keep_cols:
            dfc = dfc.rename(columns={col: '{}_counts'.format(col)})
    else:
        dfc = None

    if return_original_df:
        # 1.  dictionary of mapping values

        def map_dataframe(df, method='mean'):
            dff = df.reset_index()
            dff['mapper'] = dff['filename'].astype(str) + '_' + dff['id'].astype(str)
            dff = dff.set_index('mapper')
            dff = dff.drop(columns=['filename', 'id', 'cm_{}'.format(method)])
            mapping_dict = dff.to_dict()
            return mapping_dict

        mapper_dict = {}
        mapper_dict.update(map_dataframe(dfm, method='mean'))
        mapper_dict.update(map_dataframe(dfstd, method='std'))
        mapper_dict.update(map_dataframe(dfc, method='counts'))

        # 2. create columns for mapping in original dataframe
        df['z_mean'] = df['filename'].astype(str) + '_' + df['id'].astype(str)
        df['z_std'] = df['filename'].astype(str) + '_' + df['id'].astype(str)
        df['z_counts'] = df['filename'].astype(str) + '_' + df['id'].astype(str)

        # 3. map values from dict to new columns
        df = df.replace({'z_mean': mapper_dict['z_mean']})
        df = df.replace({'z_std': mapper_dict['z_std']})
        df = df.replace({'z_counts': mapper_dict['z_counts']})

        return df

    else:
        if all([std, count]):
            dfg = dfm.join([dfstd, dfc])
        elif std:
            dfg = dfm.join([dfstd])
        else:
            dfg = dfm

        return dfg

# ----------------------------------------- ABOVE TO BE DEPRECATED -----------------------------------------------------