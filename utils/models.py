# gdpyt-analysis: analyze
"""
Notes
"""

# imports
from os.path import join
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# ---

from scipy.optimize import curve_fit
from scipy import special

from utils import functions, fit, io
from utils import plotting
from utils.bin import *


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

# -------------------------------------------- REFERENTIAL MODELS ------------------------------------------------------

"""
Define: REFERENTIAL MODELS
    > A "model" is generated of a specific: distribution, profile, shape, etc.
    > This model then serves as a "reference" for comparison to other datasets (i.e., error evaluation).
    
"""

# Calculate error of points relative to a fitted plane measured on a different dataset

class ReferencePlane3D:
    """
    1. Initialize:
        > refPlane = models.ReferencePlane3D(dict_model)

    2. Evaluate (one 'frame' at a time):
        > fig, ax, dz, rmse, r_squared = refPlane.fit_evaluate_plot_data_on_surface(xy_data, z_data)

    """

    def __init__(self, dict_model):
        # inherent values
        self.dict_fit_plane = dict_model

        popt_plane = dict_model['popt_pixels']
        self.a = popt_plane[0]
        self.b = popt_plane[1]
        self.c = popt_plane[2]
        self.d = popt_plane[3]
        self.normal = popt_plane[4]

        self.px = dict_model['px']
        self.py = dict_model['py']
        self.pz = dict_model['pz']

        self.zf_img_xyc = dict_model['z_f_fit_plane_image_center']

        # reference points
        self.ref_xy = None
        self.ref_z = None
        self.set_reference(xy=dict_model['ref_xy'])

        # derived values
        self.popt_dz = None
        self.pz_dz = None
        self.dz = None
        self.ref_dz = None
        self.rmse = None
        self.r_square = None

    def set_reference(self, xy=None):
        if xy is not None:
            if isinstance(xy, list):
                xy = np.array(xy)

            xy = xy.flatten()
            self.ref_z = self.calculate_surface_by_xy(xy[0], xy[1])

        return self.ref_z

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
        self.pz_dz = self.popt_dz + self.pz
        self.dz = self.popt_dz + self.zf_img_xyc
        self.ref_dz = self.popt_dz + self.ref_z

        return popt[0]

    def evaluate_rmse_from_surface(self, xy_data, z_data):

        # fit to find dz
        popt = self.fit_data_to_surface(xy_data, z_data)

        # calculate z + dz
        z_surface = self.function_fit_surface_by_dz(xy=xy_data, z_fit=popt)

        # rmse and r-squared
        rmse, r_squared = fit.calculate_fit_error(fit_results=z_surface, data_fit_to=z_data)

        self.rmse = rmse
        self.r_squared = r_squared

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
        # plt.suptitle('RMSE= {}, R^2={}'.format(np.round(self.rmse, 3), np.round(self.r_squared, 3)))
        # plt.suptitle('RMSE = {}, R_sq = {}'.format(np.round(self.rmse, 3), np.round(self.r_squared, 3)))

        plt.subplots_adjust(hspace=-0.1, wspace=0.15)

        return fig, ax

    def fit_evaluate_plot_data_on_surface(self, xy_data, z_data, save_figs):
        """ fig, ax, dz, rmse, r_squared = fit_evaluate_plot_data_on_surface(self, xy_data, z_data) """
        rmse, r_squared = self.evaluate_rmse_from_surface(xy_data, z_data)

        if save_figs:
            fig, ax = self.plot_data_and_surface(xy_data, z_data)
        else:
            fig, ax = None, None

        return fig, ax, self.dz, self.ref_dz, self.rmse, self.r_squared


# ---


class ReferenceSmoothBivariateSpline:
    """
    1. Initialize:
        > refBiSpl = models.ReferenceSmoothBivariateSpline(dict_model)

    2. Evaluate (one 'frame' at a time):
        > fig, ax, dz, rmse, r_squared = refBiSpl.fit_evaluate_plot_data_on_surface(xy_data, z_data)

    """

    def __init__(self, dict_model):
        # inherent values
        self.bispl = dict_model['bispl']

        # reference points
        self.ref_xy = None
        self.ref_z = None
        self.set_reference(xy=dict_model['ref_xy'])

        # derived values
        self.popt_dz = None
        self.dz = None
        self.ref_dz = None
        self.rmse = None
        self.r_squared = None

    def calculate_surface_by_xy(self, x, y):
        z = self.bispl.ev(x, y)
        return z

    def set_reference(self, xy=None):
        if xy is not None:
            if isinstance(xy, list):
                xy = np.array(xy)

            xy = xy.flatten()
            self.ref_z = self.calculate_surface_by_xy(xy[0], xy[1])

        return self.ref_z

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
        self.dz = self.popt_dz
        self.ref_dz = self.popt_dz + self.ref_z

        return popt[0]

    def evaluate_rmse_from_surface(self, xy_data, z_data):

        popt = self.fit_data_to_surface(xy_data, z_data)

        z_surface = self.function_fit_surface_by_dz(xy=xy_data, z_fit=popt)

        # rmse and r-squared
        rmse, r_squared = fit.calculate_fit_error(fit_results=z_surface, data_fit_to=z_data)

        self.rmse = rmse
        self.r_squared = r_squared

        return rmse, r_squared

    def plot_data_and_surface(self, xy_data, z_data, grid_resolution=20):

        fig, ax = plotting.scatter_3d_and_spline(x=xy_data[:, 0],
                                                 y=xy_data[:, 1],
                                                 z=z_data,
                                                 bispl=self.bispl,
                                                 cmap='RdBu',
                                                 grid_resolution=grid_resolution,
                                                 view='multi',
                                                 bispl_z_offset=self.dz,
                                                 zlim_range=2,
                                                 scatter_size=2,
                                                 scatter_cmap='cool',
                                                 )

        #if self.rmse is not None:
        #    plt.suptitle('fit RMSE={}, R^2={}'.format(np.round(self.rmse, 3), np.round(self.r_squared, 3)))

        return fig, ax

    def fit_evaluate_plot_data_on_surface(self, xy_data, z_data, save_figs):
        rmse, r_squared = self.evaluate_rmse_from_surface(xy_data, z_data)

        if save_figs:
            fig, ax = self.plot_data_and_surface(xy_data, z_data)
        else:
            fig, ax = None, None

        return fig, ax, self.dz, self.ref_dz, self.rmse, self.r_squared

# ---