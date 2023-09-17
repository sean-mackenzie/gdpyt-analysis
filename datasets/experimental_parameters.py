# experimental and processing parameters + results

# imports
import os
from os.path import join

import numpy as np

from utils import functions


# ------------------------------------------------


class ExperimentalParametersUnpacker(object):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

        self.details = None

    def unpack(self):
        if self.dataset == 'subpix':
            # ----------------------------------------------------------------------------------------------------------
            # EXPERIMENTAL

            # optics
            mag_eff = 5.0
            numerical_aperture = 0.3
            wavelength = 600e-9
            index_of_refraction = 1.0
            pixel_size = 16
            depth_of_focus = functions.depth_of_field(mag_eff, numerical_aperture, wavelength, index_of_refraction,
                                                      pixel_size=pixel_size * 1e-6) * 1e6
            microns_per_pixel = 3.2
            exposure_time = 40e-3
            frame_rate = 24.444

            # mechanics
            E_silpuran = 500e3
            poisson = 0.5
            t_membrane = 20e-6

            # pressure application
            start_frame = 39
            start_time = start_frame / frame_rate

            # normalization
            t_membrane_norm = t_membrane * 1e6

            # ----------------------------------------------------------------------------------------------------------
            # PROCESSING

            padding_during_idpt_test_calib = 15
            image_length = 512
            img_xc = 256
            img_yc = 256

            """ --- MEMBRANE SPECIFIC PARAMETERS --- """

            # mask lower right membrane
            xc_lr, yc_lr, r_edge_lr = 423, 502, 252
            circle_coords_lr = [xc_lr, yc_lr, r_edge_lr]

            # mask upper left membrane
            xc_ul, yc_ul, r_edge_ul = 167, 35, 157
            circle_coords_ul = [xc_ul, yc_ul, r_edge_ul]

            # mask left membrane
            xc_ll, yc_ll, r_edge_ll = 12, 289, 78
            circle_coords_ll = [xc_ll, yc_ll, r_edge_ll]

            # mask middle
            xc_mm, yc_mm, r_edge_mm = 177, 261, 31
            circle_coords_mm = [xc_mm, yc_mm, r_edge_mm]

            # ----------------------------------------------------------------------------------------------------------------------
            # 2.5 IMPORTANT PHYSICAL POSITIONS

            # axial positions
            z_f_from_calib = 140
            z_offset_lr = 5
            z_offset_ul = 2
            z_inital_focal_plane_bias_errors = np.max([z_offset_lr, z_offset_ul]) + 5

            lr_w0_max = 133
            ul_w0_max = 64

            # exclude outliers
            """
            Saturated pids for ttemp <= 11: [12, 13, 18, 34, 39, 49, 66, 78]
            Saturated pids for ttemp == 13: [11, 12, 17, 33, 38, 48, 65, 77]
            """
            pids_saturated = [12, 13, 18, 34, 39, 49, 66, 78]
            exclude_pids = [39, 61]

            # ---

            self.details = {'microns_per_pixel': microns_per_pixel,
                            'membrane_radius': None,
                            't_membrane': t_membrane,
                            'E_silpuran': E_silpuran,
                            'poisson': poisson,
                            'a_membrane': None,
                            't_membrane_norm': t_membrane_norm,
                            'pids_saturated': pids_saturated,
                            'exclude_pids': exclude_pids,
                            }

            return self.details