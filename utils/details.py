# gdpyt-analysis: details
"""
Notes:
    * Read experiment details from strings or meta-data
"""

# imports
import os

import pandas as pd
import numpy as np

# scripts

# ---------------------------------------- FUNCTIONS (BELOW) -----------------------------------------------------------


def parse_filename_to_details(path_name, sort_strings, filetype='.xlsx', subset=None, startswith=None):
    """
    Parsable details:
        1. read string to get test details:
            1.1 measurement volume: full or half
            1.2 magnification: 20X or 10X
            1.3 demag: 1X or 0.5X
            1.4 particle diameter: 0.87, 2.15, 5.1, 5.61 um
            1.5 # of images in calibration averaging: 9
        2. create label
            2.1 mag, demag, pd (e.g. 20X-0.5X, 5.61 um; 10X, 2.15 um)

    :param path_name:
    :return:
    """

    calib_volume = ['coords_', 'h_']
    mag = ['h_', 'X_']
    demag = ['X_', 'demag_']
    pd = ['demag_', 'um_']
    calib_img_avg = ['um_mean', 'calib']

    # read files in directory
    if startswith is not None:
        files = [f for f in os.listdir(path_name) if f.startswith(startswith) and f.endswith(filetype)]
    else:
        files = [f for f in os.listdir(path_name) if f.endswith(filetype)]

    if len(files) == 0:
        raise ValueError("No files found at {}".format(path_name))

    if subset:
        files = files[:subset]

    files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1])[0]))
    names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1])[0]) for f in files]
    calib_volumes = [float(f.split(calib_volume[0])[-1].split(calib_volume[1])[0]) for f in files]
    mags = [float(f.split(mag[0])[-1].split(mag[1])[0]) for f in files]
    demags = [float(f.split(demag[0])[-1].split(demag[1])[0]) for f in files]
    pds = [float(f.split(pd[0])[-1].split(pd[1])[0]) for f in files]
    calib_img_avgs = [float(f.split(calib_img_avg[0])[-1].split(calib_img_avg[1])[0]) for f in files]

    details = {}
    for n, cv, m, dm, d, cia in zip(names, calib_volumes, mags, demags, pds, calib_img_avgs):

        if dm == 1.0:
            if cv == 1:
                lbl = '{}X, {}um'.format(m, d)
            if cv == 0.5:
                lbl = '{}X, {} um, h/2'.format(m, d)
        elif dm == 0.5:
            if cv == 1:
                lbl = '{}X-{}X, {} um, h'.format(m, dm, d)
            if cv == 0.5:
                lbl = '{}X-{}X, {} um, h/2'.format(m, dm, d)

        detail = {
                  'label': lbl,
                  'calib_volume': cv,
                  'mag': m,
                  'demag': dm,
                  'pd': d,
                  'calib_img_avg': cia,
                  }

        # update dictionary
        details.update({n: detail})

    return details


def read_dficts_coords_to_details(dficts, dficts_details, calib=False):
    """
    Parsable details:
        1. read test coords to get test details:
            1.1 measurement volume: z min, z max
            1.2 # of particles: p_num

    :param dficts:
    :param dficts_details:
    :return:
    """

    for name, df in dficts.items():

        if calib:
            zmin = df.z.min()
            zmax = df.z.max()
        else:
            zmin = df.z_true.min()
            zmax = df.z_true.max()

        meas_vol = zmax - zmin
        p_num = len(df.id.unique())

        # update dictionary
        dficts_details[name].update({
            'zmin': zmin,
            'zmax': zmax,
            'meas_vol': meas_vol,
            'p_num': p_num
        })

    return dficts_details




# ---------------------------------------------- END -------------------------------------------------------------------