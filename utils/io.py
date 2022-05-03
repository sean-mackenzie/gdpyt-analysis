# gdpyt-analysis: utils: read
"""
Notes
"""

# imports
import ast
from os.path import join
from os import listdir
import os
import pandas as pd
import numpy as np
from utils import modify, fit, functions


# ------------------------------------------- SIMPLE READ FUNCTIONS ----------------------------------------------------

def read_calib_coords(path_calib_coords, method):
    """
    To run:
        dfc, dfcpid, dfcpop, dfcstats = io.read_calib_coords(path_calib_coords, method)

    :param path_calib_coords:
    :param method:
    :return:
    """

    files = [f for f in listdir(path_calib_coords) if f.endswith('.xlsx')]

    # files
    file_dfc = [f for f in files if f.startswith('calib_correction_coords')]
    file_dfcpid = [f for f in files if f.startswith('calib_{}_pid_defocus_stats'.format(method))]
    file_dfcpop = [f for f in files if f.startswith('calib_{}_pop_defocus_stats'.format(method))]
    file_dfcstats = [f for f in files if f.startswith('calib_{}_stats'.format(method))]

    # read index column == 0
    index_cols = [None, None, 0, None]

    dfs = []
    for ic, file in zip(index_cols, [file_dfc, file_dfcpid, file_dfcpop, file_dfcstats]):
        if len(file) == 2:
            file_dfcpidxy = [f for f in file if f.endswith('xy.xlsx'.format(method))][0]
            df = pd.read_excel(join(path_calib_coords, file_dfcpidxy), index_col=ic)
        elif len(file) > 0:
            df = pd.read_excel(join(path_calib_coords, file[0]), index_col=ic)
        else:
            df = None
        dfs.append(df)

    dfc, dfcpid, dfcpop, dfcstats = dfs[0], dfs[1], dfs[2], dfs[3]

    if 'x' not in dfcpid.columns:
        if dfcstats is not None:
            dfcpid = modify.merge_calib_pid_defocus_and_correction_coords(path_calib_coords, method, dfs=[dfcstats,
                                                                                                          dfcpid])
        elif dfc is not None:
            dfcpid = modify.merge_calib_pid_defocus_and_correction_coords(path_calib_coords, method, dfs=[dfc,
                                                                                                          dfcpid])
        dfc = None

    return dfc, dfcpid, dfcpop, dfcstats


def read_pop_gauss_diameter_properties(dfcpop):

    if isinstance(dfcpop, str):
        dfcpop = pd.read_excel(dfcpop, index_col=0)

    mag_eff = dfcpop.loc['mag_eff', 'mean']
    zf = dfcpop.loc['zf_from_dia', 'mean']
    c1 = dfcpop.loc['pop_c1', 'mean']
    c2 = dfcpop.loc['pop_c2', 'mean']

    return mag_eff, zf, c1, c2


def read_test_coords(path_test_coords):
    files = [f for f in listdir(path_test_coords) if f.endswith('.xlsx')]
    df = pd.read_excel([join(path_test_coords, f) for f in files if f.startswith('test_coords')][0])

    return df


def read_similarity(path_similarity):
    files = [f for f in listdir(path_similarity) if f.endswith('.xlsx')]

    if len([f for f in files if f.startswith('calib_stacks_')]) > 0:
        dfsf = pd.read_excel([join(path_similarity, f) for f in files if f.startswith('calib_stacks_forward_self')][0])
        dfsm = pd.read_excel([join(path_similarity, f) for f in files if f.startswith('calib_stacks_middle_self')][0])
        if len(dfsf) < 10:
            dfsf = None
        if len(dfsm) < 10:
            dfsm = None
    else:
        dfsf, dfsm = None, None

    # read average similarity between particles per-frame
    if len([f for f in files if f.startswith('average_similarity_')]) > 0:
        dfas = pd.read_excel([join(path_similarity, f) for f in files if f.startswith('average_similarity_')][0])
        if len(dfas) < 10:
            dfas = None
    else:
        dfas = None

    # read particle similarity curves
    psc_files = [f for f in files if f.startswith('particle_similarity_curves')]

    if len(psc_files) > 1:
        dfs_sheets = []
        for sheet in psc_files:
            dfs_sheets.append(pd.read_excel(join(path_similarity, sheet)))
        dfs = pd.concat(dfs_sheets, ignore_index=True)
    elif len(psc_files) == 1:
        dfs = pd.read_excel(join(path_similarity, psc_files[0]))
    else:
        dfs = None

    # read collection similarity (every particle to every other particle in every frame)
    if len([f for f in files if f.startswith('collection_similarities_')]) > 0:
        dfcs = pd.read_excel([join(path_similarity, f) for f in files if f.startswith('collection_similarities_')][0])
        if len(dfcs) < 10:
            dfcs = None
    else:
        dfcs = None

    return dfs, dfsf, dfsm, dfas, dfcs


def export_dict_intrinsic_aberrations(dict_intrinsic_aberrations, path_results, unique_id):

    # dataframe of intrinsic aberrations values
    dfai = dict_intrinsic_aberrations['dfai']

    # dataframe of fitted values
    zfit = dict_intrinsic_aberrations['zfit']
    cmfit_cubic = dict_intrinsic_aberrations['cmfit_cubic']
    cmfit_quartic = dict_intrinsic_aberrations['cmfit_quartic']
    dfai_fit = pd.DataFrame(np.vstack([zfit, cmfit_cubic, cmfit_quartic]).T,
                            columns=['zfit', 'cmfit_cubic', 'cmfit_quartic'])

    # dataframe of intrinsic aberrations parameters
    dict_ai_params = {key: val for key, val in dict_intrinsic_aberrations.items() if key in
                      ['zf', 'num_pids', 'num_frames', 'cpopt', 'qpopt']}
    dfai_params = pd.DataFrame.from_dict(dict_ai_params, orient='index', columns=['value'])

    # export
    dfai.to_excel(path_results + '/ia_values_{}.xlsx'.format(unique_id))
    dfai_fit.to_excel(path_results + '/ia_fits_{}.xlsx'.format(unique_id))
    dfai_params.to_excel(path_results + '/ia_params_{}.xlsx'.format(unique_id))


# --------------------------------------------------- END --------------------------------------------------------------


def read_files(read_to, path_name, sort_strings, filetype='.xlsx', subset=None, startswith=None, columns=[],
               dtype=float, print_filenames=True, drop_na=False):
    """
    Notes:
        sort_strings: if sort_strings[1] == filetype, then sort_strings[1] should equal empty string, ''.

    :param path_name:
    :param sort_strings (iterable; e.g. tuple): ['z', 'um']
    :return:
    """
    # read files in directory
    if startswith is not None:
        files = [f for f in os.listdir(path_name) if f.startswith(startswith) and f.endswith(filetype)]
    else:
        files = [f for f in os.listdir(path_name) if f.endswith(filetype)]

    if len(files) == 0:
        raise ValueError("No files found at {}".format(path_name))

    if subset:
        files = files[:subset]

    # sort files and get names
    if sort_strings[1] == '':
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1]+filetype)[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1] + filetype)[0]) for f in files]
    else:
        if print_filenames:
            print(files)
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1])[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1])[0]) for f in files]

    # organize dataframes into list for iteration
    data = {}
    for n, f in zip(names, files):

        if read_to == 'df':
            df = read_dataframe(join(path_name, f), filetype, dtype=dtype)
            data.update({n: df})

        elif read_to == 'dict':
            df = read_dataframe(join(path_name, f), filetype, columns=['parameter', n], dtype=dtype)
            df = df.set_index(keys='parameter')
            data.update(df.to_dict(orient='dict'))

    return data


def read_ground_truth_files(settings_dict):

    # organize dataframes into list for iteration
    data = {}

    for n, settings in settings_dict.items():

        ground_truth_path_name = settings['test_ground_truth_image_path']
        test_basestring = settings['test_base_string']
        test_subset = settings['test_image_subset']
        test_num_images = settings['test_col_number_of_images']
        test_cropping = settings['test_cropping_params']

        if isinstance(test_subset, str):
            test_subset = ast.literal_eval(test_subset)

        # read .txt ground truth files to dictionary
        num_files = int(test_num_images)
        ground_truth_filetype = '.txt'
        ground_truth_sort_strings = [test_basestring, ground_truth_filetype]
        gt_dficts = read_dataframes(ground_truth_path_name, ground_truth_sort_strings, ground_truth_filetype, subset=num_files)
        df_ground_truth = modify.stack_dficts_by_key(gt_dficts)

        # filter according to cropping specs

        if isinstance(test_cropping, str):
            test_cropping = ast.literal_eval(test_cropping)
            df_ground_truth = df_ground_truth[df_ground_truth['x'] > test_cropping['xmin']]
            df_ground_truth = df_ground_truth[df_ground_truth['x'] < test_cropping['xmax']]
            df_ground_truth = df_ground_truth[df_ground_truth['y'] > test_cropping['ymin']]
            df_ground_truth = df_ground_truth[df_ground_truth['y'] < test_cropping['ymax']]

        data.update({n: df_ground_truth})

    return data


def read_dataframe(path_name, filetype, sort_strings=[], columns=[], dtype=float, drop_columns=None):
    """
    :param path_name:
    :param sort_strings:
    :return:
    """
    if filetype == '.xlsx':
        if len(columns) > 0:
            df = pd.read_excel(io=path_name, names=columns, dtype=dtype, skiprows=1)
        else:
            df = pd.read_excel(io=path_name, dtype=dtype)

        if drop_columns:
            df = df.drop(columns=drop_columns)

    elif filetype == '.txt':
        dataset = 'Dataset_I'

        if dataset == 'Dataset_I':
            gt = np.loadtxt(path_name)
            df = pd.DataFrame(gt)
        else:
            df = pd.read_csv(path_name, header=None, sep=" ")

        df.columns = ['x', 'y', 'z', 'p_d']

    return df


def read_excel(path_name, filetype='.xlsx', sort_strings=[], dtype=float):

    df = pd.read_excel(io=path_name, dtype=dtype)

    return df


def export_df_to_excel(df, path_name, include_index=True, index_label='index', filetype='.xlsx', drop_columns=None):

    if drop_columns:
        df = df.drop(columns=drop_columns)

    path_name = path_name + filetype
    df.to_excel(excel_writer=path_name, index=include_index, index_label=index_label)


"""
Deprecated Functions
"""


def read_dataframes(path_name, sort_strings, filetype='.xlsx', subset=None, drop_columns=None):
    """

    ----- DEPRECATED -----

    Notes:
        sort_strings: if sort_strings[1] == filetype, then sort_strings[1] should equal empty string, ''.

    :param path_name:
    :param sort_strings (iterable; e.g. tuple): ['z', 'um']
    :return:
    """
    # read files in directory
    files = [f for f in os.listdir(path_name) if f.endswith(filetype)]

    if subset:
        files = files[:subset]

    # sort files and get names
    if sort_strings[1] == '':
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1]+filetype)[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1] + filetype)[0]) for f in files]
    else:
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1])[0]))
        names = [float(f.split(sort_strings[0])[-1].split(sort_strings[1])[0]) for f in files]

    # organize dataframes into list for iteration
    data = {}
    for n, f in zip(names, files):
        df = read_dataframe(join(path_name, f), filetype, drop_columns=drop_columns)
        data.update({n: df})

    return data