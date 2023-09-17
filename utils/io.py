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

    if len(files) == 0:
        print('No files found: \n Path: {}, \n Method: {}'.format(path_calib_coords, method))

    # files
    file_dfc = [f for f in files if f.startswith('calib_correction_coords')]
    file_dfcpid = [f for f in files if f.startswith('calib_{}_pid_defocus_stats'.format(method))]
    file_dfcpop = [f for f in files if f.startswith('calib_{}_pop_defocus_stats'.format(method))]
    file_dfcstats = [f for f in files if f.startswith('calib_{}_stats'.format(method))]

    # print files that are present
    print("Files found: \n pid_defocus_stats: {}, \n pop_defocus_stats: {}, \n "
          "calib_{}_stats: {}".format(len(file_dfcpid), len(file_dfcpop), method, len(file_dfcstats))
          )

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
    zf = dfcpop.loc['zf_from_peak_int', 'mean']
    c1 = dfcpop.loc['pop_c1', 'mean']
    c2 = dfcpop.loc['pop_c2', 'mean']

    return mag_eff, zf, c1, c2


def read_test_coords(path_test_coords):
    files = [f for f in listdir(path_test_coords) if f.endswith('.xlsx')]

    df = pd.read_excel([join(path_test_coords, f) for f in files if f.startswith('test_coords')][0])

    """
    read test coords with image stats
    
    dfi = pd.read_excel(
        [join(path_test_coords, f) for f in files if f.startswith('test_coord_particle_image_stats')][0])
    if len(dfi) < 100:
        dfi = None
    """
    return df


def read_similarity(path_similarity):
    """
    To run: dfs, dfsf, dfsm, dfas, dfcs = io.read_similarity(path_similarity)

    dfsf: forward self-similarity
    dfsm: middle self-similarity
    dfas: average similarity (per-frame; average of 'collection similarity')
    dfs: particle similarity curves (test)
    dfcs: collection similarity (per-frame)

    :param path_similarity:
    :return:
    """
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


# ----------------------------------------- END SPECIAL IO FUNCTIONS ---------------------------------------------------


def read_files(read_to, path_name, sort_strings, filetype='.xlsx', subset=None, startswith=None, columns=[],
               dtype=float, print_filenames=False, drop_na=False, include=None):
    """
    Notes:
        sort_strings: if sort_strings[1] == filetype, then sort_strings[1] should equal empty string, ''.

    :param path_name:
    :param sort_strings: (iterable; e.g. tuple): ['z', 'um']
    :param include: a list of additional files to include that are outside the subset, e.g. [0, 1, 2]
    :return:
    """
    # read files in directory
    if startswith is not None:
        files = [f for f in os.listdir(path_name) if f.startswith(startswith) and f.endswith(filetype)]
    else:
        files = [f for f in os.listdir(path_name) if f.endswith(filetype)]

    if len(files) == 0:
        raise ValueError("No files found at {}".format(path_name))

    # sort files and get names
    if sort_strings[1] == '':
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1]+filetype)[0]))
        names = [int(f.split(sort_strings[0])[-1].split(sort_strings[1] + filetype)[0]) for f in files]
    else:
        if print_filenames:
            print(files)
        files = sorted(files, key=lambda x: float(x.split(sort_strings[0])[-1].split(sort_strings[1])[0]))
        names = [int(f.split(sort_strings[0])[-1].split(sort_strings[1])[0]) for f in files]

    if subset is None:
        pass
    else:
        new_names, new_files = [], []
        if isinstance(subset, (list, np.ndarray)):

            # seek subset
            for n, f in zip(names, files):
                if subset[0] <= n <= subset[1]:
                    new_names.append(n)
                    new_files.append(f)
                elif f in include:
                    new_names.append(n)
                    new_files.append(f)

        elif isinstance(subset, (int, float)):
            # seek a single file
            new_names, new_files = [], []
            for n, f in zip(names, files):
                if n == subset:
                    new_names.append(n)
                    new_files.append(f)

        else:
            raise ValueError("Subset type not understood. Should be type(list, np.ndarray) or (int, float).")

        # replace 'names' and 'files'
        names = new_names
        files = new_files

        # ---

    # print name/file if single file
    if len(names) == 1:
        print("Reading test id {}: {}".format(names[0], files[0]))

    # organize dataframes into list for iteration
    data = {}
    for n, f in zip(names, files):

        if read_to == 'df':
            if filetype == '.xlsx':
                df = read_dataframe(join(path_name, f), filetype, dtype=dtype)
            elif filetype == '.txt':
                gt = np.loadtxt(join(path_name, f))
                df = pd.DataFrame(gt)
                df.columns = ['x', 'y', 'z', 'p_d']
            else:
                raise ValueError('Filetype {} not one of: .xlsx, .txt'.format(filetype))

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
        test_baseline = settings['test_baseline_image'][:-4] + '.txt'
        test_num_images = settings['test_col_number_of_images']
        test_cropping = settings['test_cropping_params']

        if isinstance(test_subset, str):
            test_subset = ast.literal_eval(test_subset)
        elif isinstance(test_subset, list):
            test_subset = [int(i) for i in test_subset]

        # read .txt ground truth files to dictionary
        ground_truth_filetype = '.txt'
        ground_truth_sort_strings = [test_basestring, ground_truth_filetype]

        gt_dficts = read_files(read_to='df',
                               path_name=ground_truth_path_name,
                               sort_strings=ground_truth_sort_strings,
                               filetype=ground_truth_filetype,
                               subset=test_subset,
                               startswith=None,
                               columns=[],
                               dtype=float,
                               print_filenames=False,
                               drop_na=False,
                               include=[test_baseline],
                               )

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
        gt = np.loadtxt(path_name)
        df = pd.DataFrame(gt)

        # df = pd.read_csv(path_name, header=None, sep=" ")

        df.columns = ['x', 'y', 'z', 'p_d']

    return df


# --------------------------------------------------- EXCEL ------------------------------------------------------------


def read_excel(path_name, filetype='.xlsx', sort_strings=[], dtype=float):

    df = pd.read_excel(io=path_name, dtype=dtype)

    return df


def export_df_to_excel(df, path_name, include_index=True, index_label='index', filetype='.xlsx', drop_columns=None):

    if drop_columns:
        df = df.drop(columns=drop_columns)

    path_name = path_name + filetype
    df.to_excel(excel_writer=path_name, index=include_index, index_label=index_label)


# --------------------------------------------------- TEXT -------------------------------------------------------------


def read_txt_file_to_list(fp, data_type):

    txt_file = open(fp, "r")

    list_strings = txt_file.readlines()

    list_values = []
    for element in list_strings:
        if data_type == 'int':
            list_values.append(int(element.split("\n")[0]))

    txt_file.close()

    return list_values


def write_list_to_txt_file(list_values, filename, directory):

    if not filename.endswith('.txt'):
        filename = filename + '.txt'

    txt_file = open(join(directory, filename), "w")

    for element in list_values:
        txt_file.write(str(element) + "\n")

    txt_file.close()


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
        if isinstance(subset, (list, np.ndarray)):
            pass
        else:
            files = files[:subset]
            raise ValueError('Taking subset of ground truth files this way is dangerous.')

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