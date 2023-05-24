# gdpyt-analysis: filter
"""
Notes
"""

# imports
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# scripts

def get_pid_penetrance(df):
    """
    dict_penetrance = filter.get_pid_penetrance(df)

    dfp = dict_penetrance['dfp']
    penetrance_pids = dict_penetrance['penetrance_pids']

    dict_penetrance = {'max_idd_num_frames': max_num_frames,
                       'penetrance_num_frames': penetrance_num_frames,
                       'penetrance_num_pids': penetrance_num_pids,
                       'penetrance_pids': penetrance_pids,
                       'dfp': df_penetrance,
                       }
    """

    dfcounts = df.groupby('id').count().reset_index()
    max_num_frames = dfcounts.z.max()
    df_penetrance = dfcounts[dfcounts['z'] > max_num_frames * 0.8]

    penetrance_num_frames = max_num_frames * 0.8
    penetrance_num_pids = len(df_penetrance.id.unique())
    penetrance_pids = df_penetrance.id.unique()
    penetrance_pids.sort()

    df_penetrance = df[df.id.isin(penetrance_pids)]

    dict_penetrance = {'max_idd_num_frames': max_num_frames,
                       'penetrance_num_frames': penetrance_num_frames,
                       'penetrance_num_pids': penetrance_num_pids,
                       'penetrance_pids': penetrance_pids,
                       'dfp': df_penetrance,
                       }

    return dict_penetrance

# ---


def dficts_filter(dfictss, keys, values, operations='greaterthan', copy=True, only_keys=None, return_filtered=False):
    if copy:
        dficts = dfictss.copy()
    else:
        dficts = dfictss

    if return_filtered:
        dficts_filtered_out = {}

    for name, df in dficts.items():

        if only_keys:
            if name in only_keys:
                pass
            else:
                continue

        for key, value, operation in zip(keys, values, operations):

            initial_length = len(df)

            # filter
            if operation == 'equalto':
                dff = df[df[key] != value]
                df = df[df[key] == value]

            elif operation == 'notequalto':
                dff = df[df[key] == value]
                df = df[df[key] != value]

            elif operation == 'isin':
                inverse_boolean_series = ~df[key].isin(value)
                dff = df[inverse_boolean_series]
                boolean_series = df[key].isin(value)
                df = df[boolean_series]

            elif operation == 'notin':
                inverse_boolean_series = df[key].isin(value)
                dff = df[inverse_boolean_series]
                boolean_series = ~df[key].isin(value)
                df = df[boolean_series]

            elif operation == 'greaterthan':
                dff = df[df[key] < value]
                df = df[df[key] > value]

            elif operation == 'lessthan':
                dff = df[df[key] > value]
                df = df[df[key] < value]

            elif operation == 'not_between':
                dff = df[(df[key] > value[0]) & (df[key] < value[1])]
                df = df[(df[key] < value[0]) | (df[key] > value[1])]

            elif operation == 'between':
                dff = df[(df[key] < value[0]) | (df[key] > value[1])]
                df = df[(df[key] > value[0]) & (df[key] < value[1])]

            else:
                raise ValueError("{} operation not implemented.")

            filtered_length = len(df)

            print("{} rows ({}%) filtered out of ID {}: PASSING {} {} {}".format(filtered_length - initial_length,
                                                                                 np.round(100 * (
                                                                                             1 - filtered_length / initial_length),
                                                                                          1),
                                                                                 name,
                                                                                 operation,
                                                                                 value,
                                                                                 key,
                                                                                 )
                  )

        # update the dictionary
        dficts.update({name: df})

        # update the filtered out dictionary
        if return_filtered:
            dficts_filtered_out.update({name: dff})

    if return_filtered:
        return dficts, dficts_filtered_out
    else:
        return dficts


def dficts_dropna(dficts, columns=['z']):
    for name, df in dficts.items():
        i_rows = len(df)

        df = df.dropna(axis=0, subset=columns)

        f_rows = len(df)
        print("Dropped {} rows out of {} ({}%) because contained NaNs".format(i_rows - f_rows,
                                                                              i_rows,
                                                                              np.round(f_rows / i_rows * 100, 1)
                                                                              )
              )

        # update the dictionary
        dficts.update({name: df})

    return dficts


def find_nearest_neighbors(df_baseline_locations, pid_location, threshold=10, n_neighbors=1, algorithm='ball_tree'):
    baseline_xy = df_baseline_locations[['x', 'y']]

    particles = df_baseline_locations.id.unique()

    nneigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(baseline_xy.values)
    distances, indices = nneigh.kneighbors(np.array(pid_location))

    for distance, idx, particle in zip(distances, indices, particles):
        if distance < threshold:
            j = 1
        else:
            jj = 1