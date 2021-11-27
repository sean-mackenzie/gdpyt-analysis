# gdpyt-analysis: filter
"""
Notes
"""

# imports
import pandas as pd
import numpy as np

# scripts

def dficts_filter(dfictss, keys, values, operations='greaterthan', copy=True):

    if copy:
        dficts = dfictss.copy()
    else:
        dficts = dfictss

    for name, df in dficts.items():
        for key, value, operation in zip(keys, values, operations):

            initial_length = len(df)

            # filter
            if operation == 'greaterthan':
                df = df[df[key] > value]
            elif operation == 'lessthan':
                df = df[df[key] < value]
            else:
                raise ValueError("{} operation not implemented.")

            filtered_length = len(df)

            print("{} rows ({}%) filtered from id {} dataframe.".format(filtered_length-initial_length,
                                                                        np.round(100 * (1 - filtered_length/initial_length), 1),
                                                                        name))

        # update the dictionary
        dficts.update({name: df})

    return dficts


def dficts_dropna(dficts, columns=['z']):

    for item in dficts.items():

        # get name and dataframe (for readability)
        name = item[0]
        df = item[1]

        df = df.dropna(axis=0, subset=columns)

        # update the dictionary
        dficts.update({name: df})

    return dficts