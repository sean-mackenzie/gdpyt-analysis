# gdpyt-analysis: utils: modify
"""
Notes
"""

# imports
import pandas as pd
import numpy as np

# scripts


def dficts_scale(dficts, columns, multipliers):

    if len(multipliers) != len(columns):
        multipliers = np.ones_like(columns, dtype=float) * multipliers

    for name, df in dficts.items():
        for column, multiplier in zip(columns, multipliers):

            if column == 'index':
                df.index = df.index * multiplier
            else:
                df.loc[:, column] *= multiplier

        # update the dictionary
        dficts.update({name: df})

    return dficts


def dficts_shift(dficts, columns, shifts):
    for name, df in dficts.items():
        for column, shift in zip(columns, shifts):

            if column == 'index':
                df.index = df.index + shift
            else:
                df.loc[:, column] += shift

        # update the dictionary
        dficts.update({name: df})

    return dficts


def dficts_flip(dficts, column):
    for name, df in dficts.items():

        if column == 'index':
            df.index = df.index.max() - df.index
        else:
            df.loc[:, column] = df.loc[:, column].max() - df.loc[:, column]
            """df.loc[:, column] *= -1
            df.loc[:, column] += df.loc[:, column].max()"""

        # update the dictionary
        dficts.update({name: df})

    return dficts

def stack_dficts_by_key(dficts):

    dfs = []

    for name, df in dficts.items():
        df['filename'] = name
        dfs.append(df)

    dfstack = pd.concat(dfs)

    return dfstack