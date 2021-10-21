# gdpyt-analysis: filter
"""
Notes
"""

# imports
import pandas as pd
import numpy as np

# scripts

def dficts_filter(dficts, keys, values, operations='greaterthan'):

    for name, df in dficts.items():
        for key, value, operation in zip(keys, values, operations):

            # filter
            if operation == 'greaterthan':
                df = df[df[key] > value]
            elif operation == 'lessthan':
                df = df[df[key] < value]
            else:
                raise ValueError("{} operation not implemented.")

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