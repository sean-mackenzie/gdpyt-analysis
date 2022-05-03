# gdpyt-analysis: utils: verify
"""
Notes
"""

# imports
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import modify
import filter


# scripts

def verify_particle_ids_across_sets(df, baseline=None, save_plots=False, save_results=False, save_path=None):
    """
    Compare x, y, and z coordinates of particles of the same ID but from different image sets.

    :param df: should be a stacked dataframe where column 'filename' indicates the image set.
    :param baseline:
    :param save_plots:
    :return:
    """
    df = df.copy()

    pids = df.id.unique()

    drop_cols = [c for c in df.columns if c not in ['id', 'x', 'y', 'z', 'z_corr']]
    df = df.drop(columns=drop_cols)

    fig, ax = plt.subplots()

    data = []
    for pid in pids:
        dfpid = df[df['id'] == pid]
        dfgpid = modify.groupby_stats(dfpid, group_by='id', drop_columns=None)
        data.append(dfgpid.iloc[0].to_numpy())

        # plot low certainty particles
        if dfgpid.iloc[0].x_std > 5 or dfgpid.iloc[0].y_std > 5:
            fig, ax = plot_low_certainty_pids(dfpid, fig, ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    dfv = pd.DataFrame(data, columns=dfgpid.columns)

    if save_plots:
        save_path_verify_tests = join(save_path, 'tests', 'verify')
        plt.savefig(join(save_path_verify_tests, 'low_certainty_pids.png'))
        plt.show()
    plt.close()

    if save_results:
        save_path_verify_tests = join(save_path, 'tests', 'verify')
        dfv.to_excel(join(save_path_verify_tests, 'df_verify.xlsx'))

    return dfv

def plot_low_certainty_pids(dfpid, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots()

    ax.scatter(dfpid.x, dfpid.y, label=dfpid.iloc[0].id, s=1)

    return fig, ax