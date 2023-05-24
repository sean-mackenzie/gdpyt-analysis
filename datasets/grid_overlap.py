# grid overlap datasets: dz; nodz; dz-wide; nodz-wide;

# imports
import numpy as np


# ------------------------------------------------


class DatasetUnpacker(object):
    def __init__(self, dataset, key):

        self.dataset = dataset
        self.key = key

        self.details = None

    def unpack(self):
        str_method, str_dz, str_wide = 'idpt', '', ''
        # save_id = '{}_{}dz{}-overlap'.format(str_method, str_dz, str_wide)

        if self.dataset == 'grid-overlap':
            if self.key in [1, 2, 11, 12, 21]:

                # method specific parameters
                template_size = 37  # SPCT = 37; IDPT = ?

                # dataset details
                max_diameter = 31
                overlap_scaling = 1

                # particle spacing details
                no_dx = 38  # the specified "dx" for the single isolated particle
                idx_no_dx = 0
                i_dx = 5

                # full splits and keys for this dataset
                splits = np.array(
                    [39, 80, 125, 171, 217, 265, 313, 363, 413, 464, 517, 571, 625, 680, 737, 795, 853, 913, 973])
                keys = np.arange(i_dx - 1, i_dx + len(splits)) * overlap_scaling  # center-to-center overlap spacing
                keys[idx_no_dx] = no_dx

                # filters for the splits and keys according to the data
                x_filter_operation = None
                x_filter = None

                # perform filtering on splits and keys
                if x_filter is not None:
                    x_filter_low = splits < x_filter[0]
                    x_filter_high = splits > x_filter[1]
                    splits = splits[x_filter_low + x_filter_high]
                    keys = keys[x_filter_low + x_filter_high]

                # pair columns with keys
                dict_splits_to_keys = {key: value for (key, value) in zip(splits, keys)}

                # additional
                intercolumn_spacing_threshold = 30
                min_length_per_split = 100
                single_column_x = np.mean(splits[:2])  # x-coordinate filter to separate single particle

            else:
                raise ValueError('No dataset found for key {}'.format(self.key))

            # ---

            # format save_id
            if 10 < self.key < 20:
                str_method = 'spct'

            if self.key in [1, 11, 12, 21]:
                str_dz = 'no-'

            save_id = '{}-{}dz{}-overlap'.format(str_method, str_dz, str_wide)

            # ---

            self.details = {'key': self.key,
                            'splits': splits,
                            'keys': keys,
                            'dict_splits_to_keys': dict_splits_to_keys,
                            'intercolumn_spacing_threshold': intercolumn_spacing_threshold,
                            'x_filter': x_filter,
                            'x_filter_operation': x_filter_operation,
                            'min_length_per_split': min_length_per_split,
                            'single_column_dx': no_dx,
                            'single_column_x': single_column_x,
                            'save_id': save_id,
                            'template_size': template_size,
                            'max_diameter': max_diameter,
                            }

        return self.details


# Old Dataset Notes
"""
IMPORTANT NOTES ON DATASETS:

# --- third iteration dz-overlap below

            if self.key in [1, 11]:
                # Notes: max diameter = 23 pixels
                no_dx, idx_no_dx = 24, 0
                i_dx = 4

                # full splits and keys for this dataset
                splits = np.array(
                    [29, 62, 96, 131, 167, 204, 240, 279, 317, 356, 397, 437, 478, 522, 564, 608, 655, 698, 744, 792,
                     838, 888, 936])
                keys = np.arange(i_dx, i_dx + len(splits)) * 0.75  # center-to-center overlap spacing
                keys[idx_no_dx] = no_dx

                # filters for the splits and keys according to the data
                x_filter_operation = 'not_between'
                x_filter = [47, 73]

                # perform filtering on splits and keys
                x_filter_low = splits < x_filter[0]
                x_filter_high = splits > x_filter[1]
                splits = splits[x_filter_low + x_filter_high]
                keys = keys[x_filter_low + x_filter_high]

                # pair columns with keys
                dict_splits_to_keys = {key: value for (key, value) in zip(splits, keys)}

                # additional
                intercolumn_spacing_threshold = 25
                min_length_per_split = 500
                single_column_x = 47
                

# --- second iteration dz-overlap below

            elif self.key in [11, 12]:
                # full splits and keys for this dataset
                splits = np.array(
                    [31, 64, 97, 131, 165, 201, 237, 274, 312, 351, 390, 430, 471, 513, 553, 597, 641, 686, 730,
                     778, 823, 870, 916, 967])
                keys = np.arange(24) * 0.75  # center-to-center overlap spacing

                # filters for the splits and keys according to the data
                x_filter_operation = 'not_between'
                x_filter = [10, 180]

                # perform filtering on splits and keys
                x_filter_low = splits < x_filter[0]
                x_filter_high = splits > x_filter[1]
                splits = splits[x_filter_low + x_filter_high]
                keys = keys[x_filter_low + x_filter_high]

                # pair columns with keys
                dict_splits_to_keys = {key: value for (key, value) in zip(splits, keys)}

                # additional
                intercolumn_spacing_threshold = 30
                min_length_per_split = 500
                single_column_x = 40

            elif self.key in [3, 4, 13, 14]:

                x_filter_operation = None
                x_filter = None

                splits = np.array([31, 75, 128, 183, 241, 297, 359, 419, 484])
                keys = [32]
                keys.extend(np.arange(8) * 1.5 + 21)

                # pair columns with keys
                dict_splits_to_keys = {key: value for (key, value) in zip(splits, keys)}

                # additional
                intercolumn_spacing_threshold = 30
                min_length_per_split = 100
                single_column_x = 40

# --- first iteration dz-overlap below

2. dz overlap dataset using GDPT synthetic particles:
    * Currently, the first column x=110; dx=7.5 is dropped b/c it isn't paired correctly in NearestNeighbors.
    --> In this case: : x_filter = 120
        splits = np.array([245.0, 377.0, 512.0, 644.0, 778.0, 910])
        keys = np.array([2, 3, 4, 5, 6, 7]) * 7.5

    splits = np.array([110.0, 245.0, 377.0, 512.0, 644.0, 778.0, 910])
    keys = np.array([1, 2, 3, 4, 5, 6, 7]) * 7.5 

    h = 70
    z_range = [-45.001, 25.001]

    * for mapping z_adj to test_coords:
        intercolumn_spacing_threshold = 46

"""
# ---