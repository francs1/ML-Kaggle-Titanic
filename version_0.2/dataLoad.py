#dataLoad

import os

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from zlib import crc32
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


from constantInit import *


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


#split train data and validation data

##method1:
# np.random.seed(42)
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

##method2:
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]



def loadData(filename='train.csv'):
	csv_path = os.path.join(PROJECT_ROOT_DIR,'input',filename)
	return pd.read_csv(csv_path,index_col=False)



def saveData(submission,filename='gender_submission.csv'):
	csv_path = os.path.join(PROJECT_ROOT_DIR,'output',filename)
	submission.to_csv(csv_path,index=False)
	print('save file in path : ' + csv_path)


