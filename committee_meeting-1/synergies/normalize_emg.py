# Test script to troubleshoot normalizing EMG data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import NMF

def nnmf_factorize(A, k):
    """Non-Negative Matrix Factorization for Muscle Synergy Extraction
    @param A: input matrix
    @param k: number of components (muscle channels)

    @return W: motor primitives
    @return H: motor modules
    @return C: factorized matrix
    """
    nmf = NMF(n_components=k, init='random', random_state=0)
    W = nmf.fit_transform(A)
    H = nmf.components_
    C = np.dot(W, H)
    return W, H, C

def normalize_emg(emg_dataframe):
    """Normalize EMG data
    @param emg: EMG data
    @return emg_norm: normalized EMG data
    """

    # Marking the beginning and end of regions
    sw_onsets = emg_dataframe.index[emg_dataframe['45 sw onset'] == 1].tolist()
    sw_offsets = emg_dataframe.index[emg_dataframe['46 sw offset'] == 1].tolist()

    for start_idx in sw_onsets:
        for end_idx in sw_offsets:
            if end_idx > start_idx:
                emg_dataframe.loc[start_idx, 'Start_of_Region'] = 1
                emg_dataframe.loc[end_idx, 'End_of_Region'] = 1

    # Forward fill to propagate the labels within the regions
    emg_dataframe['Start_of_Region'].fillna(method='ffill', inplace=True)
    emg_dataframe['End_of_Region'].fillna(method='bfill', inplace=True)


    return emg_dataframe

# Load Data
non_normalized_data = pd.read_csv("./dtr-M5-preDTX-non-100.csv", header=None)
data = normalize_emg(non_normalized_data)

print(non_normalized_data.head())
print(data.head())



# A = data.to_numpy()


