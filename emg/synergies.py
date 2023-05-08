import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import NMF

def nnmf_factorize(A, k):
    """Non-Negative Matrix Factorization for Muscle Synergy Extraction
    @param A: input matrix
    @param k: number of components (muscle channels)

    @return W: factorized matrix
    @return H: factorized matrix
    @return C: factorized matrix
    """
    nmf = NMF(n_components=k, init='random', random_state=0)
    W = nmf.fit_transform(A)
    H = nmf.components_
    C = np.dot(W, H)
    return W, H, C

# Load Data
raw_emg = pd.read_csv('./raw-emg.csv')


# Global Variables
HPo = 4 # High Pass Filter Order
HPf = 50 # High Pass Filter Frequency (Hz)
LPo = HPo # Low Pass Filter Order
LPf = 30 # Low Pass Filter Frequency (Hz)
points = 200
cycles = np.zeros(points) # To save number of cycles considered

FILT_EMG = np.zeros(len(raw_emg))
list_names = np.vectorize(raw_emg.columns)

# Will be using froom scipy.signal.butter

for ii in range(R):













A = data.to_numpy()

# Defining set of components to use based on nnmf.py
num_components = 4

# Calculating W, H, C for our chosen components
W, H, C = nnmf_factorize(A, num_components)


