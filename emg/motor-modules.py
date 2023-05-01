import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
data = pd.read_csv("./normalized-emg.csv", header=None)
A = data.to_numpy()


# Ploting Selected number of components
num_components = 4
H, W, C = nnmf_factorize(A, num_components)

# Plotting
plt.figure()
plt.hist(C[0])
plt.show()
