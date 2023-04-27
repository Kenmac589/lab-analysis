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

weights = np.random.rand(4)
motor_modules = C
motor_modules /= np.max(np.abs(motor_modules), axis=0)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
for i in range(4):
    ax[i].hist(motor_modules[:, i])
    ax[i].set_ylim([0, 100])
    ax[i].set_title(f"Synergy {i+1}")
plt.tight_layout()
plt.show()
