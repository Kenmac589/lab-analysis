"""Non-Negative Matrix Factorization for Muscle Synergy Extraction

This program performs Non-Negative Matrix Factorization for determing
the appropriate number of components/muscle channels to use for muscle
synergy extraction.

"""
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
data = pd.read_csv("./norm-wt-m1-non.csv", header=0)
A = data.to_numpy()

# Defining set of components to use
num_components = np.array([2, 3, 4, 5, 6, 7])
R2All = np.zeros(len(num_components))

# Calculating R2 for each component
for i in range(len(R2All)):
    W, H, C = nnmf_factorize(A, num_components[i])
    R2All[i] = np.corrcoef(C.flatten(), A.flatten())[0,1]**2
    print ("$R^2$ =", i+2, ":", R2All[i])

# Calculating correlation coefficient for each component
corrcoef = np.zeros(len(num_components))
for i in range(len(R2All)):
    corrcoef[i] = np.corrcoef(num_components[0:i+2], R2All[0:i+2])[0,1]
    print("r =", i+2, ":", corrcoef[i])

# Plotting Both Methods for determining number of components
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(num_components, R2All)
plt.axhline(y=0.95, color='r', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("$R^2$ of $C^x$ fit to original matrix")
plt.title("Muscle Synergy Determinance by Percentage")
plt.subplot(1, 2, 2)
plt.scatter(num_components, corrcoef)
plt.xlabel("Number of Components")
plt.ylabel("Correlation Coefficient")
plt.title("Muscle Synergy Determinance by Linearity")
plt.show()

# Plotting Both Methods overlapping
plt.plot(num_components, R2All)
plt.axhline(y=0.95, color='r', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("$R^2$ of $C^x$ fit to original matrix")
plt.title("Muscle Synergy Determinance by Percentage")
plt.scatter(num_components, corrcoef)
plt.title("Muscle Synergy Determinance by Linearity and $R^2$")
plt.show()
