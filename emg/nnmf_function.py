
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

def nnmf_factorize(A, k):
    nmf = NMF(n_components=k, init='random', random_state=0)
    W = nmf.fit_transform(A)
    H = nmf.components_
    C = np.dot(W, H)
    return W, H, C

# Load Data
data = pd.read_excel("./test file.xlsx", header=None)
A = data.to_numpy()

W2, H2, C2 = nnmf_factorize(A, 2)
W3, H3, C3 = nnmf_factorize(A, 3)
W4, H4, C4 = nnmf_factorize(A, 4)
W5, H5, C5 = nnmf_factorize(A, 5)
W6, H6, C6 = nnmf_factorize(A, 6)
W7, H7, C7 = nnmf_factorize(A, 7)

R2All = np.zeros(6)
R2All[0] = np.corrcoef(C2.flatten(), A.flatten())[0,1]**2
R2All[1] = np.corrcoef(C3.flatten(), A.flatten())[0,1]**2
R2All[2] = np.corrcoef(C4.flatten(), A.flatten())[0,1]**2
R2All[3] = np.corrcoef(C5.flatten(), A.flatten())[0,1]**2
R2All[4] = np.corrcoef(C6.flatten(), A.flatten())[0,1]**2
R2All[5] = np.corrcoef(C7.flatten(), A.flatten())[0,1]**2

X = np.array([2, 3, 4, 5, 6, 7])
# plt.plot(X, R2All)
# plt.show()

# Using lineararity to determine number of components
# r0 = np.corrcoef([2, 3], R2All[0:2])[0,1]
# print("r0 = ", r0)
# print()
# r1 = np.corrcoef([2, 3, 4], R2All[0:3])[0,1]
# print("r1 = ", r1)
# print()
# r2 = np.corrcoef([2, 3, 4, 5], R2All[0:4])[0,1]
# print("r2 = ", r2)
# print()
# r3 = np.corrcoef([2, 3, 4, 5, 6], R2All[0:5])[0,1]
# print("r3 = ", r3)
# print()
# r4 = np.corrcoef([2, 3, 4, 5, 6, 7], R2All[0:6])[0,1]
# print("r4 = ", r4)
# print()

for i in range(len(R2All)-1):
    r = np.corrcoef(X[0:i+2], R2All[0:i+2])[0,1]
    print("r =", i+2, ":", r)
    
# Plotting Both Methods for determining number of components
plt.figure()
plt.subplot(1,2,1)
plt.plot(X, R2All)
plt.axhline(y=0.95, color='r', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("$R^2$ of $C^x$ fit to original matrix")
plt.title("")
plt.show()


