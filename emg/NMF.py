import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# Load Data
data = pd.read_excel("./test file.xlsx")
A = data.to_numpy()

# Number of components
nmf = NMF(n_components=2, init='random', random_state=0)
W2= nmf.fit_transform(A)
H2 = nmf.components_

nmf = NMF(n_components=3, init='random', random_state=0)
W3 = nmf.fit_transform(A)
H3 = nmf.components_

nmf = NMF(n_components=4, init='random', random_state=0)
W4 = nmf.fit_transform(A)
H4 = nmf.components_

nmf = NMF(n_components=5, init='random', random_state=0)
W5 = nmf.fit_transform(A)
H5 = nmf.components_

nmf = NMF(n_components=6, init='random', random_state=0)
W6 = nmf.fit_transform(A)
H6 = nmf.components_

nmf = NMF(n_components=7, init='random', random_state=0)
W7= nmf.fit_transform(A)
H7= nmf.components_

C2 = np.dot(W2, H2)
C3 = np.dot(W3, H3)
C4 = np.dot(W4, H4)
C5 = np.dot(W5, H5)
C6 = np.dot(W6, H6)
C7 = np.dot(W7, H7)

R2All = np.zeros(6)
R2All[0] = np.corrcoef(C2.flatten(), A.flatten())[0,1]**2
R2All[1] = np.corrcoef(C3.flatten(), A.flatten())[0,1]**2
R2All[2] = np.corrcoef(C4.flatten(), A.flatten())[0,1]**2
R2All[3] = np.corrcoef(C5.flatten(), A.flatten())[0,1]**2
R2All[4] = np.corrcoef(C6.flatten(), A.flatten())[0,1]**2
R2All[5] = np.corrcoef(C7.flatten(), A.flatten())[0,1]**2

X = np.array([2, 3, 4, 5, 6, 7])
plt.plot(X, R2All)
plt.show()

# Ploting Selected number of components

num_components = 4

nmf = NMF(n_components=num_components, init='random', random_state=0)
W = nmf.fit_transform(A)
H = nmf.components_

H_normalized = H / np.sum(H, axis=0)

fig, axs = plt.subplots(num_components, 1, figsize=(6, 3*num_components))
for i in range(num_components):
    axs[i].bar(range(len(H_normalized)), H_normalized[:, i])
    axs[i].set_ylabel('Synergy {}'.format(i+1))

plt.show()
