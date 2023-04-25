import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# Load Data
data = pd.read_csv("./normalized-emg.csv", header=None)
A = data.to_numpy()


# Ploting Selected number of components

num_components = 7

nmf = NMF(n_components=num_components, init='random', random_state=0)
W = nmf.fit_transform(A)
H = nmf.components_

H_normalized = H / np.sum(H, axis=0)

fig, axs = plt.subplots(num_components, 1, figsize=(6, 3*num_components))
for i in range(num_components):
    axs[i].bar(range(len(H_normalized)), H_normalized[:, i])
    axs[i].set_ylabel('Synergy {}'.format(i+1))

plt.show()
