from sklearn.decomposition import NMF
import numpy as np

# create example dataset with 3 dimensions and 100 samples
X = np.random.rand(100, 3)

# apply NMF to the dataset
nmf = NMF(n_components=3)
W = nmf.fit_transform(X)
H = nmf.components_

# plot the components
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(10, 4))

for i, ax in enumerate(axs):
    ax.plot(H[i])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Weight')
    ax.set_title(f'Component {i+1}')

plt.tight_layout()
plt.show()
