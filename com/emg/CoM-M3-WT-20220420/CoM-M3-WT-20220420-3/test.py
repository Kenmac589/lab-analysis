import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


muscle_recordings = np.random.rand(8, 1000)


model = NMF(n_components=4, init='random', random_state=0)
W = model.fit_transform(muscle_recordings.T)
H = model.components_


weights = np.random.rand(4)
motor_modules = H.T * weights
motor_modules /= np.max(np.abs(motor_modules), axis=0)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
for i in range(4):
    ax[i].hist(motor_modules[:, i])
    ax[i].set_ylim([0, 7])
    ax[i].set_title(f"Synergy {i+1}")
plt.tight_layout()
plt.show()
