import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

df = pd.read_csv("./norm-wt-m1-non.csv")

n_start = 950
n_stop = 1500
gaussian_kernal_sigma = 30
gm = df['GM'].values
gm = gm[n_start:n_stop]
x = np.arange(n_stop - n_start)

gm_smooth = gaussian_filter1d(gm, gaussian_kernal_sigma)
smooth_2nd_derivative = np.gradient(np.gradient(gm_smooth))


inflection_pointss = np.where(np.minimum(0, np.diff(np.sign(smooth_2nd_derivative))))[0]

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(x, gm / np.max(gm), label='raw-data')
ax.plot(x, gm_smooth / np.max(gm_smooth), label='smoothed data')
ax. plot(x, smooth_2nd_derivative / np.max(smooth_2nd_derivative), label='2nd deriv')
for i, infl in enumerate(inflection_pointss, 1):
    plt.axvline(x=infl, color='k')
ax.legend()
plt.show()
