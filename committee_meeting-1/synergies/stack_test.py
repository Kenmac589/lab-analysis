import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

ax = sns.kdeplot(np.random.rand(100))
kde_curve = ax.lines[0]
x = kde_curve.get_xdata()
y = kde_curve.get_ydata()
halfmax = y.max() / 2
maxpos = y.argmax()
leftpos = (np.abs(y[:maxpos] - halfmax)).argmin()
rightpos = (np.abs(y[maxpos:] - halfmax)).argmin() + maxpos
fullwidthathalfmax = x[rightpos] - x[leftpos]
ax.hlines(halfmax, x[leftpos], x[rightpos], color='crimson', ls=':')
ax.text(x[maxpos], halfmax, f'{fullwidthathalfmax:.3f}\n', color='crimson', ha='center', va='center')
ax.set_ylim(ymin=0)
plt.show()
