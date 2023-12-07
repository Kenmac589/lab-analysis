# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

conditions = ["WT", "PreDTX", "PostDTX"]
step_cycle_means = {
    'Non-Perturbation': [18.35, 18.43, 14.98],
    'Perturbation': [38.79, 48.83, 47.50],
}

x = np.arange(len(conditions))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in step_cycle_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=1)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Length (mm)')
ax.set_title('')
ax.set_xticks(x + width, conditions)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 250)

plt.show()
