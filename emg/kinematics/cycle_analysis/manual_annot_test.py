from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(10)

plt.plot(t, np.sin(t))

# plt.title("matplotlib.pyplot.ginput()\ function Example", fontweight="bold")

pair_values = plt.ginput(0, 0)
y_values = list(map(lambda x: x[1], pair_values))
print(y_values)


plt.show()
