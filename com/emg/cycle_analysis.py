import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cumulative_step_cycles = pd.read_csv('./step_cycle_comparison.csv', header=0)

# Get Data for Column
step_cycles = cumulative_step_cycles.iloc[:, 0].to_numpy()

x_values = np.arange(len(step_cycles))


# plt.bar(cumulative_step_cycles.iloc[:, 0], cumulative_step_cycles.iloc[:, 1])
# plt.bar(x_values, step_cycles) 
# plt.show()
