import pandas as pd
import matplotlib.pyplot as plt
import sys

# data_path = sys.argv[1]
data_path = './predtx-per-primitives.txt'


# Read in the frame
data_input = pd.read_csv(data_path, header=None)
print(data_input)

data_input.plot(subplots=True)
plt.show()
