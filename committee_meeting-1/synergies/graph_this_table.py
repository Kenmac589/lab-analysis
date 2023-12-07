import pandas as pd
import matplotlib.pyplot as plt
import sys

data_path = './predtx-non-primitives.txt'


# Read in the frame
data_input = pd.read_csv(data_path, header=0)
print(data_input)

data_input.plot(subplots=True)
plt.show()

# data_path = sys.argv[1]
data_path_test = './predtx-non-primitives-test.txt'


# Read in the frame
data_input_test = pd.read_csv(data_path_test, header=0)
print(data_input)

data_input_test.plot(subplots=True)
plt.show()