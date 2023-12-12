import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_quick_look(file_list):
    for i in range(len(file_list)):
        data_path = file_list[i]
        data_input = pd.read_csv(data_path, header=0)
        data_input.plot(subplots=True)
        plt.show()


# For single uses
# data_path = './predtx-non-primitives.txt'
# data_input = pd.read_csv(data_path, header=0)
# data_input.plot(subplots=True)
# plt.show()

data_list = [
    './predtx-non-primitives.txt',
    './predtx-non-primitives-test.txt',
    './predtx-per-primitives.txt',
    './predtx-per-primitives-test.txt'
]

plot_quick_look(data_list)
