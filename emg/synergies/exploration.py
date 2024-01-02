# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: lab
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import mplcatppuccin
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# %%
mpl.style.use(["ggplot", "mocha"])

data_path = './wt-m1-non-primitives.txt'
data_input = pd.read_csv(data_path, header=0)
data_input.plot(subplots=True)
plt.show()

data_path_per_test = './wt-m1-per-primitives.txt'
data_input_per_test = pd.read_csv(data_path_per_test, header=0)
data_input_per_test.plot(subplots=True)
plt.show()

# %%
data_path = './predtx-non-primitives.txt'
data_input = pd.read_csv(data_path, header=0)
data_input.plot(subplots=True)
plt.show()

data_path_test = './predtx-non-primitives-test.txt'
data_input_test = pd.read_csv(data_path_test, header=0)
data_input_test.plot(subplots=True)
plt.show()

data_path_per = './predtx-per-primitives.txt'
data_input_per = pd.read_csv(data_path_per, header=0)
data_input_per.plot(subplots=True)
plt.show()

data_path_per_test = './predtx-per-primitives-test.txt'
data_input_per_test = pd.read_csv(data_path_per_test, header=0)
data_input_per_test.plot(subplots=True)
plt.show()
# %%

