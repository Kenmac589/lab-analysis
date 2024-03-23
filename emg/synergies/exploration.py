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

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcatppuccin
import numpy as np

# %%
import pandas as pd
import scipy as sp

import motorpyrimitives as mp

# %%
# mpl.style.use(["ggplot", "mocha"])
# Comparing primitives

data_path = "./wt_data/wt_m1_non_primitives.csv"


data_input = pd.read_csv(data_path, header=None)
data_input.plot(subplots=True, title="WT Primitives direct from NMF", legend=False)
plt.savefig("/Users/kenzie_mackinnon/Downloads/nmf_no_filt.png", dpi=300)

motor_p_data = data_input.to_numpy()

motor_p_filtered = sp.ndimage.median_filter(
    motor_p_data, size=2, cval=1, mode="constant"
)
motor_p_df = pd.DataFrame(
    motor_p_filtered,
    columns=(
        "Synergy 1",
        "Synergy 2",
        "Synergy 3",
        "Synergy 4",
        "Synergy 5",
        "Synergy 6",
        "Synergy 7",
    ),
)
motor_p_df.plot(
    subplots=True,
    title="WT Primitives direct from NMF with median filter",
    legend=False,
)
plt.savefig("/Users/kenzie_mackinnon/Downloads/nmf_with_filt.png", dpi=300)

# %%
data_path = "./wt-m1-non-primitives.txt"
data_input = pd.read_csv(data_path, header=0)
data_input.plot(subplots=True, title="WT Primitives cleaned in spike", legend=False)


# %%
data_path_per_test = "./wt-m1-per-primitives.txt"
data_input_per_test = pd.read_csv(data_path_per_test, header=0)
data_input_per_test.plot(subplots=True)
plt.show()

# %%
data_path = "./predtx-non-primitives.txt"
data_input = pd.read_csv(data_path, header=0)
data_input.plot(subplots=True)
plt.show()

data_path_test = "./predtx-non-primitives-test.txt"
data_input_test = pd.read_csv(data_path_test, header=0)
data_input_test.plot(subplots=True)
plt.show()

data_path_per = "./predtx-per-primitives.txt"
data_input_per = pd.read_csv(data_path_per, header=0)
data_input_per.plot(subplots=True)
plt.show()

data_path_per_test = "./predtx-per-primitives-test.txt"
data_input_per_test = pd.read_csv(data_path_per_test, header=0)
data_input_per_test.plot(subplots=True)
plt.show()
# %%
