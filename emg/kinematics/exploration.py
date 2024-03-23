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
import pandas as pd
import scipy as sp

# %%
# mpl.style.use(["ggplot", "mocha"])
# Comparing primitives
df = pd.read_csv("./xCom-test.txt", header=0, delimiter=",")
print(df.head())

# %%
# Plotting
fig, ax1 = plt.subplots()

x = df["Time"]
y1 = df["37b CoMy (cm)"]
y2 = df["v7 xCoM"]

ax2 = ax1.twinx()

ax1.plot(x, y1, "g-", label="CoM position")
ax2.plot(x, y2, "b-", label="xCoM")
fig.suptitle("CoM Position versus xCoM")
fig.legend()
fig.show()

# %%
