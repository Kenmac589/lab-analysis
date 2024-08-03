import dlc2kinematics as dlck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dlc2kinematics import Visualizer2D
from scipy import sig


# Custom median filter from
def median_filter(arr, k):
    """
    :param arr: input numpy array
    :param k: is the size of the window you want to slide over the array.
    also considered the kernel

    :return : An array of the same length where each element is the median of
    a window centered around the index in the array.
    """
    # Initialize output array
    result = []

    # Iterate over every index in arr
    for i in range(len(arr)):
        if i < (k // 2) or i > len(arr) - (k // 2) - 1:
            # Add a placeholder for the indices before k//2 and after length of array - k//2 - 1
            result.append(np.nan)
        else:
            # Calculate median within window and append to result list
            result.append(np.median(arr[i - (k // 2) : i + (k // 2) + 1]))

    return np.array(result)


def apply_fir(data, coeff):
    """applies a simple moving average to your data array.
       The coefficients define the weights for each sample in the input sequence. In this case, they are [0.5, 0.5], which
    means that the output is calculated as the weighted average of the last two samples (in this case, `data[n]*coeff[0] + data[n-1]*coeff[1]`).
    """
    filtered = sig.lfilter(coeff, 1, data)
    return filtered


# Loading in a dataset
df, bodyparts, scorer = dlck.load_data(
    "./1-6yrRAP-M1-preDTX_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000.h5"
)
config_path = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31_analyzed/config.yaml"

# Interactive Visualizer
# non_perturbation_input = "./1-6yrRAP-M1-preDTX_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000.h5"
# viz = Visualizer2D(config_path, non_perturbation_input)
# viz.view(show_axes=True, show_grid=True)

# Compute for all bodyparts
df_vel = dlck.compute_velocity(df, bodyparts=["all"], filter_window=3, order=1)
df_speed = dlck.compute_speed(df, bodyparts=["all"], filter_window=3, order=1)

# Looking for desired bodyparts
bodyparts_to_check = [
    "mirror_lhl",
    "mirror_rhl",
    "mirror_lfl",
    "mirror_rfl",
    "mirror_com",
]
df_vel_current = dlck.compute_velocity(
    df, bodyparts=bodyparts_to_check, filter_window=7, order=1
)
df_vel_current.head(3)
# dlck.plot_velocity(df[scorer][bodyparts_to_check], df_vel_current)

lhl = df[scorer]["mirror_lhl"]
rhl = df[scorer]["mirror_rhl"]
lfl = df[scorer]["mirror_lfl"]
rfl = df[scorer]["mirror_rfl"]
com = df[scorer]["mirror_com"]

# Converting to numpy array test
lhl_np = pd.array(lhl["x"])
rhl_np = pd.array(rhl["x"])
lfl_np = pd.array(lfl["x"])
rfl_np = pd.array(rfl["x"])
com_np = pd.array(com["y"])

# Simply coefficient example
coeff = np.array([0.1, 0.2, 0.7])
coeff = np.array([0.1, 0.2, 0.7])
fir_lhl = apply_fir(lhl_np, coeff)
fir_rhl = apply_fir(rhl_np, coeff)
fir_lfl = apply_fir(lfl_np, coeff)
fir_rfl = apply_fir(rfl_np, coeff)

lhl_filtered = median_filter(lhl_np, 9)
rhl_filtered = median_filter(rhl_np, 9)
lfl_filtered = median_filter(lfl_np, 9)
rfl_filtered = median_filter(rfl_np, 9)
com_filtered = median_filter(com_np, 9)

plt.plot(fir_lhl)
plt.plot(fir_rhl)
plt.plot(fir_lfl)
plt.plot(fir_rfl)
plt.plot(com_np)

plt.legend(bodyparts_to_check)
plt.show()
