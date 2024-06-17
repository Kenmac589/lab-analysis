import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dlc2kinematics import Visualizer2D
from scipy import signal

import latstability as ls


def frame_to_time(frame_index):
    # Convert to miliseconds
    frame_mili = frame_index * 2
    # Convert to seconds
    time_seconds = frame_mili / 1000

    return time_seconds


def swing_estimation(foot_cord, width_threshold=40):
    """This approximates swing onset and offset from kinematic data
    :param : Exported channels from spike most importantly the x values for a channel

    :return swing_onset: A list of indices where swing onset occurs
    :return swing_offset: A list of indices where swing offet occurs
    """

    swing_offset, _ = signal.find_peaks(foot_cord, distance=width_threshold)
    swing_onset, _ = signal.find_peaks(-foot_cord, width=width_threshold)

    return swing_onset, swing_offset


def step_cycle_est(foot_cord, width_threshold=40):
    """This approximates swing onset and offset from kinematic data
    :param input_dataframe: Exported channels from spike most importantly the x values for a channel

    :return cycle_durations: A numpy array with the duration of each cycle
    :return average_step: A list of indices where swing offet occurs
    """

    # Calculating swing estimations
    swing_onset, _ = swing_estimation(foot_cord)

    # Converting Output to time in seconds
    time_conversion = np.vectorize(frame_to_time)
    onset_timing = time_conversion(swing_onset)

    cycle_durations = np.array([])
    for i in range(len(onset_timing) - 1):
        time_diff = onset_timing[i + 1] - onset_timing[i]
        cycle_durations = np.append(cycle_durations, time_diff)

    return cycle_durations


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


# Loading in a dataset
df, bodyparts, scorer = dlck.load_data(
    "./xinrui_M-255/2024-06-13_000001DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
)
figure_filename = "./test_kinematics-2024-06-13_000001.png"
figure_title = "Step Cycle for Video 2024-06-13_000001"
step_cycles_filename = "./step_cycles-2024-06-13_000001.csv"

# Grabbing toe marker data
toe = df[scorer]["toe"]

# Converting to numpy array
toe_np = pd.array(toe["x"])

# Filtering to clean up traces like you would in spike
toe_filtered = median_filter(toe_np, 9)
toe_roi_selection = toe_np[0:2550]  # Just to compare to original

# Cleaning up selection to region before mouse moves back
toe_roi_selection_fil = toe_filtered[0:2550]

# Calling function for swing estimation
swing_onset, swing_offset = swing_estimation(toe_filtered)

step_cyc_durations = step_cycle_est(toe_filtered)

# Saving values
np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")


# Calling function for step cycle calculation

# Some of my default plotting parameters I like
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", font_scale=1.5, rc=custom_params)

# Plot Legend
swing_legend = [
    "Limb X cord",
    "Swing offset",
    "Swing onset",
]

fig, axs = plt.subplots(2)
fig.suptitle(figure_title)

# For plotting figure demonstrating how swing estimation was done
axs[0].set_title("Swing Estimation")
axs[0].plot(toe_filtered)
axs[0].plot(swing_offset, toe_filtered[swing_offset], "^")
axs[0].plot(swing_onset, toe_filtered[swing_onset], "v")
axs[0].legend(swing_legend, loc="best")

# Showing results for step cycle timing
axs[1].set_title("Step Cycle Result")
axs[1].bar(0, np.mean(step_cyc_durations), yerr=np.std(step_cyc_durations), capsize=5)


# Saving Figure in same folder
fig = mpl.pyplot.gcf()
fig.set_size_inches(19.8, 10.80)
plt.savefig(figure_filename, dpi=300)
