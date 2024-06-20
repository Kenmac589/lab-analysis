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


def apply_fir(data, coeff):
    """applies a simple moving average to your data array.
    The coefficients define the weights for each sample in the input sequence.
    In this case, they are [0.5, 0.5], which means that the output is calculated
    as the weighted average of the last two samples (in this case, `data[n]*coeff[0] + data[n-1]*coeff[1]`).
    """
    filtered = signal.lfilter(coeff, 1, data)
    return filtered

def spike_slope(comy, p):
    """
    :param comy: numpy array of the y coordinate of the center of mass
    :param p: How many 


    """
    data = input_dataframe[comy].values

    n = len(data)
    slope = [0] * n  # initialize with zeros

    for i in range(p, n - p):
        past = data[i - p : i]
        future = data[i + 1 : i + p + 1]

        # calculate means of past and future points
        mean_past = np.mean(past)
        mean_future = np.mean(future)

        # update slope at time i using the calculated means
        slope[i] = (mean_future - mean_past) / 2

    slope = np.array(slope)

    return slope

def cop(fl_y, hl_y):
    return (fl_y + hl_y) / 2


def main():

    # Settings before running
    save_results = False
    figure_title = "Step Cycle for Video lwalk-M1"
    figure_filename = "./lr-walking/lwalk_M1.png"
    step_cycles_filename = "./lr-walking/step_cycles-lwalk-M1.csv"

    # Loading in a dataset
    df, bodyparts, scorer = dlck.load_data(
        "./lr-walking/ldir/M1_01mps_L_walking_tmDLC_resnet_50_CoM-treadmill_to_leftMar2shuffle1_1030000.h5"
    )

    # Grabbing toe marker data
    toe = df[scorer]["toe"]
    lhl = df[scorer]["Mirror lHL"]
    rhl = df[scorer]["Mirror rHL"]
    lfl = df[scorer]["Mirror lFL"]
    rfl = df[scorer]["Mirror rFL"]
    com = df[scorer]["Mirror CoM"]

    # Converting to numpy array
    toe_np = pd.array(toe["x"])
    rfl_np = pd.array(rfl["y"])
    rhl_np = pd.array(rhl["y"])
    lfl_np = pd.array(lfl["y"])
    lhl_np = pd.array(lhl["y"])
    com_np = pd.array(com["y"])

    # Filtering to clean up traces like you would in spike
    toe_med = median_filter(toe_np, 9)
    rfl_med = median_filter(rfl_np, 9)
    rhl_med = median_filter(rhl_np, 9)
    lfl_med = median_filter(lfl_np, 9)
    lhl_med = median_filter(lhl_np, 9)
    com_med = median_filter(com_np, 9)

    # Trying FIR filter
    coeff = np.array([0.1, 0.2, 0.7])
    toe_fir = apply_fir(toe_np, coeff)
    rfl_fir = apply_fir(rfl_np, coeff)
    rhl_fir = apply_fir(rhl_np, coeff)
    lfl_fir = apply_fir(lfl_np, coeff)
    lhl_fir = apply_fir(lhl_np, coeff)
    com_fir = apply_fir(com_np, coeff)

    # Cleaning up selection to region before mouse moves back
    # toe_roi_selection_fil = toe_filtered[0:2550]

    # Center of pressures
    rcop = cop(rfl_med, rhl_med)
    lcop = cop(lfl_med, lhl_med)

    # Calling function for swing estimation
    swing_onset, swing_offset = swing_estimation(toe_med)
    step_cyc_durations = step_cycle_est(toe_med)

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
    filtest_legend = [
        "Original",
        "Median",
        "FIR",
    ]

    fig, axs = plt.subplots(2)
    fig.suptitle(figure_title)

    # Showing results for step cycle timing
    axs[0].set_title("Filter test")
    # axs[0].plot(com_np)
    axs[0].plot(com_med)
    # axs[0].plot(com_fir)
    # axs[0].legend(filtest_legend, loc="best")
    # axs[0].bar(0, np.mean(step_cyc_durations), yerr=np.std(step_cyc_durations), capsize=5)

    # For plotting figure demonstrating how swing estimation was done
    axs[1].set_title("Swing Estimation")
    axs[1].plot(toe_med)
    axs[1].plot(swing_offset, toe_med[swing_offset], "^")
    axs[1].plot(swing_onset, toe_med[swing_onset], "v")
    axs[1].legend(swing_legend, loc="best")

    # Saving Figure in same folder
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(19.8, 10.80)
    plt.show()

    if save_results is True:
        # Saving plot and results
        np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")
        plt.savefig(figure_filename, dpi=300)
        print("Results saved")
    else:
        print("Results not saved")


if __name__ == "__main__":
    main()
