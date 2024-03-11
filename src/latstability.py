import csv
import os
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp


def read_all_csv(directory_path):
    data_dict = {}  # Initialize an empty dictionary to store the data

    if not os.path.isdir(directory_path):
        print(f"{directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            data = pd.read_csv(file_path)
            data_dict[filename] = data

    return data_dict


def step_duration(input_dataframe):
    """
    @param: inpo
    """
    # Define the value and column to search for
    value_to_find = 1
    column_to_search = "45 sw onset"
    column_for_time = "Time"
    column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []
    treadmill_speed = []

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iterrows():
        if row[column_to_search] == value_to_find:
            time_value = row[column_for_time]
            time_values.append(time_value)
            treadmill_value = row[column_for_treadmill]
            treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)
    treadmill_speed_array = np.array(treadmill_speed)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)

    # Applying the filter to the arrays
    adjusted_time_differences = time_differences_array[combined_filter]
    adjusted_treadmill_speeds = treadmill_speed_array[combined_filter]
    adj_time_xaxis = np.arange(0, len(adjusted_time_differences))

    # Finding average step cylce for this length
    average_step_difference = np.mean(adjusted_time_differences)

    return adjusted_time_differences, adjusted_treadmill_speeds


def extract_cycles(input_dataframe, swonset_channel="44 sw onset"):
    """Get cycle periods
    @param input_dataframe: spike file input as *.csv
    @param swonset_channel: the channel with swing onsets

    @return step_cycles: the step cycles
    """
    # Define the value and column to search for
    value_to_find = 1
    column_to_search = swonset_channel
    column_for_time = "Time"
    column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []
    treadmill_speed = []

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iterrows():
        if row[column_to_search] == value_to_find:
            time_value = row[column_for_time]
            time_values.append(time_value)
            treadmill_value = row[column_for_treadmill]
            treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)

    # Applying the filter to the arrays
    step_cycles = time_differences_array[combined_filter]

    return step_cycles


def swingon_estim():
    """Full width half maxiumum calculation
    Currently a work in progress
    @param: motor_p_full_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    # Save
    fwhl = []
    number_cycles = len(motor_p_full) // 200

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 1]

        # Find peaks
        peaks, properties = signal.find_peaks(current_primitive, distance=40, width=2)
        max_ind = np.argmax(peaks)
        # min_ind = np.argmin(mcurrent_primitive[0:max_ind])

        # half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind]) / 2

        # print("Manually Calculated", half_width_height)
        max_width = properties["widths"][max_ind]
        fwhl.append(max_width)
        # fwhl_start = properties["left_ips"][max_ind]
        # fwhl_stop = properties["right_ips"][max_ind]
        # half_width_height = properties["width_heights"][max_ind]

        print("Scipy calculated", properties["widths"][max_ind])
        # print(peaks[max_ind])
    fwhl = np.asarray(fwhl)

    return fwhl


def stance_duration(
    input_dataframe, swonset_channel="44 sw onset", swoffset_channel="45 sw offset"
):
    """Stance duration during step cycle
    @param input_dataframe: spike file input as *.csv
    @param swonset_channel: the channel with swing onsets
    @param swoffset_channel: the channel with swing offsets

    @return stance_duration_lengths: How long each stance duration is
    @return stance_duration_timings:
    """

    # Define the value and column to search for
    value_to_find = 1
    stance_begin = swoffset_channel
    # stance_end = swonset_channel
    # column_to_search = swonset_channel
    column_for_time = "Time"
    # column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []

    # Find the first stance phase to start tracking time duration
    first_stance = (
        input_dataframe[stance_begin]
        .loc[input_dataframe[stance_begin] == value_to_find]
        .index[0]
    )

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iloc[first_stance:].iterrows():
        if row[swoffset_channel] == value_to_find:
            # print("swoff found at", row[column_for_time])
            time_value = row[column_for_time]
            time_values.append(time_value)
        elif row[swonset_channel] == value_to_find:
            # print("swon found at", row[column_for_time])
            time_value = row[column_for_time]
            time_values.append(time_value)
            # treadmill_value = row[column_for_treadmill]
            # treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)

    # Applying the filter to the arrays
    stance_duration_lengths = time_differences_array[combined_filter]

    stance_duration_timings = time_values

    return stance_duration_lengths, stance_duration_timings


def spike_slope(input_dataframe, time_constant, comy="37 CoMy (cm)"):

    # Converting time constant to meaningful indices
    time_factor = int(time_constant / 2)

    # Bring in data and convert to numpy array
    comy_values = input_dataframe[comy].tolist()
    comy_values = np.array(comy_values)

    # Remove missing values
    # filter = comy_values[np.logical_not(np.isnan(comy_values))]
    filter = np.isnan(comy_values)
    comy_values = comy_values[~filter]

    # Calculate slope based on spike 2 specifications

    current_slope = 0

    for i in range(len(comy_values)):
        if i <= 5:
            values_to_consider = np.array(comy_values[0:time_factor])
            current_slope = np.mean(values_to_consider)
        else:
            previous_values = np.array(comy_values[i - time_factor : i])
            next_values = np.array(comy_values[i : i + time_factor])
            print(current_slope)


def step_width(
    input_dataframe: pd.DataFrame,
    rl_stance: Union[np.ndarray, list],
    ll_stance: Union[np.ndarray, list],
    rl_y: str,
    ll_y: str,
) -> np.array:

    start_time = time.time()

    input_dataframe_subset = input_dataframe.loc[:, ["Time", rl_y, ll_y]].set_index(
        "Time"
    )

    ll_step_placement = input_dataframe_subset.loc[ll_stance, :][ll_y].values
    rl_step_placement = input_dataframe_subset.loc[rl_stance, :][rl_y].values

    # Dealing with possible unequal amount of recorded swoffsets for each limb
    comparable_steps = 0
    if rl_step_placement.shape[0] >= ll_step_placement.shape[0]:
        comparable_steps = ll_step_placement.shape[0]
    else:
        comparable_steps = rl_step_placement.shape[0]

    step_widths = []

    # Compare step widths for each step
    for i in range(comparable_steps):
        new_width = np.abs(rl_step_placement[i] - ll_step_placement[i])
        step_widths.append(new_width)

    step_widths = np.asarray(step_widths)
    stop_time = time.time()
    print(f"Calculation duration\n{stop_time - start_time}\n")

    return step_widths


def hip_height(input_dataframe, toey="24 toey (cm)", hipy="16 Hipy (cm)"):

    # Bringing in the values for toey and hipy
    toey_values = input_dataframe[toey].tolist()
    hipy_values = input_dataframe[hipy].tolist()
    toey_values = np.array(toey_values)
    hipy_values = np.array(hipy_values)

    # Remove missing values
    toey_values = toey_values[np.logical_not(np.isnan(toey_values))]
    hipy_values = hipy_values[np.logical_not(np.isnan(hipy_values))]

    # Getting lower quartile value of toey as proxy for the ground
    toey_lowerq = np.percentile(toey_values, q=25)
    average_hip_value = np.mean(hipy_values)

    hip_height = average_hip_value - toey_lowerq
    return hip_height


def xcom(input_dataframe, hip_height, comy="37 CoMy (cm)"):

    # Bring in data
    comy_values = input_dataframe[comy].tolist()
    comy_values = np.array(comy_values)
    comy_values = comy_values[np.logical_not(np.isnan(comy_values))]

    # Getting slope of values
    vcom = np.diff(comy_values)

    # Fixing 1 off
    comy_values = np.delete(comy_values, -1)

    # Get xCoM in (cm)
    xcom = comy_values + (vcom / np.sqrt(981 / hip_height))

    print("comy", comy_values.size)
    print("vcom", vcom.size)
    print("xcom", xcom)

    x_axis = np.arange(len(comy_values))

    # Testing output
    rbf = sp.interpolate.Rbf(x_axis, vcom, function="thin_plate", smooth=2)
    xnew = np.linspace(x_axis.min(), x_axis.max(), num=100, endpoint=True)
    ynew = rbf(xnew)

    fig, axs = plt.subplots(4, 1, layout="constrained")
    axs[0].set_title("CoMy")
    axs[0].legend(loc="best")
    axs[0].plot(x_axis, comy_values)
    axs[1].set_title("vCoM")
    axs[1].plot(x_axis, vcom)
    axs[2].set_title("Radial basis funtion interpolation of vCoM")
    axs[2].plot(xnew, ynew)
    axs[0].plot(x_axis, xcom)

    plt.show()

    return xcom


def cycle_period_summary(directory_path):

    trial_list = read_all_csv(directory_path)

    cycle_results = {}
    for key in trial_list:
        cycle_results[key] = None

    # Now, you can access the data from each file like this:
    for filename, data in trial_list.items():
        step_duration_array, treadmill_speed = step_duration(data)
        cycle_results[filename] = np.mean(step_duration_array), np.mean(treadmill_speed)

    # Saving results to csv
    cycle_results_csv = "cycle_analysis.csv"

    with open(cycle_results_csv, "w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row (optional)
        writer.writerow(["Data Point", "Mean", "Standard Deviation"])

        # Write data from the dictionary
        for key, (mean, std_dev) in cycle_results.items():
            writer.writerow([key, mean, std_dev])

    print(f"Data has been saved to {cycle_results_csv}")


# Main Code Body
def main():
    # t0 = time.time()
    # # Test for speed of step width
    # wt4nondf = pd.read_csv("./wt_4_non-perturbation.csv")

    # # Getting stance duration for all 4 limbs
    # lhl_st_lengths, lhl_st_timings = stance_duration(
    #     wt4nondf, swonset_channel="57 lHL swon", swoffset_channel="58 lHL swoff"
    # )
    # lfl_st_lengths, lfl_st_timings = stance_duration(
    #     wt4nondf, swonset_channel="53 lFL swon", swoffset_channel="54 lFL swoff"
    # )
    # rhl_st_lengths, rhl_st_timings = stance_duration(
    #     wt4nondf, swonset_channel="55 rHL swon", swoffset_channel="56 rHL swoff"
    # )
    # rfl_st_lengths, rfl_st_timings = stance_duration(
    #     wt4nondf, swonset_channel="51 rFL swon", swoffset_channel="52 rFL swoff"
    # )

    # # For forelimb
    # fl_step_widths = step_width(
    #     wt4nondf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)"
    # )
    # hl_step_widths = step_width(
    #     wt4nondf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)"
    # )

    # print("Average forelimb width")
    # print(np.mean(fl_step_widths))
    # print("Average hindlimb width")
    # print(np.mean(hl_step_widths))
    # print()

    # t1 = time.time()

    # total = t1 - t0
    # print("Time Elapsed", total)

    # xCom tests
    wt4nondf = pd.read_csv("./wt_4_non-perturbation.csv")
    spike_slope(wt4nondf, 12)


if __name__ == "__main__":
    main()
# %%
